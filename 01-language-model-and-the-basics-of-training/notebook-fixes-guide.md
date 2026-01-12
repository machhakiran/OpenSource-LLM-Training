MANUAL FIX INSTRUCTIONS FOR NOTEBOOK
=====================================

Make these 3 changes in your Jupyter notebook:

FIX 1: Step 4 - Add this line after loading dataset
----------------------------------------------------
After: dataset = load_dataset("BEE-spoke-data/fineweb-literature-100k")
ADD:   dataset["train"] = dataset["train"].select(range(1000))


FIX 2: Step 9 - Change optimizer import
----------------------------------------
REMOVE: from transformers import AdamW
CHANGE: optimizer = AdamW(model.parameters(), lr=learning_rate)
TO:     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


FIX 3: Step 8 - Auto-detect CPU/GPU
------------------------------------
CHANGE: device = torch.device("cuda")
        model.cuda()
        torch.cuda.manual_seed_all(seed_val)

TO:     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_val)
