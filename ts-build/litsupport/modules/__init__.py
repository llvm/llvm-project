import importlib
import logging
import pkgutil

# Load all modules
modules = dict()
for importer, modname, ispkg in pkgutil.walk_packages(
    path=__path__, prefix=__name__ + "."
):
    module = importlib.import_module(modname)
    if not hasattr(module, "mutatePlan"):
        logging.error("Skipping %s: No mutatePlan function" % modname)
        continue
    assert modname.startswith("litsupport.modules.")
    shortname = modname[len("litsupport.modules.") :]
    modules[shortname] = module
    logging.info("Loaded test module %s" % module.__file__)
