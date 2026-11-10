# Test project "b": links to the other project with an `inv:` link. The test
# supplies the mapping as JSON in the environment, because `-D` cannot take a
# dict.
import json
import os

project = "b"
extensions = ["myst_parser", "sphinx.ext.intersphinx", "llvm_sphinx.ext.inventory"]
intersphinx_mapping = {
    name: tuple(value)
    for name, value in json.loads(
        os.environ.get("INVENTORY_TEST_MAPPING", "{}")
    ).items()
}
