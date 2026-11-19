# -*- coding: utf-8 -*-

"""Sphinx builder that writes only `objects.inv`, for hermetic intersphinx.

An inventory depends only on what the read phase learns (documents, titles,
labels), not on whether any cross-project link resolved. So every project can
emit its inventory without rendering, and a second pass can then resolve
`inv:` links against the sibling inventories in a directory, with no network
access. The `llvm_inventory_dir` config value names the directory, which holds
<project>/objects.inv (see `llvm_sphinx.LLVM_INTERSPHINX_MAPPING`):

    sphinx-build -b inventory <srcdir> <dir>/<project>   # per project
    sphinx-build -b html -D llvm_inventory_dir=<dir> ...

The write phase is skipped, so this pass never reports unresolved links.
"""

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, Set

from llvm_sphinx.help import venv_help

try:
    from sphinx.application import Sphinx
    from sphinx.builders.html import StandaloneHTMLBuilder
    from sphinx.config import Config
except ImportError as err:
    print(venv_help(err), file=sys.stderr)
    raise

__version__ = "1.0"


class InventoryBuilder(StandaloneHTMLBuilder):
    """Read all sources, then dump `objects.inv` and nothing else."""

    name = "inventory"
    epilog = "The inventory is in %(outdir)s/objects.inv."
    allow_parallel = False

    def init(self) -> None:
        # No inventories exist yet, so do not try to load (or warn about) them.
        # intersphinx reads the mapping in `builder-inited`, after this runs.
        self.config.intersphinx_mapping = {}
        super().init()

    def get_outdated_docs(self) -> str:
        # Nothing is written per document; skip the per-document stat calls.
        return "all documents"

    def prepare_writing(self, docnames: Set[str]) -> None:
        pass

    def copy_assets(self) -> None:
        pass

    def write_documents(self, docnames: Set[str]) -> None:
        pass

    def finish(self) -> None:
        self.dump_inventory()
        # Some themes (Furo) write into _static when the build finishes, without
        # checking that copy_assets created it.
        (Path(self.outdir) / "_static").mkdir(exist_ok=True)


def _configure_inventories(app: Sphinx, config: Config) -> None:
    # Only projects that load sphinx.ext.intersphinx have a mapping.
    mapping = dict(getattr(config, "intersphinx_mapping", {}))
    if config.llvm_inventory_dir:
        # Read each inventory from the directory instead of the project's site.
        # Links still point at the site. This runs before intersphinx
        # validates the mapping.
        mapping = {
            name: (
                url,
                os.path.join(config.llvm_inventory_dir, name, "objects.inv"),
            )
            for name, (url, _) in mapping.items()
        }
        config.intersphinx_mapping = mapping

    # MyST resolves `inv:` links while reading and caches the result in the
    # doctree. Sphinx can't see that dependency, so make the inventory contents
    # part of the env-affecting config: changing one re-reads the documents.
    digest = hashlib.sha256()
    for name, (_, location) in sorted(mapping.items()):
        if isinstance(location, str) and os.path.exists(location):
            with open(location, "rb") as f:
                digest.update(name.encode() + b"\0" + f.read())
    config.llvm_inventory_stamp = digest.hexdigest()


def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_builder(InventoryBuilder)
    # `-D llvm_inventory_dir=<dir>` (LLVM_SPHINX_INVENTORY_DIR in CMake) makes a
    # build read <dir>/<project>/objects.inv instead of fetching the inventories.
    app.add_config_value("llvm_inventory_dir", "", "")
    app.add_config_value("llvm_inventory_stamp", "", "env")
    app.connect("config-inited", _configure_inventories)
    return {"version": __version__, "parallel_read_safe": True}
