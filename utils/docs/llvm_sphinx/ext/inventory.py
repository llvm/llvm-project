# -*- coding: utf-8 -*-

"""Sphinx builder that writes only `objects.inv`, for hermetic intersphinx.

An inventory depends only on what the read phase learns (documents, titles,
labels), not on whether any cross-project link resolved. So every project can
emit its inventory without rendering, and a second pass can then resolve
`inv:` links against the sibling inventories named by LLVM_SPHINX_INVENTORY_DIR
(see `llvm_sphinx.LLVM_INTERSPHINX_MAPPING`) with no network access:

    sphinx-build -b inventory <srcdir> <outdir>/<project>   # per project
    LLVM_SPHINX_INVENTORY_DIR=<outdir> sphinx-build -b html ...

The write phase is skipped, so this pass never reports unresolved links.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
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
    # MyST resolves `inv:` links while reading and caches the result in the
    # doctree. Sphinx can't see that dependency, so make the inventory contents
    # part of the env-affecting config: changing one re-reads the documents.
    digest = hashlib.sha256()
    # Only projects that load sphinx.ext.intersphinx have a mapping.
    mapping = getattr(config, "intersphinx_mapping", {})
    for name, (_, location) in sorted(mapping.items()):
        if isinstance(location, str) and os.path.exists(location):
            with open(location, "rb") as f:
                digest.update(name.encode() + b"\0" + f.read())
    config.llvm_inventory_stamp = digest.hexdigest()


def setup(app: Sphinx) -> Dict[str, Any]:
    app.add_builder(InventoryBuilder)
    app.add_config_value("llvm_inventory_stamp", "", "env")
    app.connect("config-inited", _configure_inventories)
    return {"version": __version__, "parallel_read_safe": True}


# -----------------------------------------------------------------------------
# Test code only below:
# -----------------------------------------------------------------------------
def _build(srcdir: Path, tmp: Path, name: str, builder: str, mapping: dict) -> str:
    """Build `srcdir` with `builder` and return the Sphinx warnings.

    This runs `sphinx-build` as a subprocess, as the real build does: several
    Sphinx applications in one process warn about re-registered directives. The
    inventory pass is quiet (`-Q`), because MyST warns at read time about every
    link that its (empty) mapping cannot resolve yet.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[2])]
        + env.get("PYTHONPATH", "").split(os.pathsep)
    )
    env["INVENTORY_TEST_MAPPING"] = json.dumps(mapping)
    quiet = ["-Q"] if builder == "inventory" else ["-q"]
    result = subprocess.run(
        [
            sys.executable, "-m", "sphinx", *quiet, "-b", builder,
            "-d", str(tmp / "doctrees" / f"{builder}-{name}"),
            str(srcdir), str(tmp / "out" / f"{builder}-{name}"),
        ],
        env=env, capture_output=True, text=True, check=True,
    )  # fmt: skip
    return result.stderr


def run_tests() -> None:
    sites = {"a": "https://a.test/", "b": "https://b.test/"}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        src = tmp / "src"
        shutil.copytree(Path(__file__).resolve().parent / "inventory_test", src)

        # Pass 1: each project writes its inventory, though the projects link to
        # each other and neither inventory exists yet.
        for name in sites:
            _build(src / name, tmp, name, "inventory", {})
        inventories = {
            name: tmp / "out" / f"inventory-{name}" / "objects.inv" for name in sites
        }
        for name, inventory in inventories.items():
            if not inventory.exists():
                raise AssertionError(f"missing inventory for {name}")
            if list(inventory.parent.glob("*.html")):
                raise AssertionError(
                    f"the inventory builder rendered a page for {name}"
                )

        # Pass 2: with the sibling inventories, circular links resolve, to the
        # site and not to the local file.
        mapping = {n: (sites[n], str(inventories[n])) for n in sites}
        for name, other in (("a", "b"), ("b", "a")):
            warnings = _build(src / name, tmp, name, "html", mapping)
            if warnings:
                raise AssertionError(f"html build of {name} warned:\n{warnings}")
            html = (tmp / "out" / f"html-{name}" / "index.html").read_text()
            if f'href="{sites[other]}index.html#{other}-label"' not in html:
                raise AssertionError(f"link from {name} to {other} not resolved")

        # A changed inventory re-reads the documents that were not edited: the
        # link must warn again rather than keep its previously resolved target.
        (src / "b" / "index.md").write_text("# Project b\n\n(renamed)=\n## B\n")
        _build(src / "b", tmp, "b", "inventory", {})
        warnings = _build(src / "a", tmp, "a", "html", mapping)
        if "b:std:label:b-label" not in warnings:
            raise AssertionError(f"stale link was not reported:\n{warnings}")
