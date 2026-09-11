# -*- coding: utf-8 -*-

"""Check for absolute links to Sphinx documentation."""

from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path
from typing import Collection, Dict, Mapping, Sequence
from urllib.parse import unquote, urljoin, urlsplit

from llvm_sphinx.help import venv_help

try:
    from docutils import nodes
    from sphinx.application import Sphinx
    from sphinx.ext.intersphinx import InventoryAdapter
    from sphinx.util import logging
except ImportError as err:
    print(venv_help(err), file=sys.stderr)
    raise

__version__ = "1.0"

logger = logging.getLogger("llvm_sphinx.ext.absolute_links")


def setup(app: Sphinx) -> Dict[str, object]:
    app.setup_extension("sphinx.ext.intersphinx")
    app.add_config_value("llvm_sphinx_doc_url_prefixes", (), "env", [list, tuple])
    app.connect("doctree-read", check_absolute_doc_links)
    return {
        "version": __version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }


def _url_prefix_parts(prefix: str) -> tuple[str, str] | None:
    try:
        parsed = urlsplit(prefix)
        port = parsed.port
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None
    if port not in (None, 80, 443):
        return None
    path = parsed.path
    if not path.endswith("/"):
        path += "/"
    return parsed.hostname.lower(), path


def _docname_from_url(
    uri: str, prefixes: Sequence[str], found_docs: Collection[str]
) -> str | None:
    try:
        parsed = urlsplit(uri)
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or port not in (None, 80, 443)
    ):
        return None

    hostname = parsed.hostname.lower()
    path = unquote(parsed.path)
    for prefix in prefixes:
        prefix_parts = _url_prefix_parts(prefix)
        if prefix_parts is None:
            continue
        prefix_hostname, prefix_path = prefix_parts
        if hostname != prefix_hostname:
            continue
        if path.rstrip("/") == prefix_path.rstrip("/"):
            relative_path = ""
        elif path.startswith(prefix_path):
            relative_path = path[len(prefix_path) :]
        else:
            continue
        if not relative_path:
            docname = "index"
        elif relative_path.endswith(".html"):
            docname = relative_path[: -len(".html")]
        elif relative_path.endswith("/"):
            docname = relative_path + "index"
        else:
            continue

        if docname in found_docs:
            return docname
    return None


def _intersphinx_base_url(project: str, mapping: object) -> str | None:
    """Return the target URL from a validated intersphinx mapping entry."""
    if not isinstance(mapping, tuple) or len(mapping) != 2:
        return None
    name_or_url, target_or_inventory = mapping
    if name_or_url == project and isinstance(target_or_inventory, tuple):
        base_url = target_or_inventory[0]
    else:
        base_url = name_or_url
    return base_url if isinstance(base_url, str) else None


def _inventory_item_uri(item: object) -> str | None:
    uri = getattr(item, "uri", None)
    if uri is None and isinstance(item, tuple) and len(item) > 2:
        uri = item[2]
    return uri if isinstance(uri, str) else None


def _external_target_from_url(
    uri: str, base_url: str, inventory: Mapping[str, Mapping[str, object]]
) -> tuple[str, str, str] | None:
    docs = inventory.get("std:doc", {})
    docname = _docname_from_url(uri, (base_url,), docs.keys())
    if docname is None:
        return None

    fragment = unquote(urlsplit(uri).fragment)
    if not fragment:
        return docname, "doc", docname

    expected_uri = urljoin(base_url, f"{docname}.html#{fragment}")
    preferred_roles = {"std:label": "ref", "std:cmdoption": "option"}
    for object_type, objects in inventory.items():
        for target, item in objects.items():
            item_uri = _inventory_item_uri(item)
            if item_uri is None:
                continue
            item_url = urlsplit(urljoin(base_url, item_uri))
            expected_url = urlsplit(expected_uri)
            if (
                item_url.hostname,
                item_url.path,
                item_url.fragment,
            ) != (
                expected_url.hostname,
                expected_url.path,
                expected_url.fragment,
            ):
                continue
            role = preferred_roles.get(object_type, "any")
            return docname, role, target
    return None


def check_absolute_doc_links(app: Sphinx, doctree: nodes.document) -> None:
    """Diagnose absolute URLs that name documents known to Sphinx."""
    prefixes = app.config.llvm_sphinx_doc_url_prefixes
    if not prefixes:
        return

    for node in doctree.findall(nodes.reference):
        if node.get("inv_match"):
            continue
        uri = node.get("refuri")
        if not uri:
            continue
        docname = _docname_from_url(uri, prefixes, app.env.found_docs)
        if docname is not None:
            logger.warning(
                "absolute URL points to document %r in this Sphinx project; "
                "use an internal 'doc' or 'ref' role instead: %s",
                docname,
                uri,
                location=node,
                type="llvm_sphinx",
                subtype="absolute-doc-link",
            )
            continue

        inventories = InventoryAdapter(app.env).named_inventory
        for project, mapping in app.config.intersphinx_mapping.items():
            base_url = _intersphinx_base_url(project, mapping)
            if base_url is None:
                continue
            inventory = inventories.get(project, {})
            external_target = _external_target_from_url(uri, base_url, inventory)
            if external_target is None:
                continue
            docname, role, target = external_target
            logger.warning(
                "absolute URL points to document %r in the %r Sphinx project; "
                "use the 'external+%s:%s' intersphinx role with target %r "
                "instead: %s",
                docname,
                project,
                project,
                role,
                target,
                uri,
                location=node,
                type="llvm_sphinx",
                subtype="absolute-doc-link",
            )
            break


# -----------------------------------------------------------------------------
# Test code only below:
# -----------------------------------------------------------------------------
def _build_test_docs() -> str:
    """Build the reST and Markdown test inputs and return Sphinx warnings."""
    srcdir = Path(__file__).resolve().parent / "absolute_links_test"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        external_srcdir = srcdir / "external"
        external_outdir = tmp_path / "external"
        external_app = Sphinx(
            srcdir=external_srcdir,
            confdir=external_srcdir,
            outdir=external_outdir,
            doctreedir=tmp_path / "external-doctrees",
            buildername="html",
            freshenv=True,
            warningiserror=False,
            status=None,
            warning=None,
        )
        external_app.build()

        warnings = io.StringIO()
        app = Sphinx(
            srcdir=srcdir,
            confdir=srcdir,
            outdir=tmp_path / "out",
            doctreedir=tmp_path / "doctrees",
            buildername="html",
            freshenv=True,
            warningiserror=False,
            confoverrides={
                "intersphinx_mapping": {
                    "other": (
                        "https://other.example.test/docs/",
                        str(external_outdir / "objects.inv"),
                    )
                }
            },
            status=None,
            warning=warnings,
        )
        app.build()
        return warnings.getvalue()


def run_tests() -> None:
    warnings = _build_test_docs()
    expected_urls = (
        "https://example.test/docs/target.html#target-section",
        "https://example.test/docs/",
        "http://www.example.test/docs/target.html?view=1#target-section",
        "https://other.example.test/docs/target.html#external-section",
        "https://other.example.test/docs",
    )
    for url in expected_urls:
        if url not in warnings:
            raise AssertionError(f"expected a warning for {url}")
    if warnings.count("absolute URL points to document") != len(expected_urls):
        raise AssertionError(f"unexpected Sphinx warnings:\n{warnings}")
