"""Shared building blocks for `--param fn=NAMES`-driven lit substitutions.

Used by `lit.llvm.fn_selection` and `lit.llvm.fn_extract` to narrow compilation
to a subset of functions. Kept here so the two helpers stay short and share
parsing + capture-substitution wiring."""

from lit.TestingConfig import SubstituteCaptures


def parse_fn_names(lit_config, param="fn"):
    """Return the comma-separated list passed via `--param <param>=NAMES`,
    or an empty list when the param is absent or empty."""
    val = lit_config.params.get(param)
    if not val:
        return []
    return [n.strip() for n in val.split(",") if n.strip()]


def add_capture_sub(config, pattern, replacement):
    """Append a substitution that preserves regex backreferences in `replacement`."""
    config.substitutions.append((pattern, SubstituteCaptures(replacement)))


def install(config, lit_config):
    """Dispatch `--param fn=NAMES` to the right helper, and ask FileCheck to
    drop CHECKs outside the selected CHECK-LABEL sections.

    `--param fn-pass=1` opts into `lit.llvm.fn_selection` (the select-function
    pass, opt-only); otherwise `lit.llvm.fn_extract` is used (prepends
    llvm-extract, tool-agnostic)."""
    names = parse_fn_names(lit_config)
    if not names:
        return
    # Splice `--filter-label=NAMES` after any FileCheck invocation so the
    # downstream FileCheck only checks the CHECK-LABEL sections we kept.
    add_capture_sub(
        config, r"(\S*FileCheck)\b", r"\1 --filter-label=" + ",".join(names)
    )
    if lit_config.params.get("fn-pass"):
        # from lit.llvm import fn_selection
        # fn_selection.install(config, lit_config)
        pass
    else:
        # from lit.llvm import fn_extract
        # fn_extract.install(config, lit_config)
        pass
