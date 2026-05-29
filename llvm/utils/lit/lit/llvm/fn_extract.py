"""Prepend `llvm-extract --func=NAME %s -o -` to the first pipeline stage of
every RUN line that reads from `%s`, so subsequent tools (opt/llc/llvm-as/...)
operate on a module narrowed to the named functions and their transitive
dependencies. Tool-agnostic. Driven by `--param fn=NAMES`."""

from lit.llvm.fn_param import add_capture_sub, parse_fn_names


def install(config, lit_config):
    names = parse_fn_names(lit_config)
    if not names:
        return
    extract = "llvm-extract " + " ".join("--func=" + n for n in names) + " %s -o -"
    # Match: optional %dbg(...) marker, then the first pipeline stage up to
    # (but not crossing) the first `|`, ending with `< %s` or bare `%s`.
    # `\s*` immediately before `%s` is consumed so we don't leave a double
    # space after stripping the redirect/positional.
    pattern = r"^(%dbg\([^)]*\)\s*)?([^|]*?)\s*(?:<\s*)?%s(?=\s|$)"
    add_capture_sub(config, pattern, r"\1" + extract + r" | \2")
