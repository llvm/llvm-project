"""CIR check generation for update_cc_test_checks.py.

This file does two things:
  1. It writes CHECK lines for RUN lines that use -emit-cir.
  2. With --body-prefixes, it lets the CIR and LLVM RUN lines share one set of
     CHECK lines for each function (the "body"), so they are not written twice.

Example with --body-prefixes LLVM,CIR:

  // C-LABEL: @test_f(                <- only checks the function name (C)
  // CPP-LABEL: @_Z6test_fv(          <- only checks the function name (C++)
  // CIR:  ... cir.call_llvm_intrinsic "aarch64.sve.frecpe.x" ...
  // LLVM: ... call ... @llvm.aarch64.sve.frecpe.x.nxv8f16(...)

Shared bodies are checked less strictly than normal checks:
  - They use plain CHECK lines, not CHECK-NEXT.
  - They have no checks for globals.
  - They keep the function arguments, but not the return type or attributes.
"""

from __future__ import print_function

import collections
import functools
import os
import re
import subprocess
import sys

from . import common

# The signature is deliberately not captured: it mixes in attributes and debug
# locations that can't be stripped from a check line matching the raw output.
CIR_FUNCTION_RE = re.compile(
    r"^(?P<indent> *)cir\.func\s+(?P<attrs>)(?P<funcdef_attrs_and_ret>[^@\n]*)"
    r"@(?P<func>[\w.$-]+)(?P<args_and_sig>\()[^\n]*\{\n(?P<body>.*?)^(?P=indent)\}",
    flags=(re.M | re.S),
)

SCRUB_CIR_LOC_RE = re.compile(r"[ \t]+loc\((?:[^()\n]|\([^()\n]*\))*\)")


def scrub_cir_body(body):
    # Debug locations are noise, and they trail the op so the check still matches.
    body = SCRUB_CIR_LOC_RE.sub("", body)
    # "[%0" would turn into "[[[TMP0:%.*]]", which FileCheck can't parse.
    body = body.replace("[%", "{{\\[}}%")
    return common.scrub_body(body)


def make_cir_generalizer(version):
    # CIR text is MLIR: only SSA values are unstable. Types (!cir.ptr) and
    # attributes (#cir.int) look like LLVM IR metadata and attribute groups but
    # must be matched literally.
    values = [common.NamelessValue(r"TMP", "%", r"%", r"[\w$.-]+?", None)]
    prefix = r"(\s*)"
    suffix = r"([,:\s\(\)\[\]\{\}]|\Z)"

    return common.GeneralizerInfo(
        version,
        common.GeneralizerInfo.MODE_IR,
        values,
        prefix,
        suffix,
        func_keyword="cir.func",
    )


def split_list(value):
    return [v for v in value.split(",") if v]


def emits_cir(clang_args):
    return "-emit-cir" in clang_args


@functools.lru_cache(maxsize=None)
def clang_has_cir(clang):
    """Whether `clang` was built with ClangIR support."""
    result = subprocess.run(
        [clang, "-cc1", "-fclangir", "-emit-cir", "-x", "c", "-o", os.devnull, "-"],
        input="",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        universal_newlines=True,
    )
    return result.returncode == 0


def check_clang_has_cir(clang, run_list, path):
    """Error out if RUN lines use -fclangir but `clang` has no CIR support.

    Without CIR, -fclangir -emit-llvm silently falls back to classic codegen,
    which would generate classic checks for the CIR RUN lines.
    """
    if not any("-fclangir" in i[1] for i in run_list):
        return
    if not clang_has_cir(clang):
        common.error(
            "RUN lines use -fclangir, but {} was built without CIR (build it "
            "with -DCLANG_ENABLE_CIR=ON)".format(clang),
            test_file=path,
        )
        sys.exit(1)


def add_arguments(parser):
    parser.add_argument(
        "--body-prefixes",
        default="",
        help="Comma-separated check prefixes that hold function bodies shared "
        "across RUN lines (e.g. LLVM,CIR). Every other prefix only gets a -LABEL "
        "line with the function name, so that C and C++ can share one body. "
        "Shared bodies use plain CHECK lines, no -NEXT, no global checks, and "
        "no return type or attributes.",
    )


def select_run_lines(args, path, run_list):
    """Apply --body-prefixes to the RUN lines of a test.

    Returns the RUN lines to process and the body prefixes in effect. May change
    `args`, which holds the options of this one test.
    """
    filecheck_run_list = [i for i in run_list if i[0]]
    body_prefixes = split_list(args.body_prefixes)
    if body_prefixes:
        # A prefix can be shared between formats, but its globals can't.
        args.check_globals = "none"
    if any(all(p in body_prefixes for p in i[0]) for i in filecheck_run_list):
        # Without a label prefix there is nothing to share bodies under.
        # Skipping these RUN lines would leave their checks stale.
        common.error(
            "a RUN line has only body prefixes; add a label prefix (e.g. C) "
            "to every RUN line that uses --body-prefixes",
            test_file=path,
        )
        sys.exit(1)
    if body_prefixes and any(
        not any(p in body_prefixes for p in i[0]) for i in filecheck_run_list
    ):
        # Such a RUN line would only get -LABEL lines: its body goes unchecked.
        common.error(
            "a RUN line has no body prefix; add one of {} to every RUN line "
            "when using --body-prefixes".format(body_prefixes),
            test_file=path,
        )
        sys.exit(1)
    return run_list, body_prefixes


def validate_output_formats(ir_run_list, cir_run_list, body_prefixes):
    """A prefix can't hold both LLVM IR and CIR bodies: they never match."""
    ir_prefixes = {p for run in ir_run_list for p in run[0]}
    cir_prefixes = {p for run in cir_run_list for p in run[0]}
    if not cir_run_list:
        return
    message = None
    shared = ir_prefixes & cir_prefixes
    if not body_prefixes and shared:
        message = "prefixes {} are used for both LLVM IR and CIR; use separate "
        message += "prefixes or --body-prefixes"
        message = message.format(sorted(shared))
    elif shared & set(body_prefixes):
        message = "body prefixes {} are used for both LLVM IR and CIR".format(
            sorted(shared & set(body_prefixes))
        )
    if message:
        common.error(message)
        sys.exit(1)


def add_split_checks(
    output_lines, run_list, names, body_prefixes, formats, global_vars_seen_dict
):
    """Emit one -LABEL per non-body prefix, followed by the shared bodies.

    `names` are the mangled names of one source function, and `formats` maps a
    RUN line's clang args to its (func_dict, ginfo). Checks can't use -NEXT, the
    RUN lines sharing a body may differ in lines that are not checked.
    """
    label_names = collections.defaultdict(set)
    body_owners = {}
    for run in run_list:
        func_dict, ginfo = formats(run[1])
        for prefix in run[0]:
            present = names & func_dict[prefix].keys()
            if prefix in body_prefixes:
                body_owners[prefix] = (func_dict, ginfo)
            else:
                label_names[prefix] |= present

    if any(len(found) > 1 for found in label_names.values()):
        common.warn("Skipping functions with several names per prefix: " + str(names))
        return []

    printed = []
    for prefix, found in label_names.items():
        for name in found:
            output_lines.append("// {}-LABEL: @{}(".format(prefix, name))
            printed.append(prefix)

    for prefix, (func_dict, ginfo) in body_owners.items():
        bodies = [func_dict[prefix][n] for n in names if n in func_dict[prefix]]
        if not bodies:
            continue
        first = bodies[0]
        if not all(
            b and str(b) == str(first) and b.args_and_sig == first.args_and_sig
            for b in bodies
        ):
            # The RUN lines sharing this prefix disagree (e.g. classic codegen
            # and CIR lowering differ). Skipping the body would hide exactly the
            # divergence the shared prefix is meant to catch.
            common.error(
                "prefix {} has conflicting bodies for {}; use a separate prefix "
                "for the RUN lines that differ".format(prefix, sorted(names))
            )
            sys.exit(1)

        vars_seen = {}
        global_vars_seen = global_vars_seen_dict.setdefault(prefix, {})

        def generalize(lines):
            return common.generalize_check_lines(
                lines,
                ginfo,
                vars_seen,
                global_vars_seen,
                original_check_lines=[],
                no_meta_details=ginfo.no_meta_details(),
            )

        # Keep the arguments, they are the operands of the checked operations.
        # Everything after them is not what these checks are about.
        signature = first.args_and_sig
        arguments = signature[1 : signature.rfind(")") + 1]
        if arguments not in ("", ")"):
            line = generalize([arguments])[0]
            output_lines.append("// {}-SAME: {}".format(prefix, line))
        for line in generalize(str(first).splitlines()):
            if line.strip():
                output_lines.append("// {}:  {}".format(prefix, line))
        printed.append(prefix)

    output_lines.append("//")
    return printed
