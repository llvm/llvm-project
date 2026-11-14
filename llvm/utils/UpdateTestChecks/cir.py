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
import re
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
        # Without a label prefix there is nothing to share bodies under. Generate
        # LLVM IR checks only. CIR prefixes then stay out of the prefix set, which
        # leaves any existing CIR checks in the test untouched.
        common.warn(
            "A RUN line has no label prefix for --body-prefixes: "
            "generating LLVM IR checks only and skipping CIR RUN lines",
            test_file=path,
        )
        run_list = [i for i in run_list if not emits_cir(i[1])]
        args.body_prefixes = ""
        body_prefixes = []
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
            common.warn(
                "Prefix {} has conflicting bodies for {}".format(prefix, sorted(names))
            )
            continue

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
