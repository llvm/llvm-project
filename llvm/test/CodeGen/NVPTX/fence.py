# Test generator for the fence-sm*.py lit tests, which import this module and
# call main(). Emits the test IR plus structural FileCheck lines to stdout; not a
# lit test itself (excluded in lit.local.cfg).

import argparse
import sys
from string import Template
from itertools import product

fence_func = Template(
    """
define void @fence_${ordering}_${ptx_scope}() {
    fence syncscope(\"${llvm_scope}\") ${ordering}
    ret void
}
"""
)

LLVM_SCOPES = ["singlethread", "", "block", "cluster", "device"]

SCOPE_LLVM_TO_PTX = {
    "singlethread": "thread",
    "": "sys",
    "block": "cta",
    "cluster": "cluster",
    "device": "gpu",
}

ORDERINGS = ["acquire", "release", "acq_rel", "seq_cst"]


# A `fence` lowers to exactly one barrier instruction, so we check it precisely.
# The form depends on the PTX ISA version:
#   - sm_60-: legacy `membar.<scope>` with no ordering semantics.
#   - sm_70:  `fence` exists but only `.acq_rel`/`.sc`, and `.cluster` isn't
#             available so it's widened to `.cta`; we pin the scope but wildcard
#             the semantic.
#   - sm_90+: the full `fence.<sem>.<scope>` set; pinned exactly.
# A singlethread fence is a no-op and emits nothing.
def emit_func(out, sm, ordering, llvm_scope):
    scope = SCOPE_LLVM_TO_PTX[llvm_scope]
    print("; CHECK-LABEL: fence_{}_{}(".format(ordering, scope), file=out)
    if scope == "thread":
        # Match the instructions' trailing "."; the function names contain "fence" too.
        print("; CHECK-NOT: fence.", file=out)
        print("; CHECK-NOT: membar.", file=out)
    elif sm < 70:
        print("; CHECK: membar.{{[a-z]+}};", file=out)
    elif sm < 90:
        fence_scope = "cta" if scope == "cluster" else scope
        print("; CHECK: fence.{{[a-z_]+}}.%s;" % fence_scope, file=out)
    else:
        sem = "sc" if ordering == "seq_cst" else ordering
        print("; CHECK: fence.{}.{};".format(sem, scope), file=out)
    print(fence_func.substitute(
        llvm_scope=llvm_scope, ptx_scope=scope, ordering=ordering), file=out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sm", type=int, required=True)
    args = parser.parse_args()
    out = sys.stdout
    for ordering, llvm_scope in product(ORDERINGS, LLVM_SCOPES):
        emit_func(out, args.sm, ordering, llvm_scope)


if __name__ == "__main__":
    main()
