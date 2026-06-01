# Test generator for the cmpxchg-sm*.py lit tests, which import this module and
# call main(). Emits the test IR plus structural FileCheck lines to stdout; not a
# lit test itself (excluded in lit.local.cfg).

import argparse
import sys
from string import Template
from itertools import product

cmpxchg_func = Template(
    """define i$size @${success}_${failure}_i${size}_${addrspace}_${ptx_scope}(ptr${addrspace_cast} %addr, i$size %cmp, i$size %new) {
    %pairold = cmpxchg ptr${addrspace_cast} %addr, i$size %cmp, i$size %new syncscope(\"${llvm_scope}\") $success $failure
    ret i$size %new
}
"""
)

cmpxchg_func_no_scope = Template(
    """define i$size @${success}_${failure}_i${size}_${addrspace}(ptr${addrspace_cast} %addr, i$size %cmp, i$size %new) {
    %pairold = cmpxchg ptr${addrspace_cast} %addr, i$size %cmp, i$size %new $success $failure
    ret i$size %new
}
"""
)

def get_addrspace_cast(addrspace):
    if addrspace == 0:
        return ""
    else:
        return " addrspace({})".format(str(addrspace))


TESTS = [(60, 50), (70, 63), (90, 87)]

LLVM_SCOPES = ["", "block", "cluster", "device"]

SCOPE_LLVM_TO_PTX = {"": "sys", "block": "cta", "cluster": "cluster", "device": "gpu"}

SUCCESS_ORDERINGS = ["monotonic", "acquire", "release", "acq_rel", "seq_cst"]

FAILURE_ORDERINGS = ["monotonic", "acquire", "seq_cst"]

SIZES = [8, 16, 32, 64]

ADDRSPACES = [0, 1, 3]

ADDRSPACE_NUM_TO_ADDRSPACE = {0: "generic", 1: "global", 3: "shared"}

ADDRSPACE_TO_SPACE_DOT = {0: "", 1: ".global", 3: ".shared"}


# Structural check: cmpxchg always lowers to an `atom...cas` at the requested
# scope/space (or a CAS loop, for emulated sub-word sizes). We assert that and
# leave exact register-level output / fence placement to ptxas-verify. The
# success/failure ordering combinations make fence placement awkward to predict,
# so we don't check it here.
def emit_func(out, success, failure, size, addrspace, llvm_scope):
    # llvm_scope=None means no syncscope() in the IR, i.e. the system scope.
    scope = "sys" if llvm_scope is None else SCOPE_LLVM_TO_PTX[llvm_scope]
    space = ADDRSPACE_TO_SPACE_DOT[addrspace]
    space_name = ADDRSPACE_NUM_TO_ADDRSPACE[addrspace]

    if llvm_scope is None:
        name = "{}_{}_i{}_{}".format(success, failure, size, space_name)
        body = cmpxchg_func_no_scope.substitute(
            success=success, failure=failure, size=size,
            addrspace=space_name, addrspace_cast=get_addrspace_cast(addrspace),
        )
    else:
        name = "{}_{}_i{}_{}_{}".format(success, failure, size, space_name, scope)
        body = cmpxchg_func.substitute(
            success=success, failure=failure, size=size,
            addrspace=space_name, addrspace_cast=get_addrspace_cast(addrspace),
            llvm_scope=llvm_scope, ptx_scope=scope,
        )

    print("; CHECK-LABEL: {}(".format(name), file=out)
    print("; CHECK: atom.{{{{.*}}}}{}{}.cas".format(scope, space), file=out)
    print(body, file=out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sm", type=int, required=True)
    args = parser.parse_args()
    sm = args.sm
    out = sys.stdout

    # Our test space is: SIZES X SUCCESS_ORDERINGS X FAILURE_ORDERINGS X ADDRSPACES X LLVM_SCOPES
    # This is very large, so we instead test 3 slices.

    # First slice:  are all orderings correctly supported, with and without emulation loops?
    # set addrspace to global, scope to cta, generate all possible orderings, for all operation sizes
    addrspace, llvm_scope = 1, "block"
    for size, success, failure in product(SIZES, SUCCESS_ORDERINGS, FAILURE_ORDERINGS):
        emit_func(out, success, failure, size, addrspace, llvm_scope)

    # Second slice: Are all scopes correctlly supported, with and without emulation loops?
    # fix addrspace, ordering, generate all possible scopes, for operation sizes i8, i32
    addrspace, success, failure = 1, "acq_rel", "acquire"
    for size in [8, 32]:
        emit_func(out, success, failure, size, addrspace, None)

    for llvm_scope in LLVM_SCOPES:
        if sm < 90 and llvm_scope == "cluster":
            continue
        if llvm_scope == "block":
            # skip (acq_rel, acquire, global, cta)
            continue
        emit_func(out, success, failure, size, addrspace, llvm_scope)

    # Third slice: Are all address spaces correctly supported?
    # fix ordering, scope, generate all possible address spaces, for operation sizes i8, i32
    success, failure, llvm_scope = "acq_rel", "acquire", "block"
    for size, addrspace in product([8, 32], ADDRSPACES):
        if addrspace == 1:
            # skip (acq_rel, acquire, global, cta)
            continue
        emit_func(out, success, failure, size, addrspace, llvm_scope)


if __name__ == "__main__":
    main()
