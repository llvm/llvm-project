# Test generator for the atomicrmw-sm*.py lit tests, which import this module and
# call main(). Emits the test IR plus structural FileCheck lines to stdout; not a
# lit test itself (excluded in lit.local.cfg).

import argparse
import sys
from string import Template
from itertools import product

TEST_SM_ARCH_PAIRS = [(60, 50), (70, 63), (90, 87)]

SCOPE_LLVM_TO_PTX = {"": "sys", "block": "cta", "cluster": "cluster", "device": "gpu"}

ORDERINGS = ["monotonic", "acquire", "release", "acq_rel", "seq_cst"]

INTEGER_OPERATIONS = [
    "xchg",
    "add",
    "sub",
    "and",
    "nand",
    "or",
    "xor",
    "max",
    "min",
    "umax",
    "umin",
    "uinc_wrap",
    "udec_wrap",
    "usub_cond",
    "usub_sat",
]

FLOATING_POINT_OPERATIONS = ["fadd", "fsub", "fmin", "fmax", "fminimum", "fmaximum"]

ADDRSPACE_NUM_TO_ADDRSPACE = {0: "generic", 1: "global", 3: "shared"}

atomicrmw_func = Template(
    """define ${datatype} @${operation}_${ordering}_${datatype}_${addrspace}_${ptx_scope}(ptr${addrspace_cast} %addr, ${datatype} %val) {
        %retval = atomicrmw ${operation} ptr ${addrspace_cast} %addr, ${datatype} %val syncscope(\"${llvm_scope}\") ${ordering} 
        ret $datatype %retval
}
"""
)

ADDRSPACE_TO_SPACE_DOT = {0: "", 1: ".global", 3: ".shared"}


def get_addrspace_cast(addrspace):
    if addrspace == 0:
        return ""
    else:
        return " addrspace({})".format(str(addrspace))


# Structural checks. These deliberately don't pin the exact PTX (that would be a
# golden snapshot we'd have to check in); they assert the atomic lowers to an
# `atom` at the right scope/space and that the memory ordering produces the right
# fence. PTX < ISA 6.0 (sm_60) has no ordering-qualified atoms or `fence`: it uses
# `membar` and an unordered `atom`, so the checks vary by `scoped`.
def emit_func(out, scoped, operation, ordering, datatype, addrspace, llvm_scope):
    scope = SCOPE_LLVM_TO_PTX[llvm_scope]
    space = ADDRSPACE_TO_SPACE_DOT[addrspace]
    fenceword = "fence" if scoped else "membar"
    atom = "; CHECK: atom.{{{{.*}}}}{}{}{{{{.*}}}};".format(scope, space)

    print("; CHECK-LABEL: {}_{}_{}_{}_{}(".format(
        operation, ordering, datatype, ADDRSPACE_NUM_TO_ADDRSPACE[addrspace], scope), file=out)
    if ordering == "monotonic":
        print("; CHECK-NOT: {}".format(fenceword), file=out)
        if scoped:
            print("; CHECK: atom.relaxed.{}{}{{{{.*}}}};".format(scope, space), file=out)
        else:
            print(atom, file=out)
        print("; CHECK-NOT: {}".format(fenceword), file=out)
    elif ordering == "seq_cst":
        if scoped:
            print("; CHECK: fence.sc.{};".format(scope), file=out)
        else:
            print("; CHECK: membar.{};".format(scope), file=out)
        print(atom, file=out)
    else:
        print(atom, file=out)
    print(atomicrmw_func.substitute(
        operation=operation,
        ordering=ordering,
        datatype=datatype,
        addrspace=ADDRSPACE_NUM_TO_ADDRSPACE[addrspace],
        ptx_scope=scope,
        llvm_scope=llvm_scope,
        addrspace_cast=get_addrspace_cast(addrspace),
    ), file=out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sm", type=int, required=True)
    args = parser.parse_args()
    scoped = args.sm >= 70
    out = sys.stdout

    # Slice 1: Keep addrspace, llvm_scope, ordering fixed, generate all possible operations and sizes
    addrspace, llvm_scope, ordering = 1, "block", "acq_rel"
    for operation, datatype in product(INTEGER_OPERATIONS, ["i8", "i16", "i32", "i64"]):
        emit_func(out, scoped, operation, ordering, datatype, addrspace, llvm_scope)
    # Floating point
    for datatype, operation in product(
        ["float", "double", "half", "bfloat"], FLOATING_POINT_OPERATIONS
    ):
        emit_func(out, scoped, operation, ordering, datatype, addrspace, llvm_scope)

    # Slice 2: Keep addrspace, llvm_scope fixed, and generate all possible orderings for operations add and nand.
    # add is natively supported for larger bitwidths, while nand is emulated always
    addrspace, llvm_scope = 1, "block"
    for operation, datatype, ordering in product(["add", "nand"], ["i8", "i32"], ORDERINGS):
        if addrspace == 1 and llvm_scope == "block" and ordering == "acq_rel":
            # These are a part of Slice 1
            continue
        emit_func(out, scoped, operation, ordering, datatype, addrspace, llvm_scope)


if __name__ == "__main__":
    main()
