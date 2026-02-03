# For manual usage, not as a part of lit tests. Used for generating the following tests:

from string import Template
from itertools import product

TESTS = [(60, 50), (70, 63), (90, 87)]

LLVM_SCOPES = ["", "block", "cluster", "device"]

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

ADDRSPACES = [0, 1, 3]

ADDRSPACE_NUM_TO_ADDRSPACE = {0: "generic", 1: "global", 3: "shared"}

atomicrmw_func = Template(
    """define ${datatype} @${operation}_${ordering}_${datatype}_${addrspace}_${ptx_scope}(ptr${addrspace_cast} %addr, ${datatype} %val) {
        %retval = atomicrmw ${operation} ptr ${addrspace_cast} %addr, ${datatype} %val syncscope(\"${llvm_scope}\") ${ordering} 
        ret $datatype %retval
}
"""
)

run_statement = Template(
    """; RUN: llc < %s -march=nvptx64 -mcpu=sm_${sm} -mattr=+ptx${ptx} | FileCheck %s --check-prefix=SM${sm}
; RUN: %if ptxas-sm_${sm} && ptxas-isa-${ptxfp} %{ llc < %s -march=nvptx64 -mcpu=sm_${sm} -mattr=+ptx${ptx} | %ptxas-verify -arch=sm_${sm} %}
"""
)


def get_addrspace_cast(addrspace):
    if addrspace == 0:
        return ""
    else:
        return " addrspace({})".format(str(addrspace))


if __name__ == "__main__":
    for sm, ptx in TESTS:
        # Slice 1: Keep addrspace, llvm_scope, ordering fixed, generate all possible operations and sizes
        with open("atomicrmw-sm{}.ll".format(str(sm)), "w") as fp:
            print(run_statement.substitute(sm=sm, ptx=ptx, ptxfp=ptx / 10.0), file=fp)
            # Integer operations
            addrspace, llvm_scope, ordering = 1, "block", "acq_rel"
            for operation, datatype in product(
                INTEGER_OPERATIONS, ["i8", "i16", "i32", "i64"]
            ):
                print(
                    atomicrmw_func.substitute(
                        operation=operation,
                        ordering=ordering,
                        datatype=datatype,
                        addrspace=ADDRSPACE_NUM_TO_ADDRSPACE[addrspace],
                        ptx_scope=SCOPE_LLVM_TO_PTX[llvm_scope],
                        llvm_scope=llvm_scope,
                        addrspace_cast=get_addrspace_cast(addrspace),
                    ),
                    file=fp,
                )

            # Floating point add
            for datatype, operation in product(
                ["float", "double", "half", "bfloat"], FLOATING_POINT_OPERATIONS
            ):
                print(
                    atomicrmw_func.substitute(
                        operation=operation,
                        ordering=ordering,
                        datatype=datatype,
                        addrspace=ADDRSPACE_NUM_TO_ADDRSPACE[addrspace],
                        ptx_scope=SCOPE_LLVM_TO_PTX[llvm_scope],
                        llvm_scope=llvm_scope,
                        addrspace_cast=get_addrspace_cast(addrspace),
                    ),
                    file=fp,
                )

            # Slice 2: Keep addrspace, llvm_scope fixed, and generate all possible orderings for operations add and nand.
            # add is natively supported for larger bitwidths, while nand is emulated always
            addrspace, llvm_scope = 1, "block"
            for operation, datatype, ordering in product(
                ["add", "nand"], ["i8", "i32"], ORDERINGS
            ):
                if addrspace == 1 and llvm_scope == "block" and ordering == "acq_rel":
                    # These are a part of Slice 1
                    continue
                print(
                    atomicrmw_func.substitute(
                        operation=operation,
                        ordering=ordering,
                        datatype=datatype,
                        addrspace=ADDRSPACE_NUM_TO_ADDRSPACE[addrspace],
                        addrspace_cast=get_addrspace_cast(addrspace),
                        ptx_scope=SCOPE_LLVM_TO_PTX[llvm_scope],
                        llvm_scope=llvm_scope,
                    ),
                    file=fp,
                )
