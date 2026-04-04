# For manual usage, not as a part of lit tests. Used for generating fence.ll

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

run_statement = Template(
    """; RUN: llc < %s -march=nvptx64 -mcpu=sm_${sm} -mattr=+ptx${ptx} | FileCheck %s --check-prefix=SM${sm}
; RUN: %if ptxas-sm_${sm} && ptxas-isa-${ptxfp} %{ llc < %s -march=nvptx64 -mcpu=sm_${sm} -mattr=+ptx${ptx} | %ptxas-verify -arch=sm_${sm} %}"""
)

# (sm, ptx)
TESTS = [(30, 50), (70, 60), (90, 87)]

LLVM_SCOPES = ["singlethread", "", "block", "cluster", "device"]

SCOPE_LLVM_TO_PTX = {
    "singlethread": "thread",
    "": "sys",
    "block": "cta",
    "cluster": "cluster",
    "device": "gpu",
}

ORDERINGS = ["acquire", "release", "acq_rel", "seq_cst"]

if __name__ == "__main__":
    with open("fence.ll", "w") as fp:
        for sm, ptx in TESTS:
            print(run_statement.substitute(sm=sm, ptx=ptx, ptxfp=ptx / 10.0), file=fp)
        print(
            "; NOTE: Please do not modify this file manually- instead modify fence.py",
            file=fp,
        )
        for ordering, llvm_scope in product(ORDERINGS, LLVM_SCOPES):
            print(
                fence_func.substitute(
                    llvm_scope=llvm_scope,
                    ptx_scope=SCOPE_LLVM_TO_PTX[llvm_scope],
                    ordering=ordering,
                ),
                file=fp,
            )
