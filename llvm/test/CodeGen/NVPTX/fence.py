# For manual usage, not as a part of lit tests. Used for generating the following tests:
# fence-sm30.ll, fence-sm70.ll, fence-sm90.ll

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
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_${sm} -mattr=+ptx${ptx} | %ptxas-verify %}"""
)

# (sm, ptx)
TESTS = [(30, 50), (70, 60), (90, 87)]

LLVM_SCOPES_NO_CLUSTER = ["", "block", "device"]

SCOPE_LLVM_TO_PTX = {"": "sys", "block": "cta", "cluster": "cluster", "device": "gpu"}

ORDERINGS = ["acquire", "release", "acq_rel", "seq_cst"]

if __name__ == "__main__":
    # non-cluster orderings are supported on SM30, SM70 and SM90
    with open("fence-nocluster.ll", "w") as fp:
        for sm, ptx in TESTS:
            print(run_statement.substitute(sm=sm, ptx=ptx), file=fp)
        for ordering, llvm_scope in product(ORDERINGS, LLVM_SCOPES_NO_CLUSTER):
            print(
                fence_func.substitute(
                    llvm_scope=llvm_scope,
                    ptx_scope=SCOPE_LLVM_TO_PTX[llvm_scope],
                    ordering=ordering,
                ),
                file=fp,
            )

    # cluster ordering only supported on SM90
    with open("fence-cluster.ll", "w") as fp:
        print(run_statement.substitute(sm=90, ptx=87), file=fp)
        for ordering in ORDERINGS:
            print(
                fence_func.substitute(
                    llvm_scope="cluster",
                    ptx_scope=SCOPE_LLVM_TO_PTX["cluster"],
                    ordering=ordering,
                ),
                file=fp,
            )
