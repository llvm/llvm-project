# For manual usage, not as a part of lit tests. Used for generating the following tests:
# cmpxchg-sm30.ll, cmpxchg-sm70.ll, cmpxchg-sm90.ll

from string import Template
from itertools import product

cmpxchg_func = Template(
    """define i$size @${success}_${failure}_i${size}_${addrspace}(ptr${addrspace_cast} %addr, i$size %cmp, i$size %new) {
    %pairold = cmpxchg ptr${addrspace_cast} %addr, i$size %cmp, i$size %new $success $failure
    ret i$size %new
}
"""
)

run_statement = Template(
    """; RUN: llc < %s -march=nvptx64 -mcpu=sm_${sm} -mattr=+ptx${ptx} | FileCheck %s --check-prefix=SM${sm}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_${sm} -mattr=+ptx${ptx} | %ptxas-verify -arch=sm_${sm} %}
"""
)

TESTS = [(60, 50), (70, 63), (90, 87)]

LLVM_SCOPES = ["", "block", "cluster", "device"]

SCOPE_LLVM_TO_PTX = {"": "sys", "block": "cta", "cluster": "cluster", "device": "gpu"}

SUCCESS_ORDERINGS = ["monotonic", "acquire", "release", "acq_rel", "seq_cst"]

FAILURE_ORDERINGS = ["monotonic", "acquire", "seq_cst"]

SIZES = [8, 16, 32, 64]

ADDRSPACES = [0, 1, 3]

ADDRSPACE_NUM_TO_ADDRSPACE = {0: "generic", 1: "global", 3: "shared"}

if __name__ == "__main__":
    for sm, ptx in TESTS:
        with open("cmpxchg-sm{}.ll".format(str(sm)), "w") as fp:
            print(run_statement.substitute(sm=sm, ptx=ptx), file=fp)
            for size, success, failure, addrspace in product(
                SIZES, SUCCESS_ORDERINGS, FAILURE_ORDERINGS, ADDRSPACES
            ):
                if addrspace == 0:
                    addrspace_cast = ""
                else:
                    addrspace_cast = " addrspace({})".format(str(addrspace))
                print(
                    cmpxchg_func.substitute(
                        success=success,
                        failure=failure,
                        size=size,
                        addrspace=ADDRSPACE_NUM_TO_ADDRSPACE[addrspace],
                        addrspace_cast=addrspace_cast,
                    ),
                    file=fp,
                )
