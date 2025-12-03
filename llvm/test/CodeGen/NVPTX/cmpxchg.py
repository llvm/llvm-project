# For manual usage, not as a part of lit tests. Used for generating the following tests:
# cmpxchg-sm30.ll, cmpxchg-sm70.ll, cmpxchg-sm90.ll

from string import Template
from itertools import product

cmpxchg_func = Template(
    """define i$size @${strength}_${success}_${failure}_i${size}_${addrspace}_${ptx_scope}(ptr${addrspace_cast} %addr, i$size %cmp, i$size %new) {
    %pairold = cmpxchg ${weak} ptr${addrspace_cast} %addr, i$size %cmp, i$size %new syncscope(\"${llvm_scope}\") $success $failure
    %oldvalue = extractvalue { i$size, i1 } %pairold, 0
    ret i$size %oldvalue
}
"""
)

cmpxchg_func_no_scope = Template(
    """define i$size @${strength}_${success}_${failure}_i${size}_${addrspace}(ptr${addrspace_cast} %addr, i$size %cmp, i$size %new) {
    %pairold = cmpxchg ${weak} ptr${addrspace_cast} %addr, i$size %cmp, i$size %new $success $failure
    %oldvalue = extractvalue { i$size, i1 } %pairold, 0
    ret i$size %oldvalue
}
"""
)

run_statement = Template(
    """; RUN: llc < %s -march=nvptx64 -mcpu=sm_${sm} -mattr=+ptx${ptx} | FileCheck %s --check-prefix=SM${sm}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_${sm} -mattr=+ptx${ptx} | %ptxas-verify -arch=sm_${sm} %}
"""
)


def get_addrspace_cast(addrspace):
    if addrspace == 0:
        return ""
    else:
        return " addrspace({})".format(str(addrspace))


TESTS = [(60, 50), (70, 63), (90, 87)]
# We don't include (100, 90) because the codegen is identical to (90, 87)

LLVM_SCOPES = ["", "block", "cluster", "device"]

SCOPE_LLVM_TO_PTX = {"": "sys", "block": "cta", "cluster": "cluster", "device": "gpu"}

SUCCESS_ORDERINGS = ["monotonic", "acquire", "release", "acq_rel", "seq_cst"]

FAILURE_ORDERINGS = ["monotonic", "acquire", "seq_cst"]

STRENGTHS = ["weak", "strong"]

SIZES = [8, 16, 32, 64]

ADDRSPACES = [0, 1, 3]

ADDRSPACE_NUM_TO_ADDRSPACE = {0: "generic", 1: "global", 3: "shared"}

if __name__ == "__main__":
    for sm, ptx in TESTS:
        with open("cmpxchg-sm{}.ll".format(str(sm)), "w") as fp:
            print(run_statement.substitute(sm=sm, ptx=ptx), file=fp)
            # Test weak and strong cmpxchg for all slices
            for strength in STRENGTHS:
                # Our test space is: SIZES X SUCCESS_ORDERINGS X FAILURE_ORDERINGS X ADDRSPACES X LLVM_SCOPES
                # This is very large, so we instead test 3 slices.

                # First slice:  are all orderings correctly supported, with and without emulation loops?
                # set addrspace to global, scope to cta, generate all possible orderings, for all operation sizes
                addrspace, llvm_scope = 1, "block"
                for size, success, failure in product(
                    SIZES, SUCCESS_ORDERINGS, FAILURE_ORDERINGS
                ):
                    print(
                        cmpxchg_func.substitute(
                            success=success,
                            failure=failure,
                            size=size,
                            addrspace=ADDRSPACE_NUM_TO_ADDRSPACE[addrspace],
                            addrspace_cast=get_addrspace_cast(addrspace),
                            llvm_scope=llvm_scope,
                            ptx_scope=SCOPE_LLVM_TO_PTX[llvm_scope],
                            strength=strength,
                            weak="weak" if strength == "weak" else "",
                        ),
                        file=fp,
                    )

                # Second slice: Are all scopes correctly supported, with and without emulation loops?
                # fix addrspace, ordering, generate all possible scopes, for operation sizes i8, i32
                addrspace, success, failure = 1, "acq_rel", "acquire"
                for size in [8, 32]:
                    print(
                        cmpxchg_func_no_scope.substitute(
                            success=success,
                            failure=failure,
                            size=size,
                            addrspace=ADDRSPACE_NUM_TO_ADDRSPACE[addrspace],
                            addrspace_cast=get_addrspace_cast(addrspace),
                            strength=strength,
                            weak="weak" if strength == "weak" else "",
                        ),
                        file=fp,
                    )

                for llvm_scope in LLVM_SCOPES:
                    if sm < 90 and llvm_scope == "cluster":
                        continue
                    if llvm_scope == "block":
                        # skip (acq_rel, acquire, global, cta)
                        continue
                    print(
                        cmpxchg_func.substitute(
                            success=success,
                            failure=failure,
                            size=size,
                            addrspace=ADDRSPACE_NUM_TO_ADDRSPACE[addrspace],
                            addrspace_cast=get_addrspace_cast(addrspace),
                            llvm_scope=llvm_scope,
                            ptx_scope=SCOPE_LLVM_TO_PTX[llvm_scope],
                            strength=strength,
                            weak="weak" if strength == "weak" else "",
                        ),
                        file=fp,
                    )

                # Third slice: Are all address spaces correctly supported?
                # fix ordering, scope, generate all possible address spaces, for operation sizes i8, i32
                success, failure, llvm_scope = "acq_rel", "acquire", "block"
                for size, addrspace in product([8, 32], ADDRSPACES):
                    if addrspace == 1:
                        # skip (acq_rel, acquire, global, cta)
                        continue
                    print(
                        cmpxchg_func.substitute(
                            success=success,
                            failure=failure,
                            size=size,
                            addrspace=ADDRSPACE_NUM_TO_ADDRSPACE[addrspace],
                            addrspace_cast=get_addrspace_cast(addrspace),
                            llvm_scope=llvm_scope,
                            ptx_scope=SCOPE_LLVM_TO_PTX[llvm_scope],
                            strength=strength,
                            weak="weak" if strength == "weak" else "",
                        ),
                        file=fp,
                    )
