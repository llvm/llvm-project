# For manual usage, not as a part of lit tests. Used for generating the following tests:
# fence-sm30.ll, fence-sm70.ll, fence-sm90.ll

from string import Template
from itertools import product

fence_func = Template(
    """
define void @fence_${ordering}_${scope}() {
    fence syncscope(\"${scope}\") ${ordering}
    ret void
}
"""
)

run_statement = Template(
    """
; ${run}: llc < %s -march=nvptx64 -mcpu=sm_${sm} -mattr=+ptx${ptx} | FileCheck %s --check-prefix=SM${sm}
; ${run}: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_${sm} -mattr=+ptx${ptx} | %ptxas-verfy %}
"""
)

# (sm, ptx)
TESTS = [(30, 50), (70, 60), (90, 87)]

SCOPES = ["", "block", "cluster", "device"]

ORDERINGS = ["acquire", "release", "acq_rel", "seq_cst"]

if __name__ == "__main__":
    for sm, ptx in TESTS:
        with open("fence-sm{}.ll".format(sm), "w") as fp:
            print(run_statement.substitute(run="RUN", sm=sm, ptx=ptx), file=fp)
            for ordering, scope in product(ORDERINGS, SCOPES):
                if scope == "cluster" and (sm < 90 or ptx < 78):
                    print(
                        "; .cluster scope unsupported on SM = {} PTX = {}".format(
                            sm, ptx
                        ),
                        file=fp,
                    )
                else:
                    print(
                        fence_func.substitute(scope=scope, ordering=ordering), file=fp
                    )
