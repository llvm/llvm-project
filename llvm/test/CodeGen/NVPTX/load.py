from string import Template
from itertools import product

# From https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld, Target ISA notes:

# SM
# Sub-qualifier ::cta requires sm_30 or higher.
# Support for scope qualifier, .relaxed, .acquire, .weak qualifiers require sm_70 or higher.
# Support for .b128 type requires sm_70 or higher.
# Support for .cluster scope qualifier requires sm_90 or higher.
# Sub-qualifier ::cluster requires sm_90 or higher.

# PTX
# Support for scope qualifier, .relaxed, .acquire, .weak qualifiers introduced in PTX ISA version 6.0.  (prior to this, support will have to be through fences)
# Support for .cluster scope qualifier introduced in PTX ISA version 7.8.
# Support for ::cta and ::cluster sub-qualifiers introduced in PTX ISA version 7.8.
# Support for .b128 type introduced in PTX ISA version 8.3.
# Support for .sys scope with .b128 type introduced in PTX ISA version 8.4.
# Support for .volatile qualifier with .local state space introduced in PTX ISA version 9.1.

# Feature compatibility matrix
# ----------------------------
# Features (each gated by both an SM and a PTX threshold; the feature lights up
# only when both thresholds are satisfied):
#   A = scope qualifier / .relaxed / .acquire / .weak   sm_70 + PTX 6.0
#   B = .cluster scope                                  sm_90 + PTX 7.8
#   C = .b128 type                                      sm_70 + PTX 8.3
#   D = .sys + .b128                                    sm_70 + PTX 8.4
#   E = .volatile + .local                              (any)  + PTX 9.1
#
#                  PTX 5.0   PTX 6.0   PTX 7.8   PTX 8.3   PTX 8.4   PTX 9.1
#   sm_30          --        --        --        --        --        E
#   sm_70          --        A         A         A C       A C D     A C D E
#   sm_90          --        A         A B       A B C     A B C D   A B C D E
#
# Each cell lists the features valid at that (SM, PTX); blank means no feature
# from this set is supported. A feature's "subgrid" is the rectangle of cells
# containing its letter; the subgrid's top-left cell is the minimal (SM, PTX)
# pair that turns the feature on.

# One PTX version per SM (matching cmpxchg.py / atomicrmw.py). The previous
# multi-PTX sweep produced identical codegen on every slice axis -- the only
# PTX-version-sensitive lowering (volatile-on-local at PTX 9.1) is exercised
# by volatile-spin-loop.ll, which keeps its own multi-PTX RUN lines.
TEST_SM_PTX_ARCHS = [(60, 50), (70, 63), (90, 87)]

SCOPE_LLVM_TO_PTX = {"": "sys", "block": "cta", "cluster": "cluster", "device": "gpu"}

ORDERINGS = ["notatomic", "unordered", "monotonic", "acquire", "seq_cst"]

ADDRSPACE_NUM_TO_ADDRSPACE = {
    0: "generic",
    1: "global",
    3: "shared",
    4: "const",
    5: "local",
    101: "param",
}

DATATYPE_SIZE_BITS = {
    "i8": 8,
    "i16": 16,
    "i32": 32,
    "i64": 64,
    "i128": 128,
    "half": 16,
    "bfloat": 16,
    "float": 32,
    "double": 64,
}
DATATYPES = list(DATATYPE_SIZE_BITS.keys())

VOLATILES = [False, True]


def get_addrspace_cast(addrspace):
    if addrspace == 0:
        return ""
    return " addrspace({})".format(addrspace)


load_atomic_func = Template(
    """define ${datatype} @${func_name}(ptr${addrspace_cast} %addr) {
    %retval = load atomic ${volatile_kw}${datatype}, ptr${addrspace_cast} %addr syncscope("${llvm_scope}") ${ordering}, align ${align}
    ret ${datatype} %retval
}
"""
)

load_nonatomic_func = Template(
    """define ${datatype} @${func_name}(ptr${addrspace_cast} %addr) {
    %retval = load ${volatile_kw}${datatype}, ptr${addrspace_cast} %addr
    ret ${datatype} %retval
}
"""
)

local_atomic_load_func = Template(
    """define ${datatype} @${func_name}() {
    %slot = alloca ${datatype}, align ${align}, addrspace(5)
    call void asm sideeffect "", "r"(ptr addrspace(5) %slot)
    %retval = load atomic ${volatile_kw}${datatype}, ptr addrspace(5) %slot syncscope("${llvm_scope}") ${ordering}, align ${align}
    ret ${datatype} %retval
}
"""
)

local_nonatomic_load_func = Template(
    """define ${datatype} @${func_name}() {
    %slot = alloca ${datatype}, addrspace(5)
    call void asm sideeffect "", "r"(ptr addrspace(5) %slot)
    %retval = load ${volatile_kw}${datatype}, ptr addrspace(5) %slot
    ret ${datatype} %retval
}
"""
)


def get_load_func(datatype, addrspace, scope, ordering, volatile):
    """Pick template + fill slots from (datatype, addrspace, scope, ordering, volatile).

    - ordering == "notatomic" -> non-atomic (scope unused)
    - addrspace == 5          -> local (alloca + inline-asm address capture)
    - volatile                -> emits `volatile` keyword and suffixes the name
    """
    size = DATATYPE_SIZE_BITS[datatype]
    align = size // 8
    addrspace_name = ADDRSPACE_NUM_TO_ADDRSPACE[addrspace]
    is_atomic = ordering != "notatomic"
    is_local = addrspace == 5
    volatile_kw = "volatile " if volatile else ""
    volatile_suffix = "_volatile" if volatile else ""

    if is_atomic:
        ptx_scope = SCOPE_LLVM_TO_PTX[scope]
        func_name = "{}_{}_{}_{}{}".format(
            ordering, datatype, addrspace_name, ptx_scope, volatile_suffix
        )
        if is_local:
            return local_atomic_load_func.substitute(
                datatype=datatype,
                func_name=func_name,
                llvm_scope=scope,
                ordering=ordering,
                align=align,
                volatile_kw=volatile_kw,
            )
        return load_atomic_func.substitute(
            datatype=datatype,
            func_name=func_name,
            addrspace_cast=get_addrspace_cast(addrspace),
            llvm_scope=scope,
            ordering=ordering,
            align=align,
            volatile_kw=volatile_kw,
        )

    func_name = "notatomic_{}_{}{}".format(datatype, addrspace_name, volatile_suffix)
    if is_local:
        return local_nonatomic_load_func.substitute(
            datatype=datatype,
            func_name=func_name,
            volatile_kw=volatile_kw,
        )
    return load_nonatomic_func.substitute(
        datatype=datatype,
        func_name=func_name,
        addrspace_cast=get_addrspace_cast(addrspace),
        volatile_kw=volatile_kw,
    )


run_statement = Template(
    """; RUN: llc < %s -march=nvptx64 -mcpu=sm_${sm} -mattr=+ptx${ptx} | FileCheck %s
; RUN: %if ptxas-sm_${sm} && ptxas-isa-${ptxfp} %{ llc < %s -march=nvptx64 -mcpu=sm_${sm} -mattr=+ptx${ptx} | %ptxas-verify -arch=sm_${sm} %}
"""
)

if __name__ == "__main__":
    # NOTE: loads can't be `release` or `acq_rel` per LangRef. On sm < 70 the
    # backend used to reject orderings other than notatomic/monotonic, but
    # AtomicExpandPass now emulates them, so the same axes work uniformly.

    for sm, ptx in TEST_SM_PTX_ARCHS:
        with open("load-sm{}.ll".format(str(sm)), "w") as fp:
            print(run_statement.substitute(sm=sm, ptx=ptx, ptxfp=ptx / 10.0), file=fp)

            # All combinations of (ordering x addrspace x datatype x volatile x
            # scope). Atomic loads on .const/.param are skipped (read-only
            # spaces have no observable atomic semantic). i128 only at sm_90
            # -- atom.b128 requires sm_90+/PTX 8.3+, and plain ld.b128 requires
            # sm_70+/PTX 8.3+; neither holds at sm_60/70 with our test matrix.
            # For notatomic, syncscope is illegal -- emit only at scope="".
            for ordering, addrspace, datatype, volatile, scope in product(
                ORDERINGS,
                ADDRSPACE_NUM_TO_ADDRSPACE,
                DATATYPES,
                VOLATILES,
                SCOPE_LLVM_TO_PTX,
            ):
                if datatype == "i128" and sm < 90:
                    continue
                is_atomic = ordering != "notatomic"
                if is_atomic and addrspace in (4, 101):
                    continue
                if not is_atomic and scope != "":
                    continue  # scope irrelevant for notatomic
                print(
                    get_load_func(datatype, addrspace, scope, ordering, volatile),
                    file=fp,
                )
