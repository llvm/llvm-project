# For manual usage, not as a part of lit tests. Used for generating the following tests:

from string import Template
from itertools import product

TEST_SM_ARCH_PAIRS = [(60, 50), (70, 63), (90, 87)]

SCOPE_LLVM_TO_PTX = {"": "sys", "block": "cta", "cluster": "cluster", "device": "gpu"}

ORDERINGS = ["monotonic", "acquire", "release", "acq_rel", "seq_cst"]

VECTOR_SIZES = [1, 2, 4, 8]

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

DATATYPE_SUFFIXES = {
    "float": "f32",
    "double": "f64",
    "half": "f16",
    "bfloat": "bf16",
}

ADDRSPACE_NUM_TO_ADDRSPACE = {0: "generic", 1: "global", 3: "shared"}

# A small spread of operations used to exercise the scope and address-space
# qualifiers, which are emitted orthogonally to the operation: a native integer op
# at two sizes, signed/unsigned variants, an always-emulated op (lowers to a cas
# loop), and floating-point add.
REPRESENTATIVE_OPS = [
    ("add", "i32"),
    ("add", "i64"),
    ("min", "i32"),
    ("umax", "i32"),
    ("nand", "i32"),
    ("nand", "i64"),
    ("fadd", "float"),
    ("fadd", "double"),
]

atomicrmw_func = Template(
    """define ${datatype} @${operation}_${ordering}_${datatype}_${addrspace}_${ptx_scope}(ptr${addrspace_cast} %addr, ${datatype} %val) {
        %retval = atomicrmw ${operation} ptr ${addrspace_cast} %addr, ${datatype} %val syncscope(\"${llvm_scope}\") ${ordering}
        ret $datatype %retval
}
"""
)

atomicrmw_elementwise_func = Template(
    """define ${datatype} @${operation}_${ordering}_${datatype_name}_${addrspace}_${ptx_scope}(ptr${addrspace_cast} %addr, ${datatype} %val) {
        %retval = atomicrmw elementwise ${operation} ptr ${addrspace_cast} %addr, ${datatype} %val syncscope(\"${llvm_scope}\") ${ordering}
        ret $datatype %retval
}
"""
)

# atomicrmw fadd's lowering depends on the function's FTZ (denormal) mode, so we
# check codegen both with and without it. Lines common to both runs collapse to
# the SM${sm} prefix; only the FTZ-sensitive ops diverge into SM${sm}-NOFTZ /
# SM${sm}-FTZ. (-nvptx-allow-ftz-atomics is covered separately in
# atomicrmw-allow-ftz-atomics.ll.)

run_statement = Template(
    """; RUN: llc < %s -march=nvptx64 -mcpu=sm_${sm} -mattr=+ptx${ptx} | FileCheck %s --check-prefixes=SM${sm},SM${sm}-NOFTZ
; RUN: llc < %s -march=nvptx64 -mcpu=sm_${sm} -mattr=+ptx${ptx} -denormal-fp-math-f32=preserve-sign | FileCheck %s --check-prefixes=SM${sm},SM${sm}-FTZ
; RUN: %if ptxas-sm_${sm} && ptxas-isa-${ptxfp} %{ llc < %s -march=nvptx64 -mcpu=sm_${sm} -mattr=+ptx${ptx} | %ptxas-verify -arch=sm_${sm} %}
; RUN: %if ptxas-sm_${sm} && ptxas-isa-${ptxfp} %{ llc < %s -march=nvptx64 -mcpu=sm_${sm} -mattr=+ptx${ptx} -denormal-fp-math-f32=preserve-sign | %ptxas-verify -arch=sm_${sm} %}
"""
)


def get_addrspace_cast(addrspace):
    if addrspace == 0:
        return ""
    else:
        return " addrspace({})".format(str(addrspace))


def get_vector_datatype(datatype, size):
    return "<{} x {}>".format(size, datatype)


def get_vector_datatype_name(datatype, size):
    return "v{}{}".format(size, DATATYPE_SUFFIXES.get(datatype, datatype))


def print_atomicrmw(
    fp, operation, datatype, ordering, addrspace, llvm_scope, vector_size=None
):
    if vector_size is None:
        template = atomicrmw_func
        ir_datatype = datatype
        datatype_name = datatype
    else:
        template = atomicrmw_elementwise_func
        ir_datatype = get_vector_datatype(datatype, vector_size)
        datatype_name = get_vector_datatype_name(datatype, vector_size)

    print(
        template.substitute(
            operation=operation,
            ordering=ordering,
            datatype=ir_datatype,
            datatype_name=datatype_name,
            addrspace=ADDRSPACE_NUM_TO_ADDRSPACE[addrspace],
            ptx_scope=SCOPE_LLVM_TO_PTX[llvm_scope],
            llvm_scope=llvm_scope,
            addrspace_cast=get_addrspace_cast(addrspace),
        ),
        file=fp,
    )


if __name__ == "__main__":
    for sm, ptx in TEST_SM_ARCH_PAIRS:
        for elementwise in [False, True]:
            vector_sizes = VECTOR_SIZES if elementwise else [None]
            filename_suffix = "-elementwise" if elementwise else ""
            with open("atomicrmw{}-sm{}.ll".format(filename_suffix, sm), "w") as fp:
                print(
                    run_statement.substitute(sm=sm, ptx=ptx, ptxfp=ptx / 10.0),
                    file=fp,
                )

                # Slice 1: Keep addrspace, llvm_scope, and ordering fixed while
                # generating every operation and type combination.
                addrspace, llvm_scope, ordering = 1, "block", "acq_rel"
                for operation, datatype, vector_size in product(
                    INTEGER_OPERATIONS,
                    ["i8", "i16", "i32", "i64"],
                    vector_sizes,
                ):
                    print_atomicrmw(
                        fp,
                        operation,
                        datatype,
                        ordering,
                        addrspace,
                        llvm_scope,
                        vector_size,
                    )

                # Floating-point operations.
                for datatype, operation, vector_size in product(
                    ["float", "double", "half", "bfloat"],
                    FLOATING_POINT_OPERATIONS,
                    vector_sizes,
                ):
                    print_atomicrmw(
                        fp,
                        operation,
                        datatype,
                        ordering,
                        addrspace,
                        llvm_scope,
                        vector_size,
                    )

                # Slice 2: Keep addrspace and llvm_scope fixed, and generate all
                # orderings for add and nand. add is natively supported for larger
                # bitwidths, while nand is always emulated.
                addrspace, llvm_scope = 1, "block"
                for operation, datatype, ordering, vector_size in product(
                    ["add", "nand"], ["i8", "i32"], ORDERINGS, vector_sizes
                ):
                    if ordering == "acq_rel":
                        # These cases are part of Slice 1.
                        continue
                    print_atomicrmw(
                        fp,
                        operation,
                        datatype,
                        ordering,
                        addrspace,
                        llvm_scope,
                        vector_size,
                    )

                if elementwise:
                    continue

                # Slice 3: Keep addrspace (global) and ordering fixed, vary the
                # scope qualifier. block/global is already covered by Slice 1.
                addrspace, ordering = 1, "acq_rel"
                for llvm_scope in [s for s in SCOPE_LLVM_TO_PTX if s != "block"]:
                    for operation, datatype in REPRESENTATIVE_OPS:
                        print_atomicrmw(
                            fp,
                            operation,
                            datatype,
                            ordering,
                            addrspace,
                            llvm_scope,
                        )

                # Slice 4: Keep scope (block) and ordering fixed, vary the
                # address-space qualifier. global is already covered by Slice 1.
                llvm_scope, ordering = "block", "acq_rel"
                for addrspace in [a for a in ADDRSPACE_NUM_TO_ADDRSPACE if a != 1]:
                    for operation, datatype in REPRESENTATIVE_OPS:
                        print_atomicrmw(
                            fp,
                            operation,
                            datatype,
                            ordering,
                            addrspace,
                            llvm_scope,
                        )
