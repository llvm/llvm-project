# This test generates all variants of load/store instructions and verifies that
# LLVM generates correct PTX for them.

# RUN: %python %s > %t.ll
# RUN: llc < %t.ll -mtriple=nvptx -mcpu=sm_30 | FileCheck -check-prefixes=CHECK,CHECK_P32 %t.ll
# RUN: llc < %t.ll -mtriple=nvptx64 -mcpu=sm_30 | FileCheck -check-prefixes=CHECK,CHECK_P64 %t.ll
# RUN: %if ptxas-ptr32 %{ llc < %t.ll -mtriple=nvptx -mcpu=sm_30 | %ptxas-verify %}
# RUN: %if ptxas %{ llc < %t.ll -mtriple=nvptx64 -mcpu=sm_30 | %ptxas-verify %}

from __future__ import print_function

from itertools import product
from string import Template


llvm_type_to_ptx_load_type = {
    "i8": "b8",
    "i16": "b16",
    "i32": "b32",
    "i64": "b64",
    "half": "b16",
    "<2 x half>": "b32",
    "float": "b32",
    "double": "b64",
}

llvm_type_to_ptx_type = {
    "i8": "u8",
    "i16": "u16",
    "i32": "u32",
    "i64": "u64",
    "half": "b16",
    "<2 x half>": "b32",
    "float": "f32",
    "double": "f64",
}

llvm_type_to_ptx_reg = {
    "i8": "r",
    "i16": "r",
    "i32": "r",
    "i64": "rd",
    "half": "rs",
    "<2 x half>": "r",
    "float": "r",
    "double": "rd",
}

addrspace_id = {
    "": 0,
    ".global": 1,
    ".shared": 3,
    ".const": 4,
    ".local": 5,
    ".param": 101,
}


def gen_load_tests():
    load_template = """
define ${type} @${testname}(${type} addrspace(${asid})* %ptr) {
; CHECK: ${testname}
; CHECK_P32: ld${_volatile}${_volatile_as}.${ptx_load_type} %${ptx_reg}{{[0-9]+}}, [%r{{[0-9]+}}]
; CHECK_P64: ld${_volatile}${_volatile_as}.${ptx_load_type} %${ptx_reg}{{[0-9]+}}, [%rd{{[0-9]+}}]
; CHECK: ret
  %p = ${generic_ptr}
  %a = load ${volatile} ${type}, ${type}* %p
  ret ${type} %a
}
"""
    for op_type, volatile, space in product(
        ["i8", "i16", "i32", "i64", "half", "float", "double", "<2 x half>"],
        [True, False],  # volatile
        ["", ".shared", ".global", ".const", ".local", ".param"],
    ):

        # Volatile is only supported for global, shared and generic.
        if volatile and not space in ["", ".global", ".shared"]:
            continue

        # Volatile is only supported for global, shared and generic.
        # All other volatile accesses are done in generic AS.
        if volatile and not space in ["", ".global", ".shared"]:
            volatile_as = ""
        else:
            volatile_as = space

        params = {
            "type": op_type,
            "volatile": "volatile" if volatile else "",
            "_volatile": ".volatile" if volatile else "",
            "_volatile_as": volatile_as,
            "_space": space,
            "ptx_reg": llvm_type_to_ptx_reg[op_type],
            "ptx_type": llvm_type_to_ptx_type[op_type],
            "ptx_load_type": llvm_type_to_ptx_load_type[op_type],
            "asid": addrspace_id[space],
        }

        testname = Template("ld_${_volatile}${_space}.${ptx_type}").substitute(params)
        params["testname"] = testname.replace(".", "_")

        # LLVM does not accept "addrspacecast Type* addrspace(0) to Type*", so we
        # need to avoid it for generic pointer tests.
        if space:
            generic_ptr_template = (
                "addrspacecast ${type} addrspace(${asid})* %ptr " "to ${type}*"
            )
        else:
            generic_ptr_template = "select i1 true, ${type}* %ptr, ${type}* %ptr"
        params["generic_ptr"] = Template(generic_ptr_template).substitute(params)

        print(Template(load_template).substitute(params))


def main():
    gen_load_tests()


main()
