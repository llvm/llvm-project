# This test generates all variants of wmma intrinsics and verifies that LLVM
# generates correct instructions for them.

# RUN: python %s > %t.ll
# RUN: llc < %t.ll -march=nvptx64 -mcpu=sm_70 -mattr=+ptx61 | FileCheck %t.ll

from __future__ import print_function

from itertools import product
from string import Template

def make_wmma_slice_ty(abcd, itype):
  elt_ty = "<2 x half>" if itype == "f16" else "float"
  num_elts = 4 if abcd in "cd" and itype == "f16" else 8;
  return [elt_ty] * num_elts

def make_wmma_ld_ret_ty(abc, itype):
  return "{%s}" % ", ".join(make_wmma_slice_ty(abc, itype))

# returns address space
def get_aspace(space):
  space_map = {
      ".global" : 1,
      ".shared" : 3,
      ".const"  : 4,
      ".local"  : 5,
      ".param"  : 101,
      ""        : 0,
      ".generic": 0
  }
  return space_map[space];

def get_pspace(space):
  return "p%di8" % get_aspace(space);

# Convenient test patterns.
check_f16_8 = "{{%s}}" % ", *".join(["%hh[0-9]+"] * 8)
check_f16_4 = "{{%s}}" % ", *".join(["%hh[0-9]+"] * 4)
check_f32_8 = "{{%s}}" % ", *".join(["%f[0-9]+"] * 8)

known_geoms = ["m16n16k16", "m8n32k16", "m32n8k16"]

def gen_wmma_load_tests():
  load_template = """
declare ${ret_ty} @${intrinsic}(i8 ${as}* %src ${extra_args});

; CHECK-LABEL: .func {{.*}}test_${function}(
define ${ret_ty} @test_${function}(i8 ${as}* %src ${extra_args}) {
; CHECK: ${instruction}
; CHECK: {${check_result}}
; CHECK: [%rd{{[0-9]+}}]${stride_pattern}
  %v0 = call ${ret_ty} @${intrinsic}(i8 ${as}* %src ${extra_args});
  ret ${ret_ty} %v0;
}

; CHECK-LABEL: .func{{.*}}test_${function}_o(
define ${ret_ty} @test_${function}_o(i8 ${as}* %src ${extra_args}) {
; CHECK: ${instruction}
; CHECK: {${check_result}}
; CHECK: [%rd{{[0-9]+}}+128]${stride_pattern}
  %src1 = getelementptr i8, i8 ${as}* %src, i32 128;
  %v0 = call ${ret_ty} @${intrinsic}(i8 ${as}* %src1 ${extra_args});
  ret ${ret_ty} %v0;
}
"""
  intrinsic_template = "llvm.nvvm.wmma.${geom}.load.${abc}.${layout}${stride}.${itype}.${pspace}"
  instruction_template = "wmma.load.${abc}.sync.${layout}.${geom}${space}.${itype}"

  for geom, abc, layout, space, stride, itype in product(
      known_geoms,
      "abc",
      ["row","col"],
      ["",".shared",".global"],
      ["", ".stride"],
      ["f16", "f32"]):

    params = {
        "abc" : abc,
        "layout" : layout,
        "space" : space,
        "stride" : stride,
        "itype" : itype,
        "pspace" : get_pspace(space),
        "as"     : "addrspace(%d)" % get_aspace(space),
        "geom"   : geom,
    }

    if itype == "f32" and abc != "c":
      continue

    test_params = params
    test_params["intrinsic"] = Template(intrinsic_template).substitute(params)
    test_params["function"] = test_params["intrinsic"].replace(".","_")
    test_params["instruction"] = Template(instruction_template).substitute(params)
    test_params["ret_ty"] = make_wmma_ld_ret_ty(abc, itype)
    if abc == "c" :
      test_params["check_result"] = check_f16_4 if itype == "f16" else check_f32_8
    else:
      test_params["check_result"] = check_f16_8

    if stride:
      test_params["extra_args"] = ", i32 %stride";
      test_params["stride_pattern"] = ", %r{{[0-9]+}}"
    else:
      test_params["extra_args"] = ""
      test_params["stride_pattern"] = ""

    print(Template(load_template).substitute(test_params))

def make_wmma_slice_args(itype, abcd, prefix="v"):
  return ", ".join(["%s %%%s%d" % (t, prefix, i) for i,t
                  in enumerate(make_wmma_slice_ty(abcd, itype))])

def gen_wmma_store_tests():
  store_template = """
declare void @${intrinsic}(i8 ${as}* %src, ${args}${extra_args});

; CHECK-LABEL: .func {{.*}}test_${function}(
define void @test_${function}(i8 ${as}* %src, ${args}${extra_args}) {
; CHECK: ${instruction} {{.*}}[%rd{{[0-9+]}}
; CHECK: {${check_args}}
; CHECK: ${stride_pattern}
  call void @${intrinsic}(i8 ${as}* %src, ${args} ${extra_args});
  ret void
}

; CHECK-LABEL: .func{{.*}}test_${function}_o(
define void @test_${function}_o(i8 ${as}* %src, ${args}${extra_args}) {
; CHECK: ${instruction} {{.*}}[%rd{{[0-9+]}}+128]
; CHECK: ${check_args}
; CHECK: ${stride_pattern}
  %src1 = getelementptr i8, i8 ${as}* %src, i32 128;
  call void @${intrinsic}(i8 ${as}* %src1, ${args}${extra_args});
  ret void
}
"""
  intrinsic_template = "llvm.nvvm.wmma.${geom}.store.${abc}.${layout}${stride}.${itype}.${pspace}"
  instruction_template = "wmma.store.${abc}.sync.${layout}.${geom}${space}.${itype}"

  for geom, abc, layout, space, stride, itype in product(
      known_geoms,
      "d",
      ["row","col"],
      ["",".shared",".global"],
      ["", ".stride"],
      ["f16", "f32"]):

    params = {
        "abc" : abc,
        "layout" : layout,
        "space" : space,
        "stride" : stride,
        "itype" : itype,
        "pspace" : get_pspace(space),
        "as"     : "addrspace(%d)" % get_aspace(space),
        "geom"   : geom,
    }

    test_params = params
    test_params["intrinsic"] = Template(intrinsic_template).substitute(params)
    test_params["function"] = test_params["intrinsic"].replace(".","_")
    test_params["instruction"] = Template(instruction_template).substitute(params)
    test_params["ret_ty"] = make_wmma_ld_ret_ty(abc, itype)
    test_params["check_args"] = check_f16_4 if itype == "f16" else check_f32_8
    if stride:
      test_params["extra_args"] = ", i32 %stride";
      test_params["stride_pattern"] = ", %r{{[0-9]+}};"
    else:
      test_params["extra_args"] = ""
      test_params["stride_pattern"] = ";"
    test_params["args"] = make_wmma_slice_args(itype, "d");

    print(Template(store_template).substitute(test_params))

def gen_wmma_mma_tests():
  mma_template = """
declare ${ret_ty} @${intrinsic}(
        ${args});

; CHECK-LABEL: .func {{.*}}test_${function}(
define ${ret_ty} @test_${function}(
        ${args}) {
; CHECK: ${instruction}
; CHECK-NEXT: ${check_d}
; CHECK-NEXT: ${check_ab}
; CHECK-NEXT: ${check_ab}
; CHECK-NEXT: ${check_c}
  %r = call ${ret_ty} @${intrinsic}(
        ${args});
  ret ${ret_ty} %r;
}
"""
  intrinsic_template = "llvm.nvvm.wmma.${geom}.mma.${alayout}.${blayout}.${dtype}.${ctype}${satf}"
  instruction_template = "wmma.mma.sync.${alayout}.${blayout}.${geom}.${dtype}.${ctype}${satf}"

  for geom, alayout, blayout, ctype, dtype, satf in product(
      known_geoms,
      ["row","col"],
      ["row","col"],
      ["f16", "f32"],
      ["f16", "f32"],
      [".satfinite", ""]):

    params = {
        "alayout" : alayout,
        "blayout" : blayout,
        "ctype" : ctype,
        "dtype" : dtype,
        "satf"  : satf,
        "geom"  : geom,
    }

    test_params = params
    test_params["intrinsic"] = Template(intrinsic_template).substitute(params)
    test_params["function"] = test_params["intrinsic"].replace(".", "_")
    test_params["instruction"] = Template(instruction_template).substitute(params)
    test_params["ret_ty"] = make_wmma_ld_ret_ty("d", dtype)
    test_params["check_ab"] = check_f16_8
    test_params["check_c"] = check_f16_4 if ctype == "f16" else check_f32_8
    test_params["check_d"] = check_f16_4 if dtype == "f16" else check_f32_8
    args = ",\n        ".join(make_wmma_slice_args(t, abcd, prefix=abcd)
                              for abcd, t in (("a", "f16"),
                                              ("b", "f16"),
                                              ("c", ctype)))
    test_params["args"] = args
    print(Template(mma_template).substitute(test_params))

def main():
  gen_wmma_load_tests()
  gen_wmma_store_tests()
  gen_wmma_mma_tests()

main()
