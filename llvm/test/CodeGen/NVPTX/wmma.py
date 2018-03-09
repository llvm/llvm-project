# This test generates all variants of wmma intrinsics and verifies that LLVM
# generates correct instructions for them.

# RUN: python %s > %t.ll
# RUN: llc < %t.ll -march=nvptx64 -mcpu=sm_70 -mattr=+ptx60 | FileCheck %t.ll

from itertools import product
from string import Template

def make_wmma_slice_ty(abcd, itype):
  elt_ty = "<2 x half>" if itype == "f16" else "float"
  num_elts = 4 if abcd in "cd" and itype == "f16" else 8;
  return [elt_ty] * num_elts

def make_wmma_ld_ret_ty(abc, itype):
  return "{%s}" % ", ".join(make_wmma_slice_ty(abc, itype))

# Convenient test patterns.
check_f16_8 = "{{%s}}" % ", *".join(["%hh[0-9]+"] * 8)
check_f16_4 = "{{%s}}" % ", *".join(["%hh[0-9]+"] * 4)
check_f32_8 = "{{%s}}" % ", *".join(["%f[0-9]+"] * 8)

def gen_wmma_load_tests():
  load_template = """
declare ${ret_ty} @llvm.nvvm.wmma.load.$intrinsic_suffix(i8* %src ${extra_args});

; CHECK-LABEL: .func {{.*}}test_wmma_load_${function_suffix}(
define ${ret_ty} @test_wmma_load_${function_suffix}(i8* %src ${extra_args}) {
; CHECK wmma.load.${intrinsic_suffix}
; CHECK: {${check_result}}
; CHECK: [%rd{{[0-9]+}}]${stride_pattern}
  %v0 = call ${ret_ty} @llvm.nvvm.wmma.load.${intrinsic_suffix}(i8* %src ${extra_args});
  ret ${ret_ty} %v0;
}

; CHECK-LABEL: .func{{.*}}test_wmma_load_${function_suffix}_o(
define ${ret_ty} @test_wmma_load_${function_suffix}_o(i8* %src ${extra_args}) {
; CHECK wmma.load.${intrinsic_suffix}
; CHECK: {${check_result}}
; CHECK: [%rd{{[0-9]+}}+128]${stride_pattern}
  %src1 = getelementptr i8, i8* %src, i32 128;
  %v0 = call ${ret_ty} @llvm.nvvm.wmma.load.${intrinsic_suffix}(i8* %src1 ${extra_args});
  ret ${ret_ty} %v0;
}
"""
  suffix_template = "${abc}.sync.${layout}.m16n16k16${space}${stride}.${itype}"
  instruction_template = "${abc}.sync.${layout}.m16n16k16${space}.${itype}"

  for abc, layout, space, stride, itype in product(
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
        "itype" : itype
    }

    if itype == "f32" and abc != "c":
      continue

    test_params = params
    test_params["intrinsic_suffix"] = Template(suffix_template).substitute(params)
    test_params["function_suffix"] = test_params["intrinsic_suffix"].replace(".","_")
    test_params["instruction_suffix"] = Template(instruction_template).substitute(params)
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
declare void @llvm.nvvm.wmma.store.$intrinsic_suffix(i8* %src, ${args}${extra_args});

; CHECK-LABEL: .func {{.*}}test_wmma_store_${function_suffix}(
define void @test_wmma_store_${function_suffix}(i8* %src, ${args}${extra_args}) {
; CHECK wmma.store.${intrinsic_suffix} {{.*}}[%rd{{[0-9+]}}
; CHECK: {${check_args}}
; CHECK: ${stride_pattern}
  call void @llvm.nvvm.wmma.store.${intrinsic_suffix}(i8* %src, ${args} ${extra_args});
  ret void
}

; CHECK-LABEL: .func{{.*}}test_wmma_store_${function_suffix}_o(
define void @test_wmma_store_${function_suffix}_o(i8* %src, ${args}${extra_args}) {
; CHECK wmma.store.${intrinsic_suffix} {{.*}}[%rd{{[0-9+]}}+128]
; CHECK: ${check_args}
; CHECK: ${stride_pattern}
  %src1 = getelementptr i8, i8* %src, i32 128;
  call void @llvm.nvvm.wmma.store.${intrinsic_suffix}(i8* %src1, ${args}${extra_args});
  ret void
}
"""
  suffix_template = "${abc}.sync.${layout}.m16n16k16${space}${stride}.${itype}"
  instruction_template = "${abc}.sync.${layout}.m16n16k16${space}.${itype}"

  for abc, layout, space, stride, itype in product(
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
        "itype" : itype
    }

    test_params = params
    test_params["intrinsic_suffix"] = Template(suffix_template).substitute(params)
    test_params["function_suffix"] = test_params["intrinsic_suffix"].replace(".","_")
    test_params["instruction_suffix"] = Template(instruction_template).substitute(params)
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
declare ${ret_ty} @llvm.nvvm.wmma.mma.sync.$intrinsic_suffix(
        ${args});

; CHECK-LABEL: .func {{.*}}test_wmma_mma_${function_suffix}(
define ${ret_ty} @test_wmma_mma_${function_suffix}(
        ${args}) {
; CHECK wmma.mma.${intrinsic_suffix} {{.*}}[%rd{{[0-9+]}}
; CHECK ${check_d}
; CHECK ${check_ab}
; CHECK ${check_ab}
; CHECK ${check_c}
  %r = call ${ret_ty} @llvm.nvvm.wmma.mma.sync.${intrinsic_suffix}(
        ${args});
  ret ${ret_ty} %r;
}
"""
  suffix_template = "${alayout}.${blayout}.m16n16k16.${dtype}.${ctype}${satf}"

  for alayout, blayout, ctype, dtype, satf in product(
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
        "satf"  : satf
    }

    test_params = params
    test_params["intrinsic_suffix"] = Template(suffix_template).substitute(params)
    test_params["function_suffix"] = test_params["intrinsic_suffix"].replace(".", "_")
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
