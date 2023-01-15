; RUN: llc -mtriple=riscv64 -mattr=+v -o - %s | FileCheck %s --check-prefix=CHECK-ASM
; RUN: llc -mtriple=riscv64 -mattr=+v -filetype=obj -o - %s \
; RUN:   | llvm-readobj --symbols - | FileCheck %s --check-prefix=CHECK-OBJ

define i32 @base_cc() {
; CHECK-ASM-LABEL: base_cc:
; CHECK-ASM-NOT: .variant_cc
; CHECK-OBJ-LABEL: Name: base_cc
; CHECK-OBJ: Other: 0
  ret i32 42
}

define <4 x i32> @fixed_vector_cc_1(<4 x i32> %arg) {
; CHECK-ASM: .variant_cc fixed_vector_cc_1
; CHECK-ASM-NEXT: fixed_vector_cc_1:
; CHECK-OBJ-LABEL: Name: fixed_vector_cc_1
; CHECK-OBJ: Other [ (0x80)
  ret <4 x i32> %arg
}

define <vscale x 4 x i32> @rvv_vector_cc_1() {
; CHECK-ASM: .variant_cc rvv_vector_cc_1
; CHECK-ASM-NEXT: rvv_vector_cc_1:
; CHECK-OBJ-LABEL: Name: rvv_vector_cc_1
; CHECK-OBJ: Other [ (0x80)
  ret <vscale x 4 x i32> undef
}

define <vscale x 4 x i1> @rvv_vector_cc_2() {
; CHECK-ASM: .variant_cc rvv_vector_cc_2
; CHECK-ASM-NEXT: rvv_vector_cc_2:
; CHECK-OBJ-LABEL: Name: rvv_vector_cc_2
; CHECK-OBJ: Other [ (0x80)
  ret <vscale x 4 x i1> undef
}

define void @rvv_vector_cc_3(<vscale x 4 x i32> %arg) {
; CHECK-ASM: .variant_cc rvv_vector_cc_3
; CHECK-ASM-NEXT: rvv_vector_cc_3:
; CHECK-OBJ-LABEL: Name: rvv_vector_cc_3
; CHECK-OBJ: Other [ (0x80)
  ret void
}

define void @rvv_vector_cc_4(<vscale x 4 x i1> %arg) {
; CHECK-ASM: .variant_cc rvv_vector_cc_4
; CHECK-ASM-NEXT: rvv_vector_cc_4:
; CHECK-OBJ-LABEL: Name: rvv_vector_cc_4
; CHECK-OBJ: Other [ (0x80)
  ret void
}
