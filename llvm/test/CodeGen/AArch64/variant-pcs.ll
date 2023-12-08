; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -o - %s | FileCheck %s --check-prefix=CHECK-ASM --strict-whitespace
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve -filetype=obj -o - %s \
; RUN:   | llvm-readobj --symbols - | FileCheck %s --check-prefix=CHECK-OBJ

define i32 @base_pcs() {
; CHECK-ASM-LABEL: base_pcs:
; CHECK-ASM-NOT: .variant_pcs
; CHECK-OBJ-LABEL: Name: base_pcs
; CHECK-OBJ: Other: 0
  ret i32 42
}

define aarch64_vector_pcs <4 x i32> @neon_vector_pcs_1(<4 x i32> %arg) {
; CHECK-ASM: .variant_pcs	neon_vector_pcs_1
; CHECK-ASM-NEXT: neon_vector_pcs_1:
; CHECK-OBJ-LABEL: Name: neon_vector_pcs_1
; CHECK-OBJ: Other [ (0x80)
  ret <4 x i32> %arg
}

define <vscale x 4 x i32> @sve_vector_pcs_1() {
; CHECK-ASM: .variant_pcs	sve_vector_pcs_1
; CHECK-ASM-NEXT: sve_vector_pcs_1:
; CHECK-OBJ-LABEL: Name: sve_vector_pcs_1
; CHECK-OBJ: Other [ (0x80)
  ret <vscale x 4 x i32> undef
}

define <vscale x 4 x i1> @sve_vector_pcs_2() {
; CHECK-ASM: .variant_pcs	sve_vector_pcs_2
; CHECK-ASM-NEXT: sve_vector_pcs_2:
; CHECK-OBJ-LABEL: Name: sve_vector_pcs_2
; CHECK-OBJ: Other [ (0x80)
  ret <vscale x 4 x i1> undef
}

define void @sve_vector_pcs_3(<vscale x 4 x i32> %arg) {
; CHECK-ASM: .variant_pcs	sve_vector_pcs_3
; CHECK-ASM-NEXT: sve_vector_pcs_3:
; CHECK-OBJ-LABEL: Name: sve_vector_pcs_3
; CHECK-OBJ: Other [ (0x80)
  ret void
}

define void @sve_vector_pcs_4(<vscale x 4 x i1> %arg) {
; CHECK-ASM: .variant_pcs	sve_vector_pcs_4
; CHECK-ASM-NEXT: sve_vector_pcs_4:
; CHECK-OBJ-LABEL: Name: sve_vector_pcs_4
; CHECK-OBJ: Other [ (0x80)
  ret void
}
