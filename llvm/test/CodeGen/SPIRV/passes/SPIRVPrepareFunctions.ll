; RUN: opt -S -passes=spirv-prepare-functions -mtriple=spirv64-unknown-unknown < %s | FileCheck %s

; @llvm.bswap.* is replaced with a call to a SPIR-V helper function whose
; body implements the byte-swap with shifts/masks/ors.
define i32 @bswap_i32(i32 %x) {
; CHECK-LABEL: define i32 @bswap_i32(
; CHECK: call i32 @spirv.llvm_bswap_i32(i32 %x)
  %r = call i32 @llvm.bswap.i32(i32 %x)
  ret i32 %r
}

; @llvm.fshl is wrapped in a generated SPIR-V helper function.
define i32 @fshl_i32(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: define i32 @fshl_i32(
; CHECK: call i32 @spirv.llvm_fshl_i32(i32 %a, i32 %b, i32 %c)
  %r = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 %c)
  ret i32 %r
}

; @llvm.fshr is wrapped in a generated SPIR-V helper function.
define i32 @fshr_i32(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: define i32 @fshr_i32(
; CHECK: call i32 @spirv.llvm_fshr_i32(i32 %a, i32 %b, i32 %c)
  %r = call i32 @llvm.fshr.i32(i32 %a, i32 %b, i32 %c)
  ret i32 %r
}

; The bswap helper is materialized with the standard shift/mask/or unrolling.
; CHECK-LABEL: define i32 @spirv.llvm_bswap_i32(i32 %0)
; CHECK-DAG:   shl i32 %0, 24
; CHECK-DAG:   shl i32 %0, 8
; CHECK-DAG:   lshr i32 %0, 8
; CHECK-DAG:   lshr i32 %0, 24
; CHECK-DAG:   and i32 {{.*}}, 16711680
; CHECK-DAG:   and i32 {{.*}}, 65280
; CHECK:       ret i32

; The fshl helper is materialized once with the spir_func calling convention
; and computes the rotate body in terms of urem/shl/sub/lshr/or.
; CHECK-LABEL: define spir_func i32 @spirv.llvm_fshl_i32(i32 %0, i32 %1, i32 %2)
; CHECK:       %[[#MOD:]] = urem i32 %2, 32
; CHECK:       %[[#A:]]   = shl i32 %0, %[[#MOD]]
; CHECK:       %[[#SUB:]] = sub i32 32, %[[#MOD]]
; CHECK:       %[[#B:]]   = lshr i32 %1, %[[#SUB]]
; CHECK:       %[[#OR:]]  = or i32 %[[#A]], %[[#B]]
; CHECK:       ret i32 %[[#OR]]

; The fshr helper mirrors fshl but shifts the LSB operand right first.
; CHECK-LABEL: define spir_func i32 @spirv.llvm_fshr_i32(i32 %0, i32 %1, i32 %2)
; CHECK:       %[[#MOD:]] = urem i32 %2, 32
; CHECK:       %[[#A:]]   = lshr i32 %1, %[[#MOD]]
; CHECK:       %[[#SUB:]] = sub i32 32, %[[#MOD]]
; CHECK:       %[[#B:]]   = shl i32 %0, %[[#SUB]]
; CHECK:       %[[#OR:]]  = or i32 %[[#A]], %[[#B]]
; CHECK:       ret i32 %[[#OR]]

declare i32 @llvm.bswap.i32(i32)
declare i32 @llvm.fshl.i32(i32, i32, i32)
declare i32 @llvm.fshr.i32(i32, i32, i32)
