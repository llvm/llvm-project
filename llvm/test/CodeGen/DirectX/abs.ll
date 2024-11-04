; RUN: opt -S  -dxil-intrinsic-expansion  -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefixes=CHECK,EXPCHECK
; RUN: opt -S  -dxil-op-lower  -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefixes=CHECK,DOPCHECK

; Make sure dxil operation function calls for abs are generated for int16_t/int/int64_t.

; CHECK-LABEL: abs_i16
define noundef i16 @abs_i16(i16 noundef %a) {
entry:
; CHECK: sub i16 0, %a
; EXPCHECK: call i16 @llvm.smax.i16(i16 %a, i16 %{{.*}})
; DOPCHECK: call i16 @dx.op.binary.i16(i32 37, i16 %a, i16 %{{.*}})
  %elt.abs = call i16 @llvm.abs.i16(i16 %a, i1 false)
  ret i16 %elt.abs
}

; CHECK-LABEL: abs_i32
define noundef i32 @abs_i32(i32 noundef %a) {
entry:
; CHECK: sub i32 0, %a
; EXPCHECK: call i32 @llvm.smax.i32(i32 %a, i32 %{{.*}})
; DOPCHECK: call i32 @dx.op.binary.i32(i32 37, i32 %a, i32 %{{.*}})
  %elt.abs = call i32 @llvm.abs.i32(i32 %a, i1 false)
  ret i32 %elt.abs
}

; CHECK-LABEL: abs_i64
define noundef i64 @abs_i64(i64 noundef %a) {
entry:
; CHECK: sub i64 0, %a
; EXPCHECK: call i64 @llvm.smax.i64(i64 %a, i64 %{{.*}})
; DOPCHECK: call i64 @dx.op.binary.i64(i32 37, i64 %a, i64 %{{.*}})
  %elt.abs = call i64 @llvm.abs.i64(i64 %a, i1 false)
  ret i64 %elt.abs
}

declare i16 @llvm.abs.i16(i16, i1 immarg)
declare i32 @llvm.abs.i32(i32, i1 immarg)
declare i64 @llvm.abs.i64(i64, i1 immarg)
