; RUN: opt -mtriple amdgcn-unknown-amdhsa -mcpu=gfx90a -passes='print<divergence>' -disable-output %s 2>&1 | FileCheck %s

; CHECK-LABEL: Divergence Analysis' for function 'read_register_exec':
; CHECK-NOT: DIVERGENT
define i64 @read_register_exec() {
  %reg = call i64 @llvm.read_register.i64(metadata !0)
  ret i64 %reg
}

; CHECK-LABEL: Divergence Analysis' for function 'read_register_m0':
; CHECK-NOT: DIVERGENT
define i32 @read_register_m0() {
  %reg = call i32 @llvm.read_register.i32(metadata !1)
  ret i32 %reg
}

; CHECK-LABEL: Divergence Analysis' for function 'read_register_s17':
; CHECK-NOT: DIVERGENT
define i32 @read_register_s17() {
  %reg = call i32 @llvm.read_register.i32(metadata !2)
  ret i32 %reg
}

; CHECK-LABEL: Divergence Analysis' for function 'read_register_s17_i17':
; CHECK-NOT: DIVERGENT
define i17 @read_register_s17_i17() {
  %reg = call i17 @llvm.read_register.i17(metadata !2)
  ret i17 %reg
}

; CHECK-LABEL: Divergence Analysis' for function 'read_register_v0':
; CHECK: DIVERGENT
define i32 @read_register_v0() {
  %reg = call i32 @llvm.read_register.i32(metadata !3)
  ret i32 %reg
}

; CHECK-LABEL: Divergence Analysis' for function 'read_register_v0_v1':
; CHECK: DIVERGENT
define i64 @read_register_v0_v1() {
  %reg = call i64 @llvm.read_register.i64(metadata !4)
  ret i64 %reg
}

; CHECK-LABEL: Divergence Analysis' for function 'read_register_a0':
; CHECK: DIVERGENT
define i32 @read_register_a0() {
  %reg = call i32 @llvm.read_register.i32(metadata !5)
  ret i32 %reg
}

; CHECK-LABEL: Divergence Analysis' for function 'read_register_a0_a1':
; CHECK: DIVERGENT
define i64 @read_register_a0_a1() {
  %reg = call i64 @llvm.read_register.i64(metadata !6)
  ret i64 %reg
}

; CHECK-LABEL: Divergence Analysis' for function 'read_register_vcc_i64':
; CHECK-NOT: DIVERGENT
define i64 @read_register_vcc_i64() {
  %reg = call i64 @llvm.read_register.i64(metadata !7)
  ret i64 %reg
}

; CHECK-LABEL: Divergence Analysis' for function 'read_register_vcc_i1':
; CHECK: DIVERGENT
define i1 @read_register_vcc_i1() {
  %reg = call i1 @llvm.read_register.i1(metadata !7)
  ret i1 %reg
}

; CHECK-LABEL: Divergence Analysis' for function 'read_register_invalid_reg':
; CHECK-NOT: DIVERGENT
define i64 @read_register_invalid_reg() {
  %reg = call i64 @llvm.read_register.i64(metadata !8)
  ret i64 %reg
}

; CHECK-LABEL: Divergence Analysis' for function 'read_register_flat_scratch':
; CHECK-NOT: DIVERGENT
define i32 @read_register_flat_scratch() {
  %reg = call i32 @llvm.read_register.i32(metadata !9)
  ret i32 %reg
}

; CHECK-LABEL: Divergence Analysis' for function 'read_register_vcc_lo_i32':
; CHECK-NOT: DIVERGENT
define i32 @read_register_vcc_lo_i32() {
  %reg = call i32 @llvm.read_register.i32(metadata !10)
  ret i32 %reg
}

; CHECK-LABEL: Divergence Analysis' for function 'read_register_vcc_hi_i32':
; CHECK-NOT: DIVERGENT
define i32 @read_register_vcc_hi_i32() {
  %reg = call i32 @llvm.read_register.i32(metadata !11)
  ret i32 %reg
}

; CHECK-LABEL: Divergence Analysis' for function 'read_register_exec_lo_i32':
; CHECK-NOT: DIVERGENT
define i32 @read_register_exec_lo_i32() {
  %reg = call i32 @llvm.read_register.i32(metadata !12)
  ret i32 %reg
}

; CHECK-LABEL: Divergence Analysis' for function 'read_register_exec_hi_i32':
; CHECK-NOT: DIVERGENT
define i32 @read_register_exec_hi_i32() {
  %reg = call i32 @llvm.read_register.i32(metadata !13)
  ret i32 %reg
}

; FIXME: Why does the verifier allow this?
; CHECK-LABEL: Divergence Analysis' for function 'read_register_empty_str_i32':
; CHECK-NOT: DIVERGENT
define i32 @read_register_empty_str_i32() {
  %reg = call i32 @llvm.read_register.i32(metadata !14)
  ret i32 %reg
}

declare i64 @llvm.read_register.i64(metadata)
declare i32 @llvm.read_register.i32(metadata)
declare i17 @llvm.read_register.i17(metadata)
declare i1 @llvm.read_register.i1(metadata)

!0 = !{!"exec"}
!1 = !{!"m0"}
!2 = !{!"s17"}
!3 = !{!"v0"}
!4 = !{!"v[0:1]"}
!5 = !{!"a0"}
!6 = !{!"a[0:1]"}
!7 = !{!"vcc"}
!8 = !{!"not a register"}
!9 = !{!"flat_scratch"}
!10 = !{!"vcc_lo"}
!11 = !{!"vcc_hi"}
!12 = !{!"exec_lo"}
!13 = !{!"exec_hi"}
!14 = !{!""}
