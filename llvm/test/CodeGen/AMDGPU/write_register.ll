; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck %s

declare void @llvm.write_register.i32(metadata, i32) #0
declare void @llvm.write_register.i64(metadata, i64) #0

; CHECK-LABEL: {{^}}test_write_m0:
define void @test_write_m0(i32 %val) #0 {
  call void @llvm.write_register.i32(metadata !0, i32 0)
  call void @llvm.write_register.i32(metadata !0, i32 -1)
  call void @llvm.write_register.i32(metadata !0, i32 %val)
  ret void
}

; CHECK-LABEL: {{^}}test_write_exec:
; CHECK: s_mov_b64 exec, 0
; CHECK: s_mov_b64 exec, -1
; CHECK: s_mov_b64 exec, s{{\[[0-9]+:[0-9]+\]}}
define void @test_write_exec(i64 %val) #0 {
  call void @llvm.write_register.i64(metadata !1, i64 0)
  call void @llvm.write_register.i64(metadata !1, i64 -1)
  call void @llvm.write_register.i64(metadata !1, i64 %val)
  ret void
}

; CHECK-LABEL: {{^}}test_write_flat_scratch:
; CHECK: s_mov_b64 flat_scratch, 0
; CHECK: s_mov_b64 flat_scratch, -1
; CHECK: s_mov_b64 flat_scratch, s{{\[[0-9]+:[0-9]+\]}}
define void @test_write_flat_scratch(i64 %val) #0 {
  call void @llvm.write_register.i64(metadata !2, i64 0)
  call void @llvm.write_register.i64(metadata !2, i64 -1)
  call void @llvm.write_register.i64(metadata !2, i64 %val)
  ret void
}

; CHECK-LABEL: {{^}}test_write_flat_scratch_lo:
; CHECK: s_mov_b32 flat_scratch_lo, 0
; CHECK: s_mov_b32 flat_scratch_lo, s{{[0-9]+}}
define void @test_write_flat_scratch_lo(i32 %val) #0 {
  call void @llvm.write_register.i32(metadata !3, i32 0)
  call void @llvm.write_register.i32(metadata !3, i32 %val)
  ret void
}

; CHECK-LABEL: {{^}}test_write_flat_scratch_hi:
; CHECK: s_mov_b32 flat_scratch_hi, 0
; CHECK: s_mov_b32 flat_scratch_hi, s{{[0-9]+}}
define void @test_write_flat_scratch_hi(i32 %val) #0 {
  call void @llvm.write_register.i32(metadata !4, i32 0)
  call void @llvm.write_register.i32(metadata !4, i32 %val)
  ret void
}

; CHECK-LABEL: {{^}}test_write_exec_lo:
; CHECK: s_mov_b32 exec_lo, 0
; CHECK: s_mov_b32 exec_lo, s{{[0-9]+}}
define void @test_write_exec_lo(i32 %val) #0 {
  call void @llvm.write_register.i32(metadata !5, i32 0)
  call void @llvm.write_register.i32(metadata !5, i32 %val)
  ret void
}

; CHECK-LABEL: {{^}}test_write_exec_hi:
; CHECK: s_mov_b32 exec_hi, 0
; CHECK: s_mov_b32 exec_hi, s{{[0-9]+}}
define void @test_write_exec_hi(i32 %val) #0 {
  call void @llvm.write_register.i32(metadata !6, i32 0)
  call void @llvm.write_register.i32(metadata !6, i32 %val)
  ret void
}

attributes #0 = { nounwind }

!0 = !{!"m0"}
!1 = !{!"exec"}
!2 = !{!"flat_scratch"}
!3 = !{!"flat_scratch_lo"}
!4 = !{!"flat_scratch_hi"}
!5 = !{!"exec_lo"}
!6 = !{!"exec_hi"}
