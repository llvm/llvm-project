; RUN: not llc -global-isel=0 -mtriple=amdgpu6.00 -filetype=null < %s 2>&1 | FileCheck --implicit-check-not=error %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu6.00 -filetype=null < %s 2>&1 | FileCheck --implicit-check-not=error %s

; flat_scratch{,_lo,_hi} require the flat address space, so named-register
; access to them should be rejected on earlier subtargets.

; CHECK: error: <unknown>:0:0: invalid register "flat_scratch" for llvm.read_register
define void @test_read_flat_scratch() {
  %v = call i64 @llvm.read_register.i64(metadata !0)
  call void asm sideeffect "; use $0", "s"(i64 %v)
  ret void
}

; CHECK: error: <unknown>:0:0: invalid register "flat_scratch_lo" for llvm.read_register
define void @test_read_flat_scratch_lo() {
  %v = call i32 @llvm.read_register.i32(metadata !1)
  call void asm sideeffect "; use $0", "s"(i32 %v)
  ret void
}

; CHECK: error: <unknown>:0:0: invalid register "flat_scratch_hi" for llvm.read_register
define void @test_read_flat_scratch_hi() {
  %v = call i32 @llvm.read_register.i32(metadata !2)
  call void asm sideeffect "; use $0", "s"(i32 %v)
  ret void
}

; CHECK: error: <unknown>:0:0: invalid register "flat_scratch" for llvm.write_register
define void @test_write_flat_scratch(i64 inreg %val) {
  call void @llvm.write_register.i64(metadata !0, i64 %val)
  ret void
}

; CHECK: error: <unknown>:0:0: invalid register "flat_scratch_lo" for llvm.write_register
define void @test_write_flat_scratch_lo(i32 inreg %val) {
  call void @llvm.write_register.i32(metadata !1, i32 %val)
  ret void
}

; CHECK: error: <unknown>:0:0: invalid register "flat_scratch_hi" for llvm.write_register
define void @test_write_flat_scratch_hi(i32 inreg %val) {
  call void @llvm.write_register.i32(metadata !2, i32 %val)
  ret void
}

!0 = !{!"flat_scratch"}
!1 = !{!"flat_scratch_lo"}
!2 = !{!"flat_scratch_hi"}
