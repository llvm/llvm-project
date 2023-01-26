; RUN: opt -passes=globalopt --mtriple=amdgcn-amd-amdhsa < %s -S | FileCheck %s
; REQUIRES: amdgpu-registered-target

@gvar = internal unnamed_addr global i32 undef
@lvar = internal unnamed_addr addrspace(3) global i32 undef

; Should optimize @gvar.
; CHECK-NOT: @gvar

; Negative test for AS(3). Skip shrink global to bool optimization.
; CHECK: @lvar = internal unnamed_addr addrspace(3) global i32 undef

define void @test_global_var(i1 %i) {
; CHECK-LABEL: @test_global_var(
; CHECK:    store volatile i32 10, ptr undef, align 4
;
entry:
  br i1 %i, label %bb1, label %exit
bb1:
  store i32 10, ptr @gvar
  br label %exit
exit:
  %ld = load i32, ptr @gvar
  store volatile i32 %ld, ptr undef
  ret void
}

define void @test_lds_var(i1 %i) {
; CHECK-LABEL: @test_lds_var(
; CHECK:    store i32 10, ptr addrspace(3) @lvar, align 4
; CHECK:    [[LD:%.*]] = load i32, ptr addrspace(3) @lvar, align 4
; CHECK:    store volatile i32 [[LD]], ptr undef, align 4
;
entry:
  br i1 %i, label %bb1, label %exit
bb1:
  store i32 10, ptr addrspace(3) @lvar
  br label %exit
exit:
  %ld = load i32, ptr addrspace(3) @lvar
  store volatile i32 %ld, ptr undef
  ret void
}
