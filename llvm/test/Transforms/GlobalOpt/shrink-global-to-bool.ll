; RUN: opt -passes=globalopt --mtriple=amdgcn-amd-amdhsa < %s -S | FileCheck %s
; REQUIRES: amdgpu-registered-target

@gvar = internal unnamed_addr global i32 undef
@lvar = internal unnamed_addr addrspace(3) global i32 undef

; Should optimize @gvar.
; CHECK-NOT: @gvar

; Negative test for AS(3). Skip shrink global to bool optimization.
; CHECK: @lvar = internal unnamed_addr addrspace(3) global i32 undef

define void @test_global_var() {
; CHECK-LABEL: @test_global_var(
; CHECK:    store volatile i32 10, i32* undef, align 4
;
entry:
  store i32 10, i32* @gvar
  br label %exit
exit:
  %ld = load i32, i32* @gvar
  store volatile i32 %ld, i32* undef
  ret void
}

define void @test_lds_var() {
; CHECK-LABEL: @test_lds_var(
; CHECK:    store i32 10, i32 addrspace(3)* @lvar, align 4
; CHECK:    [[LD:%.*]] = load i32, i32 addrspace(3)* @lvar, align 4
; CHECK:    store volatile i32 [[LD]], i32* undef, align 4
;
entry:
  store i32 10, i32 addrspace(3)* @lvar
  br label %exit
exit:
  %ld = load i32, i32 addrspace(3)* @lvar
  store volatile i32 %ld, i32* undef
  ret void
}
