; RUN: sed -e '/^; MARK$/,$d' %s | not llc -mtriple=amdgcn -mcpu=gfx942 2>&1 | FileCheck %s
; RUN: sed -e '1,/^; MARK$/d' %s | llc -mtriple=amdgcn -mcpu=gfx942 -filetype=null

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.init.exec.from.input(i32, i32 immarg) #0
attributes #0 = { convergent nocallback nofree nounwind willreturn }

; CHECK: only inreg arguments to the parent function are valid as inputs to this intrinsic
; CHECK-NEXT: call void @llvm.amdgcn.init.exec.from.input(i32 0, i32 0)
define amdgpu_ps void @init_exec_from_input_fail_immarg(i32 inreg %a, i32 %b) {
  call void @llvm.amdgcn.init.exec.from.input(i32 0, i32 0)
  ret void
}

; CHECK: only inreg arguments to the parent function are valid as inputs to this intrinsic
; CHECK-NEXT: call void @llvm.amdgcn.init.exec.from.input(i32 %b, i32 0)
define amdgpu_ps void @init_exec_from_input_fail_not_inreg(i32 inreg %a, i32 %b) {
  call void @llvm.amdgcn.init.exec.from.input(i32 %b, i32 0)
  ret void
}

; MARK

define amdgpu_ps void @init_exec_from_input_success(i32 inreg %a, i32 %b) {
  call void @llvm.amdgcn.init.exec.from.input(i32 %a, i32 0)
  ret void
}
