; RUN: not llvm-as -o /dev/null 2>&1 %s | FileCheck %s

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
