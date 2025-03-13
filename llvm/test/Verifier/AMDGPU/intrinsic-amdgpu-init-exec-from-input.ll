; RUN: not llvm-as -disable-output 2>&1 %s | FileCheck %s

declare void @llvm.amdgcn.init.exec.from.input(i32, i32 immarg)

; CHECK: only inreg arguments to the parent function are valid as inputs to this intrinsic
; CHECK-NEXT: call void @llvm.amdgcn.init.exec.from.input(i32 0, i32 0)
define void @init_exec_from_input_fail_constant() {
  call void @llvm.amdgcn.init.exec.from.input(i32 0, i32 0)
  ret void
}

; CHECK: only inreg arguments to the parent function are valid as inputs to this intrinsic
; CHECK-NEXT: call void @llvm.amdgcn.init.exec.from.input(i32 %b, i32 0)
define void @init_exec_from_input_fail_not_inreg(i32 inreg %a, i32 %b) {
  call void @llvm.amdgcn.init.exec.from.input(i32 %b, i32 0)
  ret void
}

; CHECK: only inreg arguments to the parent function are valid as inputs to this intrinsic
; CHECK-NEXT: call void @llvm.amdgcn.init.exec.from.input(i32 %c, i32 0)
define void @init_exec_from_input_fail_instruction(i32 inreg %a, i32 %b) {
  %c = add i32 %a, %b
  call void @llvm.amdgcn.init.exec.from.input(i32 %c, i32 0)
  ret void
}
