; RUN: split-file %s %t
; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %t/top-level-bool.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %t/nested-bool.ll -o /dev/null 2>&1 | FileCheck %s

; CHECK: llvm.spv.abort message type must be a concrete SPIR-V type

;--- top-level-bool.ll
declare void @llvm.spv.abort.i1(i1) #0
define void @abort_with_bool(i1 %b) {
entry:
  call void @llvm.spv.abort.i1(i1 %b)
  unreachable
}
attributes #0 = { noreturn }

;--- nested-bool.ll
%B = type { i32, i1 }
declare void @llvm.spv.abort.s_Bs(%B) #0
define void @abort_with_struct_of_bool(i1 %b) {
entry:
  %s0 = insertvalue %B poison, i32 0, 0
  %s1 = insertvalue %B %s0, i1 %b, 1
  call void @llvm.spv.abort.s_Bs(%B %s1)
  unreachable
}
attributes #0 = { noreturn }
