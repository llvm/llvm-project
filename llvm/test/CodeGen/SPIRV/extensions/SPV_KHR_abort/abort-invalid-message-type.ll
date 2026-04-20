; RUN: split-file %s %t
; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %t/top-level.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %t/nested-struct.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %t/nested-array.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %t/top-level-bool.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %t/nested-bool.ll -o /dev/null 2>&1 | FileCheck %s

; CHECK: llvm.spv.abort message type must be a concrete SPIR-V type

;--- top-level.ll
declare void @llvm.spv.abort(...) #0
define void @abort_with_pointer(ptr %p) {
entry:
  call void (...) @llvm.spv.abort(ptr %p)
  unreachable
}
attributes #0 = { noreturn }

;--- nested-struct.ll
%S = type { i32, ptr }
declare void @llvm.spv.abort(...) #0
define void @abort_with_struct_of_pointer(ptr %p) {
entry:
  %s0 = insertvalue %S poison, i32 0, 0
  %s1 = insertvalue %S %s0, ptr %p, 1
  call void (...) @llvm.spv.abort(%S %s1)
  unreachable
}
attributes #0 = { noreturn }

;--- nested-array.ll
declare void @llvm.spv.abort(...) #0
define void @abort_with_array_of_pointer(ptr %p) {
entry:
  %a0 = insertvalue [2 x ptr] poison, ptr %p, 0
  %a1 = insertvalue [2 x ptr] %a0, ptr %p, 1
  call void (...) @llvm.spv.abort([2 x ptr] %a1)
  unreachable
}
attributes #0 = { noreturn }

;--- top-level-bool.ll
declare void @llvm.spv.abort(...) #0
define void @abort_with_bool(i1 %b) {
entry:
  call void (...) @llvm.spv.abort(i1 %b)
  unreachable
}
attributes #0 = { noreturn }

;--- nested-bool.ll
%B = type { i32, i1 }
declare void @llvm.spv.abort(...) #0
define void @abort_with_struct_of_bool(i1 %b) {
entry:
  %s0 = insertvalue %B poison, i32 0, 0
  %s1 = insertvalue %B %s0, i1 %b, 1
  call void (...) @llvm.spv.abort(%B %s1)
  unreachable
}
attributes #0 = { noreturn }
