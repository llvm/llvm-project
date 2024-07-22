; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK: llvm.func @f()
declare void @f()

; CHECK-LABEL: @call_convergent
define void @call_convergent() {
; CHECK: llvm.call @f() {convergent}
  call void @f() convergent
  ret void
}

// -----

; CHECK: llvm.func @f()
declare void @f()

; CHECK-LABEL: @call_no_unwind
define void @call_no_unwind() {
; CHECK: llvm.call @f() {no_unwind}
  call void @f() nounwind
  ret void
}

// -----

; CHECK: llvm.func @f()
declare void @f()

; CHECK-LABEL: @call_will_return
define void @call_will_return() {
; CHECK: llvm.call @f() {will_return}
  call void @f() willreturn
  ret void
}

// -----

; CHECK: llvm.func @f()
declare void @f()

; CHECK-LABEL: @call_memory_effects
define void @call_memory_effects() {
; CHECK: llvm.call @f() {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>}
  call void @f() memory(none)
; CHECK: llvm.call @f() {memory = #llvm.memory_effects<other = none, argMem = write, inaccessibleMem = read>}
  call void @f() memory(none, argmem: write, inaccessiblemem: read)
; CHECK: llvm.call @f() {memory = #llvm.memory_effects<other = write, argMem = none, inaccessibleMem = write>}
  call void @f() memory(write, argmem: none)
; CHECK: llvm.call @f() {memory = #llvm.memory_effects<other = readwrite, argMem = readwrite, inaccessibleMem = read>}
  call void @f() memory(readwrite, inaccessiblemem: read)
; CHECK: llvm.call @f()
; CHECK-NOT: #llvm.memory_effects
; CHECK-SAME: : () -> ()
  call void @f() memory(readwrite)
  ret void
}
