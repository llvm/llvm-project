; RUN: opt -mtriple=amdgcn-amd-amdhsa -load-store-vectorizer -S -o - %s | FileCheck %s

; CHECK-LABEL: @no_implicit_float(
; CHECK: store i32
; CHECK: store i32
; CHECK: store i32
; CHECK: store i32
define void @no_implicit_float(i32 addrspace(1)* %out) #0 {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  %out.gep.2 = getelementptr i32, i32 addrspace(1)* %out, i32 2
  %out.gep.3 = getelementptr i32, i32 addrspace(1)* %out, i32 3

  store i32 123, i32 addrspace(1)* %out.gep.1
  store i32 456, i32 addrspace(1)* %out.gep.2
  store i32 333, i32 addrspace(1)* %out.gep.3
  store i32 1234, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind noimplicitfloat }
