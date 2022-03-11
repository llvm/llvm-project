; RUN: not llvm-as -opaque-pointers < %s 2>&1 | FileCheck %s

; CHECK: invalid forward reference to function 'f' with wrong type: expected 'ptr' but was 'ptr addrspace(1)'

@a = alias void (), ptr addrspace(1) @f

define void @f() {
  ret void
}
