; RUN: not llvm-as --data-layout="A5" -disable-output %s 2>&1 | FileCheck %s

; CHECK: llvm.sponentry must return a pointer to the stack
define ptr addrspace(0) @p0()  {
entry:
  %res = tail call ptr addrspace(0) @llvm.sponentry()
  ret ptr addrspace(0) %res
}

; CHECK: llvm.sponentry must return a pointer to the stack
define ptr addrspace(1) @p1()  {
entry:
  %res = tail call ptr addrspace(1) @llvm.sponentry()
  ret ptr addrspace(1) %res
}

; CHECK: llvm.sponentry must return a pointer to the stack
define ptr addrspace(2) @p2()  {
entry:
  %res = tail call ptr addrspace(2) @llvm.sponentry()
  ret ptr addrspace(2) %res
}

; CHECK: llvm.sponentry must return a pointer to the stack
define ptr addrspace(3) @p3()  {
entry:
  %res = tail call ptr addrspace(3) @llvm.sponentry()
  ret ptr addrspace(3) %res
}

; CHECK: llvm.sponentry must return a pointer to the stack
define ptr addrspace(4) @p4()  {
entry:
  %res = tail call ptr addrspace(4) @llvm.sponentry()
  ret ptr addrspace(4) %res
}

; CHECK: llvm.sponentry must return a pointer to the stack
define ptr addrspace(6) @p6()  {
entry:
  %res = tail call ptr addrspace(6) @llvm.sponentry()
  ret ptr addrspace(6) %res
}

; CHECK: llvm.sponentry must return a pointer to the stack
define ptr addrspace(7) @p7()  {
entry:
  %res = tail call ptr addrspace(7) @llvm.sponentry()
  ret ptr addrspace(7) %res
}

; CHECK: llvm.sponentry must return a pointer to the stack
define ptr addrspace(8) @p8()  {
entry:
  %res = tail call ptr addrspace(8) @llvm.sponentry()
  ret ptr addrspace(8) %res
}

; CHECK: llvm.sponentry must return a pointer to the stack
define ptr @no_as()  {
entry:
  %res = tail call ptr @llvm.sponentry()
  ret ptr %res
}
