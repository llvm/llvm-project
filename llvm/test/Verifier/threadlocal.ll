; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

@var = global i32 0
@tlsvar = thread_local addrspace(1) global i32 0

define void @fail0(ptr %arg) {
; CHECK: llvm.threadlocal.address first argument must be a GlobalValue
  %p0 = call ptr @llvm.threadlocal.address(ptr %arg)
  store i32 42, ptr %p0, align 4
  ret void
}

define void @fail1() {
; CHECK: llvm.threadlocal.address first argument must be a GlobalValue
  %p0 = call ptr @llvm.threadlocal.address.p0(ptr addrspacecast (ptr addrspace(1) @tlsvar to ptr addrspace(0)))
  store i32 42, ptr %p0, align 4
  ret void
}



define void @fail2() {
; CHECK: llvm.threadlocal.address operand isThreadLocal() must be true
  %p0 = call ptr @llvm.threadlocal.address(ptr @var)
  store i32 42, ptr %p0, align 4
  ret void
}

define void @fail3() {
; CHECK: llvm.threadlocal.address operand isThreadLocal() must be true
  %p0 = call ptr @llvm.threadlocal.address(ptr @fail2)
  store i32 42, ptr %p0, align 4
  ret void
}
