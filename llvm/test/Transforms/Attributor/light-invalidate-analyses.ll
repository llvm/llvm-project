; RUN: opt -disable-output -passes='function(require<memoryssa>),cgscc(attributor-light-cgscc),function(print<memoryssa>)' %s 2>&1 | FileCheck %s

; Make sure function analyses cached before attributor-light are invalidated
; when it only changes attributes. Once @callee is known to not access memory,
; the call to it in @caller must not be a MemoryDef.

define internal void @callee(ptr %p) {
  ret void
}

; CHECK-LABEL: MemorySSA for function: caller
; CHECK: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i32 0, ptr %q
; CHECK-NEXT: call void @callee(
; CHECK-NEXT: ; MemoryUse(1)
; CHECK-NEXT: %v = load i32, ptr %q
define void @caller(ptr %p, ptr %q) {
  store i32 0, ptr %q
  call void @callee(ptr %p)
  %v = load i32, ptr %q
  store i32 %v, ptr %p
  ret void
}
