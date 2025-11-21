; Remove possible memory effects from functions that are invalidated by
; AddressSanitizer instrumentation.

; RUN: opt -passes='asan<use-after-scope>' -S %s | FileCheck %s

; CHECK: @foo(ptr writeonly) #[[ATTRS_FOO:[0-9]+]]
declare void @foo(ptr writeonly) memory(argmem: write)

; CHECK: @bar() #[[ATTRS_BAR:[0-9]+]]
define void @bar() sanitize_address {
entry:
  %x = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr %x)
  call void @foo(ptr %x)
  call void @llvm.lifetime.end.p0(ptr %x)
  ret void
}

; CHECK: attributes #[[ATTRS_FOO]] = { nobuiltin }
; CHECK: attributes #[[ATTRS_BAR]] = { sanitize_address }
