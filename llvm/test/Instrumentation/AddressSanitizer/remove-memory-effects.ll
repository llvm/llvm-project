; Remove possible memory effects from functions that are invalidated by
; AddressSanitizer instrumentation.

; RUN: opt -passes='asan<use-after-scope>' -S %s | FileCheck %s

; CHECK: @foo(ptr writeonly) #[[ATTRS_FOO:[0-9]+]]
declare void @foo(ptr writeonly) memory(argmem: write)

; CHECK: @foo2(ptr writeonly) #[[ATTRS_FOO2:[0-9]+]]
declare void @foo2(ptr writeonly) memory(argmem: readwrite, inaccessiblemem: write)

; CHECK: @foo3(ptr) #[[ATTRS_FOO3:[0-9]+]]
declare void @foo3(ptr) memory(inaccessiblemem: write)

; CHECK: @bar() #[[ATTRS_BAR:[0-9]+]]
define void @bar() sanitize_address {
entry:
  %x = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr %x)
  call void @foo(ptr %x)
  call void @llvm.lifetime.end.p0(ptr %x)
  ret void
}

; CHECK: attributes #[[ATTRS_FOO]] = { nobuiltin memory(readwrite, argmem: write, inaccessiblemem: none
; CHECK: attributes #[[ATTRS_FOO2]] = { nobuiltin memory(readwrite, inaccessiblemem: write
; CHECK: attributes #[[ATTRS_FOO3]] = { memory(inaccessiblemem: write) }
; CHECK: attributes #[[ATTRS_BAR]] = { sanitize_address }
