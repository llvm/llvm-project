; Remove possible memory effects from functions that are invalidated by
; AddressSanitizer instrumentation.

; RUN: opt < %s -passes=asan -asan-use-after-scope -S | FileCheck %s

target datalayout = "e-i64:64-f80:128-s:64-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @foo(ptr) #[[ATTRS_FOO:[0-9]+]]
declare void @foo(ptr) memory(argmem: write)

; CHECK: @bar() #[[ATTRS_BAR:[0-9]+]]
define void @bar() sanitize_address {
entry:
  %x = alloca i32, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr %x)
  call void @foo(ptr %x)
  call void @llvm.lifetime.end.p0(i64 4, ptr %x)
  ret void
}

; CHECK: attributes #[[ATTRS_FOO]] = { nobuiltin }
; CHECK: attributes #[[ATTRS_BAR]] = { sanitize_address }
