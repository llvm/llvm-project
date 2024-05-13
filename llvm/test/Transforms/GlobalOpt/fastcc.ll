; RUN: opt < %s -passes=globalopt -S | FileCheck %s

declare token @llvm.call.preallocated.setup(i32)
declare ptr @llvm.call.preallocated.arg(token, i32)

define internal i32 @f(ptr %m) {
; CHECK-LABEL: define internal fastcc i32 @f
  %v = load i32, ptr %m
  ret i32 %v
}

define internal x86_thiscallcc i32 @g(ptr %m) {
; CHECK-LABEL: define internal fastcc i32 @g
  %v = load i32, ptr %m
  ret i32 %v
}

; Leave this one alone, because the user went out of their way to request this
; convention.
define internal coldcc i32 @h(ptr %m) {
; CHECK-LABEL: define internal coldcc i32 @h
  %v = load i32, ptr %m
  ret i32 %v
}

define internal i32 @j(ptr %m) {
; CHECK-LABEL: define internal i32 @j
  %v = load i32, ptr %m
  ret i32 %v
}

define internal i32 @inalloca(ptr inalloca(i32) %p) {
; CHECK-LABEL: define internal fastcc i32 @inalloca(ptr %p)
  %rv = load i32, ptr %p
  ret i32 %rv
}

define i32 @inalloca2_caller(ptr inalloca(i32) %p) {
  %rv = musttail call i32 @inalloca2(ptr inalloca(i32) %p)
  ret i32 %rv
}
define internal i32 @inalloca2(ptr inalloca(i32) %p) {
; Because of the musttail caller, this inalloca cannot be dropped.
; CHECK-LABEL: define internal i32 @inalloca2(ptr inalloca(i32) %p)
  %rv = load i32, ptr %p
  ret i32 %rv
}

define internal i32 @preallocated(ptr preallocated(i32) %p) {
; CHECK-LABEL: define internal fastcc i32 @preallocated(ptr %p)
  %rv = load i32, ptr %p
  ret i32 %rv
}

define void @call_things() {
  %m = alloca i32
  call i32 @f(ptr %m)
  call x86_thiscallcc i32 @g(ptr %m)
  call coldcc i32 @h(ptr %m)
  call i32 @j(ptr %m)
  %args = alloca inalloca i32
  call i32 @inalloca(ptr inalloca(i32) %args)
  %c = call token @llvm.call.preallocated.setup(i32 1)
  %N = call ptr @llvm.call.preallocated.arg(token %c, i32 0) preallocated(i32)
  call i32 @preallocated(ptr preallocated(i32) %N) ["preallocated"(token %c)]
  ret void
}
; CHECK-LABEL: define void @call_things()
; CHECK: call fastcc i32 @f
; CHECK: call fastcc i32 @g
; CHECK: call coldcc i32 @h
; CHECK: call i32 @j
; CHECK: call fastcc i32 @inalloca(ptr %args)
; CHECK-NOT: llvm.call.preallocated
; CHECK: call fastcc i32 @preallocated(ptr %paarg)

@llvm.used = appending global [1 x ptr] [
   ptr @j
], section "llvm.metadata"
