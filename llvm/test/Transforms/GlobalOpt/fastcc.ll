; RUN: opt < %s -globalopt -S | FileCheck %s

declare token @llvm.call.preallocated.setup(i32)
declare i8* @llvm.call.preallocated.arg(token, i32)

define internal i32 @f(i32* %m) {
; CHECK-LABEL: define internal fastcc i32 @f
  %v = load i32, i32* %m
  ret i32 %v
}

define internal x86_thiscallcc i32 @g(i32* %m) {
; CHECK-LABEL: define internal fastcc i32 @g
  %v = load i32, i32* %m
  ret i32 %v
}

; Leave this one alone, because the user went out of their way to request this
; convention.
define internal coldcc i32 @h(i32* %m) {
; CHECK-LABEL: define internal coldcc i32 @h
  %v = load i32, i32* %m
  ret i32 %v
}

define internal i32 @j(i32* %m) {
; CHECK-LABEL: define internal i32 @j
  %v = load i32, i32* %m
  ret i32 %v
}

define internal i32 @inalloca(i32* inalloca(i32) %p) {
; CHECK-LABEL: define internal fastcc i32 @inalloca(i32* %p)
  %rv = load i32, i32* %p
  ret i32 %rv
}

define i32 @inalloca2_caller(i32* inalloca(i32) %p) {
  %rv = musttail call i32 @inalloca2(i32* inalloca(i32) %p)
  ret i32 %rv
}
define internal i32 @inalloca2(i32* inalloca(i32) %p) {
; Because of the musttail caller, this inalloca cannot be dropped.
; CHECK-LABEL: define internal i32 @inalloca2(i32* inalloca(i32) %p)
  %rv = load i32, i32* %p
  ret i32 %rv
}

define internal i32 @preallocated(i32* preallocated(i32) %p) {
; CHECK-LABEL: define internal fastcc i32 @preallocated(i32* %p)
  %rv = load i32, i32* %p
  ret i32 %rv
}

define void @call_things() {
  %m = alloca i32
  call i32 @f(i32* %m)
  call x86_thiscallcc i32 @g(i32* %m)
  call coldcc i32 @h(i32* %m)
  call i32 @j(i32* %m)
  %args = alloca inalloca i32
  call i32 @inalloca(i32* inalloca(i32) %args)
  %c = call token @llvm.call.preallocated.setup(i32 1)
  %N = call i8* @llvm.call.preallocated.arg(token %c, i32 0) preallocated(i32)
  %n = bitcast i8* %N to i32*
  call i32 @preallocated(i32* preallocated(i32) %n) ["preallocated"(token %c)]
  ret void
}
; CHECK-LABEL: define void @call_things()
; CHECK: call fastcc i32 @f
; CHECK: call fastcc i32 @g
; CHECK: call coldcc i32 @h
; CHECK: call i32 @j
; CHECK: call fastcc i32 @inalloca(i32* %args)
; CHECK-NOT: llvm.call.preallocated
; CHECK: call fastcc i32 @preallocated(i32* %n)

@llvm.used = appending global [1 x i8*] [
   i8* bitcast (i32(i32*)* @j to i8*)
], section "llvm.metadata"
