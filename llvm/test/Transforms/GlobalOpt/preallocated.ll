; RUN: opt < %s -globalopt -S | FileCheck %s

declare token @llvm.call.preallocated.setup(i32)
declare i8* @llvm.call.preallocated.arg(token, i32)
declare i32 @__CxxFrameHandler3(...)

; Don't touch functions with any musttail calls
define internal i32 @preallocated_musttail(i32* preallocated(i32) %p) {
; CHECK-LABEL: define internal i32 @preallocated_musttail(i32* preallocated(i32) %p)
  %rv = load i32, i32* %p
  ret i32 %rv
}

define i32 @call_preallocated_musttail(i32* preallocated(i32) %a) {
  %r = musttail call i32 @preallocated_musttail(i32* preallocated(i32) %a)
  ret i32 %r
}
; CHECK-LABEL: define i32 @call_preallocated_musttail(i32* preallocated(i32) %a)
; CHECK: musttail call i32 @preallocated_musttail(i32* preallocated(i32) %a)

define i32 @call_preallocated_musttail_without_musttail() {
  %c = call token @llvm.call.preallocated.setup(i32 1)
  %N = call i8* @llvm.call.preallocated.arg(token %c, i32 0) preallocated(i32)
  %n = bitcast i8* %N to i32*
  %r = call i32 @preallocated_musttail(i32* preallocated(i32) %n) ["preallocated"(token %c)]
  ret i32 %r
}
; CHECK-LABEL: define i32 @call_preallocated_musttail_without_musttail()
; CHECK: call i32 @preallocated_musttail(i32* preallocated(i32) %n)

; Check that only one alloca per preallocated arg
define internal i32 @preallocated(i32* preallocated(i32) %a) {
; CHECK-LABEL: define internal fastcc i32 @preallocated(i32* %a)
  %rv = load i32, i32* %a
  ret i32 %rv
}

declare void @foo(i8*)

define i32 @call_preallocated_multiple_args() {
; CHECK-LABEL: define i32 @call_preallocated_multiple_args()
; CHECK-NEXT: [[SS:%[0-9a-zA-Z_]+]] = call i8* @llvm.stacksave()
; CHECK-NEXT: [[ARG0:%[0-9a-zA-Z_]+]] = alloca i32
; CHECK-NEXT: [[ARG1:%[0-9a-zA-Z_]+]] = bitcast i32* [[ARG0]] to i8*
; CHECK-NEXT: call void @foo(i8* [[ARG1]])
; CHECK-NEXT: call void @foo(i8* [[ARG1]])
; CHECK-NEXT: call void @foo(i8* [[ARG1]])
; CHECK-NEXT: [[ARG2:%[0-9a-zA-Z_]+]] = bitcast i8* [[ARG1]] to i32*
; CHECK-NEXT: call fastcc i32 @preallocated(i32* [[ARG2]])
; CHECK-NEXT: call void @llvm.stackrestore(i8* [[SS]])
; CHECK-NEXT: ret
  %c = call token @llvm.call.preallocated.setup(i32 1)
  %a1 = call i8* @llvm.call.preallocated.arg(token %c, i32 0) preallocated(i32)
  call void @foo(i8* %a1)
  %a2 = call i8* @llvm.call.preallocated.arg(token %c, i32 0) preallocated(i32)
  call void @foo(i8* %a2)
  %a3 = call i8* @llvm.call.preallocated.arg(token %c, i32 0) preallocated(i32)
  call void @foo(i8* %a3)
  %b = bitcast i8* %a3 to i32*
  %r = call i32 @preallocated(i32* preallocated(i32) %b) ["preallocated"(token %c)]
  ret i32 %r
}

; Don't touch functions with any invokes
define internal i32 @preallocated_invoke(i32* preallocated(i32) %p) {
; CHECK-LABEL: define internal i32 @preallocated_invoke(i32* preallocated(i32) %p)
  %rv = load i32, i32* %p
  ret i32 %rv
}

define i32 @call_preallocated_invoke() personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
  %c = call token @llvm.call.preallocated.setup(i32 1)
  %a = call i8* @llvm.call.preallocated.arg(token %c, i32 0) preallocated(i32)
  %b = bitcast i8* %a to i32*
  %r = invoke i32 @preallocated_invoke(i32* preallocated(i32) %b) ["preallocated"(token %c)]
       to label %conta unwind label %contb
conta:
  ret i32 %r
contb:
  %s = catchswitch within none [label %catch] unwind to caller
catch:
  %p = catchpad within %s []
  catchret from %p to label %cont
cont:
  ret i32 42
}
; CHECK-LABEL: define i32 @call_preallocated_invoke()
; CHECK: invoke i32 @preallocated_invoke(i32* preallocated(i32) %b)
