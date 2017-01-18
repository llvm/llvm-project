; RUN: opt < %s -inline -instcombine -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline,function(instcombine))' -S | FileCheck %s

; We can't inline this thunk yet, but one day we will be able to.  And when we
; do, this test case will be ready.

declare void @ext_method(i8*, i32)

define linkonce_odr void @thunk(i8* %this, ...) {
  %this_adj = getelementptr i8, i8* %this, i32 4
  musttail call void (i8*, ...) bitcast (void (i8*, i32)* @ext_method to void (i8*, ...)*)(i8* %this_adj, ...)
  ret void
}

define void @thunk_caller(i8* %p) {
  call void (i8*, ...) @thunk(i8* %p, i32 42)
  ret void
}
; CHECK-LABEL: define void @thunk_caller(i8* %p)
; CHECK: call void (i8*, ...) @thunk(i8* %p, i32 42)

; FIXME: Inline the thunk. This should be significantly easier than inlining
; general varargs functions.
