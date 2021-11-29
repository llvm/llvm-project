; RUN: opt < %s -passes=globalopt -S -enable-coldcc-stress-test -mtriple=powerpc64le-unknown-linux-gnu | FileCheck %s -check-prefix=COLDCC
; RUN: opt < %s -passes=globalopt -S | FileCheck %s -check-prefix=CHECK

define internal i32 @callee_default(i32* %m) {
; COLDCC-LABEL: define internal coldcc i32 @callee_default
; CHECK-LABEL: define internal fastcc i32 @callee_default
  %v = load i32, i32* %m
  ret i32 %v
}

define internal fastcc i32 @callee_fastcc(i32* %m) {
; COLDCC-LABEL: define internal fastcc i32 @callee_fastcc
; CHECK-LABEL: define internal fastcc i32 @callee_fastcc
  %v = load i32, i32* %m
  ret i32 %v
}

define internal coldcc i32 @callee_coldcc(i32* %m) {
; COLDCC-LABEL: define internal coldcc i32 @callee_coldcc
; CHECK-LABEL: define internal coldcc i32 @callee_coldcc
  %v = load i32, i32* %m
  ret i32 %v
}

define i32 @callee(i32* %m) {
  %v = load i32, i32* %m
  ret i32 %v
}

define void @caller() {
  %m = alloca i32
  call i32 @callee_default(i32* %m)
  call fastcc i32 @callee_fastcc(i32* %m)
  call coldcc i32 @callee_coldcc(i32* %m)
  call i32 @callee(i32* %m)
  ret void
}

; COLDCC-LABEL: define void @caller()
; COLDCC: call coldcc i32 @callee_default
; COLDCC: call fastcc i32 @callee_fastcc
; COLDCC: call coldcc i32 @callee_coldcc
; COLDCC: call i32 @callee
; CHECK-LABEL: define void @caller()
; CHECK: call fastcc i32 @callee_default
; CHECK: call fastcc i32 @callee_fastcc
; CHECK: call coldcc i32 @callee_coldcc
; CHECK: call i32 @callee
