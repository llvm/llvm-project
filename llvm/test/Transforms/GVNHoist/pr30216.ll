; RUN: opt -S -passes=gvn-hoist < %s | FileCheck %s

; Make sure the two stores @B do not get hoisted past the load @B.

; CHECK-LABEL: define ptr @Foo
; CHECK: store
; CHECK: store
; CHECK: load
; CHECK: store

@A = external global i8
@B = external global ptr

define ptr @Foo() {
  store i8 0, ptr @A
  br i1 undef, label %if.then, label %if.else

if.then:
  store ptr null, ptr @B
  ret ptr null

if.else:
  %1 = load ptr, ptr @B
  store ptr null, ptr @B
  ret ptr %1
}

; Make sure the two stores @B do not get hoisted past the store @GlobalVar.

; CHECK-LABEL: define ptr @Fun
; CHECK: store
; CHECK: store
; CHECK: store
; CHECK: store
; CHECK: load

@GlobalVar = internal global i8 0

define ptr @Fun() {
  store i8 0, ptr @A
  br i1 undef, label %if.then, label %if.else

if.then:
  store ptr null, ptr @B
  ret ptr null

if.else:
  store i8 0, ptr @GlobalVar
  store ptr null, ptr @B
  %1 = load ptr, ptr @B
  ret ptr %1
}
