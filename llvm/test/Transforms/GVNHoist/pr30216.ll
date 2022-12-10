; RUN: opt -S -passes=gvn-hoist < %s | FileCheck %s

; Make sure the two stores @B do not get hoisted past the load @B.

; CHECK-LABEL: define i8* @Foo
; CHECK: store
; CHECK: store
; CHECK: load
; CHECK: store

@A = external global i8
@B = external global i8*

define i8* @Foo() {
  store i8 0, i8* @A
  br i1 undef, label %if.then, label %if.else

if.then:
  store i8* null, i8** @B
  ret i8* null

if.else:
  %1 = load i8*, i8** @B
  store i8* null, i8** @B
  ret i8* %1
}

; Make sure the two stores @B do not get hoisted past the store @GlobalVar.

; CHECK-LABEL: define i8* @Fun
; CHECK: store
; CHECK: store
; CHECK: store
; CHECK: store
; CHECK: load

@GlobalVar = internal global i8 0

define i8* @Fun() {
  store i8 0, i8* @A
  br i1 undef, label %if.then, label %if.else

if.then:
  store i8* null, i8** @B
  ret i8* null

if.else:
  store i8 0, i8* @GlobalVar
  store i8* null, i8** @B
  %1 = load i8*, i8** @B
  ret i8* %1
}
