; RUN: opt -mergefunc -S < %s | FileCheck %s

; This test makes sure that the mergefunc pass, uses extract and insert value
; to convert the struct result type; as struct types cannot be bitcast.

target datalayout = "e-m:w-p:32:32-i64:64-f80:32-n8:16:32-S32"

%kv1 = type { i32*, i32* }
%kv2 = type { i8*, i8* }

declare void @noop()

define %kv1 @fn1() {
; CHECK-LABEL: @fn1(
  %tmp = alloca %kv1
  %v1 = getelementptr %kv1, %kv1* %tmp, i32 0, i32 0
  store i32* null, i32** %v1
  %v2 = getelementptr %kv1, %kv1* %tmp, i32 0, i32 0
  store i32* null, i32** %v2
  call void @noop()
  %v3 = load %kv1, %kv1* %tmp
  ret %kv1 %v3
}

define %kv2 @fn2() {
; CHECK-LABEL: @fn2(
; CHECK: %1 = tail call %kv1 @fn1()
; CHECK: %2 = extractvalue %kv1 %1, 0
; CHECK: %3 = bitcast i32* %2 to i8*
; CHECK: %4 = insertvalue %kv2 poison, i8* %3, 0
  %tmp = alloca %kv2
  %v1 = getelementptr %kv2, %kv2* %tmp, i32 0, i32 0
  store i8* null, i8** %v1
  %v2 = getelementptr %kv2, %kv2* %tmp, i32 0, i32 0
  store i8* null, i8** %v2
  call void @noop()

  %v3 = load %kv2, %kv2* %tmp
  ret %kv2 %v3
}
