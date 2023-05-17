; RUN: opt -passes=mergefunc -S < %s | FileCheck %s

; This test makes sure that the mergefunc pass, uses extract and insert value
; to convert the struct result type; as struct types cannot be bitcast.

target datalayout = "e-m:w-p:32:32-i64:64-f80:32-n8:16:32-S32"

%kv1 = type { ptr, ptr }
%kv2 = type { ptr, ptr }

declare void @noop()

define %kv1 @fn1() {
; CHECK-LABEL: @fn1(
  %tmp = alloca %kv1
  store ptr null, ptr %tmp
  store ptr null, ptr %tmp
  call void @noop()
  %v3 = load %kv1, ptr %tmp
  ret %kv1 %v3
}

define %kv2 @fn2() {
; CHECK-LABEL: @fn2(
; CHECK: %1 = tail call %kv1 @fn1()
; CHECK: %2 = extractvalue %kv1 %1, 0
; CHECK: %3 = insertvalue %kv2 poison, ptr %2, 0
  %tmp = alloca %kv2
  store ptr null, ptr %tmp
  store ptr null, ptr %tmp
  call void @noop()

  %v3 = load %kv2, ptr %tmp
  ret %kv2 %v3
}
