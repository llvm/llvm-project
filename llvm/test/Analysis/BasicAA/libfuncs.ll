; RUN: opt -inferattrs -basic-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7"

; CHECK-LABEL: Function: test_memcmp_const_size
; CHECK:      Just Ref:  Ptr: i8* %a	<->  %res = tail call i32 @memcmp(i8* %a, i8* %b, i64 4)
; CHECK-NEXT: Just Ref:  Ptr: i8* %b	<->  %res = tail call i32 @memcmp(i8* %a, i8* %b, i64 4)
; CHECK-NEXT: Just Ref:  Ptr: i8* %a.gep.1	<->  %res = tail call i32 @memcmp(i8* %a, i8* %b, i64 4)
; CHECK-NEXT: NoModRef:  Ptr: i8* %a.gep.5	<->  %res = tail call i32 @memcmp(i8* %a, i8* %b, i64 4)
; CHECK-NEXT: Just Ref:  Ptr: i8* %b.gep.1	<->  %res = tail call i32 @memcmp(i8* %a, i8* %b, i64 4)
; CHECK-NEXT: NoModRef:  Ptr: i8* %b.gep.5	<->  %res = tail call i32 @memcmp(i8* %a, i8* %b, i64 4)
define i32 @test_memcmp_const_size(i8* noalias %a, i8* noalias %b) {
entry:
  %res = tail call i32 @memcmp(i8* %a, i8* %b, i64 4)
  %a.gep.1 = getelementptr i8, i8* %a, i32 1
  store i8 0, i8* %a.gep.1
  %a.gep.5 = getelementptr i8, i8* %a, i32 5
  store i8 1, i8* %a.gep.5
  %b.gep.1 = getelementptr i8, i8* %b, i32 1
  store i8 2, i8* %b.gep.1
  %b.gep.5 = getelementptr i8, i8* %b, i32 5
  store i8 3, i8* %b.gep.5
  ret i32 %res
}

; CHECK-LABEL: Function: test_memcmp_variable_size
; CHECK:      Just Ref:  Ptr: i8* %a	<->  %res = tail call i32 @memcmp(i8* %a, i8* %b, i64 %n)
; CHECK-NEXT: Just Ref:  Ptr: i8* %b	<->  %res = tail call i32 @memcmp(i8* %a, i8* %b, i64 %n)
; CHECK-NEXT: Just Ref:  Ptr: i8* %a.gep.1	<->  %res = tail call i32 @memcmp(i8* %a, i8* %b, i64 %n)
; CHECK-NEXT: Just Ref:  Ptr: i8* %a.gep.5	<->  %res = tail call i32 @memcmp(i8* %a, i8* %b, i64 %n)
; CHECK-NEXT: Just Ref:  Ptr: i8* %b.gep.1	<->  %res = tail call i32 @memcmp(i8* %a, i8* %b, i64 %n)
; CHECK-NEXT: Just Ref:  Ptr: i8* %b.gep.5	<->  %res = tail call i32 @memcmp(i8* %a, i8* %b, i64 %n)
define i32 @test_memcmp_variable_size(i8* noalias %a, i8* noalias %b, i64 %n) {
entry:
  %res = tail call i32 @memcmp(i8* %a, i8* %b, i64 %n)
  %a.gep.1 = getelementptr i8, i8* %a, i32 1
  store i8 0, i8* %a.gep.1
  %a.gep.5 = getelementptr i8, i8* %a, i32 5
  store i8 1, i8* %a.gep.5
  %b.gep.1 = getelementptr i8, i8* %b, i32 1
  store i8 2, i8* %b.gep.1
  %b.gep.5 = getelementptr i8, i8* %b, i32 5
  store i8 3, i8* %b.gep.5
  ret i32 %res
}


declare i32 @memcmp(i8*, i8*, i64)
