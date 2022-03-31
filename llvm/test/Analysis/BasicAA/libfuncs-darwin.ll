; RUN: opt -mtriple=x86_64-apple-macosx10.7 -aa-pipeline=basic-aa -passes=inferattrs,aa-eval -print-all-alias-modref-info -disable-output 2>&1 %s | FileCheck %s

declare void @memset_pattern4(i8*, i8*, i64)
declare void @memset_pattern8(i8*, i8*, i64)
declare void @memset_pattern16(i8*, i8*, i64)

define void @test_memset_pattern4_const_size(i8* noalias %a, i8* noalias %pattern) {
; CHECK-LABEL: Function: test_memset_pattern4_const_size
; CHECK:       Just Mod:  Ptr: i8* %a	<->  call void @memset_pattern4(i8* %a, i8* %pattern, i64 17)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %pattern	<->  call void @memset_pattern4(i8* %a, i8* %pattern, i64 17)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %a.gep.1	<->  call void @memset_pattern4(i8* %a, i8* %pattern, i64 17)
; CHECK-NEXT:  NoModRef:  Ptr: i8* %a.gep.17	<->  call void @memset_pattern4(i8* %a, i8* %pattern, i64 17)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %pattern.gep.3	<->  call void @memset_pattern4(i8* %a, i8* %pattern, i64 17)
; CHECK-NEXT:  NoModRef:  Ptr: i8* %pattern.gep.4	<->  call void @memset_pattern4(i8* %a, i8* %pattern, i64 17)
;
entry:
  load i8, i8* %a
  load i8, i8* %pattern
  call void @memset_pattern4(i8* %a, i8* %pattern, i64 17)
  %a.gep.1 = getelementptr i8, i8* %a, i32 1
  store i8 0, i8* %a.gep.1
  %a.gep.17 = getelementptr i8, i8* %a, i32 17
  store i8 1, i8* %a.gep.17

  %pattern.gep.3 = getelementptr i8, i8* %pattern, i32 3
  store i8 1, i8* %pattern.gep.3
  %pattern.gep.4 = getelementptr i8, i8* %pattern, i32 4
  store i8 1, i8* %pattern.gep.4
  ret void
}

define void @test_memset_pattern4_variable_size(i8* noalias %a, i8* noalias %pattern, i64 %n) {
; CHECK-LABEL: Function: test_memset_pattern4_variable_size
; CHECK:       Just Mod:  Ptr: i8* %a	<->  call void @memset_pattern4(i8* %a, i8* %pattern, i64 %n)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %pattern	<->  call void @memset_pattern4(i8* %a, i8* %pattern, i64 %n)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %a.gep.1	<->  call void @memset_pattern4(i8* %a, i8* %pattern, i64 %n)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %a.gep.17	<->  call void @memset_pattern4(i8* %a, i8* %pattern, i64 %n)
;
entry:
  load i8, i8* %a
  load i8, i8* %pattern
  call void @memset_pattern4(i8* %a, i8* %pattern, i64 %n)
  %a.gep.1 = getelementptr i8, i8* %a, i32 1
  store i8 0, i8* %a.gep.1
  %a.gep.17 = getelementptr i8, i8* %a, i32 17
  store i8 1, i8* %a.gep.17
  ret void
}

define void @test_memset_pattern8_const_size(i8* noalias %a, i8* noalias %pattern) {
; CHECK-LABEL: Function: test_memset_pattern8_const_size
; CHECK:       Just Mod:  Ptr: i8* %a	<->  call void @memset_pattern8(i8* %a, i8* %pattern, i64 17)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %pattern	<->  call void @memset_pattern8(i8* %a, i8* %pattern, i64 17)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %a.gep.1	<->  call void @memset_pattern8(i8* %a, i8* %pattern, i64 17)
; CHECK-NEXT:  NoModRef:  Ptr: i8* %a.gep.17	<->  call void @memset_pattern8(i8* %a, i8* %pattern, i64 17)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %pattern.gep.7	<->  call void @memset_pattern8(i8* %a, i8* %pattern, i64 17)
; CHECK-NEXT:  NoModRef:  Ptr: i8* %pattern.gep.8	<->  call void @memset_pattern8(i8* %a, i8* %pattern, i64 17)
;
entry:
  load i8, i8* %a
  load i8, i8* %pattern
  call void @memset_pattern8(i8* %a, i8* %pattern, i64 17)
  %a.gep.1 = getelementptr i8, i8* %a, i32 1
  store i8 0, i8* %a.gep.1
  %a.gep.17 = getelementptr i8, i8* %a, i32 17
  store i8 1, i8* %a.gep.17

  %pattern.gep.7 = getelementptr i8, i8* %pattern, i32 7
  store i8 1, i8* %pattern.gep.7
  %pattern.gep.8 = getelementptr i8, i8* %pattern, i32 8
  store i8 1, i8* %pattern.gep.8
  ret void
}

define void @test_memset_pattern8_variable_size(i8* noalias %a, i8* noalias %pattern, i64 %n) {
; CHECK-LABEL: Function: test_memset_pattern8_variable_size
; CHECK:       Just Mod:  Ptr: i8* %a	<->  call void @memset_pattern8(i8* %a, i8* %pattern, i64 %n)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %pattern	<->  call void @memset_pattern8(i8* %a, i8* %pattern, i64 %n)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %a.gep.1	<->  call void @memset_pattern8(i8* %a, i8* %pattern, i64 %n)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %a.gep.17	<->  call void @memset_pattern8(i8* %a, i8* %pattern, i64 %n)
;
entry:
  load i8, i8* %a
  load i8, i8* %pattern
  call void @memset_pattern8(i8* %a, i8* %pattern, i64 %n)
  %a.gep.1 = getelementptr i8, i8* %a, i32 1
  store i8 0, i8* %a.gep.1
  %a.gep.17 = getelementptr i8, i8* %a, i32 17
  store i8 1, i8* %a.gep.17
  ret void
}

define void @test_memset_pattern16_const_size(i8* noalias %a, i8* noalias %pattern) {
; CHECK-LABEL: Function: test_memset_pattern16_const_size
; CHECK:       Just Mod:  Ptr: i8* %a	<->  call void @memset_pattern16(i8* %a, i8* %pattern, i64 17)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %pattern	<->  call void @memset_pattern16(i8* %a, i8* %pattern, i64 17)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %a.gep.1	<->  call void @memset_pattern16(i8* %a, i8* %pattern, i64 17)
; CHECK-NEXT:  NoModRef:  Ptr: i8* %a.gep.17	<->  call void @memset_pattern16(i8* %a, i8* %pattern, i64 17)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %pattern.gep.15	<->  call void @memset_pattern16(i8* %a, i8* %pattern, i64 17)
; CHECK-NEXT:  NoModRef:  Ptr: i8* %pattern.gep.16	<->  call void @memset_pattern16(i8* %a, i8* %pattern, i64 17)
;
entry:
  load i8, i8* %a
  load i8, i8* %pattern
  call void @memset_pattern16(i8* %a, i8* %pattern, i64 17)
  %a.gep.1 = getelementptr i8, i8* %a, i32 1
  store i8 0, i8* %a.gep.1
  %a.gep.17 = getelementptr i8, i8* %a, i32 17
  store i8 1, i8* %a.gep.17

  %pattern.gep.15 = getelementptr i8, i8* %pattern, i32 15
  store i8 1, i8* %pattern.gep.15
  %pattern.gep.16 = getelementptr i8, i8* %pattern, i32 16
  store i8 1, i8* %pattern.gep.16
  ret void
}

define void @test_memset_pattern16_variable_size(i8* noalias %a, i8* noalias %pattern, i64 %n) {
; CHECK-LABEL: Function: test_memset_pattern16_variable_size
; CHECK:       Just Mod:  Ptr: i8* %a	<->  call void @memset_pattern16(i8* %a, i8* %pattern, i64 %n)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %pattern	<->  call void @memset_pattern16(i8* %a, i8* %pattern, i64 %n)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %a.gep.1	<->  call void @memset_pattern16(i8* %a, i8* %pattern, i64 %n)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %a.gep.17	<->  call void @memset_pattern16(i8* %a, i8* %pattern, i64 %n)
;
entry:
  load i8, i8* %a
  load i8, i8* %pattern
  call void @memset_pattern16(i8* %a, i8* %pattern, i64 %n)
  %a.gep.1 = getelementptr i8, i8* %a, i32 1
  store i8 0, i8* %a.gep.1
  %a.gep.17 = getelementptr i8, i8* %a, i32 17
  store i8 1, i8* %a.gep.17
  ret void
}
