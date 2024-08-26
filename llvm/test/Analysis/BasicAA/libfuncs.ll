; RUN: opt -mtriple=x86_64-pc-linux-gnu -aa-pipeline=basic-aa -passes=inferattrs,aa-eval -print-all-alias-modref-info -disable-output 2>&1 %s | FileCheck %s

; CHECK-LABEL: Function: test_memcmp_const_size
; CHECK:      Just Ref:  Ptr: i8* %a	<->  %res = tail call i32 @memcmp(ptr %a, ptr %b, i64 4)
; CHECK-NEXT: Just Ref:  Ptr: i8* %b	<->  %res = tail call i32 @memcmp(ptr %a, ptr %b, i64 4)
; CHECK-NEXT: Just Ref:  Ptr: i8* %a.gep.1	<->  %res = tail call i32 @memcmp(ptr %a, ptr %b, i64 4)
; CHECK-NEXT: NoModRef:  Ptr: i8* %a.gep.5	<->  %res = tail call i32 @memcmp(ptr %a, ptr %b, i64 4)
; CHECK-NEXT: Just Ref:  Ptr: i8* %b.gep.1	<->  %res = tail call i32 @memcmp(ptr %a, ptr %b, i64 4)
; CHECK-NEXT: NoModRef:  Ptr: i8* %b.gep.5	<->  %res = tail call i32 @memcmp(ptr %a, ptr %b, i64 4)
define i32 @test_memcmp_const_size(ptr noalias %a, ptr noalias %b) {
entry:
  load i8, ptr %a
  load i8, ptr %b
  %res = tail call i32 @memcmp(ptr %a, ptr %b, i64 4)
  %a.gep.1 = getelementptr i8, ptr %a, i32 1
  store i8 0, ptr %a.gep.1
  %a.gep.5 = getelementptr i8, ptr %a, i32 5
  store i8 1, ptr %a.gep.5
  %b.gep.1 = getelementptr i8, ptr %b, i32 1
  store i8 2, ptr %b.gep.1
  %b.gep.5 = getelementptr i8, ptr %b, i32 5
  store i8 3, ptr %b.gep.5
  ret i32 %res
}

; CHECK-LABEL: Function: test_memcmp_variable_size
; CHECK:      Just Ref:  Ptr: i8* %a	<->  %res = tail call i32 @memcmp(ptr %a, ptr %b, i64 %n)
; CHECK-NEXT: Just Ref:  Ptr: i8* %b	<->  %res = tail call i32 @memcmp(ptr %a, ptr %b, i64 %n)
; CHECK-NEXT: Just Ref:  Ptr: i8* %a.gep.1	<->  %res = tail call i32 @memcmp(ptr %a, ptr %b, i64 %n)
; CHECK-NEXT: Just Ref:  Ptr: i8* %a.gep.5	<->  %res = tail call i32 @memcmp(ptr %a, ptr %b, i64 %n)
; CHECK-NEXT: Just Ref:  Ptr: i8* %b.gep.1	<->  %res = tail call i32 @memcmp(ptr %a, ptr %b, i64 %n)
; CHECK-NEXT: Just Ref:  Ptr: i8* %b.gep.5	<->  %res = tail call i32 @memcmp(ptr %a, ptr %b, i64 %n)
define i32 @test_memcmp_variable_size(ptr noalias %a, ptr noalias %b, i64 %n) {
entry:
  load i8, ptr %a
  load i8, ptr %b
  %res = tail call i32 @memcmp(ptr %a, ptr %b, i64 %n)
  %a.gep.1 = getelementptr i8, ptr %a, i32 1
  store i8 0, ptr %a.gep.1
  %a.gep.5 = getelementptr i8, ptr %a, i32 5
  store i8 1, ptr %a.gep.5
  %b.gep.1 = getelementptr i8, ptr %b, i32 1
  store i8 2, ptr %b.gep.1
  %b.gep.5 = getelementptr i8, ptr %b, i32 5
  store i8 3, ptr %b.gep.5
  ret i32 %res
}

declare i32 @memcmp(ptr, ptr, i64)
declare i32 @bcmp(ptr, ptr, i64)

; CHECK-LABEL: Function: test_bcmp_const_size
; CHECK:      Just Ref:  Ptr: i8* %a	<->  %res = tail call i32 @bcmp(ptr %a, ptr %b, i64 4)
; CHECK-NEXT: Just Ref:  Ptr: i8* %b	<->  %res = tail call i32 @bcmp(ptr %a, ptr %b, i64 4)
; CHECK-NEXT: Just Ref:  Ptr: i8* %a.gep.1	<->  %res = tail call i32 @bcmp(ptr %a, ptr %b, i64 4)
; CHECK-NEXT: NoModRef:  Ptr: i8* %a.gep.5	<->  %res = tail call i32 @bcmp(ptr %a, ptr %b, i64 4)
; CHECK-NEXT: Just Ref:  Ptr: i8* %b.gep.1	<->  %res = tail call i32 @bcmp(ptr %a, ptr %b, i64 4)
; CHECK-NEXT: NoModRef:  Ptr: i8* %b.gep.5	<->  %res = tail call i32 @bcmp(ptr %a, ptr %b, i64 4)
define i32 @test_bcmp_const_size(ptr noalias %a, ptr noalias %b) {
entry:
  load i8, ptr %a
  load i8, ptr %b
  %res = tail call i32 @bcmp(ptr %a, ptr %b, i64 4)
  %a.gep.1 = getelementptr i8, ptr %a, i32 1
  store i8 0, ptr %a.gep.1
  %a.gep.5 = getelementptr i8, ptr %a, i32 5
  store i8 1, ptr %a.gep.5
  %b.gep.1 = getelementptr i8, ptr %b, i32 1
  store i8 2, ptr %b.gep.1
  %b.gep.5 = getelementptr i8, ptr %b, i32 5
  store i8 3, ptr %b.gep.5
  ret i32 %res
}

; CHECK-LABEL: Function: test_bcmp_variable_size
; CHECK:      Just Ref:  Ptr: i8* %a	<->  %res = tail call i32 @bcmp(ptr %a, ptr %b, i64 %n)
; CHECK-NEXT: Just Ref:  Ptr: i8* %b	<->  %res = tail call i32 @bcmp(ptr %a, ptr %b, i64 %n)
; CHECK-NEXT: Just Ref:  Ptr: i8* %a.gep.1	<->  %res = tail call i32 @bcmp(ptr %a, ptr %b, i64 %n)
; CHECK-NEXT: Just Ref:  Ptr: i8* %a.gep.5	<->  %res = tail call i32 @bcmp(ptr %a, ptr %b, i64 %n)
; CHECK-NEXT: Just Ref:  Ptr: i8* %b.gep.1	<->  %res = tail call i32 @bcmp(ptr %a, ptr %b, i64 %n)
; CHECK-NEXT: Just Ref:  Ptr: i8* %b.gep.5	<->  %res = tail call i32 @bcmp(ptr %a, ptr %b, i64 %n)
define i32 @test_bcmp_variable_size(ptr noalias %a, ptr noalias %b, i64 %n) {
entry:
  load i8, ptr %a
  load i8, ptr %b
  %res = tail call i32 @bcmp(ptr %a, ptr %b, i64 %n)
  %a.gep.1 = getelementptr i8, ptr %a, i32 1
  store i8 0, ptr %a.gep.1
  %a.gep.5 = getelementptr i8, ptr %a, i32 5
  store i8 1, ptr %a.gep.5
  %b.gep.1 = getelementptr i8, ptr %b, i32 1
  store i8 2, ptr %b.gep.1
  %b.gep.5 = getelementptr i8, ptr %b, i32 5
  store i8 3, ptr %b.gep.5
  ret i32 %res
}

declare ptr @memchr(ptr, i32, i64)

; CHECK-LABEL: Function: test_memchr_const_size
; CHECK: Just Ref:  Ptr: i8* %res      <->  %res = call ptr @memchr(ptr %a, i32 42, i64 4)
; CHECK-NEXT: Just Ref:  Ptr: i8* %a.gep.1  <->  %res = call ptr @memchr(ptr %a, i32 42, i64 4)
; CHECK-NEXT: NoModRef:  Ptr: i8* %a.gep.5  <->  %res = call ptr @memchr(ptr %a, i32 42, i64 4)
define ptr @test_memchr_const_size(ptr noalias %a) {
entry:
  %res = call ptr @memchr(ptr %a, i32 42, i64 4)
  load i8, ptr %res
  %a.gep.1 = getelementptr i8, ptr %a, i32 1
  store i8 0, ptr %a.gep.1
  %a.gep.5 = getelementptr i8, ptr %a, i32 5
  store i8 1, ptr %a.gep.5
  ret ptr %res
}

declare ptr @memccpy(ptr, ptr, i32, i64)

; CHECK-LABEL: Function: test_memccpy_const_size
; CHECK:      Just Mod:  Ptr: i8* %a        <->  %res = call ptr @memccpy(ptr %a, ptr %b, i32 42, i64 4)
; CHECK-NEXT: Just Ref:  Ptr: i8* %b        <->  %res = call ptr @memccpy(ptr %a, ptr %b, i32 42, i64 4)
; CHECK-NEXT: Just Mod:  Ptr: i8* %res      <->  %res = call ptr @memccpy(ptr %a, ptr %b, i32 42, i64 4)
; CHECK-NEXT: Just Mod:  Ptr: i8* %a.gep.1  <->  %res = call ptr @memccpy(ptr %a, ptr %b, i32 42, i64 4)
; CHECK-NEXT: NoModRef:  Ptr: i8* %a.gep.5  <->  %res = call ptr @memccpy(ptr %a, ptr %b, i32 42, i64 4)
; CHECK-NEXT: Just Ref:  Ptr: i8* %b.gep.1  <->  %res = call ptr @memccpy(ptr %a, ptr %b, i32 42, i64 4)
; CHECK-NEXT: NoModRef:  Ptr: i8* %b.gep.5  <->  %res = call ptr @memccpy(ptr %a, ptr %b, i32 42, i64 4)

define ptr @test_memccpy_const_size(ptr noalias %a, ptr noalias %b) {
entry:
  load i8, ptr %a
  load i8, ptr %b
  %res = call ptr @memccpy(ptr %a, ptr %b, i32 42, i64 4)
  load i8, ptr %res
  %a.gep.1 = getelementptr i8, ptr %a, i32 1
  store i8 0, ptr %a.gep.1
  %a.gep.5 = getelementptr i8, ptr %a, i32 5
  store i8 1, ptr %a.gep.5
  %b.gep.1 = getelementptr i8, ptr %b, i32 1
  store i8 2, ptr %b.gep.1
  %b.gep.5 = getelementptr i8, ptr %b, i32 5
  store i8 3, ptr %b.gep.5
  ret ptr %res
}

declare ptr @strcat(ptr, ptr)

define ptr @test_strcat_read_write_after(ptr noalias %a, ptr noalias %b) {
; CHECK-LABEL: Function: test_strcat_read_write_after
; CHECK:       NoModRef:  Ptr: i8* %a	<->  %res = tail call ptr @strcat(ptr %a.gep.1, ptr %b.gep.1)
; CHECK-NEXT:  NoModRef:  Ptr: i8* %b	<->  %res = tail call ptr @strcat(ptr %a.gep.1, ptr %b.gep.1)
; CHECK-NEXT:  Both ModRef:  Ptr: i8* %a.gep.1	<->  %res = tail call ptr @strcat(ptr %a.gep.1, ptr %b.gep.1)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %b.gep.1	<->  %res = tail call ptr @strcat(ptr %a.gep.1, ptr %b.gep.1)
; CHECK-NEXT:  Both ModRef:  Ptr: i8* %res	<->  %res = tail call ptr @strcat(ptr %a.gep.1, ptr %b.gep.1)
; CHECK-NEXT:  Both ModRef:  Ptr: i8* %a.gep.5	<->  %res = tail call ptr @strcat(ptr %a.gep.1, ptr %b.gep.1)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %b.gep.5	<->  %res = tail call ptr @strcat(ptr %a.gep.1, ptr %b.gep.1)
;
entry:
  store i8 0, ptr %a
  store i8 2, ptr %b
  %a.gep.1 = getelementptr i8, ptr %a, i32 1
  load i8, ptr %a.gep.1
  %b.gep.1 = getelementptr i8, ptr %b, i32 1
  load i8, ptr %b.gep.1
  %res = tail call ptr @strcat(ptr %a.gep.1, ptr %b.gep.1)
  load i8, ptr %res
  %a.gep.5 = getelementptr i8, ptr %a, i32 5
  store i8 1, ptr %a.gep.5
  %b.gep.5 = getelementptr i8, ptr %b, i32 5
  store i8 3, ptr %b.gep.5
  ret ptr %res
}

declare ptr @strncat(ptr, ptr, i64)

define ptr @test_strncat_read_write_after(ptr noalias %a, ptr noalias %b, i64 %n) {
; CHECK-LABEL: Function: test_strncat_read_write_after
; CHECK:       NoModRef:  Ptr: i8* %a	<->  %res = tail call ptr @strncat(ptr %a.gep.1, ptr %b.gep.1, i64 %n)
; CHECK-NEXT:  NoModRef:  Ptr: i8* %b	<->  %res = tail call ptr @strncat(ptr %a.gep.1, ptr %b.gep.1, i64 %n)
; CHECK-NEXT:  Both ModRef:  Ptr: i8* %a.gep.1	<->  %res = tail call ptr @strncat(ptr %a.gep.1, ptr %b.gep.1, i64 %n)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %b.gep.1	<->  %res = tail call ptr @strncat(ptr %a.gep.1, ptr %b.gep.1, i64 %n)
; CHECK-NEXT:  Both ModRef:  Ptr: i8* %res	<->  %res = tail call ptr @strncat(ptr %a.gep.1, ptr %b.gep.1, i64 %n)
; CHECK-NEXT:  Both ModRef:  Ptr: i8* %a.gep.5	<->  %res = tail call ptr @strncat(ptr %a.gep.1, ptr %b.gep.1, i64 %n)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %b.gep.5	<->  %res = tail call ptr @strncat(ptr %a.gep.1, ptr %b.gep.1, i64 %n)
;
entry:
  store i8 0, ptr %a
  store i8 2, ptr %b
  %a.gep.1 = getelementptr i8, ptr %a, i32 1
  load i8, ptr %a.gep.1
  %b.gep.1 = getelementptr i8, ptr %b, i32 1
  load i8, ptr %b.gep.1
  %res = tail call ptr @strncat(ptr %a.gep.1, ptr %b.gep.1, i64 %n)
  load i8, ptr %res
  %a.gep.5 = getelementptr i8, ptr %a, i32 5
  store i8 1, ptr %a.gep.5
  %b.gep.5 = getelementptr i8, ptr %b, i32 5
  store i8 3, ptr %b.gep.5
  ret ptr %res
}

declare ptr @strcpy(ptr, ptr)

define ptr @test_strcpy_read_write_after(ptr noalias %a, ptr noalias %b) {
; CHECK-LABEL: Function: test_strcpy_read_write_after
; CHECK:       NoModRef:  Ptr: i8* %a	<->  %res = tail call ptr @strcpy(ptr %a.gep.1, ptr %b.gep.1)
; CHECK-NEXT:  NoModRef:  Ptr: i8* %b	<->  %res = tail call ptr @strcpy(ptr %a.gep.1, ptr %b.gep.1)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %a.gep.1	<->  %res = tail call ptr @strcpy(ptr %a.gep.1, ptr %b.gep.1)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %b.gep.1	<->  %res = tail call ptr @strcpy(ptr %a.gep.1, ptr %b.gep.1)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %res	<->  %res = tail call ptr @strcpy(ptr %a.gep.1, ptr %b.gep.1)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %a.gep.5	<->  %res = tail call ptr @strcpy(ptr %a.gep.1, ptr %b.gep.1)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %b.gep.5	<->  %res = tail call ptr @strcpy(ptr %a.gep.1, ptr %b.gep.1)
;
entry:
  store i8 0, ptr %a
  store i8 2, ptr %b
  %a.gep.1 = getelementptr i8, ptr %a, i32 1
  load i8, ptr %a.gep.1
  %b.gep.1 = getelementptr i8, ptr %b, i32 1
  load i8, ptr %b.gep.1
  %res = tail call ptr @strcpy(ptr %a.gep.1, ptr %b.gep.1)
  load i8, ptr %res
  %a.gep.5 = getelementptr i8, ptr %a, i32 5
  store i8 1, ptr %a.gep.5
  %b.gep.5 = getelementptr i8, ptr %b, i32 5
  store i8 3, ptr %b.gep.5
  ret ptr %res
}

declare ptr @strncpy(ptr, ptr, i64)

define ptr @test_strncpy_const_size(ptr noalias %a, ptr noalias %b) {
; CHECK-LABEL: Function: test_strncpy_const_size
; CHECK:       Just Mod:  Ptr: i8* %a	<->  %res = tail call ptr @strncpy(ptr %a, ptr %b, i64 4)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %b	<->  %res = tail call ptr @strncpy(ptr %a, ptr %b, i64 4)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %res	<->  %res = tail call ptr @strncpy(ptr %a, ptr %b, i64 4)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %a.gep.1	<->  %res = tail call ptr @strncpy(ptr %a, ptr %b, i64 4)
; CHECK-NEXT:  NoModRef:  Ptr: i8* %a.gep.5	<->  %res = tail call ptr @strncpy(ptr %a, ptr %b, i64 4)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %b.gep.1	<->  %res = tail call ptr @strncpy(ptr %a, ptr %b, i64 4)
; CHECK-NEXT:  NoModRef:  Ptr: i8* %b.gep.5	<->  %res = tail call ptr @strncpy(ptr %a, ptr %b, i64 4)
;
entry:
  load i8, ptr %a
  load i8, ptr %b
  %res = tail call ptr @strncpy(ptr %a, ptr %b, i64 4)
  load i8, ptr %res
  %a.gep.1 = getelementptr i8, ptr %a, i32 1
  store i8 0, ptr %a.gep.1
  %a.gep.5 = getelementptr i8, ptr %a, i32 5
  store i8 1, ptr %a.gep.5
  %b.gep.1 = getelementptr i8, ptr %b, i32 1
  store i8 2, ptr %b.gep.1
  %b.gep.5 = getelementptr i8, ptr %b, i32 5
  store i8 3, ptr %b.gep.5
  ret ptr %res
}

define ptr @test_strncpy_variable_size(ptr noalias %a, ptr noalias %b, i64 %n) {
; CHECK-LABEL: Function: test_strncpy_variable_size
; CHECK:       Just Mod:  Ptr: i8* %a	<->  %res = tail call ptr @strncpy(ptr %a, ptr %b, i64 %n)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %b	<->  %res = tail call ptr @strncpy(ptr %a, ptr %b, i64 %n)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %res	<->  %res = tail call ptr @strncpy(ptr %a, ptr %b, i64 %n)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %a.gep.1	<->  %res = tail call ptr @strncpy(ptr %a, ptr %b, i64 %n)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %a.gep.5	<->  %res = tail call ptr @strncpy(ptr %a, ptr %b, i64 %n)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %b.gep.1	<->  %res = tail call ptr @strncpy(ptr %a, ptr %b, i64 %n)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %b.gep.5	<->  %res = tail call ptr @strncpy(ptr %a, ptr %b, i64 %n)
;
entry:
  load i8, ptr %a
  load i8, ptr %b
  %res = tail call ptr @strncpy(ptr %a, ptr %b, i64 %n)
  load i8, ptr %res
  %a.gep.1 = getelementptr i8, ptr %a, i32 1
  store i8 0, ptr %a.gep.1
  %a.gep.5 = getelementptr i8, ptr %a, i32 5
  store i8 1, ptr %a.gep.5
  %b.gep.1 = getelementptr i8, ptr %b, i32 1
  store i8 2, ptr %b.gep.1
  %b.gep.5 = getelementptr i8, ptr %b, i32 5
  store i8 3, ptr %b.gep.5
  ret ptr %res
}

declare ptr @__memset_chk(ptr writeonly, i32, i64, i64)

; CHECK-LABEL: Function: test_memset_chk_const_size
define ptr @test_memset_chk_const_size(ptr noalias %a, i64 %n) {
; CHECK:       Just Mod:  Ptr: i8* %a	<->  %res = tail call ptr @__memset_chk(ptr %a, i32 0, i64 4, i64 %n)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %res	<->  %res = tail call ptr @__memset_chk(ptr %a, i32 0, i64 4, i64 %n)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %a.gep.1	<->  %res = tail call ptr @__memset_chk(ptr %a, i32 0, i64 4, i64 %n)
; CHECK-NEXT:  NoModRef:  Ptr: i8* %a.gep.5	<->  %res = tail call ptr @__memset_chk(ptr %a, i32 0, i64 4, i64 %n)
;
entry:
  load i8, ptr %a
  %res = tail call ptr @__memset_chk(ptr %a, i32 0, i64 4, i64 %n)
  load i8, ptr %res
  %a.gep.1 = getelementptr i8, ptr %a, i32 1
  store i8 0, ptr %a.gep.1
  %a.gep.5 = getelementptr i8, ptr %a, i32 5
  store i8 1, ptr %a.gep.5
  ret ptr %res
}

define ptr @test_memset_chk_variable_size(ptr noalias %a, i64 %n.1, i64 %n.2) {
; CHECK-LABEL: Function: test_memset_chk_variable_size
; CHECK:       Just Mod:  Ptr: i8* %a	<->  %res = tail call ptr @__memset_chk(ptr %a, i32 0, i64 %n.1, i64 %n.2)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %res	<->  %res = tail call ptr @__memset_chk(ptr %a, i32 0, i64 %n.1, i64 %n.2)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %a.gep.1	<->  %res = tail call ptr @__memset_chk(ptr %a, i32 0, i64 %n.1, i64 %n.2)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %a.gep.5	<->  %res = tail call ptr @__memset_chk(ptr %a, i32 0, i64 %n.1, i64 %n.2)
;
entry:
  load i8, ptr %a
  %res = tail call ptr @__memset_chk(ptr %a, i32 0, i64 %n.1, i64 %n.2)
  load i8, ptr %res
  %a.gep.1 = getelementptr i8, ptr %a, i32 1
  store i8 0, ptr %a.gep.1
  %a.gep.5 = getelementptr i8, ptr %a, i32 5
  store i8 1, ptr %a.gep.5
  ret ptr %res
}

declare ptr @__memcpy_chk(ptr writeonly, ptr readonly, i64, i64)

define ptr @test_memcpy_chk_const_size(ptr noalias %a, ptr noalias %b, i64 %n) {
; CHECK-LABEL: Function: test_memcpy_chk_const_size
; CHECK:       Just Mod:  Ptr: i8* %a	<->  %res = tail call ptr @__memcpy_chk(ptr %a, ptr %b, i64 4, i64 %n)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %res	<->  %res = tail call ptr @__memcpy_chk(ptr %a, ptr %b, i64 4, i64 %n)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %a.gep.1	<->  %res = tail call ptr @__memcpy_chk(ptr %a, ptr %b, i64 4, i64 %n)
; CHECK-NEXT:  NoModRef:  Ptr: i8* %a.gep.5	<->  %res = tail call ptr @__memcpy_chk(ptr %a, ptr %b, i64 4, i64 %n)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %b.gep.1	<->  %res = tail call ptr @__memcpy_chk(ptr %a, ptr %b, i64 4, i64 %n)
; CHECK-NEXT:  NoModRef:  Ptr: i8* %b.gep.5	<->  %res = tail call ptr @__memcpy_chk(ptr %a, ptr %b, i64 4, i64 %n)
;
entry:
  load i8, ptr %a
  %res = tail call ptr @__memcpy_chk(ptr %a, ptr %b, i64 4, i64 %n)
  load i8, ptr %res
  %a.gep.1 = getelementptr i8, ptr %a, i32 1
  store i8 0, ptr %a.gep.1
  %a.gep.5 = getelementptr i8, ptr %a, i32 5
  store i8 1, ptr %a.gep.5
  %b.gep.1 = getelementptr i8, ptr %b, i32 1
  store i8 0, ptr %b.gep.1
  %b.gep.5 = getelementptr i8, ptr %b, i32 5
  store i8 1, ptr %b.gep.5
  ret ptr %res
}

define ptr @test_memcpy_chk_variable_size(ptr noalias %a, ptr noalias %b, i64 %n.1, i64 %n.2) {
; CHECK-LABEL: Function: test_memcpy_chk_variable_size
; CHECK:       Just Mod:  Ptr: i8* %a	<->  %res = tail call ptr @__memcpy_chk(ptr %a, ptr %b, i64 %n.1, i64 %n.2)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %res	<->  %res = tail call ptr @__memcpy_chk(ptr %a, ptr %b, i64 %n.1, i64 %n.2)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %a.gep.1	<->  %res = tail call ptr @__memcpy_chk(ptr %a, ptr %b, i64 %n.1, i64 %n.2)
; CHECK-NEXT:  Just Mod:  Ptr: i8* %a.gep.5	<->  %res = tail call ptr @__memcpy_chk(ptr %a, ptr %b, i64 %n.1, i64 %n.2)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %b.gep.1	<->  %res = tail call ptr @__memcpy_chk(ptr %a, ptr %b, i64 %n.1, i64 %n.2)
; CHECK-NEXT:  Just Ref:  Ptr: i8* %b.gep.5	<->  %res = tail call ptr @__memcpy_chk(ptr %a, ptr %b, i64 %n.1, i64 %n.2)
;
entry:
  load i8, ptr %a
  %res = tail call ptr @__memcpy_chk(ptr %a, ptr %b, i64 %n.1, i64 %n.2)
  load i8, ptr %res
  %a.gep.1 = getelementptr i8, ptr %a, i32 1
  store i8 0, ptr %a.gep.1
  %a.gep.5 = getelementptr i8, ptr %a, i32 5
  store i8 1, ptr %a.gep.5
  %b.gep.1 = getelementptr i8, ptr %b, i32 1
  store i8 0, ptr %b.gep.1
  %b.gep.5 = getelementptr i8, ptr %b, i32 5
  store i8 1, ptr %b.gep.5
  ret ptr %res
}
