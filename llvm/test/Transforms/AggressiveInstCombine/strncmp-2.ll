; RUN: opt -S -passes=aggressive-instcombine -strncmp-inline-threshold=3 < %s | FileCheck %s
; RUN: opt -S -passes=aggressive-instcombine -strncmp-inline-threshold=2 < %s | FileCheck --check-prefix TH-2 %s
; RUN: opt -S -passes=aggressive-instcombine -strncmp-inline-threshold=1 < %s | FileCheck --check-prefix TH-1 %s
; RUN: opt -S -passes=aggressive-instcombine -strncmp-inline-threshold=0 < %s | FileCheck --check-prefix TH-0 %s

declare i32 @strcmp(ptr nocapture, ptr nocapture)
declare i32 @strncmp(ptr nocapture, ptr nocapture, i64)

@s1 = constant [1 x i8] c"\00", align 1
@s2n = constant [2 x i8] c"aa", align 1
@s3 = constant [3 x i8] c"aa\00", align 1
@s4 = constant [4 x i8] c"aab\00", align 1

; strncmp(s, "aa", 1)
define i1 @test_strncmp_0(i8* nocapture readonly %s) {
entry:
  %call = tail call i32 @strncmp(ptr nonnull dereferenceable(1) %s, ptr nonnull dereferenceable(3) @s3, i64 1)
  %cmp = icmp eq i32 %call, 0
  ret i1 %cmp
}
; CHECK-LABEL: @test_strncmp_0(
; CHECK: @strncmp

; strncmp(s, "aa", 2)
define i1 @test_strncmp_1(i8* nocapture readonly %s) {
entry:
  %call = tail call i32 @strncmp(ptr nonnull dereferenceable(1) %s, ptr nonnull dereferenceable(3) @s3, i64 2)
  %cmp = icmp eq i32 %call, 0
  ret i1 %cmp
}
; CHECK-LABEL: @test_strncmp_1(
; CHECK-NOT: @strncmp

; TH-2-LABEL: @test_strncmp_1(
; TH-2-NOT: @strncmp
; TH-1-LABEL: @test_strncmp_1(
; TH-1: @strncmp
; TH-0-LABEL: @test_strncmp_1(
; TH-0: @strncmp

define i1 @test_strncmp_1_dereferenceable(i8* nocapture readonly dereferenceable(2) %s) {
entry:
  %call = tail call i32 @strncmp(ptr nonnull %s, ptr nonnull dereferenceable(3) @s3, i64 2)
  %cmp = icmp eq i32 %call, 0
  ret i1 %cmp
}
; CHECK-LABEL: @test_strncmp_1_dereferenceable(
; CHECK: @strncmp

; TH-2-LABEL: @test_strncmp_1_dereferenceable(
; TH-1-LABEL: @test_strncmp_1_dereferenceable(
; TH-0-LABEL: @test_strncmp_1_dereferenceable(

define i32 @test_strncmp_1_not_comparision(i8* nocapture readonly %s) {
entry:
  %call = tail call i32 @strncmp(ptr nonnull dereferenceable(1) %s, ptr nonnull dereferenceable(3) @s3, i64 2)
  ret i32 %call
}
; CHECK-LABEL: @test_strncmp_1_not_comparision(
; CHECK: @strncmp

; strncmp(s, "aa", 3)
define i1 @test_strncmp_2(i8* nocapture readonly %s) {
entry:
  %call = tail call i32 @strncmp(ptr nonnull dereferenceable(1) %s, ptr nonnull dereferenceable(3) @s3, i64 3)
  %cmp = icmp eq i32 %call, 0
  ret i1 %cmp
}
; CHECK-LABEL: @test_strncmp_2(
; CHECK-NOT: @strncmp

; TH-2-LABEL: @test_strncmp_2(
; TH-2: @strncmp
; TH-1-LABEL: @test_strncmp_2(
; TH-1: @strncmp
; TH-0-LABEL: @test_strncmp_2(
; TH-0: @strncmp

; strncmp(s, "aab", 3)
define i1 @test_strncmp_3(i8* nocapture readonly %s) {
entry:
  %call = tail call i32 @strncmp(ptr nonnull dereferenceable(1) %s, ptr nonnull dereferenceable(4) @s4, i64 3)
  %cmp = icmp eq i32 %call, 0
  ret i1 %cmp
}
; CHECK-LABEL: @test_strncmp_3(
; CHECK-NOT: @strncmp

; TH-2-LABEL: @test_strncmp_3(
; TH-1-LABEL: @test_strncmp_3(
; TH-0-LABEL: @test_strncmp_3(

; strncmp(s, "aab", 4)
define i1 @test_strncmp_4(i8* nocapture readonly %s) {
entry:
  %call = tail call i32 @strncmp(ptr nonnull dereferenceable(1) %s, ptr nonnull dereferenceable(4) @s4, i64 4)
  %cmp = icmp eq i32 %call, 0
  ret i1 %cmp
}
; CHECK-LABEL: @test_strncmp_4(
; CHECK: @strncmp

; strncmp(s, "aa", 2)
define i1 @test_strncmp_5(i8* nocapture readonly %s) {
entry:
  %call = tail call i32 @strncmp(ptr nonnull dereferenceable(1) %s, ptr nonnull dereferenceable(3) @s3, i64 2)
  %cmp = icmp eq i32 %call, 0
  ret i1 %cmp
}
; CHECK-LABEL: @test_strncmp_5(
; CHECK-NOT: @strncmp

; char s2[] = {'a', 'a'}
; strncmp(s1, s2, 2)
define i1 @test_strncmp_6(i8* nocapture readonly %s1) {
entry:
  %call = tail call i32 @strncmp(ptr nonnull dereferenceable(1) %s1, ptr nonnull dereferenceable(3) @s2n, i64 2)
  %cmp = icmp eq i32 %call, 0
  ret i1 %cmp
}
; CHECK-LABEL: @test_strncmp_6(
; CHECK-NOT: @strncmp

; char s2[] = {'a', 'a'}
; strncmp(s1, s2, 3)
define i1 @test_strncmp_7(i8* nocapture readonly %s1) {
entry:
  %call = tail call i32 @strncmp(ptr nonnull dereferenceable(1) %s1, ptr nonnull dereferenceable(3) @s2n, i64 3)
  %cmp = icmp eq i32 %call, 0
  ret i1 %cmp
}
; CHECK-LABEL: @test_strncmp_7(
; CHECK: @strncmp

; strcmp(s, "")
define i1 @test_strcmp_0(i8* nocapture readonly %s) {
entry:
  %call = tail call i32 @strcmp(ptr nonnull dereferenceable(1) %s, ptr nonnull dereferenceable(1) @s1)
  %cmp = icmp eq i32 %call, 0
  ret i1 %cmp
}
; CHECK-LABEL: @test_strcmp_0(
; CHECK: @strcmp

; strcmp(s, "aa")
define i1 @test_strcmp_1(i8* nocapture readonly %s) {
entry:
  %call = tail call i32 @strcmp(ptr nonnull dereferenceable(1) %s, ptr nonnull dereferenceable(3) @s3)
  %cmp = icmp eq i32 %call, 0
  ret i1 %cmp
}
; CHECK-LABEL: @test_strcmp_1(
; CHECK-NOT: @strcmp

; strcmp(s, "aab")
define i1 @test_strcmp_2(i8* nocapture readonly %s) {
entry:
  %call = tail call i32 @strcmp(ptr nonnull dereferenceable(1) %s, ptr nonnull dereferenceable(4) @s4)
  %cmp = icmp eq i32 %call, 0
  ret i1 %cmp
}
; CHECK-LABEL: @test_strcmp_2(
; CHECK: @strcmp
