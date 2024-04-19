; RUN: opt -S -passes=aggressive-instcombine -strncmp-inline-threshold=3 < %s | FileCheck %s

declare i32 @strncmp(ptr nocapture, ptr nocapture, i64)
declare i32 @strcmp(ptr nocapture, ptr nocapture)

@.str = private unnamed_addr constant [3 x i8] c"aa\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"aab\00", align 1
@__const.test_strncmp_8.s2 = private unnamed_addr constant [2 x i8] c"aa", align 1

; int test_strncmp_1(const char *s) {
;   if (!strncmp(s, "aa", 2))
;     return 11;
;   return 41;
; }
define i32 @test_strncmp_1(i8* nocapture readonly %s) {
entry:
  %call = tail call i32 @strncmp(ptr nonnull dereferenceable(1) %s, ptr nonnull dereferenceable(3) @.str, i64 2)
  %tobool.not = icmp eq i32 %call, 0
  %retval.0 = select i1 %tobool.not, i32 11, i32 41
  ret i32 %retval.0
}
; CHECK-LABEL: @test_strncmp_1(
; CHECK-NOT: @strncmp

; int test_strncmp_2(const char *s) {
;   if (!strncmp(s, "aa", 3))
;     return 11;
;   return 41;
; }
define i32 @test_strncmp_2(i8* nocapture readonly %s) {
entry:
  %call = tail call i32 @strncmp(ptr nonnull dereferenceable(1) %s, ptr nonnull dereferenceable(3) @.str, i64 3)
  %tobool.not = icmp eq i32 %call, 0
  %retval.0 = select i1 %tobool.not, i32 11, i32 41
  ret i32 %retval.0
}
; CHECK-LABEL: @test_strncmp_2(
; CHECK-NOT: @strncmp

; int test_strncmp_3(const char *s) {
;   if (!strncmp(s, "aab", 3))
;     return 11;
;   return 41;
; }
define i32 @test_strncmp_3(i8* nocapture readonly %s) {
entry:
  %call = tail call i32 @strncmp(ptr nonnull dereferenceable(1) %s, ptr nonnull dereferenceable(4) @.str.1, i64 3)
  %tobool.not = icmp eq i32 %call, 0
  %retval.0 = select i1 %tobool.not, i32 11, i32 41
  ret i32 %retval.0
}
; CHECK-LABEL: @test_strncmp_3(
; CHECK-NOT: @strncmp

; int test_strncmp_4(const char *s) {
;   if (!strncmp(s, "aab", 4))
;     return 11;
;   return 41;
; }
define i32 @test_strncmp_4(i8* nocapture readonly %s) {
entry:
  %call = tail call i32 @strncmp(ptr nonnull dereferenceable(1) %s, ptr nonnull dereferenceable(4) @.str.1, i64 4)
  %tobool.not = icmp eq i32 %call, 0
  %retval.0 = select i1 %tobool.not, i32 11, i32 41
  ret i32 %retval.0
}
; CHECK-LABEL: @test_strncmp_4(
; CHECK: @strncmp

; int test_strncmp_5(const char *s) {
;   if (strncmp(s, "aa", 2) < 0)
;     return 11;
;   return 41;
; }
define i32 @test_strncmp_5(i8* nocapture readonly %s) {
entry:
  %call = tail call i32 @strncmp(ptr nonnull dereferenceable(1) %s, ptr nonnull dereferenceable(3) @.str, i64 2)
  %cmp = icmp slt i32 %call, 0
  %retval.0 = select i1 %cmp, i32 11, i32 41
  ret i32 %retval.0
}
; CHECK-LABEL: @test_strncmp_5(
; CHECK-NOT: @strncmp

; int test_strncmp_6(const char *s1) {
;   char s2[] = {'a', 'a'};
;   if (strncmp(s1, s2, 2) < 0)
;     return 11;
;   return 41;
; }
define i32 @test_strncmp_6(i8* nocapture readonly %s1) {
entry:
  %call = tail call i32 @strncmp(ptr nonnull dereferenceable(1) %s1, ptr nonnull dereferenceable(3) @__const.test_strncmp_8.s2, i64 2)
  %cmp = icmp slt i32 %call, 0
  %retval.0 = select i1 %cmp, i32 11, i32 41
  ret i32 %retval.0
}
; CHECK-LABEL: @test_strncmp_6(
; CHECK-NOT: @strncmp

; int test_strncmp_7(const char *s1) {
;   char s2[] = {'a', 'a'};
;   if (strncmp(s1, s2, 3) < 0)
;     return 11;
;   return 41;
; }
define i32 @test_strncmp_7(i8* nocapture readonly %s1) {
entry:
  %call = tail call i32 @strncmp(ptr nonnull dereferenceable(1) %s1, ptr nonnull dereferenceable(3) @__const.test_strncmp_8.s2, i64 3)
  %cmp = icmp slt i32 %call, 0
  %retval.0 = select i1 %cmp, i32 11, i32 41
  ret i32 %retval.0
}
; CHECK-LABEL: @test_strncmp_7(
; CHECK: @strncmp

; int test_strcmp_1(const char *s) {
;   if (!strcmp(s, "aa"))
;     return 11;
;   return 41;
; }
define i32 @test_strcmp_1(i8* nocapture readonly %s) {
entry:
  %call = tail call i32 @strcmp(ptr nonnull dereferenceable(1) %s, ptr nonnull dereferenceable(3) @.str)
  %tobool.not = icmp eq i32 %call, 0
  %retval.0 = select i1 %tobool.not, i32 11, i32 41
  ret i32 %retval.0
}
; CHECK-LABEL: @test_strcmp_1(
; CHECK-NOT: @strcmp

; int test_strcmp_2(const char *s) {
;   if (!strcmp(s, "aab"))
;     return 11;
;   return 41;
; }
define i32 @test_strcmp_2(i8* nocapture readonly %s) {
entry:
  %call = tail call i32 @strcmp(ptr nonnull dereferenceable(1) %s, ptr nonnull dereferenceable(4) @.str.1)
  %tobool.not = icmp eq i32 %call, 0
  %retval.0 = select i1 %tobool.not, i32 11, i32 41
  ret i32 %retval.0
}
; CHECK-LABEL: @test_strcmp_2(
; CHECK: @strcmp
