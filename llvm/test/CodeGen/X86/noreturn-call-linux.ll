; RUN: llc < %s -mtriple=x86_64-linux-gnu | FileCheck %s

; PR43155, we used to emit dead stack adjustments for noreturn calls with stack
; arguments.

; Original source code:
; __attribute__((noreturn)) void exit_manyarg(int, int, int, int, int, int, int, int, int, int);
; struct ByVal {
;   int vals[10];
; };
; struct ByVal getbyval();
; void make_push_unprofitable(struct ByVal);
; int foo(int c) {
;   if (c)
;     exit_manyarg(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
;   make_push_unprofitable(getbyval());
;   make_push_unprofitable(getbyval());
;   make_push_unprofitable(getbyval());
;   return 0;
; }

%struct.ByVal = type { [10 x i32] }

define dso_local i32 @foo(i32 %c) {
entry:
  %agg.tmp = alloca %struct.ByVal, align 8
  %agg.tmp1 = alloca %struct.ByVal, align 8
  %agg.tmp2 = alloca %struct.ByVal, align 8
  %tobool = icmp eq i32 %c, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  tail call void @exit_manyarg(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10) #3
  unreachable

if.end:                                           ; preds = %entry
  call void @getbyval(ptr nonnull sret(%struct.ByVal) %agg.tmp) #4
  call void @make_push_unprofitable(ptr nonnull byval(%struct.ByVal) align 8 %agg.tmp) #4
  call void @getbyval(ptr nonnull sret(%struct.ByVal) %agg.tmp1) #4
  call void @make_push_unprofitable(ptr nonnull byval(%struct.ByVal) align 8 %agg.tmp1) #4
  call void @getbyval(ptr nonnull sret(%struct.ByVal) %agg.tmp2) #4
  call void @make_push_unprofitable(ptr nonnull byval(%struct.ByVal) align 8 %agg.tmp2) #4
  ret i32 0
}

; CHECK-LABEL: foo:
;   The main body is not important.
; CHECK: callq exit_manyarg
; CHECK-NOT: sub
; CHECK-NOT: add
; CHECK: # -- End function

; Function Attrs: noreturn
declare dso_local void @exit_manyarg(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) noreturn

declare dso_local void @make_push_unprofitable(ptr byval(%struct.ByVal) align 8)

declare dso_local void @getbyval(ptr sret(%struct.ByVal))

