; RUN: opt -passes=gvn -S < %s | FileCheck %s
;
; The Global Value Numbering pass (GVN) propagates boolean values
; that are constant in dominated basic blocks to all the uses
; in these basic blocks. However, we don't want the constant propagated
; into fake.use intrinsics since this would render the intrinsic useless
; with respect to keeping the variable live up until the fake.use.
; This test checks that we don't generate any fake.uses with constant 0.
;
; Reduced from the following test case, generated with clang -O2 -S -emit-llvm -fextend-lifetimes test.c
;
; extern void func1();
; extern int bar();
; extern void baz(int);
;
; int foo(int i, float f, int *punused)
; {
;   int j = 3*i;
;   if (j > 0) {
;     int m = bar(i);
;     if (m) {
;       char b = f;
;       baz(b);
;       if (b)
;         goto lab;
;       func1();
;     }
; lab:
;     func1();
;   }
;   return 1;
; }

;; GVN should propagate a constant value through to a regular call, but not to
;; a fake use, which should continue to track the original value.
; CHECK: %[[CONV_VAR:[a-zA-Z0-9]+]] = fptosi
; CHECK: call {{.+}} @bees(i8 0)
; CHECK: call {{.+}} @llvm.fake.use(i8 %[[CONV_VAR]])

define i32 @foo(float %f) optdebug {
  %conv = fptosi float %f to i8
  %tobool3 = icmp eq i8 %conv, 0
  br i1 %tobool3, label %if.end, label %lab

if.end:
  tail call void (...) @bees(i8 %conv)
  tail call void (...) @llvm.fake.use(i8 %conv)
  br label %lab

lab:
  ret i32 1
}

declare i32 @bar(...)

declare void @baz(i32)

declare void @bees(i32)

declare void @func1(...)
