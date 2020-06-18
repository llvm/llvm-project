; RUN: llc -emit-call-site-info -mtriple aarch64-linux-gnu -debug-entry-values %s -o - -stop-before=finalize-isel | FileCheck %s
; Verify that Selection DAG knows how to recognize simple function parameter forwarding registers.
; Produced from:
; extern int fn1(int,int,int);
; int fn2(int a, int b, int c) {
;   int local = fn1(a+b, c, 10);
;   if (local > 10)
;     return local + 10;
;   return local;
; }
; clang -g -O2 -target aarch64-linux-gnu -S -emit-llvm %s
; CHECK: callSites:
; CHECK-NEXT:   - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs:
; CHECK-NEXT:       - { arg: 0, reg: '$w0' }
; CHECK-NEXT:       - { arg: 1, reg: '$w1' }
; CHECK-NEXT:       - { arg: 2, reg: '$w2' } }

; ModuleID = 'call-site-info-output.c'
source_filename = "call-site-info-output.c"
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nounwind
define dso_local i32 @fn2(i32 %a, i32 %b, i32 %c) local_unnamed_addr{
entry:
  %add = add nsw i32 %b, %a
  %call = tail call i32 @fn1(i32 %add, i32 %c, i32 10)
  %cmp = icmp sgt i32 %call, 10
  %add1 = add nsw i32 %call, 10
  %retval.0 = select i1 %cmp, i32 %add1, i32 %call
  ret i32 %retval.0
}

declare dso_local i32 @fn1(i32, i32, i32) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.ident = !{!0}

!0 = !{!"clang version 10.0.0"}
