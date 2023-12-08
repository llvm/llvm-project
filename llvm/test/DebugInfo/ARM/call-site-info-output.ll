; RUN: llc -emit-call-site-info -mtriple arm-linux-gnu -debug-entry-values %s -o - -stop-before=finalize-isel | FileCheck %s
; Verify that Selection DAG knows how to recognize simple function parameter forwarding registers.
; Produced from:
; extern int fn1(int,int,int);
; int fn2(int a, int b, int c) {
;   int local = fn1(a+b, c, 10);
;   if (local > 10)
;     return local + 10;
;   return local;
; }
; clang -g -O2 -target arm-linux-gnu -S -emit-llvm %s
; CHECK: callSites:
; CHECK-NEXT:   - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs:
; CHECK-NEXT:       - { arg: 0, reg: '$r0' }
; CHECK-NEXT:       - { arg: 1, reg: '$r1' }
; CHECK-NEXT:       - { arg: 2, reg: '$r2' } }

; ModuleID = 'call-site-info-output.c'
source_filename = "call-site-info-output.c"
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv4t-unknown-linux-gnu"

; Function Attrs: nounwind
define dso_local arm_aapcscc i32 @fn2(i32 %a, i32 %b, i32 %c) {
entry:
  %add = add nsw i32 %b, %a
  %call = tail call arm_aapcscc i32 @fn1(i32 %add, i32 %c, i32 10)
  %cmp = icmp sgt i32 %call, 10
  %add1 = select i1 %cmp, i32 %c, i32 0
  %retval.0 = add nsw i32 %add1, %call
  ret i32 %retval.0
}

declare dso_local arm_aapcscc i32 @fn1(i32, i32, i32) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.ident = !{!0}

!0 = !{!"clang version 10.0.0"}
