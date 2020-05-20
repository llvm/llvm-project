;; Test mips32:
; RUN: llc -mtriple=mips-linux-gnu -emit-call-site-info %s -stop-before=finalize-isel -o -| \
; RUN: llc -mtriple=mips-linux-gnu -emit-call-site-info -x='mir' -run-pass=finalize-isel -o -| FileCheck %s
;; Test mips64:
; RUN: llc -mtriple=mips64-linux-gnu -emit-call-site-info %s -stop-before=finalize-isel -o -| \
; RUN: llc -mtriple=mips64-linux-gnu -emit-call-site-info -x='mir' -run-pass=finalize-isel -o -| FileCheck %s --check-prefix=CHECK64
;; Test mipsel:
; RUN: llc -mtriple=mipsel-linux-gnu -emit-call-site-info %s -stop-before=finalize-isel -o -| \
; RUN: llc -mtriple=mipsel-linux-gnu -emit-call-site-info -x='mir' -run-pass=finalize-isel -o -| FileCheck %s
;; Test mips64el:
; RUN: llc -mtriple=mips64el-linux-gnu -emit-call-site-info %s -stop-before=finalize-isel -o -| \
; RUN: llc -mtriple=mips64el-linux-gnu -emit-call-site-info -x='mir' -run-pass=finalize-isel -o -| FileCheck %s --check-prefix=CHECK64

;; Test call site info MIR parser and printer. Parser assertions and machine
;; verifier will check the rest.
;; There is no need to verify call instruction location since it will be
;; checked by the MIR parser.
;; Verify that we are able to parse output mir and that we are getting valid call site info.

;; Source:
;; extern int fn1(int,int,int);
;; int fn2(int a, int b, int c) {
;;   int local = fn1(a+b, c, 10);
;;   if (local > 10)
;;     return local + 10;
;;   return local;
;; }

;; Test mips32 and mips32el:
; CHECK: name: fn2
; CHECK: callSites:
; CHECK-NEXT: bb: {{.*}}, offset: {{.*}}, fwdArgRegs:
; CHECK-NEXT:   arg: 0, reg: '$a0'
; CHECK-NEXT:   arg: 1, reg: '$a1'
; CHECK-NEXT:   arg: 2, reg: '$a2'

;; Test mips64 and mips64el:
; CHECK64: name: fn2
; CHECK64: callSites:
; CHECK64-NEXT: bb: {{.*}}, offset: {{.*}}, fwdArgRegs:
; CHECK64-NEXT:   arg: 0, reg: '$a0_64'
; CHECK64-NEXT:   arg: 1, reg: '$a1_64'
; CHECK64-NEXT:   arg: 2, reg: '$a2_64'

; ModuleID = 'test/CodeGen/Mips/call-site-info-output.c'
source_filename = "test/CodeGen/Mips/call-site-info-output.c"
target datalayout = "E-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64"
target triple = "mips-unknown-linux-gnu"
; Function Attrs: nounwind
define dso_local i32 @fn2(i32 signext %a, i32 signext %b, i32 signext %c) local_unnamed_addr {
entry:
  %add = add nsw i32 %b, %a
  %call = tail call i32 @fn1(i32 signext %add, i32 signext %c, i32 signext 10)
  %cmp = icmp sgt i32 %call, 10
  %add1 = add nsw i32 %call, 10
  %retval.0 = select i1 %cmp, i32 %add1, i32 %call
  ret i32 %retval.0
}
declare dso_local i32 @fn1(i32 signext, i32 signext, i32 signext) local_unnamed_addr

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 11.0.0"}
