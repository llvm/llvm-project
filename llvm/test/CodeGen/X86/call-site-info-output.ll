; Test call site info MIR printer and parser.Parser assertions and machine
; verifier will check the rest;
; RUN: llc -emit-call-site-info %s -stop-before=finalize-isel -o %t.mir
; RUN: cat %t.mir | FileCheck %s
; CHECK: name: fn2
; CHECK: callSites:
; There is no need to verify call instruction location since it will be
; checked by the MIR parser in the next RUN.
; CHECK-NEXT: bb: {{.*}}, offset: {{.*}}, fwdArgRegs:
; CHECK-NEXT:   arg: 0, reg: '$edi'
; CHECK-NEXT:   arg: 1, reg: '$esi'
; CHECK-NEXT:   arg: 2, reg: '$edx'
; RUN: llc -emit-call-site-info %t.mir -run-pass=finalize-isel -o -| FileCheck %s --check-prefix=PARSER
; Verify that we are able to parse output mir and that we are getting the same result.
; PARSER: name: fn2
; PARSER: callSites:
; PARSER-NEXT: bb: {{.*}}, offset: {{.*}}, fwdArgRegs:
; PARSER-NEXT:   arg: 0, reg: '$edi'
; PARSER-NEXT:   arg: 1, reg: '$esi'
; PARSER-NEXT:   arg: 2, reg: '$edx'

; ModuleID = 'test/CodeGen/X86/call-site-info-output.c'
source_filename = "test/CodeGen/X86/call-site-info-output.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind uwtable
define dso_local i64 @fn2(i32 %a, i32 %b, i32 %c) local_unnamed_addr {
entry:
  %call = tail call i32 (i32, i32, i32, ...) bitcast (i32 (...)* @fn1 to i32 (i32, i32, i32, ...)*)(i32 -50, i32 50, i32 -7)
  %add = mul i32 %a, 3
  %sub = sub i32 %add, %b
  %add2 = add i32 %sub, %c
  %conv4 = sext i32 %add2 to i64
  ret i64 %conv4
}

declare dso_local i32 @fn1(...) local_unnamed_addr

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 9.0.0"}
