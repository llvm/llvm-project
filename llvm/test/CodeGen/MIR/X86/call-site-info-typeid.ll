; Test MIR printer and parser for type id field in call site info. Test that
; it works well with/without --emit-call-site-info.

; Multiplex --call-graph-section and -emit-call-site-info as both utilize
; CallSiteInfo and callSites.

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Test printer and parser with --call-graph-section only.

; Test printer.
; Verify that fwdArgRegs is not set, typeId is set.
; Verify the exact typeId value to ensure it is not garbage but the value
; computed as the type id from the type operand bundle.
; RUN: llc --call-graph-section %s -stop-before=finalize-isel -o %t1.mir
; RUN: cat %t1.mir | FileCheck %s --check-prefix=PRINTER_CGS
; PRINTER_CGS: name: main
; PRINTER_CGS: callSites:
; PRINTER_CGS-NEXT: - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs: [], typeId:
; PRINTER_CGS-NEXT: 7854600665770582568 }


; Test parser.
; Verify that we get the same result.
; RUN: llc --call-graph-section %t1.mir -run-pass=finalize-isel -o - \
; RUN: | FileCheck %s --check-prefix=PARSER_CGS
; PARSER_CGS: name: main
; PARSER_CGS: callSites:
; PARSER_CGS-NEXT: - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs: [], typeId:
; PARSER_CGS-NEXT: 7854600665770582568 }

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Test printer and parser with -emit-call-site-info only.

; Test printer.
; Verify that fwdArgRegs is set, typeId is not set.
; RUN: llc -emit-call-site-info %s -stop-before=finalize-isel -o %t2.mir
; RUN: cat %t2.mir | FileCheck %s --check-prefix=PRINTER_CSI
; PRINTER_CSI: name: main
; PRINTER_CSI: callSites:
; PRINTER_CSI-NEXT: - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs:
; PRINTER_CSI-NEXT: { arg: 0, reg: '$edi' }
; PRINTER_CSI-NOT: typeId:


; Test parser.
; Verify that we get the same result.
; RUN: llc -emit-call-site-info %t2.mir -run-pass=finalize-isel -o - \
; RUN: | FileCheck %s --check-prefix=PARSER_CSI
; PARSER_CSI: name: main
; PARSER_CSI: callSites:
; PARSER_CSI-NEXT: - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs:
; PARSER_CSI-NEXT: { arg: 0, reg: '$edi' }
; PARSER_CSI-NOT: typeId:

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Test printer and parser with both -emit-call-site-info and --call-graph-section.

; Test printer.
; Verify both fwdArgRegs and typeId are set.
; Verify the exact typeId value to ensure it is not garbage but the value
; computed as the type id from the type operand bundle.
; RUN: llc --call-graph-section -emit-call-site-info %s -stop-before=finalize-isel -o %t2.mir
; RUN: cat %t2.mir | FileCheck %s --check-prefix=PRINTER_CGS_CSI
; PRINTER_CGS_CSI: name: main
; PRINTER_CGS_CSI: callSites:
; PRINTER_CGS_CSI-NEXT: - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs:
; PRINTER_CGS_CSI-NEXT: { arg: 0, reg: '$edi' }, typeId:
; PRINTER_CGS_CSI-NEXT:   7854600665770582568 }


; Test parser.
; Verify that we get the same result.
; RUN: llc --call-graph-section -emit-call-site-info %t2.mir -run-pass=finalize-isel -o - \
; RUN: | FileCheck %s --check-prefix=PARSER_CGS_CSI
; PARSER_CGS_CSI: name: main
; PARSER_CGS_CSI: callSites:
; PARSER_CGS_CSI-NEXT: - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs:
; PARSER_CGS_CSI-NEXT: { arg: 0, reg: '$edi' }, typeId:
; PARSER_CGS_CSI-NEXT:   7854600665770582568 }

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @foo(i8 signext %a) !type !3 {
entry:
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() !type !4 {
entry:
  %retval = alloca i32, align 4
  %fp = alloca void (i8)*, align 8
  store i32 0, i32* %retval, align 4
  store void (i8)* @foo, void (i8)** %fp, align 8
  %0 = load void (i8)*, void (i8)** %fp, align 8
  call void %0(i8 signext 97) [ "type"(metadata !"_ZTSFvcE.generalized") ]
  ret i32 0
}

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i64 0, !"_ZTSFvcE.generalized"}
!4 = !{i64 0, !"_ZTSFiE.generalized"}
