;; Test MIR printer and parser for type id field in call site info. Test that
;; it works well with/without --emit-call-site-info.

;; Multiplex --call-graph-section and -emit-call-site-info as both utilize
;; CallSiteInfo and callSites.

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Test printer and parser with --call-graph-section only.

;; Test printer.
;; Verify that fwdArgRegs is not set, calleeTypeIds is set.
;; Verify the exact calleeTypeIds value to ensure it is not garbage but the value
;; computed as the type id from the callee_type metadata.
; RUN: llc --call-graph-section %s -stop-after=finalize-isel -o %t1.mir
; RUN: cat %t1.mir | FileCheck %s --check-prefix=PRINTER_CGS
; PRINTER_CGS: name: main
; PRINTER_CGS: callSites:
; PRINTER_CGS-NEXT: - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs: [], calleeTypeIds:
; PRINTER_CGS-NEXT: [ 7854600665770582568 ] }


;; Test parser.
;; Verify that we get the same result.
; RUN: llc --call-graph-section %t1.mir -run-pass=finalize-isel -o - \
; RUN: | FileCheck %s --check-prefix=PARSER_CGS
; PARSER_CGS: name: main
; PARSER_CGS: callSites:
; PARSER_CGS-NEXT: - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs: [], calleeTypeIds:
; PARSER_CGS-NEXT: [ 7854600665770582568 ] }

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Test printer and parser with -emit-call-site-info only.

;; Test printer.
;; Verify that fwdArgRegs is set, calleeTypeIds is not set.
; RUN: llc -emit-call-site-info %s -stop-after=finalize-isel -o %t2.mir
; RUN: cat %t2.mir | FileCheck %s --check-prefix=PRINTER_CSI
; PRINTER_CSI: name: main
; PRINTER_CSI: callSites:
; PRINTER_CSI-NEXT: - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs:
; PRINTER_CSI-NEXT: { arg: 0, reg: '$edi' }
; PRINTER_CSI-NOT: calleeTypeIds:


;; Test parser.
;; Verify that we get the same result.
; RUN: llc -emit-call-site-info %t2.mir -run-pass=finalize-isel -o - \
; RUN: | FileCheck %s --check-prefix=PARSER_CSI
; PARSER_CSI: name: main
; PARSER_CSI: callSites:
; PARSER_CSI-NEXT: - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs:
; PARSER_CSI-NEXT: { arg: 0, reg: '$edi' }
; PARSER_CSI-NOT: calleeTypeIds:

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Test printer and parser with both -emit-call-site-info and --call-graph-section.

;; Test printer.
;; Verify both fwdArgRegs and calleeTypeIds are set.
;; Verify the exact calleeTypeIds value to ensure it is not garbage but the value
;; computed as the type id from the callee_type metadata.
; RUN: llc --call-graph-section -emit-call-site-info %s -stop-after=finalize-isel -o %t2.mir
; RUN: cat %t2.mir | FileCheck %s --check-prefix=PRINTER_CGS_CSI
; PRINTER_CGS_CSI: name: main
; PRINTER_CGS_CSI: callSites:
; PRINTER_CGS_CSI-NEXT: - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs:
; PRINTER_CGS_CSI-NEXT: { arg: 0, reg: '$edi' }, calleeTypeIds:
; PRINTER_CGS_CSI-NEXT:   [ 7854600665770582568 ] }


;; Test parser.
;; Verify that we get the same result.
; RUN: llc --call-graph-section -emit-call-site-info %t2.mir -run-pass=finalize-isel -o - \
; RUN: | FileCheck %s --check-prefix=PARSER_CGS_CSI
; PARSER_CGS_CSI: name: main
; PARSER_CGS_CSI: callSites:
; PARSER_CGS_CSI-NEXT: - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs:
; PARSER_CGS_CSI-NEXT: { arg: 0, reg: '$edi' }, calleeTypeIds:
; PARSER_CGS_CSI-NEXT:   [ 7854600665770582568 ] }

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @foo(i8 signext %a) !type !3 {
entry:
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main() !type !4 {
entry:
  %retval = alloca i32, align 4
  %fp = alloca ptr, align 8
  store i32 0, ptr %retval, align 4
  store ptr @foo, ptr %fp, align 8
  %fp_val = load ptr, ptr %fp, align 8
  call void %fp_val(i8 signext 97), !callee_type !5
  ret i32 0
}

!3 = !{i64 0, !"_ZTSFvcE.generalized"}
!4 = !{i64 0, !"_ZTSFiE.generalized"}
!5 = !{!3}
