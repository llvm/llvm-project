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
; RUN: llc -mtriple=x86_64 --call-graph-section %s -stop-after=finalize-isel -o %t1.mir
; RUN: cat %t1.mir | FileCheck %s --check-prefix=PRINTER_CGS
; PRINTER_CGS: name: main
; PRINTER_CGS: callSites:
; PRINTER_CGS-NEXT: - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs: [], calleeTypeIds:
; PRINTER_CGS-NEXT: [ 7854600665770582568 ] }


;; Test parser.
;; Verify that we get the same result.
; RUN: llc -mtriple=x86_64 --call-graph-section %t1.mir -run-pass=finalize-isel -o - \
; RUN: | FileCheck %s --check-prefix=PARSER_CGS
; PARSER_CGS: name: main
; PARSER_CGS: callSites:
; PARSER_CGS-NEXT: - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs: [], calleeTypeIds:
; PARSER_CGS-NEXT: [ 7854600665770582568 ] }

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Test printer and parser with -emit-call-site-info only.

;; Test printer.
;; Verify that fwdArgRegs is set, calleeTypeIds is not set.
; RUN: llc -mtriple=x86_64 -emit-call-site-info %s -stop-after=finalize-isel -o %t2.mir
; RUN: cat %t2.mir | FileCheck %s --check-prefix=PRINTER_CSI
; PRINTER_CSI: name: main
; PRINTER_CSI: callSites:
; PRINTER_CSI-NEXT: - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs:
; PRINTER_CSI-NEXT: { arg: 0, reg: {{.*}} }
; PRINTER_CSI-NOT: calleeTypeIds:


;; Test parser.
;; Verify that we get the same result.
; RUN: llc -mtriple=x86_64 -emit-call-site-info %t2.mir -run-pass=finalize-isel -o - \
; RUN: | FileCheck %s --check-prefix=PARSER_CSI
; PARSER_CSI: name: main
; PARSER_CSI: callSites:
; PARSER_CSI-NEXT: - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs:
; PARSER_CSI-NEXT: { arg: 0, reg: {{.*}} }
; PARSER_CSI-NOT: calleeTypeIds:

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Test printer and parser with both -emit-call-site-info and --call-graph-section.

;; Test printer.
;; Verify both fwdArgRegs and calleeTypeIds are set.
;; Verify the exact calleeTypeIds value to ensure it is not garbage but the value
;; computed as the type id from the callee_type metadata.
; RUN: llc -mtriple=x86_64 --call-graph-section -emit-call-site-info %s -stop-after=finalize-isel -o %t2.mir
; RUN: cat %t2.mir | FileCheck %s --check-prefix=PRINTER_CGS_CSI
; PRINTER_CGS_CSI: name: main
; PRINTER_CGS_CSI: callSites:
; PRINTER_CGS_CSI-NEXT: - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs:
; PRINTER_CGS_CSI-NEXT: { arg: 0, reg: {{.*}} }, calleeTypeIds:
; PRINTER_CGS_CSI-NEXT:   [ 7854600665770582568 ] }


;; Test parser.
;; Verify that we get the same result.
; RUN: llc -mtriple=x86_64 --call-graph-section -emit-call-site-info %t2.mir -run-pass=finalize-isel -o - \
; RUN: | FileCheck %s --check-prefix=PARSER_CGS_CSI
; PARSER_CGS_CSI: name: main
; PARSER_CGS_CSI: callSites:
; PARSER_CGS_CSI-NEXT: - { bb: {{.*}}, offset: {{.*}}, fwdArgRegs:
; PARSER_CGS_CSI-NEXT: { arg: 0, reg: {{.*}} }, calleeTypeIds:
; PARSER_CGS_CSI-NEXT:   [ 7854600665770582568 ] }

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define i32 @main() {
entry:
  %fn = load ptr, ptr null, align 8
  call void %fn(i8 0), !callee_type !0
  ret i32 0
}

!0 = !{!1}
!1 = !{i64 0, !"_ZTSFvcE.generalized"}
