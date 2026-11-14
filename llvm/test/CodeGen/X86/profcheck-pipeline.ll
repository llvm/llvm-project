; RUN: llc -enable-new-pm -mtriple=x86_64 -O2 -print-pipeline-passes=tree -enable-profcheck-in-pipeline < %s 2>&1 \
; RUN:   | FileCheck %s

; CHECK:      require<libcall-lowering-info>
; CHECK-NEXT: function
; CHECK-NEXT:   prof-inject

; CHECK:      prof-verify
; CHECK-NEXT: function
; CHECK-NEXT:   verify
; CHECK-NEXT: x86-asm-printer-begin
