; RUN: opt %loadNPMPolly '-passes=polly-custom<prepare;ast>' -polly-print-ast -S < %s | FileCheck %s

; This testcase tests plugin registration. Check-lines below serve to verify
; that the passes actually ran.

; CHECK-LABEL: void @foo
; CHECK-NEXT: entry:
; CHECK-NEXT: br label %entry.split
define void @foo() {
entry:
  ret void
}
