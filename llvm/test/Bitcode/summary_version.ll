; Check summary versioning
; RUN: opt  -module-summary  %s -o - | llvm-bcanalyzer -dump | FileCheck %s

; CHECK: <GLOBALVAL_SUMMARY_BLOCK
; CHECK: <VERSION op0=6/>



; Need a function for the summary to be populated.
define void @foo() {
    ret void
}
