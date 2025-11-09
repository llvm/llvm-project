; REQUIRES: asserts

; RUN: llc -mtriple=arm64-apple-macosx --save-stats=obj -o %t.s %s && cat %t.stats | FileCheck %s
; RUN: llc -mtriple=arm64-apple-macosx --save-stats=cwd -o %t.s %s && cat %{t:stem}.tmp.stats | FileCheck %s
; RUN: llc -mtriple=arm64-apple-macosx --save-stats -o %t.s %s && cat %{t:stem}.tmp.stats | FileCheck %s
; RUN: not llc -mtriple=arm64-apple-macosx --save-stats=invalid -o %t.s %s 2>&1 | FileCheck %s --check-prefix=INVALID_ARG

; CHECK: {
; CHECK: "asm-printer.EmittedInsts":
; CHECK: }

; INVALID_ARG: {{.*}}llc{{.*}}: for the --save-stats option: Cannot find option named 'invalid'!
define i32 @func() {
  ret i32 0
}
