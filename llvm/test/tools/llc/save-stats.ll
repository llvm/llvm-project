; REQUIRES: asserts

; Ensure the test runs in a temp directory. See https://github.com/llvm/llvm-project/pull/167403#event-20848739526
; RUN: rm -rf %t.dir && mkdir -p %t.dir && cd %t.dir

; RUN: llc --save-stats=obj -o %t.s %s && cat %t.stats | FileCheck %s
; RUN: llc --save-stats=cwd -o %t.s %s && cat %{t:stem}.tmp.stats | FileCheck %s
; RUN: llc --save-stats -o %t.s %s && cat %{t:stem}.tmp.stats | FileCheck %s
; RUN: not llc --save-stats=invalid -o %t.s %s 2>&1 | FileCheck %s --check-prefix=INVALID_ARG

; CHECK: {
; CHECK: "asm-printer.EmittedInsts":
; CHECK: }

; INVALID_ARG: {{.*}}llc{{.*}}: for the --save-stats option: Cannot find option named 'invalid'!
define i32 @func() {
  ret i32 0
}
