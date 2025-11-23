; REQUIRES: asserts

; Ensure the test runs in a temp directory. See https://github.com/llvm/llvm-project/pull/167403#event-20848739526
; RUN: rm -rf %t.dir && mkdir -p %t.dir && cd %t.dir

; RUN: opt -S -passes=instcombine --save-stats=obj -o %t.ll %s && cat %t.stats | FileCheck %s
; RUN: opt -S -passes=instcombine --save-stats=cwd -o %t.ll %s && cat %{t:stem}.tmp.stats | FileCheck %s
; RUN: opt -S -passes=instcombine --save-stats -o %t.ll %s && cat %{t:stem}.tmp.stats | FileCheck %s
; RUN: not opt -S --save-stats=invalid -o %t.ll %s 2>&1 | FileCheck %s --check-prefix=INVALID_ARG

; CHECK: {
; CHECK: "instcombine.NumWorklistIterations":
; CHECK: }

; INVALID_ARG: {{.*}}opt{{.*}}: for the --save-stats option: Cannot find option named 'invalid'!
define i32 @func() {
  ret i32 0
}
