; RUN: rm -f %t.stats && llc -mtriple=arm64-apple-macosx --save-stats=obj -o %t.s %s && test -f %t.stats
; RUN: rm -f %{t:stem}.tmp.stats && llc -mtriple=arm64-apple-macosx --save-stats=cwd -o %t.s %s && test -f %{t:stem}.tmp.stats
; RUN: rm -f %{t:stem}.tmp.stats && llc -mtriple=arm64-apple-macosx --save-stats -o %t.s %s && test -f %{t:stem}.tmp.stats
; RUN: not llc -mtriple=arm64-apple-macosx --save-stats=invalid -o %t.s %s 2>&1 | FileCheck %s --check-prefix=INVALID_ARG

; INVALID_ARG: {{.*}}llc{{.*}}: for the --save-stats option: Cannot find option named 'invalid'!
define i32 @func() {
  ret i32 0
}
