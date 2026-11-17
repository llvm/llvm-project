; RUN: llvm-profdata merge %S/Inputs/diag.proftext -o %t.profdata
; RUN: opt < %s -passes='pgo-instr-use,pgo-force-function-attrs' -pgo-kind=pgo-instr-use-pipeline -pgo-test-profile-file=%t.profdata -pgo-cold-func-opt=minsize -pgo-warn-missing-function=true -S 2>&1 | FileCheck %s --check-prefixes=CHECK,COLD-DEFAULT

; RUN: opt < %s -passes='pgo-instr-use,pgo-force-function-attrs' -pgo-kind=pgo-instr-use-pipeline -pgo-test-profile-file=%t.profdata -pgo-cold-func-opt=minsize -S 2>&1 | FileCheck %s --check-prefixes=DEFAULT,COLD-DEFAULT

; RUN: opt < %s -passes='pgo-instr-use,pgo-force-function-attrs' -pgo-kind=pgo-instr-use-pipeline -pgo-test-profile-file=%t.profdata -pgo-cold-func-opt=minsize -pgo-use-treat-unknown-as-cold=false -S 2>&1 | FileCheck %s --check-prefixes=DEFAULT,NO-COLD

; CHECK: warning: {{.+}}: no profile data available for function bar
; DEFAULT-NOT: no profile data available for function bar

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Unprofiled instrumentable function becomes cold + minsize + entry_count 0 by default.
; COLD-DEFAULT: Function Attrs: cold minsize
; COLD-DEFAULT-NEXT: define i32 @bar() #{{[0-9]+}} !prof ![[ZERO_ENTRY:[0-9]+]] {
; NO-COLD-NOT: Function Attrs:{{.*}}cold
; NO-COLD: define i32 @bar() {
define i32 @bar() {
entry:
  ret i32 0 
}

; Functions skipped by PGO instrumentation (noprofile, skipprofile) or marked hot
; must NOT be marked cold when missing from the profile.
; COLD-DEFAULT: Function Attrs: noprofile
; COLD-DEFAULT-NEXT: define i32 @bar_noprofile() #{{[0-9]+}} {
define i32 @bar_noprofile() noprofile {
entry:
  ret i32 0
}

; COLD-DEFAULT: Function Attrs: skipprofile
; COLD-DEFAULT-NEXT: define i32 @bar_skipprofile() #{{[0-9]+}} {
define i32 @bar_skipprofile() skipprofile {
entry:
  ret i32 0
}

; COLD-DEFAULT: Function Attrs: hot
; COLD-DEFAULT-NEXT: define i32 @bar_hot() #{{[0-9]+}} {
define i32 @bar_hot() hot {
entry:
  ret i32 0
}

; COLD-DEFAULT: ![[ZERO_ENTRY]] = !{!"function_entry_count", i64 0}
