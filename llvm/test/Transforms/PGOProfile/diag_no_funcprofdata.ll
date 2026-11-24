; RUN: llvm-profdata merge %S/Inputs/diag.proftext -o %t.profdata
; RUN: opt < %s -passes='pgo-instr-use,pgo-force-function-attrs' -pgo-kind=pgo-instr-use-pipeline -pgo-test-profile-file=%t.profdata -pgo-cold-func-opt=minsize -pgo-warn-missing-function=true -S 2>&1 | FileCheck %s --check-prefixes=CHECK,NONE

; RUN: opt < %s -passes='pgo-instr-use,pgo-force-function-attrs' -pgo-kind=pgo-instr-use-pipeline -pgo-test-profile-file=%t.profdata -pgo-cold-func-opt=minsize -S 2>&1 | FileCheck %s --check-prefixes=DEFAULT,NONE

; RUN: opt < %s -passes='pgo-instr-use,pgo-force-function-attrs' -pgo-kind=pgo-instr-use-pipeline -pgo-test-profile-file=%t.profdata -pgo-cold-func-opt=minsize -pgo-use-missing-func=none -S 2>&1 | FileCheck %s --check-prefixes=DEFAULT,NONE

; RUN: opt < %s -passes='pgo-instr-use,pgo-force-function-attrs' -pgo-kind=pgo-instr-use-pipeline -pgo-test-profile-file=%t.profdata -pgo-cold-func-opt=minsize -pgo-use-missing-func=cold -S 2>&1 | FileCheck %s --check-prefixes=DEFAULT,COLD

; RUN: opt < %s -passes='pgo-instr-use,pgo-force-function-attrs' -pgo-kind=pgo-instr-use-pipeline -pgo-test-profile-file=%t.profdata -pgo-cold-func-opt=minsize -pgo-use-missing-func=hot -S 2>&1 | FileCheck %s --check-prefixes=DEFAULT,HOT

; CHECK: warning: {{.+}}: no profile data available for function bar
; DEFAULT-NOT: no profile data available for function bar

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; By default (-pgo-use-missing-func=none), unprofiled functions stay unannotated.
; With cold/hot, they get cold+minsize+entry_count=0 or hot respectively.
; NONE-NOT: Function Attrs:{{.*}}(cold|hot)
; NONE: define i32 @bar() {
; COLD: Function Attrs: cold minsize
; COLD-NEXT: define i32 @bar() #{{[0-9]+}} !prof ![[ZERO_ENTRY:[0-9]+]] {
; HOT: Function Attrs: hot
; HOT-NEXT: define i32 @bar() #{{[0-9]+}} {
define i32 @bar() {
entry:
  ret i32 0 
}

; Functions skipped by PGO instrumentation (noprofile, skipprofile) or marked
; with conflicting user attributes (hot/cold) must not be overridden.
; COLD: Function Attrs: noprofile
; COLD-NEXT: define i32 @bar_noprofile() #{{[0-9]+}} {
; HOT: Function Attrs: noprofile
; HOT-NEXT: define i32 @bar_noprofile() #{{[0-9]+}} {
define i32 @bar_noprofile() noprofile {
entry:
  ret i32 0
}

; COLD: Function Attrs: skipprofile
; COLD-NEXT: define i32 @bar_skipprofile() #{{[0-9]+}} {
; HOT: Function Attrs: skipprofile
; HOT-NEXT: define i32 @bar_skipprofile() #{{[0-9]+}} {
define i32 @bar_skipprofile() skipprofile {
entry:
  ret i32 0
}

; COLD: Function Attrs: hot
; COLD-NEXT: define i32 @bar_hot() #{{[0-9]+}} {
define i32 @bar_hot() hot {
entry:
  ret i32 0
}

; HOT: Function Attrs: cold minsize
; HOT-NEXT: define i32 @bar_cold() #{{[0-9]+}} {
define i32 @bar_cold() cold {
entry:
  ret i32 0
}

; COLD: ![[ZERO_ENTRY]] = !{!"function_entry_count", i64 0}
