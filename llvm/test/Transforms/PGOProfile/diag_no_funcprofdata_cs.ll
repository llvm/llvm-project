; RUN: llvm-profdata merge %S/Inputs/cs_vs_nocs.proftext -o %t.profdata
; RUN: opt < %s -passes=pgo-instr-use -pgo-warn-missing-function=true -pgo-test-profile-file=%t.profdata -S 2>&1 | FileCheck %s
; RUN: opt < %s -passes='default<O2>' -pgo-kind=pgo-instr-use-pipeline -cspgo-kind=cspgo-instr-use-pipeline -pgo-use-missing-func=cold -profile-file=%t.profdata -S | FileCheck %s --check-prefixes=CSPGO,CSPGO-COLD
; RUN: opt < %s -passes='default<O2>' -pgo-kind=pgo-instr-use-pipeline -cspgo-kind=cspgo-instr-use-pipeline -pgo-use-missing-func=hot -profile-file=%t.profdata -S | FileCheck %s --check-prefixes=CSPGO,CSPGO-HOT

; CHECK: warning: {{.+}}: no profile data available for function bar
; CHECK-NOT: warning: {{.+}}: function control flow change detected (hash mismatch)

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; @bar is present in the CSIR profile (entry_count = 100000), so it is matched
; during CS-PGO use and not treated as missing.
define i32 @bar() {
entry:
  ret i32 0
}

; Unprofiled function missing from both base and CS profiles is annotated
; according to -pgo-use-missing-func.
; CSPGO-COLD: Function Attrs: {{.*}}cold
; CSPGO-COLD-NEXT: define {{.*}}i32 @missing()
; CSPGO-HOT: Function Attrs: {{.*}}hot
; CSPGO-HOT-NEXT: define {{.*}}i32 @missing()
define i32 @missing() {
entry:
  ret i32 2
}

; Existing entry count from base PGO must not be overwritten or marked hot/cold
; when missing from the context-sensitive profile.
; CSPGO-NOT: Function Attrs:{{.*}}(cold|hot)
; CSPGO: define {{.*}}i32 @already_profiled() {{.*}}!prof ![[EXISTING_COUNT:[0-9]+]]
define i32 @already_profiled() !prof !0 {
entry:
  ret i32 1
}

!0 = !{!"function_entry_count", i64 50}
; CSPGO: ![[EXISTING_COUNT]] = !{!"function_entry_count", i64 50}
