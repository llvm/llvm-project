; RUN: opt < %s -disable-output -instcombine -time-passes 2>&1 | FileCheck %s --check-prefix=TIME --check-prefix=TIME-LEGACY
;
; TIME: Pass execution timing report
; TIME: Total Execution Time:
; TIME: Name
; TIME-LEGACY-DAG:   Combine redundant instructions
; TIME-LEGACY-DAG:   Dominator Tree Construction
; TIME-LEGACY-DAG:   Module Verifier
; TIME-LEGACY-DAG:   Target Library Information
; TIME: 100{{.*}} Total{{$}}

define i32 @foo() {
  %res = add i32 5, 4
  ret i32 %res
}
