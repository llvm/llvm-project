; RUN: not --crash opt -O2 -opt-pipeline-trigger-crash %s -disable-output 2>&1 | \
; RUN: FileCheck %s --check-prefix=OPT
;
; RUN: not --crash llc -codegen-pipeline-trigger-crash %s -o /dev/null 2>&1 | \
; RUN: FileCheck %s --check-prefix=CODEGEN

; OPT: trigger-crash-function

; CODEGEN: TriggerCrashFunctionPass

define void @foo() {
  ret void
}
