; REQUIRES: asserts
; REQUIRES: x86-registered-target
; UNSUPPORTED: system-windows
;
; RUN: rm -rf %t.pre %t.sched
; RUN: mkdir -p %t.pre %t.sched
;
; RUN: llc -O0 \
; RUN:   -mtriple=x86_64-unknown-linux-gnu \
; RUN:   -fast-isel=false \
; RUN:   -view-isel-dags \
; RUN:   -dag-file-location=%t.pre \
; RUN:   -no-open-dag-viewer \
; RUN:   %s -o /dev/null
; RUN: find %t.pre -name '*.dot' -exec cat {} + | \
; RUN:   FileCheck %s --check-prefix=PRE \
; RUN:   --implicit-check-not='llvm_is_machine_opcode="true"'
;
; RUN: llc -O0 \
; RUN:   -mtriple=x86_64-unknown-linux-gnu \
; RUN:   -fast-isel=false \
; RUN:   -view-sched-dags \
; RUN:   -dag-file-location=%t.sched \
; RUN:   -no-open-dag-viewer \
; RUN:   %s -o /dev/null
; RUN: find %t.sched -name '*.dot' -exec cat {} + | \
; RUN:   FileCheck %s --check-prefix=SCHED
;
; PRE-DAG: llvm_opcode="{{[0-9]+}}",llvm_is_machine_opcode="false",label="{{.*}}add
; PRE-DAG: llvm_opcode="{{[0-9]+}}",llvm_is_machine_opcode="false",label="{{.*}}sub
; PRE-DAG: llvm_opcode="{{[0-9]+}}",llvm_is_machine_opcode="false",label="{{.*}}and
; PRE-DAG: llvm_opcode="{{[0-9]+}}",llvm_is_machine_opcode="false",label="{{.*}}shl
;
; SCHED-DAG: llvm_opcode="{{[0-9]+}}",llvm_is_machine_opcode="false",label="{{.*}}CopyFromReg
; SCHED-DAG: llvm_opcode="{{[0-9]+}}",llvm_is_machine_opcode="true",label="{{.*}}ADD32rr
; SCHED-DAG: llvm_opcode="{{[0-9]+}}",llvm_is_machine_opcode="true",label="{{.*}}SUB32rr
; SCHED-DAG: llvm_opcode="{{[0-9]+}}",llvm_is_machine_opcode="true",label="{{.*}}AND32rr
; SCHED-DAG: llvm_opcode="{{[0-9]+}}",llvm_is_machine_opcode="true",label="{{.*}}SHL32ri
;

define void @mixed_ops(ptr %out, i32 %a, i32 %b) {
entry:
  %added = add i32 %a, %b
  store volatile i32 %added, ptr %out

  %out1 = getelementptr i32, ptr %out, i64 1
  %subtracted = sub i32 %a, %b
  store volatile i32 %subtracted, ptr %out1

  %out2 = getelementptr i32, ptr %out, i64 2
  %masked = and i32 %a, %b
  store volatile i32 %masked, ptr %out2

  %out3 = getelementptr i32, ptr %out, i64 3
  %shifted = shl i32 %a, 2
  store volatile i32 %shifted, ptr %out3

  ret void
}