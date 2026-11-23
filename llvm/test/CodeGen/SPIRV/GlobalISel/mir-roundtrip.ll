; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -verify-machineinstrs -stop-after=spirv-postlegalizer %s -o %t.mir
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -run-pass=none %t.mir -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -verify-machineinstrs %s -o %t.ref.s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown -verify-machineinstrs -start-after=spirv-postlegalizer %t.mir -o %t.s
; RUN: diff %t.ref.s %t.s

; The kernel needs the environment on MIR input to emit ContractionOff.
; The i1 add checks that finalizeLowering runs on MIR input.

; CHECK-LABEL: name: foo
; CHECK: machineFunctionInfo:
; CHECK-NEXT: vregTypes:
; CHECK-NEXT: - { reg: '[[P:%[0-9]+]]', type: '[[PTR:%[0-9]+]]' }
; CHECK-NEXT: - { reg: '[[A:%[0-9]+]]', type: '[[I32:%[0-9]+]]' }
; CHECK-NEXT: - { reg: '[[B:%[0-9]+]]', type: '[[I32]]' }
; CHECK-NEXT: - { reg: '[[X:%[0-9]+]]', type: '[[BOOL:%[0-9]+]]' }
; CHECK-NEXT: - { reg: '[[Y:%[0-9]+]]', type: '[[BOOL]]' }
; CHECK-NEXT: - { reg: '[[SUM:%[0-9]+]]', type: '[[I32]]' }
; CHECK-NEXT: - { reg: '[[T:%[0-9]+]]', type: '[[BOOL]]' }
; CHECK-NEXT: - { reg: '[[V:%[0-9]+]]', type: '[[I32]]' }
; CHECK-NEXT: - { reg: '[[ADD:%[0-9]+]]', type: '[[I32]]' }
; CHECK-NEXT: - { reg: '[[TADD:%[0-9]+]]', type: '[[BOOL]]' }
; CHECK-NEXT: - { reg: '[[SEL:%[0-9]+]]', type: '[[I32]]' }
; CHECK-NEXT: body:
; CHECK-DAG: [[PTR]]:type(s64) = OpTypePointer 5, [[I32]](s64)
; CHECK-DAG: [[BOOL]]:type(s64) = OpTypeBool
; CHECK-DAG: [[I32]]:type(s64) = OpTypeInt 32, 0
; CHECK-DAG: [[P]]:pid(p1) = OpFunctionParameter [[PTR]](s64)
; CHECK-DAG: [[A]]:iid(i32) = OpFunctionParameter [[I32]](s64)
; CHECK-DAG: [[B]]:iid(i32) = OpFunctionParameter [[I32]](s64)
; CHECK-DAG: [[X]]:iid(i1) = OpFunctionParameter [[BOOL]](s64)
; CHECK-DAG: [[Y]]:iid(i1) = OpFunctionParameter [[BOOL]](s64)
; CHECK-DAG: [[ADD]]:iid(i32) = G_ADD [[A]], [[B]]
; CHECK-DAG: [[SUM]]:iid(i32) = ASSIGN_TYPE [[ADD]](i32), [[I32]](s64)
; CHECK-DAG: [[TADD]]:iid(i1) = G_ADD [[X]], [[Y]]
; CHECK-DAG: [[T]]:iid(i1) = ASSIGN_TYPE [[TADD]](i1), [[BOOL]](s64)
; CHECK-DAG: [[SEL]]:iid(i32) = G_SELECT [[T]](i1), [[SUM]], [[A]]
; CHECK-DAG: [[V]]:iid(i32) = ASSIGN_TYPE [[SEL]](i32), [[I32]](s64)

define spir_kernel void @foo(ptr addrspace(1) %p, i32 %a, i32 %b, i1 %x, i1 %y) {
  %s = add i32 %a, %b
  %t = add i1 %x, %y
  %v = select i1 %t, i32 %s, i32 %a
  store i32 %v, ptr addrspace(1) %p
  ret void
}
