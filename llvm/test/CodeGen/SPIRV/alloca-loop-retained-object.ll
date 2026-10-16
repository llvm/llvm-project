; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,TYPED --implicit-check-not=VariableLengthArrayINTEL
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,TYPED --implicit-check-not=VariableLengthArrayINTEL
; RUN: llc -O2 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,TYPED --implicit-check-not=VariableLengthArrayINTEL
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown -spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s --check-prefixes=CHECK,UNTYPED --implicit-check-not=VariableLengthArrayINTEL

; The previous iteration's object remains live when the next one is allocated.
; This function returns 0 in LLVM IR; reusing a single object would return 1.
; Preserve the allocation after the loop's phis. Non-entry Function variables
; are not permitted by core SPIR-V, so this case cannot pass spirv-val.

; CHECK-DAG: %[[#INT:]] = OpTypeInt 32 0
; TYPED-DAG: %[[#PTR:]] = OpTypePointer Function %[[#INT]]
; UNTYPED-DAG: %[[#PTR:]] = OpTypeUntypedPointerKHR Function
; CHECK-LABEL: Begin function retained_object
; CHECK: %[[#ENTRY:]] = OpLabel
; TYPED-NEXT: %[[#INITIAL:]] = OpVariable %[[#PTR]] Function
; UNTYPED-NEXT: %[[#INITIAL:]] = OpUntypedVariableKHR %[[#PTR]] Function %[[#INT]]
; CHECK-NOT: {{OpVariable|OpUntypedVariableKHR}}
; CHECK: OpStore %[[#INITIAL]]
; CHECK-NOT: {{OpVariable|OpUntypedVariableKHR}}
; CHECK: OpBranch %[[#LOOP:]]
; CHECK: %[[#LOOP]] = OpLabel
; CHECK: %[[#PREVIOUS:]] = OpPhi %[[#PTR]] %[[#INITIAL]] %[[#ENTRY]] %[[#CURRENT:]] %[[#LOOP]]
; TYPED: %[[#CURRENT]] = OpVariable %[[#PTR]] Function
; UNTYPED: %[[#CURRENT]] = OpUntypedVariableKHR %[[#PTR]] Function %[[#INT]]
; CHECK: OpStore %[[#CURRENT]]
; CHECK: %[[#OLD:]] = OpLoad %[[#INT]] %[[#PREVIOUS]]
; CHECK: OpBranchConditional
; CHECK: OpReturnValue %[[#OLD]]
define spir_func i32 @retained_object() {
entry:
  %initial = alloca i32, align 4
  store volatile i32 99, ptr %initial, align 4
  br label %loop

loop:
  %previous = phi ptr [ %initial, %entry ], [ %current, %loop ]
  %index = phi i32 [ 0, %entry ], [ %next, %loop ]
  %current = alloca i32, align 4
  store volatile i32 %index, ptr %current, align 4
  %old = load volatile i32, ptr %previous, align 4
  %next = add i32 %index, 1
  %repeat = icmp ult i32 %next, 2
  br i1 %repeat, label %loop, label %exit

exit:
  ret i32 %old
}
