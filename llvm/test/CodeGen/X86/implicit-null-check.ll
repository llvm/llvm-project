; RUN: llc -verify-machineinstrs -O3 -mtriple=x86_64-apple-macosx -enable-implicit-null-checks < %s | FileCheck %s

; RUN: llc < %s -mtriple=x86_64-apple-macosx -enable-implicit-null-checks \
; RUN:    | llvm-mc -triple x86_64-apple-macosx -filetype=obj -o - \
; RUN:    | llvm-objdump -triple x86_64-apple-macosx -fault-map-section - \
; RUN:    | FileCheck %s -check-prefix OBJDUMP

; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -enable-implicit-null-checks \
; RUN:    | llvm-mc -triple x86_64-unknown-linux-gnu -filetype=obj -o - \
; RUN:    | llvm-objdump -triple x86_64-unknown-linux-gnu -fault-map-section - \
; RUN:    | FileCheck %s -check-prefix OBJDUMP

define i32 @imp_null_check_load(i32* %x) {
; CHECK-LABEL: _imp_null_check_load:
; CHECK: [[BB0_imp_null_check_load:L[^:]+]]:
; CHECK: movl (%rdi), %eax
; CHECK: retq
; CHECK: [[BB1_imp_null_check_load:LBB0_[0-9]+]]:
; CHECK: movl $42, %eax
; CHECK: retq

 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t = load i32, i32* %x
  ret i32 %t
}

define i32 @imp_null_check_gep_load(i32* %x) {
; CHECK-LABEL: _imp_null_check_gep_load:
; CHECK: [[BB0_imp_null_check_gep_load:L[^:]+]]:
; CHECK: movl 128(%rdi), %eax
; CHECK: retq
; CHECK: [[BB1_imp_null_check_gep_load:LBB1_[0-9]+]]:
; CHECK: movl $42, %eax
; CHECK: retq

 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %x.gep = getelementptr i32, i32* %x, i32 32
  %t = load i32, i32* %x.gep
  ret i32 %t
}

define i32 @imp_null_check_add_result(i32* %x, i32 %p) {
; CHECK-LABEL: _imp_null_check_add_result:
; CHECK: [[BB0_imp_null_check_add_result:L[^:]+]]:
; CHECK: addl (%rdi), %esi
; CHECK: movl %esi, %eax
; CHECK: retq
; CHECK: [[BB1_imp_null_check_add_result:LBB2_[0-9]+]]:
; CHECK: movl $42, %eax
; CHECK: retq

 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t = load i32, i32* %x
  %p1 = add i32 %t, %p
  ret i32 %p1
}

define i32 @imp_null_check_hoist_over_unrelated_load(i32* %x, i32* %y, i32* %z) {
; CHECK-LABEL: _imp_null_check_hoist_over_unrelated_load:
; CHECK: [[BB0_imp_null_check_hoist_over_unrelated_load:L[^:]+]]:
; CHECK: movl (%rdi), %eax
; CHECK: movl (%rsi), %ecx
; CHECK: movl %ecx, (%rdx)
; CHECK: retq
; CHECK: [[BB1_imp_null_check_hoist_over_unrelated_load:LBB3_[0-9]+]]:
; CHECK: movl	$42, %eax
; CHECK: retq

 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t0 = load i32, i32* %y
  %t1 = load i32, i32* %x
  store i32 %t0, i32* %z
  ret i32 %t1
}

define i32 @imp_null_check_via_mem_comparision(i32* %x, i32 %val) {
; CHECK-LABEL: _imp_null_check_via_mem_comparision
; CHECK: [[BB0_imp_null_check_via_mem_comparision:L[^:]+]]:
; CHECK: cmpl   %esi, 4(%rdi)
; CHECK: jge    LBB4_2
; CHECK: movl   $100, %eax
; CHECK: retq
; CHECK: [[BB1_imp_null_check_via_mem_comparision:LBB4_[0-9]+]]:
; CHECK: movl   $42, %eax
; CHECK: retq
; CHECK: LBB4_2:
; CHECK: movl   $200, %eax
; CHECK: retq

 entry:
  %c = icmp eq i32* %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %x.loc = getelementptr i32, i32* %x, i32 1
  %t = load i32, i32* %x.loc
  %m = icmp slt i32 %t, %val
  br i1 %m, label %ret_100, label %ret_200

 ret_100:
  ret i32 100

 ret_200:
  ret i32 200
}

!0 = !{}

; CHECK-LABEL: __LLVM_FaultMaps:

; Version:
; CHECK-NEXT: .byte 1

; Reserved x2
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .short 0

; # functions:
; CHECK-NEXT: .long 5

; FunctionAddr:
; CHECK-NEXT: .quad _imp_null_check_add_result
; NumFaultingPCs
; CHECK-NEXT: .long 1
; Reserved:
; CHECK-NEXT: .long 0
; Fault[0].Type:
; CHECK-NEXT: .long 1
; Fault[0].FaultOffset:
; CHECK-NEXT: .long [[BB0_imp_null_check_add_result]]-_imp_null_check_add_result
; Fault[0].HandlerOffset:
; CHECK-NEXT: .long [[BB1_imp_null_check_add_result]]-_imp_null_check_add_result

; FunctionAddr:
; CHECK-NEXT: .quad _imp_null_check_gep_load
; NumFaultingPCs
; CHECK-NEXT: .long 1
; Reserved:
; CHECK-NEXT: .long 0
; Fault[0].Type:
; CHECK-NEXT: .long 1
; Fault[0].FaultOffset:
; CHECK-NEXT: .long [[BB0_imp_null_check_gep_load]]-_imp_null_check_gep_load
; Fault[0].HandlerOffset:
; CHECK-NEXT: .long [[BB1_imp_null_check_gep_load]]-_imp_null_check_gep_load

; FunctionAddr:
; CHECK-NEXT: .quad _imp_null_check_hoist_over_unrelated_load
; NumFaultingPCs
; CHECK-NEXT: .long 1
; Reserved:
; CHECK-NEXT: .long 0
; Fault[0].Type:
; CHECK-NEXT: .long 1
; Fault[0].FaultOffset:
; CHECK-NEXT: .long [[BB0_imp_null_check_hoist_over_unrelated_load]]-_imp_null_check_hoist_over_unrelated_load
; Fault[0].HandlerOffset:
; CHECK-NEXT: .long [[BB1_imp_null_check_hoist_over_unrelated_load]]-_imp_null_check_hoist_over_unrelated_load

; FunctionAddr:
; CHECK-NEXT: .quad _imp_null_check_load
; NumFaultingPCs
; CHECK-NEXT: .long 1
; Reserved:
; CHECK-NEXT: .long 0
; Fault[0].Type:
; CHECK-NEXT: .long 1
; Fault[0].FaultOffset:
; CHECK-NEXT: .long [[BB0_imp_null_check_load]]-_imp_null_check_load
; Fault[0].HandlerOffset:
; CHECK-NEXT: .long [[BB1_imp_null_check_load]]-_imp_null_check_load

; FunctionAddr:
; CHECK-NEXT: .quad     _imp_null_check_via_mem_comparision
; NumFaultingPCs
; CHECK-NEXT: .long   1
; Reserved:
; CHECK-NEXT: .long   0
; Fault[0].Type:
; CHECK-NEXT: .long   1
; Fault[0].FaultOffset:
; CHECK-NEXT: .long   [[BB0_imp_null_check_via_mem_comparision]]-_imp_null_check_via_mem_comparision
; Fault[0].HandlerOffset:
; CHECK-NEXT: .long   [[BB1_imp_null_check_via_mem_comparision]]-_imp_null_check_via_mem_comparision

; OBJDUMP: FaultMap table:
; OBJDUMP-NEXT: Version: 0x1
; OBJDUMP-NEXT: NumFunctions: 5
; OBJDUMP-NEXT: FunctionAddress: 0x000000, NumFaultingPCs: 1
; OBJDUMP-NEXT: Fault kind: FaultingLoad, faulting PC offset: 0, handling PC offset: 5
; OBJDUMP-NEXT: FunctionAddress: 0x000000, NumFaultingPCs: 1
; OBJDUMP-NEXT: Fault kind: FaultingLoad, faulting PC offset: 0, handling PC offset: 7
; OBJDUMP-NEXT: FunctionAddress: 0x000000, NumFaultingPCs: 1
; OBJDUMP-NEXT: Fault kind: FaultingLoad, faulting PC offset: 0, handling PC offset: 7
; OBJDUMP-NEXT: FunctionAddress: 0x000000, NumFaultingPCs: 1
; OBJDUMP-NEXT: Fault kind: FaultingLoad, faulting PC offset: 0, handling PC offset: 3
