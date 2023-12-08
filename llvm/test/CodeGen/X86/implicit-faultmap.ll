; RUN: llc -verify-machineinstrs -O3 -mtriple=x86_64-apple-macosx -enable-implicit-null-checks < %s | FileCheck %s

; RUN: llc < %s -mtriple=x86_64-apple-macosx -enable-implicit-null-checks \
; RUN:    | llvm-mc -triple x86_64-apple-macosx -filetype=obj -o - \
; RUN:    | llvm-objdump --triple=x86_64-apple-macosx --fault-map-section - \
; RUN:    | FileCheck %s -check-prefix OBJDUMP

; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -enable-implicit-null-checks \
; RUN:    | llvm-mc -triple x86_64-unknown-linux-gnu -filetype=obj -o - \
; RUN:    | llvm-objdump --triple=x86_64-unknown-linux-gnu --fault-map-section - \
; RUN:    | FileCheck %s -check-prefix OBJDUMP

;; The tests in this file exist just to check basic validity of the FaultMap
;; section.  Please don't add to this file unless you're testing the FaultMap
;; serialization code itself.

define i32 @imp_null_check_load(ptr %x) {
; CHECK-LABEL: _imp_null_check_load:
; CHECK: [[BB0_imp_null_check_load:L[^:]+]]:
; CHECK: movl (%rdi), %eax
; CHECK: retq
; CHECK: [[BB1_imp_null_check_load:LBB0_[0-9]+]]:
; CHECK: movl $42, %eax
; CHECK: retq

 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret i32 42

 not_null:
  %t = load i32, ptr %x
  ret i32 %t
}

define void @imp_null_check_store(ptr %x) {
; CHECK-LABEL: _imp_null_check_store:
; CHECK: [[BB0_imp_null_check_store:L[^:]+]]:
; CHECK: movl $1, (%rdi)
; CHECK: retq
; CHECK: [[BB1_imp_null_check_store:LBB1_[0-9]+]]:
; CHECK: retq

 entry:
  %c = icmp eq ptr %x, null
  br i1 %c, label %is_null, label %not_null, !make.implicit !0

 is_null:
  ret void

 not_null:
  store i32 1, ptr %x
  ret void
}

!0 = !{}

; CHECK-LABEL: __LLVM_FaultMaps:

; Version:
; CHECK-NEXT: .byte 1

; Reserved x2
; CHECK-NEXT: .byte 0
; CHECK-NEXT: .short 0

; # functions:
; CHECK-NEXT: .long 2

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
; CHECK-NEXT: .quad _imp_null_check_store
; NumFaultingPCs
; CHECK-NEXT: .long 1
; Reserved:
; CHECK-NEXT: .long 0
; Fault[0].Type:
; CHECK-NEXT: .long 3
; Fault[0].FaultOffset:
; CHECK-NEXT: .long [[BB0_imp_null_check_store]]-_imp_null_check_store
; Fault[0].HandlerOffset:
; CHECK-NEXT: .long [[BB1_imp_null_check_store]]-_imp_null_check_store

; OBJDUMP: FaultMap table:
; OBJDUMP-NEXT: Version: 0x1
; OBJDUMP-NEXT: NumFunctions: 2
; OBJDUMP-NEXT: FunctionAddress: 0x000000, NumFaultingPCs: 1
; OBJDUMP-NEXT: Fault kind: FaultingLoad, faulting PC offset: 0, handling PC offset: 3
; OBJDUMP-NEXT: FunctionAddress: 0x000000, NumFaultingPCs: 1
; OBJDUMP-NEXT: Fault kind: FaultingStore, faulting PC offset: 0, handling PC offset: 7
