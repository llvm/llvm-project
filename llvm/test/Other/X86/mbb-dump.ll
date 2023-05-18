; RUN: llc < %s 2>&1 -print-after=machine-scheduler -mtriple=x86_64-unknown-unknown | FileCheck %s

; expected MBB dump output
; # *** IR Dump After Machine Instruction Scheduler ***:
; # Machine code for function foo: NoPHIs, TracksLiveness
; 
; 0B	bb.0 (%ir-block.0):
; 	  successors: %bb.1(0x80000000); %bb.1(100.00%)
; 
; 16B	bb.1.next:
; 	; predecessors: %bb.0

; previously, it was broken as
; 	  successors: %bb.1(0x80000000); %bb.1(200.00%)

define void @foo(){
; CHECK: IR Dump After Machine Instruction Scheduler
; CHECK: bb.0
; CHECK: 100.0
; CHECK: bb.1
  br label %next

next:
  ret void
}
