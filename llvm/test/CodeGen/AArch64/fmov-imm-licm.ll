; RUN: llc -mtriple=aarch64-linux-gnu -o - %s | FileCheck %s

; The purpose of this test is to check that an FMOV instruction that
; only materializes an immediate is not MachineLICM'd out of a loop.
; We check this in two ways: by looking for the FMOV inside the loop,
; and also by checking that we're not spilling any FP callee-saved
; registers.

%struct.Node = type { %struct.Node*, i8* }

define void @process_nodes(%struct.Node* %0) {
; CHECK-LABEL: process_nodes:
; CHECK-NOT:   stp {{d[0-9]+}}
; CHECK-LABEL: .LBB0_2:
; CHECK:       fmov s0, #1.00000000
; CHECK:       bl do_it
entry:
  %1 = icmp eq %struct.Node* %0, null
  br i1 %1, label %exit, label %loop

loop:
  %2 = phi %struct.Node* [ %4, %loop ], [ %0, %entry ]
  tail call void @do_it(float 1.000000e+00, %struct.Node* nonnull %2)
  %3 = getelementptr inbounds %struct.Node, %struct.Node* %2, i64 0, i32 0
  %4 = load %struct.Node*, %struct.Node** %3, align 8
  %5 = icmp eq %struct.Node* %4, null
  br i1 %5, label %exit, label %loop

exit:
  ret void
}

declare void @do_it(float, %struct.Node*)
