; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-linux -stop-after=early-machinelicm -o - | FileCheck %s -check-prefix=MIR

; This tests should fail as MachineLICM does not compute register pressure
; correctly. More details: llvm.org/PR23143

; It however does not show any spills because leaq is rematerialized instead
; of spilling.

; Stopping after MachineLICM however exposes all ADD64ri8 instructions
; to be hoisted which still has to be avoided.

; XFAIL: *

; MachineLICM should take register pressure into account.
; CHECK-LABEL: {{^}}test:
; CHECK-NOT:     Spill
; CHECK-COUNT-4: leaq
; CHECK-NOT:     Spill
; CHECK:         [[LOOP:\.LBB[0-9_]+]]:
; CHECK-NOT:     Reload
; CHECK-COUNT-2: leaq
; CHECK-NOT:     Reload
; CHECK:         jne [[LOOP]]

; MIR-LABEL: name: test
; MIR:         bb.0.entry:
; MIR-COUNT-4: ADD64ri8
; MIR:         bb.1.loop-body:
; MIR-COUNT-2: ADD64ri8
; MIR:         JCC_1 %bb.1

%struct.A = type { i32, i32, i32, i32, i32, i32, i32 }

define void @test(i1 %b, %struct.A* %a) nounwind {
entry:
  br label %loop-header

loop-header:
  br label %loop-body

loop-body:
  %0 = getelementptr inbounds %struct.A, %struct.A* %a, i64 0, i32 0
  %1 = getelementptr inbounds %struct.A, %struct.A* %a, i64 0, i32 1
  %2 = getelementptr inbounds %struct.A, %struct.A* %a, i64 0, i32 2
  %3 = getelementptr inbounds %struct.A, %struct.A* %a, i64 0, i32 3
  %4 = getelementptr inbounds %struct.A, %struct.A* %a, i64 0, i32 4
  %5 = getelementptr inbounds %struct.A, %struct.A* %a, i64 0, i32 5
  %6 = getelementptr inbounds %struct.A, %struct.A* %a, i64 0, i32 6
  call void @assign(i32* %0)
  call void @assign(i32* %1)
  call void @assign(i32* %2)
  call void @assign(i32* %3)
  call void @assign(i32* %4)
  call void @assign(i32* %5)
  call void @assign(i32* %6)
  br i1 %b, label %loop-body, label %loop-exit

loop-exit:
  ret void
}

declare void @assign(i32*)
