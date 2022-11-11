; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s
; This tests currently fails as MachineLICM does not compute register pressure
; correctly. More details: llvm.org/PR23143
; XFAIL: *

; MachineLICM should take register pressure into account.
; CHECK-NOT: Spill

%struct.A = type { i32, i32, i32, i32, i32, i32, i32 }

define void @test(i1 %b, ptr %a) nounwind {
entry:
  br label %loop-header

loop-header:
  br label %loop-body

loop-body:
  %0 = getelementptr inbounds %struct.A, ptr %a, i64 0, i32 1
  %1 = getelementptr inbounds %struct.A, ptr %a, i64 0, i32 2
  %2 = getelementptr inbounds %struct.A, ptr %a, i64 0, i32 3
  %3 = getelementptr inbounds %struct.A, ptr %a, i64 0, i32 4
  %4 = getelementptr inbounds %struct.A, ptr %a, i64 0, i32 5
  %5 = getelementptr inbounds %struct.A, ptr %a, i64 0, i32 6
  call void @assign(ptr %a)
  call void @assign(ptr %0)
  call void @assign(ptr %1)
  call void @assign(ptr %2)
  call void @assign(ptr %3)
  call void @assign(ptr %4)
  call void @assign(ptr %5)
  br i1 %b, label %loop-body, label %loop-exit

loop-exit:
  ret void
}

declare void @assign(ptr)
