; RUN: llc < %s -march=x86 | FileCheck %s
; RUN: llc < %s -march=x86-64 | FileCheck %s

; CHECK: test1
; CHECK-NOT: mov

declare void @bar()
define void @test1(i32 %i0, i32 %i1, i32 %i2, i32 %i3, i32 %i4, i32 %i5, void()* %arg) nounwind {
	call void @bar()
	call void %arg()
	ret void
}

; PR14739
; CHECK: test2
; CHECK: mov{{.*}} $0, ([[REGISTER:%[a-z]+]])
; CHECK-NOT: jmp{{.*}} *([[REGISTER]])

%struct.X = type { void ()* }
define void @test2(%struct.X* nocapture %x) {
entry:
  %f = getelementptr inbounds %struct.X, %struct.X* %x, i64 0, i32 0
  %0 = load void ()*, void ()** %f
  store void ()* null, void ()** %f
  tail call void %0()
  ret void
}
