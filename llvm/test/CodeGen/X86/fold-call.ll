; RUN: llc < %s -mtriple=i686-- | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-- | FileCheck %s

; CHECK: test1
; CHECK-NOT: mov

declare void @bar()
define void @test1(i32 %i0, i32 %i1, i32 %i2, i32 %i3, i32 %i4, i32 %i5, ptr %arg) nounwind {
	call void @bar()
	call void %arg()
	ret void
}

; PR14739
; CHECK: test2
; CHECK: mov{{.*}} $0, ([[REGISTER:%[a-z]+]])
; CHECK-NOT: jmp{{.*}} *([[REGISTER]])

%struct.X = type { ptr }
define void @test2(ptr nocapture %x) {
entry:
  %0 = load ptr, ptr %x
  store ptr null, ptr %x
  tail call void %0()
  ret void
}

; Don't fold the load+call if there's inline asm in between.
; CHECK: test3
; CHECK: mov{{.*}}
; CHECK: jmp{{.*}}
define void @test3(ptr nocapture %x) {
entry:
  %0 = load ptr, ptr %x
  call void asm sideeffect "", ""()  ; It could do anything.
  tail call void %0()
  ret void
}
