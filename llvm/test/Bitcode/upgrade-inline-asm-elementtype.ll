; RUN: llvm-dis < %s.bc | FileCheck %s

; CHECK: call void asm "", "=*rm,r"(i32* elementtype(i32) %p1, i32* %p2)
define void @test_call(i32* %p1, i32* %p2) {
	call void asm "", "=*rm,r"(i32* %p1, i32* %p2)
  ret void
}

; CHECK: invoke void asm "", "=*rm,r"(i32* elementtype(i32) %p1, i32* %p2)
define void @test_invoke(i32* %p1, i32* %p2) personality i8* null {
	invoke void asm "", "=*rm,r"(i32* %p1, i32* %p2)
      to label %cont unwind label %lpad

lpad:
  %lp = landingpad i32
      cleanup
  ret void

cont:
  ret void
}

; CHECK: callbr void asm "", "=*rm,r"(i32* elementtype(i32) %p1, i32* %p2)
define void @test_callbr(i32* %p1, i32* %p2) {
	callbr void asm "", "=*rm,r"(i32* %p1, i32* %p2)
      to label %cont []

cont:
  ret void
}
