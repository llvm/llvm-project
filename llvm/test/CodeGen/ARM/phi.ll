; RUN: llc -march=arm -mattr=+v4t < %s | FileCheck %s
; <rdar://problem/8686347>

define i32 @test1(i1 %a, i32* %b) {
; CHECK: test1
entry:
  br i1 %a, label %lblock, label %rblock

lblock:
  %lbranch = getelementptr i32* %b, i32 1
  br label %end

rblock:
  %rbranch = getelementptr i32* %b, i32 1
  br label %end
  
end:
; CHECK: ldr	r0, [r1, #4]
  %gep = phi i32* [%lbranch, %lblock], [%rbranch, %rblock]
  %r = load i32* %gep
; CHECK-NEXT: bx	lr
  ret i32 %r
}
