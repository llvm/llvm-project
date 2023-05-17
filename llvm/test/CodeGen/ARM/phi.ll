; RUN: llc -mtriple=arm-eabi -mattr=+v4t %s -o - | FileCheck %s

; <rdar://problem/8686347>

define i32 @test1(i1 %a, ptr %b) {
; CHECK: test1
entry:
  br i1 %a, label %lblock, label %rblock

lblock:
  %lbranch = getelementptr i32, ptr %b, i32 1
  br label %end

rblock:
  %rbranch = getelementptr i32, ptr %b, i32 1
  br label %end
  
end:
; CHECK: ldr	r0, [r1, #4]
  %gep = phi ptr [%lbranch, %lblock], [%rbranch, %rblock]
  %r = load i32, ptr %gep
; CHECK-NEXT: bx	lr
  ret i32 %r
}
