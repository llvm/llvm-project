; RUN: llc < %s -mtriple=thumbv7-linux-gnueabihf -O1 %s -o - | FileCheck %s

; CHECK-LABEL: test_jump_table:
; CHECK: b{{.*}} .LBB
; CHECK-NOT: tbh

define i32 @test_jump_table(i32 %x, float %in) {

h1:

 %b0 = fadd float %in, 1234.5
 %b1 = fptoui float %b0 to i32
  
  switch i32 %x, label %h2 [
    i32 0, label %h3
    i32 2, label %h4
    i32 4, label %h5
    i32 6, label %h6
  ]

h2:
  %a0 = add i32 %x, 5
  br label %h3

h3:
  %d2 = phi i32 [%b1, %h1], [%a0, %h2]
  %d3 = add i32 %d2, 3
  br label %h4

h4:
  %c2 = phi i32 [%b1, %h1], [%d3, %h3]
  %c3 = add i32 %c2, 5
  br label %h5

h5:
  %a2 = phi i32 [%b1, %h1], [%c3, %h4]
  %a3 = add i32 %a2, 6
  br label %h6

h6:
  %y = phi i32 [0, %h1], [%a3, %h5]
  call i32 @llvm.arm.space(i32 2000, i32 undef)
  ret i32 %y
  
}

declare i32 @llvm.arm.space(i32, i32)
