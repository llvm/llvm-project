; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -no-integrated-as
;
; Check that TwoAddressInstructionPass does not crash after sinking (and
; revisiting) an instruction that was lowered by TII->convertToThreeAddress()
; which contains a %noreg operand.

define i32 @f23(i32 %old) {
  %and1 = and i32 %old, 14
  %and2 = and i32 %old, 254
  %res1 = call i32 asm "stepa $1, $2, $3", "=h,r,r,0"(i32 %old, i32 %and1, i32 %and2)
  %and3 = and i32 %res1, 127
  %and4 = and i32 %res1, 128
  %res2 = call i32 asm "stepb $1, $2, $3", "=r,h,h,0"(i32 %res1, i32 %and3, i32 %and4)
  ret i32 %res2
}
