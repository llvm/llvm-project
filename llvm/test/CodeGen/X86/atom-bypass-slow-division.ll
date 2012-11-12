; RUN: llc < %s -mcpu=atom -mtriple=i686-linux  | FileCheck %s

define i32 @test_get_quotient(i32 %a, i32 %b) nounwind {
; CHECK: test_get_quotient
; CHECK: orl %ecx, %edx
; CHECK-NEXT: testl $-256, %edx
; CHECK-NEXT: je
; CHECK: idivl
; CHECK: ret
; CHECK: divb
; CHECK: ret
  %result = sdiv i32 %a, %b
  ret i32 %result
}

define i32 @test_get_remainder(i32 %a, i32 %b) nounwind {
; CHECK: test_get_remainder
; CHECK: orl %ecx, %edx
; CHECK-NEXT: testl $-256, %edx
; CHECK-NEXT: je
; CHECK: idivl
; CHECK: ret
; CHECK: divb
; CHECK: ret
  %result = srem i32 %a, %b
  ret i32 %result
}

define i32 @test_get_quotient_and_remainder(i32 %a, i32 %b) nounwind {
; CHECK: test_get_quotient_and_remainder
; CHECK: orl %ecx, %edx
; CHECK-NEXT: testl $-256, %edx
; CHECK-NEXT: je
; CHECK: idivl
; CHECK: divb
; CHECK: addl
; CHECK: ret
; CEECK-NOT: idivl
; CHECK-NOT: divb
  %resultdiv = sdiv i32 %a, %b
  %resultrem = srem i32 %a, %b
  %result = add i32 %resultdiv, %resultrem
  ret i32 %result
}

define i32 @test_use_div_and_idiv(i32 %a, i32 %b) nounwind {
; CHECK: test_use_div_and_idiv
; CHECK: idivl
; CHECK: divb
; CHECK: divl
; CHECK: divb
; CHECK: addl
; CHECK: ret
  %resultidiv = sdiv i32 %a, %b
  %resultdiv = udiv i32 %a, %b
  %result = add i32 %resultidiv, %resultdiv
  ret i32 %result
}

define i32 @test_use_div_imm_imm() nounwind {
; CHECK: test_use_div_imm_imm
; CHECK: movl $64
  %resultdiv = sdiv i32 256, 4
  ret i32 %resultdiv
}

define i32 @test_use_div_reg_imm(i32 %a) nounwind {
; CHECK: test_use_div_reg_imm
; CEHCK-NOT: test
; CHECK-NOT: idiv
; CHECK-NOT: divb
  %resultdiv = sdiv i32 %a, 33
  ret i32 %resultdiv
}

define i32 @test_use_rem_reg_imm(i32 %a) nounwind {
; CHECK: test_use_rem_reg_imm
; CEHCK-NOT: test
; CHECK-NOT: idiv
; CHECK-NOT: divb
  %resultrem = srem i32 %a, 33
  ret i32 %resultrem
}

define i32 @test_use_divrem_reg_imm(i32 %a) nounwind {
; CHECK: test_use_divrem_reg_imm
; CEHCK-NOT: test
; CHECK-NOT: idiv
; CHECK-NOT: divb
  %resultdiv = sdiv i32 %a, 33
  %resultrem = srem i32 %a, 33
  %result = add i32 %resultdiv, %resultrem
  ret i32 %result
}

define i32 @test_use_div_imm_reg(i32 %a) nounwind {
; CHECK: test_use_div_imm_reg
; CHECK: test
; CHECK: idiv
; CHECK: divb
  %resultdiv = sdiv i32 4, %a
  ret i32 %resultdiv
}

define i32 @test_use_rem_imm_reg(i32 %a) nounwind {
; CHECK: test_use_rem_imm_reg
; CHECK: test
; CHECK: idiv
; CHECK: divb
  %resultdiv = sdiv i32 4, %a
  ret i32 %resultdiv
}
