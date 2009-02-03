; RUN: llvm-as < %s | opt -instcombine | llvm-dis > %t
; RUN: grep {icmp eq} %t
; RUN: grep {ret i1 false} %t | count 2
; RUN: grep {ret i1 true} %t | count 2
; RUN: grep {icmp ne} %t
; RUN: not grep {icmp slt} %t
; PR3021

; When inst combining an FCMP with the LHS coming from a uitofp instruction, we
; can't lower it to signed ICMP instructions.

define i1 @test1(i32 %val) {
  %1 = uitofp i32 %val to double
  %2 = fcmp ole double %1, 0.000000e+00
  ret i1 %2
}

define i1 @test2(i32 %val) {
  %1 = uitofp i32 %val to double
  %2 = fcmp olt double %1, 0.000000e+00
  ret i1 %2
}

define i1 @test3(i32 %val) {
  %1 = uitofp i32 %val to double
  %2 = fcmp oge double %1, 0.000000e+00
  ret i1 %2
}

define i1 @test4(i32 %val) {
  %1 = uitofp i32 %val to double
  %2 = fcmp ogt double %1, 0.000000e+00
  ret i1 %2
}

define i1 @test5(i32 %val) {
  %1 = uitofp i32 %val to double
  %2 = fcmp ogt double %1, -4.400000e+00
  ret i1 %2
}

define i1 @test6(i32 %val) {
  %1 = uitofp i32 %val to double
  %2 = fcmp olt double %1, -4.400000e+00
  ret i1 %2
}
