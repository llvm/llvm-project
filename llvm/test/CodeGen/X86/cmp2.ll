; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep ucomisd | grep CPI | count 2

define i32 @test(double %A) nounwind  {
 entry:
 %tmp2 = fcmp ogt double %A, 1.500000e+02; <i1> [#uses=1]
 %tmp5 = fcmp ult double %A, 7.500000e+01; <i1> [#uses=1]
 %bothcond = or i1 %tmp2, %tmp5; <i1> [#uses=1]
 br i1 %bothcond, label %bb8, label %bb12

 bb8:; preds = %entry
 %tmp9 = tail call i32 (...)* @foo( ) nounwind ; <i32> [#uses=1]
 ret i32 %tmp9

 bb12:; preds = %entry
 ret i32 32
}

declare i32 @foo(...)
