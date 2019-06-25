; RUN: not llc -mtriple=armv8.1-m-eabi -mattr=+mve %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: inline assembly requires more registers than available
define arm_aapcs_vfpcc <4 x i32> @t-constraint-i32-vectors-too-few-regs(<4 x i32> %a, <4 x i32> %b) {
entry:
	%0 = tail call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>,
                         <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> }
                       asm "",
             "=t,=t,=t,=t,=t,=t,=t,=t,=t,=t,t,t"(<4 x i32> %a, <4 x i32> %b)
	%asmresult = extractvalue { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>,
                                    <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32>,
                                    <4 x i32>, <4 x i32> } %0, 0
	ret <4 x i32> %asmresult
}
