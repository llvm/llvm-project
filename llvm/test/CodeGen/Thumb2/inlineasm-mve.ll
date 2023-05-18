; RUN: llc -mtriple=armv8.1-m-eabi -mattr=+mve %s -o - | FileCheck %s

define i32 @test1(i32 %tmp54) {
	%tmp56 = tail call i32 asm "uxtb16 $0,$1", "=r,r"( i32 %tmp54 )
	ret i32 %tmp56
}

define void @test2() {
	tail call void asm sideeffect "/* number: ${0:c} */", "i"( i32 1 )
	ret void
}

define arm_aapcs_vfpcc <4 x i32> @mve-t-constraint-128bit(<4 x i32>, <4 x i32>) {
; CHECK-LABEL: mve-t-constraint-128bit
; CHECK: vadd.i32 q{{[0-7]}}, q{{[0-7]}}, q{{[0-7]}}
  %ret = tail call <4 x i32>
         asm "vadd.i32 $0, $1, $2", "=t,t,t"
         (<4 x i32> %0, <4 x i32> %1)
  ret <4 x i32> %ret
}

define i32 @even-GPR-constraint() {
entry:
	; CHECK-LABEL: even-GPR-constraint
	; CHECK: add [[REG:r1*[0, 2, 4, 6, 8]]], [[REG]], #1
	; CHECK: add [[REG:r1*[0, 2, 4, 6, 8]]], [[REG]], #2
	; CHECK: add [[REG:r1*[0, 2, 4, 6, 8]]], [[REG]], #3
	; CHECK: add [[REG:r1*[0, 2, 4, 6, 8]]], [[REG]], #4
	%0 = tail call { i32, i32, i32, i32 }
             asm "add $0, #1\0Aadd $1, #2\0Aadd $2, #3\0Aadd $3, #4\0A", "=^Te,=^Te,=^Te,=^Te,0,1,2,3"
             (i32 0, i32 0, i32 0, i32 0)
	%asmresult = extractvalue { i32, i32, i32, i32 } %0, 0
	ret i32 %asmresult
}

define i32 @odd-GPR-constraint() {
entry:
	; CHECK-LABEL: odd-GPR-constraint
	; CHECK: add [[REG:r1*[1, 3, 5, 7, 9]]], [[REG]], #1
	; CHECK: add [[REG:r1*[1, 3, 5, 7, 9]]], [[REG]], #2
	; CHECK: add [[REG:r1*[1, 3, 5, 7, 9]]], [[REG]], #3
	; CHECK: add [[REG:r1*[1, 3, 5, 7, 9]]], [[REG]], #4
	%0 = tail call { i32, i32, i32, i32 }
             asm "add $0, #1\0Aadd $1, #2\0Aadd $2, #3\0Aadd $3, #4\0A", "=^To,=^To,=^To,=^To,0,1,2,3"
             (i32 0, i32 0, i32 0, i32 0)
	%asmresult = extractvalue { i32, i32, i32, i32 } %0, 0
	ret i32 %asmresult
}
