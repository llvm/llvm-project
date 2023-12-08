; Check the miscellaneous logical vector operations added in P8
; 
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 -mattr=-vsx < %s | FileCheck %s
; Test x eqv y
define <4 x i32> @test_veqv(<4 x i32> %x, <4 x i32> %y) nounwind {
       %tmp = xor <4 x i32> %x, %y
       %ret_val = xor <4 x i32> %tmp, < i32 -1, i32 -1, i32 -1, i32 -1>
       ret <4 x i32> %ret_val
; CHECK: veqv 2, 2, 3
}

; Test x vnand y
define <4 x i32> @test_vnand(<4 x i32> %x, <4 x i32> %y) nounwind {
       %tmp = and <4 x i32> %x, %y
       %ret_val = xor <4 x i32> %tmp, <i32 -1, i32 -1, i32 -1, i32 -1>
       ret <4 x i32> %ret_val
; CHECK: vnand 2, 2, 3
}

; Test x vorc y and variants
define <4 x i32> @test_vorc(<4 x i32> %x, <4 x i32> %y) nounwind {
       %tmp1 = xor <4 x i32> %y, <i32 -1, i32 -1, i32 -1, i32 -1>
       %tmp2 = or <4 x i32> %x, %tmp1
; CHECK: vorc 3, 2, 3      
       %tmp3 = xor <4 x i32> %tmp2, <i32 -1, i32 -1, i32 -1, i32 -1>
       %tmp4 = or <4 x i32> %tmp3, %x
; CHECK: vorc 2, 2, 3
       ret <4 x i32> %tmp4
}
