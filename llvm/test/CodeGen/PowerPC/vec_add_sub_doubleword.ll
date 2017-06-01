; Check VMX 64-bit integer operations
;
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 -mattr=-vsx < %s | FileCheck %s

define <2 x i64> @test_add(<2 x i64> %x, <2 x i64> %y) nounwind {
       %result = add <2 x i64> %x, %y
       ret <2 x i64> %result
; CHECK: vaddudm 2, 2, 3
}

define <2 x i64> @increment_by_one(<2 x i64> %x) nounwind {
       %result = add <2 x i64> %x, <i64 1, i64 1>
       ret <2 x i64> %result
; CHECK: vaddudm 2, 2, 3
}

define <2 x i64> @increment_by_val(<2 x i64> %x, i64 %val) nounwind {
       %tmpvec = insertelement <2 x i64> <i64 0, i64 0>, i64 %val, i32 0
       %tmpvec2 = insertelement <2 x i64> %tmpvec, i64 %val, i32 1
       %result = add <2 x i64> %x, %tmpvec2
       ret <2 x i64> %result
; CHECK: vaddudm 2, 2, 3
; FIXME: This is currently generating the following instruction sequence
;
;        std 5, -8(1)
;        std 5, -16(1)
;        addi 3, 1, -16
;        ori 2, 2, 0
;        lxvd2x 35, 0, 3
;        vaddudm 2, 2, 3
;        blr
;        
;        This will almost certainly cause a load-hit-store hazard.
;        Since val is a value parameter, it should not need to be
;        saved onto the stack at all (unless we're using this to set
;        up the vector register). Instead, it would be better to splat
;        the value into a vector register.
}

define <2 x i64> @test_sub(<2 x i64> %x, <2 x i64> %y) nounwind {
       %result = sub <2 x i64> %x, %y
       ret <2 x i64> %result
; CHECK: vsubudm 2, 2, 3
}

define <2 x i64> @decrement_by_one(<2 x i64> %x) nounwind {
       %result = sub <2 x i64> %x, <i64 -1, i64 -1>
       ret <2 x i64> %result
; CHECK: vsubudm 2, 2, 3
}

define <2 x i64> @decrement_by_val(<2 x i64> %x, i64 %val) nounwind {
       %tmpvec = insertelement <2 x i64> <i64 0, i64 0>, i64 %val, i32 0
       %tmpvec2 = insertelement <2 x i64> %tmpvec, i64 %val, i32 1
       %result = sub <2 x i64> %x, %tmpvec2
       ret <2 x i64> %result
; CHECK: vsubudm 2, 2, 3
}



