; Check VMX 128-bit integer operations
;
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 -mattr=-vsx < %s | FileCheck %s

define <1 x i128> @out_of_bounds_insertelement(<1 x i128> %x, i128 %val) nounwind {
       %tmpvec = insertelement <1 x i128> <i128 0>, i128 %val, i32 1
       %result = add <1 x i128> %x, %tmpvec
       ret <1 x i128> %result
; CHECK-LABEL: @out_of_bounds_insertelement
; CHECK: # BB#0:
; CHECK-NEXT: blr
}

define <1 x i128> @test_add(<1 x i128> %x, <1 x i128> %y) nounwind {
       %result = add <1 x i128> %x, %y
       ret <1 x i128> %result
; CHECK-LABEL: @test_add
; CHECK: vadduqm 2, 2, 3
}

define <1 x i128> @increment_by_one(<1 x i128> %x) nounwind {
       %result = add <1 x i128> %x, <i128 1>
       ret <1 x i128> %result
; CHECK-LABEL: @increment_by_one
; CHECK: vadduqm 2, 2, 3
}

define <1 x i128> @increment_by_val(<1 x i128> %x, i128 %val) nounwind {
       %tmpvec = insertelement <1 x i128> <i128 0>, i128 %val, i32 0
       %result = add <1 x i128> %x, %tmpvec
       ret <1 x i128> %result
; CHECK-LABEL: @increment_by_val
; CHECK: vadduqm 2, 2, 3
}

define <1 x i128> @test_sub(<1 x i128> %x, <1 x i128> %y) nounwind {
       %result = sub <1 x i128> %x, %y
       ret <1 x i128> %result
; CHECK-LABEL: @test_sub
; CHECK: vsubuqm 2, 2, 3
}

define <1 x i128> @decrement_by_one(<1 x i128> %x) nounwind {
       %result = sub <1 x i128> %x, <i128 1>
       ret <1 x i128> %result
; CHECK-LABEL: @decrement_by_one
; CHECK: vsubuqm 2, 2, 3
}

define <1 x i128> @decrement_by_val(<1 x i128> %x, i128 %val) nounwind {
       %tmpvec = insertelement <1 x i128> <i128 0>, i128 %val, i32 0
       %result = sub <1 x i128> %x, %tmpvec
       ret <1 x i128> %result
; CHECK-LABEL: @decrement_by_val
; CHECK: vsubuqm 2, 2, 3
}

declare <1 x i128> @llvm.ppc.altivec.vaddeuqm(<1 x i128> %x,
                                              <1 x i128> %y,
                                              <1 x i128> %z) nounwind readnone
declare <1 x i128> @llvm.ppc.altivec.vaddcuq(<1 x i128> %x,
                                             <1 x i128> %y) nounwind readnone
declare <1 x i128> @llvm.ppc.altivec.vaddecuq(<1 x i128> %x,
                                              <1 x i128> %y,
                                              <1 x i128> %z) nounwind readnone
declare <1 x i128> @llvm.ppc.altivec.vsubeuqm(<1 x i128> %x,
                                              <1 x i128> %y,
                                              <1 x i128> %z) nounwind readnone
declare <1 x i128> @llvm.ppc.altivec.vsubcuq(<1 x i128> %x,
                                             <1 x i128> %y) nounwind readnone
declare <1 x i128> @llvm.ppc.altivec.vsubecuq(<1 x i128> %x,
                                              <1 x i128> %y,
                                              <1 x i128> %z) nounwind readnone

define <1 x i128> @test_vaddeuqm(<1 x i128> %x,
       	    	                 <1 x i128> %y,
                                 <1 x i128> %z) nounwind {
  %tmp = tail call <1 x i128> @llvm.ppc.altivec.vaddeuqm(<1 x i128> %x,
                                                         <1 x i128> %y,
                                                         <1 x i128> %z)
  ret <1 x i128> %tmp
; CHECK-LABEL: @test_vaddeuqm
; CHECK: vaddeuqm 2, 2, 3, 4
}

define <1 x i128> @test_vaddcuq(<1 x i128> %x,
       	    	                <1 x i128> %y) nounwind {
  %tmp = tail call <1 x i128> @llvm.ppc.altivec.vaddcuq(<1 x i128> %x,
                                                        <1 x i128> %y)
  ret <1 x i128> %tmp
; CHECK-LABEL: @test_vaddcuq
; CHECK: vaddcuq 2, 2, 3
}

define <1 x i128> @test_vaddecuq(<1 x i128> %x,
       	    	                 <1 x i128> %y,
                                 <1 x i128> %z) nounwind {
  %tmp = tail call <1 x i128> @llvm.ppc.altivec.vaddecuq(<1 x i128> %x,
                                                         <1 x i128> %y,
                                                         <1 x i128> %z)
  ret <1 x i128> %tmp
; CHECK-LABEL: @test_vaddecuq
; CHECK: vaddecuq 2, 2, 3, 4
}

define <1 x i128> @test_vsubeuqm(<1 x i128> %x,
       	    	                 <1 x i128> %y,
                                 <1 x i128> %z) nounwind {
  %tmp = tail call <1 x i128> @llvm.ppc.altivec.vsubeuqm(<1 x i128> %x,
                                                         <1 x i128> %y,
                                                         <1 x i128> %z)
  ret <1 x i128> %tmp
; CHECK-LABEL: test_vsubeuqm
; CHECK: vsubeuqm 2, 2, 3, 4
}

define <1 x i128> @test_vsubcuq(<1 x i128> %x,
       	    	                <1 x i128> %y) nounwind {
  %tmp = tail call <1 x i128> @llvm.ppc.altivec.vsubcuq(<1 x i128> %x,
                                                        <1 x i128> %y)
  ret <1 x i128> %tmp
; CHECK-LABEL: test_vsubcuq
; CHECK: vsubcuq 2, 2, 3
}

define <1 x i128> @test_vsubecuq(<1 x i128> %x,
       	    	                 <1 x i128> %y,
                                 <1 x i128> %z) nounwind {
  %tmp = tail call <1 x i128> @llvm.ppc.altivec.vsubecuq(<1 x i128> %x,
                                                         <1 x i128> %y,
                                                         <1 x i128> %z)
  ret <1 x i128> %tmp
; CHECK-LABEL: test_vsubecuq
; CHECK: vsubecuq 2, 2, 3, 4
}

