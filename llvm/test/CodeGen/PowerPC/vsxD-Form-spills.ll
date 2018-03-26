; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr9 < %s | FileCheck -check-prefix=CHECK-PWR9 %s

define <4 x i32> @testSpill(<4 x i32> %a, <4 x i32> %b) {

; CHECK-LABEL: testSpill:
; CHECK:    li 11, 80
; CHECK:    li 12, 96
; CHECK:    li 3, 48
; CHECK:    li 10, 64
; CHECK:    stxvd2x 62, 1, 11 # 16-byte Folded Spill
; CHECK:    stxvd2x 63, 1, 12 # 16-byte Folded Spill
; CHECK:    stxvd2x 60, 1, 3 # 16-byte Folded Spill
; CHECK:    stxvd2x 61, 1, 10 # 16-byte Folded Spill
; CHECK:    li 9, 96
; CHECK:    li 10, 80
; CHECK:    li 11, 64
; CHECK:    li 12, 48
; CHECK:    lxvd2x 63, 1, 9 # 16-byte Folded Reload
; CHECK:    lxvd2x 62, 1, 10 # 16-byte Folded Reload
; CHECK:    lxvd2x 61, 1, 11 # 16-byte Folded Reload
; CHECK:    lxvd2x 60, 1, 12 # 16-byte Folded Reload
; CHECK:    mtlr 0
; CHECK-NEXT:    blr
;
; CHECK-PWR9-LABEL: testSpill:
; CHECK-PWR9:    stxv 62, 64(1) # 16-byte Folded Spill
; CHECK-PWR9:    stxv 63, 80(1) # 16-byte Folded Spill
; CHECK-PWR9:    stxv 60, 32(1) # 16-byte Folded Spill
; CHECK-PWR9:    stxv 61, 48(1) # 16-byte Folded Spill
; CHECK-PWR9:    lxv 63, 80(1) # 16-byte Folded Reload
; CHECK-PWR9:    lxv 62, 64(1) # 16-byte Folded Reload
; CHECK-PWR9:    lxv 61, 48(1) # 16-byte Folded Reload
; CHECK-PWR9:    lxv 60, 32(1) # 16-byte Folded Reload
; CHECK-PWR9:    mtlr 0
; CHECK-PWR9-NEXT:    blr

entry:
  %0 = tail call i32 @llvm.ppc.altivec.vcmpgtsw.p(i32 2, <4 x i32> %a, <4 x i32> %b)
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  %call = tail call <4 x i32> @test1(<4 x i32> %a, <4 x i32> %b)
  br label %if.end

if.else:                                          ; preds = %entry
  %call1 = tail call <4 x i32> @test2(<4 x i32> %b, <4 x i32> %a)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %c.0.in = phi <4 x i32> [ %call, %if.then ], [ %call1, %if.else ]
  %call3 = tail call <4 x i32> @test1(<4 x i32> %b, <4 x i32> %a)
  %call5 = tail call <4 x i32> @test2(<4 x i32> %a, <4 x i32> %b)
  %add4 = add <4 x i32> %a, <i32 0, i32 0, i32 2, i32 2>
  %add6 = add <4 x i32> %add4, %c.0.in
  %c.0 = add <4 x i32> %add6, %call3
  %add7 = add <4 x i32> %c.0, %call5
  ret <4 x i32> %add7
}

; Function Attrs: nounwind readnone
declare i32 @llvm.ppc.altivec.vcmpgtsw.p(i32, <4 x i32>, <4 x i32>)
declare <4 x i32> @test1(<4 x i32>, <4 x i32>)
declare <4 x i32> @test2(<4 x i32>, <4 x i32>)
