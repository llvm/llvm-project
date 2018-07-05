; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr9 < %s | FileCheck -check-prefix=CHECK-PWR9 %s

define <4 x i32> @testSpill(<4 x i32> %a, <4 x i32> %b) {

; CHECK-LABEL: testSpill:
; CHECK-DAG:    li [[REG48:[0-9]+]], 48
; CHECK-DAG:    li [[REG64:[0-9]+]], 64
; CHECK-DAG:    li [[REG80:[0-9]+]], 80
; CHECK-DAG:    li [[REG96:[0-9]+]], 96
; CHECK-DAG:    stxvd2x 60, 1, [[REG48]] # 16-byte Folded Spill
; CHECK-DAG:    stxvd2x 61, 1, [[REG64]] # 16-byte Folded Spill
; CHECK-DAG:    stxvd2x 62, 1, [[REG80]] # 16-byte Folded Spill
; CHECK-DAG:    stxvd2x 63, 1, [[REG96]] # 16-byte Folded Spill
; CHECK:        .LBB0_3
; CHECK-DAG:    li [[REG96_LD:[0-9]+]], 96
; CHECK-DAG:    li [[REG80_LD:[0-9]+]], 80
; CHECK-DAG:    li [[REG64_LD:[0-9]+]], 64
; CHECK-DAG:    li [[REG48_LD:[0-9]+]], 48
; CHECK-DAG:    lxvd2x 63, 1, [[REG96_LD]] # 16-byte Folded Reload
; CHECK-DAG:    lxvd2x 62, 1, [[REG80_LD]] # 16-byte Folded Reload
; CHECK-DAG:    lxvd2x 61, 1, [[REG64_LD]] # 16-byte Folded Reload
; CHECK-DAG:    lxvd2x 60, 1, [[REG48_LD]] # 16-byte Folded Reload
; CHECK:    mtlr 0
; CHECK-NEXT:    blr
;
; CHECK-PWR9-LABEL: testSpill:
; CHECK-PWR9-DAG:    stxv 62, 64(1) # 16-byte Folded Spill
; CHECK-PWR9-DAG:    stxv 63, 80(1) # 16-byte Folded Spill
; CHECK-PWR9-DAG:    stxv 60, 32(1) # 16-byte Folded Spill
; CHECK-PWR9-DAG:    stxv 61, 48(1) # 16-byte Folded Spill
; CHECK-PWR9-NOT:    NOT
; CHECK-PWR9-DAG:    lxv 63, 80(1) # 16-byte Folded Reload
; CHECK-PWR9-DAG:    lxv 62, 64(1) # 16-byte Folded Reload
; CHECK-PWR9-DAG:    lxv 61, 48(1) # 16-byte Folded Reload
; CHECK-PWR9-DAG:    lxv 60, 32(1) # 16-byte Folded Reload
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
