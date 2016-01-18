; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i1:32-i64:64-a:0-v32:32-n16:32"
target triple = "hexagon"

; CHECK-LABEL: test1:
; CHECK: r0 = ##1073741824
define i32 @test1() #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.S2.asr.i.r.rnd(i32 2147483647, i32 0)
  ret i32 %0
}

; CHECK-LABEL: test2:
; CHECK: r0 = ##1073741824
define i32 @test2() #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.S2.asr.i.r.rnd.goodsyntax(i32 2147483647, i32 1)
  ret i32 %0
}

; CHECK-LABEL: test3:
; CHECK: r1:0 = #1
define i64 @test3() #0 {
entry:
  %0 = tail call i64 @llvm.hexagon.S4.extractp(i64 -1, i32 63, i32 63)
  ret i64 %0
}

; CHECK-LABEL: test4:
; CHECK: r0 = #1
define i32 @test4() #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.S4.extract(i32 -1, i32 31, i32 31)
  ret i32 %0
}

; CHECK-LABEL: test5:
; CHECK: r0 = ##-1073741569
define i32 @test5() #0 {
entry:
  %0 = tail call i32 @llvm.hexagon.S4.subi.lsr.ri(i32 255, i32 -2147483648, i32 1)
  ret i32 %0
}

declare i32 @llvm.hexagon.S2.asr.i.r.rnd(i32, i32) #0
declare i32 @llvm.hexagon.S2.asr.i.r.rnd.goodsyntax(i32, i32) #0
declare i64 @llvm.hexagon.S4.extractp(i64, i32, i32) #0
declare i32 @llvm.hexagon.S4.extract(i32, i32, i32) #0
declare i32 @llvm.hexagon.S4.subi.lsr.ri(i32, i32, i32) #0

attributes #0 = { nounwind readnone }

