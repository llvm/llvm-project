; RUN: llc -mtriple=hexagon -mcpu=hexagonv73 < %s | FileCheck %s

target triple = "hexagon"

; CHECK-LABEL: test1:
; CHECK: r0 = mask(#25,#2)
; Function Attrs: optsize
define i32 @test1() #1 {
entry:
  %0 = call i32 @llvm.hexagon.A2.tfr(i32 134217724)
  ret i32 %0
}

declare i32 @llvm.hexagon.A2.tfr(i32) #0

attributes #0 = { nounwind readnone }
attributes #1 = { optsize }
