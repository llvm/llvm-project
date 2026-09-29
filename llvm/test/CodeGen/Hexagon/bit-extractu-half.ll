; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; Pick lsr (in bit-simplification) for extracting high halfword.
; CHECK: lsr{{.*}}#16

define i32 @foo(i32 %x) #0 {
  %a = call i32 @llvm.hexagon.S2.extractu(i32 %x, i32 16, i32 16)
  ret i32 %a
}

declare i32 @llvm.hexagon.S2.extractu(i32, i32, i32) #0

attributes #0 = { nounwind readnone }

