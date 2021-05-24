; Expect the constant propagation to evaluate signed and unsigned bit extract.
; RUN: llc -march=hexagon -O2 < %s | FileCheck %s

target triple = "hexagon"

@x = common global i32 0, align 4
@y = common global i32 0, align 4

define void @foo() #0 {
entry:
  ; extractu(0x000ABCD0, 16, 4)
  ; should evaluate to 0xABCD (dec 43981)
  %0 = call i32 @llvm.hexagon.S2.extractu(i32 703696, i32 16, i32 4)
; CHECK: 43981
; CHECK-NOT: extractu
  store i32 %0, i32* @x, align 4
  ; extract(0x000ABCD0, 16, 4)
  ; should evaluate to 0xFFFFABCD (dec 4294945741 or -21555)
  %1 = call i32 @llvm.hexagon.S4.extract(i32 703696, i32 16, i32 4)
; CHECK: -21555
; CHECK-NOT: extract
  store i32 %1, i32* @y, align 4
  ret void
}

declare i32 @llvm.hexagon.S2.extractu(i32, i32, i32) #1

declare i32 @llvm.hexagon.S4.extract(i32, i32, i32) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
