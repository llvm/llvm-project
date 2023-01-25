; A copy of 2012-06-08-APIntCrash.ll with arch explicitly set to hexagon.

; RUN: llc -march=hexagon < %s

define void @test1(ptr %ptr) {
  %1 = load <8 x i32>, ptr %ptr, align 32
  %2 = and <8 x i32> %1, <i32 0, i32 0, i32 0, i32 -1, i32 0, i32 0, i32 0, i32 -1>
  store <8 x i32> %2, ptr %ptr, align 16
  ret void
}
