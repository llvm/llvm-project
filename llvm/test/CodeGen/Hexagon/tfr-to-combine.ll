; RUN: llc -march=hexagon -mcpu=hexagonv5  -O3 -disable-hsdr < %s | FileCheck %s

; Check that we combine TFRs and TFRIs into COMBINEs.

@a = external global i16
@b = external global i16
@c = external global i16

; Function Attrs: nounwind
define i64 @test1() #0 {
; CHECK: combine(#10,#0)
entry:
  store i16 0, i16* @a, align 2
  store i16 10, i16* @b, align 2
  ret i64 10
}

; Function Attrs: nounwind
define i64 @test2() #0 {
; CHECK: combine(#0,r{{[0-9]+}})
entry:
  store i16 0, i16* @a, align 2
  %0 = load i16, i16* @c, align 2
  %conv2 = zext i16 %0 to i64
  ret i64 %conv2
}

; Function Attrs: nounwind
define i64 @test4() #0 {
; CHECK: combine(#0,#100)
entry:
  store i16 100, i16* @b, align 2
  store i16 0, i16* @a, align 2
  ret i64 0
}
