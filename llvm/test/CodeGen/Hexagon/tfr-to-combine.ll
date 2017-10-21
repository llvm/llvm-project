; RUN: llc -march=hexagon -mcpu=hexagonv5  -O3 -disable-hsdr < %s | FileCheck %s

; Check that we combine TFRs and TFRIs into COMBINEs.

@a = external global i16
@b = external global i16
@c = external global i16

declare void @test0a(i32, i32) #0
declare void @test0b(i32, i32, i32, i32) #0

; CHECK-LABEL: test1:
; CHECK: combine(#10,#0)
define i32 @test1() #0 {
entry:
  call void @test0a(i32 0, i32 10) #0
  ret i32 10
}

; CHECK-LABEL: test2:
; CHECK: combine(#0,r{{[0-9]+}})
define i32 @test2() #0 {
entry:
  %t0 = load i16, i16* @c, align 2
  %t1 = zext i16 %t0 to i32
  call void @test0b(i32 %t1, i32 0, i32 %t1, i32 0)
  ret i32 0
}

; CHECK-LABEL: test3:
; CHECK: combine(#0,#100)
define i32 @test3() #0 {
entry:
  call void @test0a(i32 100, i32 0)
  ret i32 0
}

attributes #0 = { nounwind }
