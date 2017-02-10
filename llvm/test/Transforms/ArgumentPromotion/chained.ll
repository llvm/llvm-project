; RUN: opt < %s -argpromotion -S | FileCheck %s
; RUN: opt < %s -passes=argpromotion -S | FileCheck %s

@G1 = constant i32 0
@G2 = constant i32* @G1

define internal i32 @test(i32** %x) {
; CHECK-LABEL: define internal i32 @test(
; CHECK: i32 %{{.*}})
entry:
  %y = load i32*, i32** %x
  %z = load i32, i32* %y
; CHECK-NOT: load
  ret i32 %z
; CHECK: ret i32
}

define i32 @caller() {
; CHECK-LABEL: define i32 @caller()
entry:
  %x = call i32 @test(i32** @G2)
; CHECK: %[[Y:.*]] = load i32*, i32** @G2
; CHECK: %[[Z:.*]] = load i32, i32* %[[Y]]
; CHECK: call i32 @test(i32 %[[Z]])
  ret i32 %x
}

