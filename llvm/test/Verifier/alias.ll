; RUN:  not llvm-as %s -o /dev/null 2>&1 | FileCheck %s


declare void @f()
@fa = alias void (), void ()* @f
; CHECK: Alias must point to a definition
; CHECK-NEXT: @fa

@g = external global i32
@ga = alias i32, i32* @g
; CHECK: Alias must point to a definition
; CHECK-NEXT: @ga

define available_externally void @f2() {
  ret void
}
@fa2 = alias void(), void()* @f2
; CHECK: Alias must point to a definition
; CHECK-NEXT: @fa2

@test2_a = alias i32, i32* @test2_b
@test2_b = alias i32, i32* @test2_a
; CHECK:      Aliases cannot form a cycle
; CHECK-NEXT: i32* @test2_a
; CHECK-NEXT: Aliases cannot form a cycle
; CHECK-NEXT: i32* @test2_b


@test3_a = global i32 42
@test3_b = weak alias i32, i32* @test3_a
@test3_c = alias i32, i32* @test3_b
; CHECK: Alias cannot point to an interposable alias
; CHECK-NEXT: i32* @test3_c
