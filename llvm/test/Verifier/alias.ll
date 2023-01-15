; RUN:  not llvm-as %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=alias --implicit-check-not=Alias


declare void @f()
@fa = alias void (), ptr @f
; CHECK: Alias must point to a definition
; CHECK-NEXT: @fa

@g = external global i32
@ga = alias i32, ptr @g
; CHECK: Alias must point to a definition
; CHECK-NEXT: @ga

define available_externally void @f2() {
  ret void
}
@fa2 = alias void(), ptr @f2
; CHECK: Alias must point to a definition
; CHECK-NEXT: @fa2

@test2_a = alias i32, ptr @test2_b
@test2_b = alias i32, ptr @test2_a
; CHECK:      Aliases cannot form a cycle
; CHECK-NEXT: ptr @test2_a
; CHECK-NEXT: Aliases cannot form a cycle
; CHECK-NEXT: ptr @test2_b


@test3_a = global i32 42
@test3_b = weak alias i32, ptr @test3_a
@test3_c = alias i32, ptr @test3_b
; CHECK: Alias cannot point to an interposable alias
; CHECK-NEXT: ptr @test3_c

@test4_a = available_externally global i32 42
@test4_b = available_externally alias i32, ptr @test4_a
@test4_c = available_externally alias void(), ptr @f2
@test4_d = available_externally alias i32, ptr @test4_b

@test4_e = available_externally alias i32, ptr @test3_a
@test4_f = available_externally alias i32, inttoptr (i64 sub (i64 ptrtoint (ptr @test4_a to i64), i64 ptrtoint (ptr @test4_a to i64)) to ptr)
; CHECK:      available_externally alias must point to available_externally global value
; CHECK-NEXT: ptr @test4_e
; CHECK:      available_externally alias must point to available_externally global value
; CHECK-NEXT: ptr @test4_f
