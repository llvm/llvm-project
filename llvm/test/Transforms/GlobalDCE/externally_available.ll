; RUN: opt < %s -passes=globaldce -S | FileCheck %s

; test_global should not be emitted to the .s file.
; CHECK-NOT: @test_global =
@test_global = available_externally global i32 4

; test_global2 is a normal global using an available externally function.
; CHECK: @test_global2 =
@test_global2 = global i32 ()* @test_function2

; test_function should not be emitted to the .s file.
; CHECK-NOT: define {{.*}} @test_function()
define available_externally i32 @test_function() {
  ret i32 4
}

; test_function2 isn't actually dead even though it's available externally.
; CHECK: define available_externally i32 @test_function2()
define available_externally i32 @test_function2() {
  ret i32 4
}
