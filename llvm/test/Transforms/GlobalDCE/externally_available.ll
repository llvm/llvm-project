; RUN: opt < %s -globaldce -S | FileCheck %s

; test_function should not be emitted to the .s file.
; CHECK-NOT: test_function
define available_externally i32 @test_function() {
  ret i32 4
}

; test_global should not be emitted to the .s file.
; CHECK-NOT: test_global
@test_global = available_externally global i32 4

