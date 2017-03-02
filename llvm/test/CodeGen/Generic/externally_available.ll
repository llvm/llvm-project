; RUN: llc -verify-machine-dom-info < %s | not grep test_

; test_function should not be emitted to the .s file.
define available_externally i32 @test_function() {
  ret i32 4
}

; test_global should not be emitted to the .s file.
@test_global = available_externally global i32 4

