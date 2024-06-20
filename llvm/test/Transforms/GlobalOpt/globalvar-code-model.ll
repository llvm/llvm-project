; RUN: opt -passes=globalopt -S < %s | FileCheck %s

@G = internal global i32 5, code_model "large"

define i32 @test() norecurse {
  %a = load i32, ptr @G
  store i32 4, ptr @G
  ret i32 %a
}

; CHECK: @G = internal unnamed_addr global i1 false, code_model "large"
