; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: clause argument must be a constant

define void @test(i32 %in) personality ptr null {
  landingpad {} filter i32 %in
}
