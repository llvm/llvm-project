; RUN: llvm-as < %s | llvm-dis --materialize-metadata --show-annotations | FileCheck %s

; CHECK: ; Materializable
; CHECK-NEXT: define dso_local i32 @test() {}
define dso_local i32 @test() {
entry:
  ret i32 0
}

