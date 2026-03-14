; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK: declare void @decl(ptr sret(i64))
; CHECK: call void @decl(ptr %arg)

declare void @decl(i64* sret(i64))

define void @test(i64* %arg) {
  call void @decl(i64* %arg)
  ret void
}
