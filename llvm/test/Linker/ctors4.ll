; RUN: llvm-link -S %s -o - | FileCheck %s

define void @f() {
  ret void
}

; We lazy link @v, which causes llvm.global_ctors to have the corresponding
; entry.
@v = linkonce global i8 42

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @f, ptr @v }]

; CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @f, ptr @v }]

