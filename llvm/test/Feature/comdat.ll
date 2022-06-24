; RUN: llvm-as --opaque-pointers=0 < %s | llvm-dis --opaque-pointers=0 | FileCheck %s

$f = comdat any
; CHECK: $f = comdat any

$f2 = comdat any
; CHECK-NOT: f2

@v = global i32 0, comdat($f)
; CHECK: @v = global i32 0, comdat($f)

@a = alias i32, i32* @v
; CHECK: @a = alias i32, i32* @v{{$}}

define void @f() comdat($f) {
  ret void
}
; CHECK: define void @f() comdat {

$i = comdat largest
@i = internal global i32 0, comdat($i)
