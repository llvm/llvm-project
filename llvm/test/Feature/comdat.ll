; RUN: llvm-as  < %s | llvm-dis  | FileCheck %s

$f = comdat any
; CHECK: $f = comdat any

$f2 = comdat any
; CHECK-NOT: f2

@v = global i32 0, comdat($f)
; CHECK: @v = global i32 0, comdat($f)

@a = alias i32, ptr @v
; CHECK: @a = alias i32, ptr @v{{$}}

define void @f() comdat($f) {
  ret void
}
; CHECK: define void @f() comdat {

$i = comdat largest
@i = internal global i32 0, comdat($i)
