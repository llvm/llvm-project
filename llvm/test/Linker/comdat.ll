; RUN: llvm-link %s %p/Inputs/comdat.ll -S -o - | FileCheck %s
target datalayout = "e-m:w-p:32:32-i64:64-f80:32-n8:16:32-S32"
target triple = "i686-pc-windows-msvc"

$foo = comdat largest
@foo = global i32 42, comdat($foo)

define i32 @bar() comdat($foo) {
  ret i32 42
}

$qux = comdat largest
@qux = global i64 12, comdat($qux)

define i32 @baz() comdat($qux) {
  ret i32 12
}

$any = comdat any
@any = global i64 6, comdat($any)

; CHECK: $qux = comdat largest
; CHECK: $foo = comdat largest
; CHECK: $any = comdat any

; CHECK: @foo = global i64 43, comdat{{$}}
; CHECK: @qux = global i64 12, comdat{{$}}
; CHECK: @any = global i64 6, comdat{{$}}
; CHECK-NOT: @in_unselected_group = global i32 13, comdat $qux

; CHECK: define i32 @baz() comdat($qux)
; CHECK: define i32 @bar() comdat($foo)
