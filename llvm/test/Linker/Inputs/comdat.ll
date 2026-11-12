target datalayout = "e-m:w-p:32:32-i64:64-f80:32-n8:16:32-S32"
target triple = "i686-pc-windows-msvc"

$foo = comdat largest
@foo = global i64 43, comdat($foo)

define i32 @bar() comdat($foo) {
  ret i32 43
}

$qux = comdat largest
@qux = global i32 13, comdat($qux)
@in_unselected_group = global i32 13, comdat($qux)

define i32 @baz() comdat($qux) {
  ret i32 13
}

$any = comdat any
@any = global i64 7, comdat($any)
