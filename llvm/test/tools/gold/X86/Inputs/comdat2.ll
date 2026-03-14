target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

$foo = comdat any
@foo = global i8 1, comdat
define void @zed() {
  call void @bar()
  ret void
}
define void @bar() comdat($foo) {
  ret void
}
