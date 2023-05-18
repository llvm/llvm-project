source_filename = "foo.c"

$dat = comdat any

define void @main() comdat($dat) {
  call void @foo()
  ret void
}

declare void @foo()
