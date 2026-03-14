source_filename = "foo.c"

$com = comdat any

define void @main() comdat($com) {
  call void @bar()
  ret void
}

declare void @bar()
