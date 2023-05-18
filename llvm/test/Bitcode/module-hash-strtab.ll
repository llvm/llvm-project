; RUN: opt -module-hash %s -o - | llvm-bcanalyzer -dump | grep '<HASH' > %t
; RUN: opt -module-hash %S/Inputs/module-hash-strtab1.ll -o - | llvm-bcanalyzer -dump | grep '<HASH' >> %t
; RUN: opt -module-hash %S/Inputs/module-hash-strtab2.ll -o - | llvm-bcanalyzer -dump | grep '<HASH' >> %t
; RUN: sort %t | uniq | count 3

source_filename = "foo.c"

$com = comdat any

define void @main() comdat($com) {
  call void @foo()
  ret void
}

declare void @foo()
