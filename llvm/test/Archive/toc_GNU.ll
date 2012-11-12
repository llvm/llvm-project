;This isn't really an assembly file, its just here to run the test.
;This test just makes sure that llvm-ar can generate a table of contents for
;GNU style archives
;RUN: llvm-ar t %p/GNU.a | FileCheck %s
;CHECK:      evenlen
;CHECK-NEXT: oddlen
;CHECK-NEXT: very_long_bytecode_file_name.bc
;CHECK-NEXT: IsNAN.o
