;This isn't really an assembly file, its just here to run the test.
;This test just makes sure that llvm-ar can generate a table of contents for
;SVR4 style archives
;RUN: llvm-ar t %p/SVR4.a | FileCheck %s
;CHECK:      evenlen
;CHECK-NEXT: oddlen
;CHECK-NEXT: very_long_bytecode_file_name.bc
;CHECK-NEXT: IsNAN.o
