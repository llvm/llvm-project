;This isn't really an assembly file, its just here to run the test.
;This test just makes sure that llvm-ar can generate a table of contents for
;MacOSX style archives
;RUN: llvm-ar t %p/MacOSX.a | FileCheck %s
;CHECK:      __.SYMDEF SORTED
;CHECK-NEXT: evenlen
;CHECK-NEXT: oddlen
;CHECK-NEXT: very_long_bytecode_file_name.bc
;CHECK-NEXT: IsNAN.o
