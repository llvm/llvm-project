; This isn't really an assembly file, its just here to run the test.

; This test just makes sure that llvm-ar can extract bytecode members
; from various style archives.

; RUN: llvm-ar p %p/GNU.a very_long_bytecode_file_name.bc | \
; RUN:   cmp -s %p/very_long_bytecode_file_name.bc -

; RUN: llvm-ar p %p/MacOSX.a very_long_bytecode_file_name.bc | \
; RUN:   cmp -s %p/very_long_bytecode_file_name.bc -

; RUN: llvm-ar p %p/SVR4.a very_long_bytecode_file_name.bc | \
; RUN:   cmp -s %p/very_long_bytecode_file_name.bc -

; RUN: llvm-ar p %p/xpg4.a very_long_bytecode_file_name.bc |\
; RUN:   cmp -s %p/very_long_bytecode_file_name.bc -
