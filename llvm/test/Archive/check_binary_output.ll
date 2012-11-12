; This is not an assembly file, this is just to run the test.
; The test verifies that llvm-ar produces a binary output.

;RUN: llvm-ar p %p/GNU.a very_long_bytecode_file_name.bc | cmp -s %p/very_long_bytecode_file_name.bc -
