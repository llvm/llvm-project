# RUN: llvm-mc %s -filetype obj -triple x86_64-apple-darwin -o - \
# RUN: | llvm-dwarfdump -verify - \
# RUN: | FileCheck %s

# CHECK-NOT: Verifying .apple_names...

# This test is meant to verify that the -verify option 
# in llvm-dwarfdump doesn't produce any .apple_names related
# output when there's no such section in the object.
# The test was manually modified to exclude the 
# .apple_names section from the apple_names_verify_num_atoms.s
# test file in the same directory.

  .section  __TEXT,__text,regular,pure_instructions
  .file 1 "basic.c"
  .comm _i,4,2                  ## @i
  .comm _j,4,2                  ## @j
  .section  __DWARF,__debug_str,regular,debug
Linfo_string:
  .asciz  "Apple LLVM version 8.1.0 (clang-802.0.35)" ## string offset=0
  .asciz  "basic.c"               ## string offset=42
  .asciz  "/Users/sgravani/Development/tests" ## string offset=50
  .asciz  "i"                     ## string offset=84
  .asciz  "int"                   ## string offset=86
  .asciz  "j"                     ## string offset=90
  
  .section  __DWARF,__debug_info,regular,debug
Lsection_info:

.subsections_via_symbols
  .section  __DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:
