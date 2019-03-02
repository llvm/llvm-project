# REQUIRES: x86-registered-target

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o

# RUN: llvm-symbolizer 0 --obj=%t.o | FileCheck %s --check-prefix=LINKAGE
# RUN: llvm-symbolizer 0 --functions --obj=%t.o | FileCheck %s --check-prefix=LINKAGE
# RUN: llvm-symbolizer 0 --functions=linkage --obj=%t.o | FileCheck %s --check-prefix=LINKAGE
# RUN: llvm-symbolizer 0 --functions=short --obj=%t.o | FileCheck %s --check-prefix=SHORT
# RUN: llvm-symbolizer 0 --functions=none --obj=%t.o | FileCheck %s --check-prefix=NONE

## Characterise behaviour for no '=' sign. llvm-symbolizer treats the next option as an
## input address, and just prints it.
# RUN: llvm-symbolizer 0 --functions none --obj=%t.o | FileCheck %s --check-prefixes=LINKAGE,ERR

# LINKAGE:      {{^}}foo(int){{$}}
# LINKAGE-NEXT: functions.cpp:2:0

# SHORT:      {{^}}foo{{$}}
# SHORT-NEXT: functions.cpp:2:0

# NONE-NOT: foo
# NONE:     functions.cpp:2:0

# ERR: none

# The assembly below is a stripped down version of the output of:
#   clang -S -g --target=x86_64-pc-linux
# for the following C++ source:
#   void foo(int bar) {}
  .type _Z3fooi,@function
_Z3fooi:
.Lfunc_begin0:
  .file   1 "/llvm-symbolizer/Inputs" "functions.cpp"
  .loc    1 2 0                 # functions.cpp:2:0
  nop
  .loc  1 2 20 prologue_end     # functions.cpp:2:20
.Lfunc_end0:

  .section    .debug_str,"MS",@progbits,1
.Linfo_string1:
  .asciz  "functions.cpp"
.Linfo_string2:
  .asciz  "/llvm-symbolizer/Inputs"
.Linfo_string3:
  .asciz  "_Z3fooi"
.Linfo_string4:
  .asciz  "foo"

  .section  .debug_abbrev,"",@progbits
  .byte 1                       # Abbreviation Code
  .byte 17                      # DW_TAG_compile_unit
  .byte 1                       # DW_CHILDREN_yes
  .byte 3                       # DW_AT_name
  .byte 14                      # DW_FORM_strp
  .byte 16                      # DW_AT_stmt_list
  .byte 23                      # DW_FORM_sec_offset
  .byte 27                      # DW_AT_comp_dir
  .byte 14                      # DW_FORM_strp
  .byte 17                      # DW_AT_low_pc
  .byte 1                       # DW_FORM_addr
  .byte 18                      # DW_AT_high_pc
  .byte 6                       # DW_FORM_data4
  .byte 0                       # EOM(1)
  .byte 0                       # EOM(2)
  .byte 2                       # Abbreviation Code
  .byte 46                      # DW_TAG_subprogram
  .byte 0                       # DW_CHILDREN_no
  .byte 17                      # DW_AT_low_pc
  .byte 1                       # DW_FORM_addr
  .byte 18                      # DW_AT_high_pc
  .byte 6                       # DW_FORM_data4
  .byte 110                     # DW_AT_linkage_name
  .byte 14                      # DW_FORM_strp
  .byte 3                       # DW_AT_name
  .byte 14                      # DW_FORM_strp
  .byte 58                      # DW_AT_decl_file
  .byte 11                      # DW_FORM_data1
  .byte 59                      # DW_AT_decl_line
  .byte 11                      # DW_FORM_data1
  .byte 0                       # EOM(1)
  .byte 0                       # EOM(2)
  .byte 0                       # EOM(3)

  .section  .debug_info,"",@progbits
.Lcu_begin0:
  .long .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
  .short  4                     # DWARF version number
  .long .debug_abbrev           # Offset Into Abbrev. Section
  .byte 8                       # Address Size (in bytes)
  .byte 1                       # Abbrev [1] 0xb:0x4f DW_TAG_compile_unit
  .long .Linfo_string1          # DW_AT_name
  .long .Lline_table_start0     # DW_AT_stmt_list
  .long .Linfo_string2          # DW_AT_comp_dir
  .quad .Lfunc_begin0           # DW_AT_low_pc
  .long .Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
  .byte 2                       # Abbrev [2] 0x2a:0x28 DW_TAG_subprogram
  .quad .Lfunc_begin0           # DW_AT_low_pc
  .long .Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
  .long .Linfo_string3          # DW_AT_linkage_name
  .long .Linfo_string4          # DW_AT_name
  .byte 1                       # DW_AT_decl_file
  .byte 2                       # DW_AT_decl_line
  .byte 0                       # End Of Children Mark
  .byte 0                       # End Of Children Mark
.Ldebug_info_end0:

  .section  .debug_line,"",@progbits
.Lline_table_start0:
