# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux --defsym CASE1=0 -o %t1.o
# RUN: not llvm-dwarfdump -debug-loclists %t1.o 2>&1 | FileCheck %s --check-prefix=ULEB -DOFFSET=0x0000000d

# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux --defsym CASE2=0 -o %t2.o
# RUN: not llvm-dwarfdump -debug-loclists %t2.o 2>&1 | FileCheck %s --check-prefix=ULEB -DOFFSET=0x0000000e

# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux --defsym CASE3=0 -o %t3.o
# RUN: not llvm-dwarfdump -debug-loclists %t3.o 2>&1 | FileCheck %s --check-prefix=ULEB -DOFFSET=0x0000000f

# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux --defsym CASE4=0 -o %t4.o
# RUN: not llvm-dwarfdump -debug-loclists %t4.o 2>&1 | FileCheck %s

# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux --defsym CASE5=0 -o %t5.o
# RUN: not llvm-dwarfdump -debug-loclists %t5.o 2>&1 | FileCheck %s

# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux --defsym CASE6=0 -o %t6.o
# RUN: not llvm-dwarfdump -debug-loclists %t6.o 2>&1 | FileCheck %s

# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux --defsym CASE7=0 -o %t7.o
# RUN: not llvm-dwarfdump -debug-loclists %t7.o 2>&1 | FileCheck %s --check-prefix=UNIMPL

# CHECK: error: unexpected end of data
# ULEB: error: unable to decode LEB128 at offset [[OFFSET]]: malformed uleb128, extends past end
# UNIMPL: error: LLE of kind 47 not supported

.section  .debug_loclists,"",@progbits
  .long  .Ldebug_loclist_table_end0-.Ldebug_loclist_table_start0
.Ldebug_loclist_table_start0:
 .short 5                        # Version.
 .byte 8                         # Address size.
 .byte 0                         # Segment selector size.
 .long 0                         # Offset entry count.
.Lloclists_table_base0:
.Ldebug_loc0:
.ifdef CASE1
  .byte  4                       # DW_LLE_offset_pair
.endif
.ifdef CASE2
  .byte  4                       # DW_LLE_offset_pair
  .uleb128 0x0                   #   starting offset
.endif
.ifdef CASE3
  .byte  4                       # DW_LLE_offset_pair
  .uleb128 0x0                   #   starting offset
  .uleb128 0x10                  #   ending offset
.endif
.ifdef CASE4
  .byte  4                       # DW_LLE_offset_pair
  .uleb128 0x0                   #   starting offset
  .uleb128 0x10                  #   ending offset
  .byte  1                       # Loc expr size
.endif
.ifdef CASE5
  .byte  4                       # DW_LLE_offset_pair
  .uleb128 0x0                   #   starting offset
  .uleb128 0x10                  #   ending offset
  .byte  1                       # Loc expr size
  .byte  117                     # DW_OP_breg5
.endif
.ifdef CASE6
  .byte  4                       # DW_LLE_offset_pair
  .uleb128 0x0                   #   starting offset
  .uleb128 0x10                  #   ending offset
  .uleb128 0xdeadbeef            # Loc expr size
.endif
.ifdef CASE7
  .byte 0x47
.endif

.Ldebug_loclist_table_end0:

