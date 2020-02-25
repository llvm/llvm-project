# REQUIRES: x86

# RUN: llvm-mc --filetype=obj --triple x86_64-pc-linux %s -o %t --defsym MAIN=0
# RUN: llvm-mc --filetype=obj --triple x86_64-pc-linux %s -o %t.dwp --defsym DWP=0
# RUN: %lldb %t -o "target variable A" -o "image lookup -v -n F1" -b | FileCheck %s
# RUN: lldb-test symbols %t | FileCheck %s --check-prefix=SYMBOLS

# CHECK-LABEL: target variable A
# CHECK: (INT0) A = 0
# CHECK: (INT1) A = 1
# CHECK: (INT2) A = 2
# CHECK: (INT3) A = 3

# CHECK-LABEL: image lookup -v -n F1
# CHECK: CompileUnit: id = {0x00000001}, file = "1.c", language = "unknown"
# CHECK: Function: {{.*}}, name = "F1", range = [0x0000000000000001-0x0000000000000002)
# CHECK: Variable: {{.*}}, name = "x", type = "int", location = DW_OP_reg1 RDX

# SYMBOLS:      Compile units:
# SYMBOLS-NEXT: CompileUnit{0x00000000}, language = "unknown", file = '0.c'
# SYMBOLS-NEXT:   Variable{{.*}}, name = "A", {{.*}}, location = DW_OP_GNU_addr_index 0x0
# SYMBOLS-NEXT:   Function{{.*}}, demangled = F0
# SYMBOLS-NEXT:   Block{{.*}}, ranges = [0x00000000-0x00000001)
# SYMBOLS-NEXT:     Variable{{.*}}, name = "x", {{.*}}, location = 
# SYMBOLS-NEXT:       DW_LLE_startx_length   (0x0000000000000001, 0x0000000000000001): DW_OP_reg0 RAX
# SYMBOLS-EMPTY:
# SYMBOLS-NEXT: CompileUnit{0x00000001}, language = "unknown", file = '1.c'
# SYMBOLS-NEXT:   Variable{{.*}}, name = "A", {{.*}}, location = DW_OP_GNU_addr_index 0x2
# SYMBOLS-NEXT:   Function{{.*}}, demangled = F1
# SYMBOLS-NEXT:   Block{{.*}}, ranges = [0x00000001-0x00000002)
# SYMBOLS-NEXT:     Variable{{.*}}, name = "x", {{.*}}, location = 
# SYMBOLS-NEXT:       DW_LLE_startx_length   (0x0000000000000003, 0x0000000000000001): DW_OP_reg1 RDX
# SYMBOLS-EMPTY:
# SYMBOLS-NEXT: CompileUnit{0x00000002}, language = "unknown", file = '2.c'
# SYMBOLS-NEXT:   Variable{{.*}}, name = "A", {{.*}}, location = DW_OP_GNU_addr_index 0x4
# SYMBOLS-NEXT:   Function{{.*}}, demangled = F2
# SYMBOLS-NEXT:   Block{{.*}}, ranges = [0x00000002-0x00000003)
# SYMBOLS-NEXT:     Variable{{.*}}, name = "x", {{.*}}, location = 
# SYMBOLS-NEXT:       DW_LLE_startx_length   (0x0000000000000005, 0x0000000000000001): DW_OP_reg2 RCX
# SYMBOLS-EMPTY:
# SYMBOLS-NEXT: CompileUnit{0x00000003}, language = "unknown", file = '3.c'
# SYMBOLS-NEXT:   Variable{{.*}}, name = "A", {{.*}}, location = DW_OP_GNU_addr_index 0x6
# SYMBOLS-NEXT:   Function{{.*}}, demangled = F3
# SYMBOLS-NEXT:   Block{{.*}}, ranges = [0x00000003-0x00000004)
# SYMBOLS-NEXT:     Variable{{.*}}, name = "x", {{.*}}, location = 
# SYMBOLS-NEXT:       DW_LLE_startx_length   (0x0000000000000007, 0x0000000000000001): DW_OP_reg3 RBX
# SYMBOLS-EMPTY:
# SYMBOLS-NEXT: CompileUnit{0x00000004}, language = "unknown", file = ''
# SYMBOLS-EMPTY:

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   0                       # DW_CHILDREN_no
        .ascii  "\260B"                 # DW_AT_GNU_dwo_name
        .byte   8                       # DW_FORM_string
        .ascii  "\261B"                 # DW_AT_GNU_dwo_id
        .byte   7                       # DW_FORM_data8
        .ascii  "\263B"                 # DW_AT_GNU_addr_base
        .byte   23                      # DW_FORM_sec_offset
        .byte   85                      # DW_AT_ranges
        .byte   23                      # DW_FORM_sec_offset
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)

.ifdef MAIN
.irpc I,01234
        .data
A\I:
        .long \I

        .text
F\I:
        nop

        .section        .debug_info,"",@progbits
.Lcu_begin\I:
        .long   .Ldebug_info_end\I-.Ldebug_info_start\I # Length of Unit
.Ldebug_info_start\I:
        .short  4                       # DWARF version number
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .byte   1                       # Abbrev [1] 0xb:0x25 DW_TAG_compile_unit
        .asciz  "A.dwo"                 # DW_AT_GNU_dwo_name
        .quad   \I                      # DW_AT_GNU_dwo_id
        .long   .debug_addr             # DW_AT_GNU_addr_base
        .long   .Lranges\I              # DW_AT_ranges
.Ldebug_info_end\I:

        .section        .debug_addr,"",@progbits
        .quad   A\I
        .quad   F\I

        .section        .debug_ranges,"",@progbits
.Lranges\I:
        .quad   F\I
        .quad   F\I+1
        .quad   0
        .quad   0
.endr
.endif

.ifdef DWP
# This deliberately excludes compile unit 4 to check test the case of a missing
# split unit.
.irpc I,0123
        .section        .debug_str.dwo,"MSe",@progbits,1
.Lstr\I:
        .byte   'I', 'N', 'T', '0'+\I, 0

        .section        .debug_str_offsets.dwo,"e",@progbits
.Lstr_offsets\I:
        .long   .Lstr\I-.debug_str.dwo
.Lstr_offsets_end\I:

        .section        .debug_loc.dwo,"e",@progbits
.Lloc\I:
        .byte   3                       # DW_LLE_startx_length
        .uleb128 \I*2+1
        .long   1                
        .short 1                      # Expression size
        .byte   80+\I                   # super-register DW_OP_reg0+\I
        .byte   0                       # DW_LLE_end_of_list
.Lloc_end\I:

        .section        .debug_abbrev.dwo,"e",@progbits
.Labbrev\I:
        .byte   \I*10+1                 # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   37                      # DW_AT_producer
        .byte   8                       # DW_FORM_string
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   \I*10+2                 # Abbreviation Code
        .byte   52                      # DW_TAG_variable
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   2                       # DW_AT_location
        .byte   24                      # DW_FORM_exprloc
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   \I*10+3                 # Abbreviation Code
        .byte   36                      # DW_TAG_base_type
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   62                      # DW_AT_encoding
        .byte   11                      # DW_FORM_data1
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   \I*10+4                 # Abbreviation Code
        .byte   22                      # DW_TAG_typedef
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .uleb128 0x1f02                 # DW_FORM_GNU_str_index
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   \I*10+5                 # Abbreviation Code
        .byte   46                      # DW_TAG_subprogram
        .byte   1                       # DW_CHILDREN_yes
        .byte   17                      # DW_AT_low_pc
        .uleb128 0x1f01                 # DW_FORM_GNU_addr_index
        .byte   18                      # DW_AT_high_pc
        .byte   11                      # DW_FORM_data1
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   \I*10+6                 # Abbreviation Code
        .byte   5                       # DW_TAG_formal_parameter
        .byte   0                       # DW_CHILDREN_no
        .byte   2                       # DW_AT_location
        .byte   23                      # DW_FORM_sec_offset
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)
.Labbrev_end\I:

        .section        .debug_info.dwo,"e",@progbits
.Lcu_begin\I:
        .long   .Ldebug_info_end\I-.Ldebug_info_start\I # Length of Unit
.Ldebug_info_start\I:
        .short  4                       # DWARF version number
        .long   0                       # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .byte   \I*10+1                 # Abbrev DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"    # DW_AT_producer
        .byte   '0'+\I, '.', 'c', 0     # DW_AT_name
        .byte   \I*10+2                 # Abbrev DW_TAG_variable
        .asciz  "A"                     # DW_AT_name
        .long   .Ltypedef\I-.Lcu_begin\I# DW_AT_type
        .byte   2                       # DW_AT_location
        .byte   0xfb                    # DW_OP_GNU_addr_index
        .byte   \I*2
.Ltypedef\I:
        .byte   \I*10+4                 # Abbrev DW_TAG_typedef
        .byte   0                       # DW_AT_name
        .long   .Ltype\I-.Lcu_begin\I   # DW_AT_type
.Ltype\I:
        .byte   \I*10+3                 # Abbrev DW_TAG_base_type
        .asciz  "int"                   # DW_AT_name
        .byte   5                       # DW_AT_encoding
        .byte   4                       # DW_AT_byte_size

        .byte   \I*10+5                 # Abbrev DW_TAG_subprogram
        .uleb128 \I*2+1                 # DW_AT_low_pc
        .byte   1                       # DW_AT_high_pc
        .byte   'F', '0'+\I, 0          # DW_AT_name

        .byte   \I*10+6                 # Abbrev DW_TAG_formal_parameter
        .long   0                       # DW_AT_location
        .asciz  "x"                     # DW_AT_name
        .long   .Ltype\I-.Lcu_begin\I   # DW_AT_type

        .byte   0                       # End Of Children Mark
        .byte   0                       # End Of Children Mark
.Ldebug_info_end\I:
.endr

        .section        .debug_cu_index,"",@progbits
        .short  2                       # DWARF version number
        .short  0                       # Reserved
        .long   4                       # Section count
        .long   4                       # Unit count
        .long   8                       # Slot count

        .quad   0, 1, 2, 3, 0, 0, 0, 0  # Hash table
        .long   1, 2, 3, 4, 0, 0, 0, 0  # Index table

        .long   1                       # DW_SECT_INFO
        .long   3                       # DW_SECT_ABBREV
        .long   5                       # DW_SECT_LOC
        .long   6                       # DW_SECT_STR_OFFSETS

.irpc I,0123
        .long .Lcu_begin\I-.debug_info.dwo
        .long .Labbrev\I-.debug_abbrev.dwo
        .long .Lloc\I-.debug_loc.dwo
        .long .Lstr_offsets\I-.debug_str_offsets.dwo
.endr
.irpc I,0123
        .long .Ldebug_info_end\I-.Lcu_begin\I
        .long .Labbrev_end\I-.Labbrev\I
        .long .Lloc_end\I-.Lloc\I
        .long .Lstr_offsets_end\I-.Lstr_offsets\I
.endr

.endif
