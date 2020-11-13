# REQUIRES: x86

# RUN: llvm-mc --filetype=obj --triple x86_64-pc-linux %s -o %t --defsym MAIN=0
# RUN: llvm-mc --filetype=obj --triple x86_64-pc-linux %s -o %t.dwp --defsym DWP=0
# RUN: %lldb %t -o "type lookup ENUM0" -o "target variable A" -b | FileCheck %s
# RUN: lldb-test symbols %t | FileCheck %s --check-prefix=SYMBOLS

# CHECK-LABEL: type lookup ENUM0
# CHECK-NEXT: enum ENUM0 {
# CHECK-NEXT:   case0
# CHECK-NEXT: }

# CHECK-LABEL: target variable A
# CHECK: (ENUM0) A = case0
# CHECK: (ENUM1) A = case0

# Make sure each entity is present in the index only once.
# SYMBOLS:      Globals and statics:
# SYMBOLS-DAG: INFO/00000023 "A"
# SYMBOLS-DAG: INFO/0000005a "A"
# SYMBOLS-EMPTY:

# SYMBOLS: Types:
# SYMBOLS-DAG: TYPE/00000018 "ENUM0"
# SYMBOLS-DAG: TYPE/0000004d "ENUM1"
# SYMBOLS-DAG: TYPE/0000002d "int"
# SYMBOLS-DAG: TYPE/00000062 "int"
# SYMBOLS-EMPTY:

.ifdef MAIN
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
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)

.irpc I,01
        .data
A\I:
        .long 0

        .section        .debug_info,"",@progbits
.Lcu_begin\I:
        .long   .Ldebug_info_end\I-.Ldebug_info_start\I # Length of Unit
.Ldebug_info_start\I:
        .short  4                       # DWARF version number
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .byte   1                       # Abbrev [1] DW_TAG_compile_unit
        .asciz  "A.dwo"                 # DW_AT_GNU_dwo_name
        .quad   \I                      # DW_AT_GNU_dwo_id
        .long   .debug_addr             # DW_AT_GNU_addr_base
.Ldebug_info_end\I:

        .section        .debug_addr,"",@progbits
        .quad   A\I

.endr
.endif

.ifdef DWP
.irpc I,01
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
        .byte   4                       # DW_TAG_enumeration_type
        .byte   0                       # DW_CHILDREN_no
        .byte   60                      # DW_AT_declaration
        .byte   25                      # DW_FORM_flag_present
        .byte   105                     # DW_AT_signature
        .byte   32                      # DW_FORM_ref_sig8
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   \I*10+4                 # Abbreviation Code
        .byte   4                       # DW_TAG_enumeration_type
        .byte   1                       # DW_CHILDREN_yes
        .byte   73                      # DW_AT_type
        .byte   19                      # DW_FORM_ref4
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   11                      # DW_AT_byte_size
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   \I*10+5                 # Abbreviation Code
        .byte   40                      # DW_TAG_enumerator
        .byte   0                       # DW_CHILDREN_no
        .byte   3                       # DW_AT_name
        .byte   8                       # DW_FORM_string
        .byte   28                      # DW_AT_const_value
        .byte   15                      # DW_FORM_udata
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   \I*10+6                 # Abbreviation Code
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
        .byte   \I*10+7                 # Abbreviation Code
        .byte   65                      # DW_TAG_type_unit
        .byte   1                       # DW_CHILDREN_yes
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
        .long   .Lenum_ref\I-.Lcu_begin\I# DW_AT_type
        .byte   2                       # DW_AT_location
        .byte   0xfb                    # DW_OP_GNU_addr_index
        .byte   \I

.Lenum_ref\I:
        .byte   \I*10+3                 # Abbrev DW_TAG_enumeration_type
        .quad   \I                      # DW_AT_signature

        .byte   0                       # End Of Children Mark
.Ldebug_info_end\I:

        .section        .debug_types.dwo,"e",@progbits
.Ltu_begin\I:
        .long   .Ltype_info_end\I-.Ltype_info_start\I # Length of Unit
.Ltype_info_start\I:
        .short  4                       # DWARF version number
        .long   0                       # Offset Into Abbrev. Section
        .byte   8                       # Address Size (in bytes)
        .quad   \I                      # Type Signature
        .long   .Lenum\I-.Ltu_begin\I   # Type DIE Offset
        .byte   \I*10+7                 # Abbrev DW_TAG_type_unit

.Lenum\I:
        .byte   \I*10+4                 # Abbrev DW_TAG_enumeration_type
        .long   .Lint\I-.Ltu_begin\I    # DW_AT_type
        .byte   'E', 'N', 'U', 'M', '0'+\I, 0# DW_AT_name
        .byte   4                       # DW_AT_byte_size

        .byte   \I*10+5                 # Abbrev DW_TAG_enumerator
        .asciz  "case0"                 # DW_AT_name
        .byte   0                       # DW_AT_const_value
        .byte   0                       # End Of Children Mark

.Lint\I:
        .byte   \I*10+6                 # Abbrev DW_TAG_base_type
        .asciz  "int"                   # DW_AT_name
        .byte   7                       # DW_AT_encoding
        .byte   4                       # DW_AT_byte_size

        .byte   0                       # End Of Children Mark
.Ltype_info_end\I:
.endr

.macro index dw_sect, section, contrib, contrib_end
        .short  2                       # DWARF version number
        .short  0                       # Reserved
        .long   2                       # Section count
        .long   2                       # Unit count
        .long   4                       # Slot count

        .quad   0, 1, 0, 0              # Hash table
        .long   1, 2, 0, 0              # Index table

        .long   \dw_sect
        .long   3                       # DW_SECT_ABBREV

.irpc I,01
        .long   \contrib\I-\section
        .long    .Labbrev\I-.debug_abbrev.dwo
.endr
.irpc I,01
        .long   \contrib_end\I-\contrib\I
        .long   .Labbrev_end\I-.Labbrev\I
.endr
.endmacro

        .section        .debug_cu_index,"",@progbits
        index 1, .debug_info.dwo, .Lcu_begin, .Ldebug_info_end

        .section        .debug_tu_index,"",@progbits
        index 2, .debug_types.dwo, .Ltu_begin, .Ltype_info_end
.endif
