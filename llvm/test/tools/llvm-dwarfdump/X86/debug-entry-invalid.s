## Test llvm-dwarfdump detects and reports invalid DWARF format of the file.

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj --defsym=CUEND=1 \
# RUN:   | llvm-dwarfdump - 2>&1 | FileCheck --check-prefix=CUEND %s
# CUEND: warning: DWARF unit from offset 0x0000000c incl. to offset 0x0000002b excl. tries to read DIEs at offset 0x0000002b

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj --defsym=ABBREVSETINVALID=1 \
# RUN:   | llvm-dwarfdump - 2>&1 | FileCheck --check-prefix=ABBREVSETINVALID %s
# ABBREVSETINVALID: warning: DWARF unit at offset 0x0000000c contains invalid abbreviation set offset 0x0

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj --defsym=ABBREVNO=2 \
# RUN:   | llvm-dwarfdump - 2>&1 | FileCheck --check-prefix=ABBREVNO %s
# ABBREVNO: warning: DWARF unit at offset 0x0000000c contains invalid abbreviation 2 at offset 0x00000018, valid abbreviations are 1, 5, 3-4

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj --defsym=FORMNO=0xdead \
# RUN:   | llvm-dwarfdump - 2>&1 | FileCheck --check-prefix=FORMNO %s
# FORMNO: warning: DWARF unit at offset 0x0000000c contains invalid FORM_* 0xdead at offset 0x00000018

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj --defsym=SHORTINITLEN=1 \
# RUN:   | llvm-dwarfdump - 2>&1 | FileCheck --check-prefix=SHORTINITLEN %s
# SHORTINITLEN:      warning: DWARF unit at 0x0000002c cannot be parsed:
# SHORTINITLEN-NEXT: warning: unexpected end of data at offset 0x2d while reading [0x2c, 0x30)

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj --defsym=BADTYPEUNIT=1 \
# RUN:   | llvm-dwarfdump - 2>&1 | FileCheck --check-prefix=BADTYPEUNITBEFORE %s
# BADTYPEUNITBEFORE: warning: DWARF type unit at offset 0x0000002c has its relocated type_offset 0x0000002d pointing inside the header

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj --defsym=BADTYPEUNIT=0x100 \
# RUN:   | llvm-dwarfdump - 2>&1 | FileCheck --check-prefix=BADTYPEUNITAFTER %s
# BADTYPEUNITAFTER: warning: DWARF type unit from offset 0x0000002c incl. to offset 0x00000045 excl. has its relocated type_offset 0x0000012c pointing past the unit end

# RUN: llvm-mc -triple x86_64-pc-linux %s -filetype=obj --defsym=TOOLONG=1 \
# RUN:   | llvm-dwarfdump - 2>&1 | FileCheck --check-prefix=TOOLONG %s
# TOOLONG: warning: DWARF unit from offset 0x0000000c incl. to offset 0x0000002d excl. extends past section size 0x0000002c

        .section .debug_abbrev,"",@progbits
.ifndef ABBREVSETINVALID
        .uleb128 1                      # Abbreviation Code
        .uleb128 17                     # DW_TAG_compile_unit
        .uleb128 1                      # DW_CHILDREN_yes
        .uleb128 37                     # DW_AT_producer
.ifndef FORMNO
        .uleb128 8                      # DW_FORM_string
.else
        .uleb128 FORMNO       
.endif
        .uleb128 0                      # end abbrev 1 DW_AT_*
        .uleb128 0                      # end abbrev 1 DW_FORM_*
        .uleb128 5                      # Abbreviation Code
        .uleb128 10                     # DW_TAG_label
        .uleb128 0                      # DW_CHILDREN_no
        .uleb128 0                      # end abbrev 4 DW_AT_*
        .uleb128 0                      # end abbrev 4 DW_FORM_*
        .uleb128 3                      # Abbreviation Code
        .uleb128 10                     # DW_TAG_label
        .uleb128 0                      # DW_CHILDREN_no
        .uleb128 0                      # end abbrev 3 DW_AT_*
        .uleb128 0                      # end abbrev 3 DW_FORM_*
        .uleb128 4                      # Abbreviation Code
        .uleb128 10                     # DW_TAG_label
        .uleb128 0                      # DW_CHILDREN_no
        .uleb128 0                      # end abbrev 4 DW_AT_*
        .uleb128 0                      # end abbrev 4 DW_FORM_*
        .uleb128 0                      # end abbrevs section
.endif

        .section .debug_info,"",@progbits
## The first CU is here to shift the next CU being really tested to non-zero CU
## offset to check more for error messages.
        .long    .Lcu_endp-.Lcu_startp  # Length of Unit
.Lcu_startp:
        .short   4                      # DWARF version number
        .long    .debug_abbrev          # Offset Into Abbrev. Section
        .byte    8                      # Address Size (in bytes)
        .uleb128 0                      # End Of Children Mark
.Lcu_endp:

.ifndef TOOLONG
.equ TOOLONG, 0
.endif
        .long    .Lcu_end0-.Lcu_start0 + TOOLONG  # Length of Unit
.Lcu_start0:
        .short   4                      # DWARF version number
        .long    .debug_abbrev          # Offset Into Abbrev. Section
        .byte    8                      # Address Size (in bytes)
.ifndef ABBREVNO
        .uleb128 1                      # Abbrev [1] DW_TAG_compile_unit
.else
        .uleb128 ABBREVNO       
.endif
        .asciz  "hand-written DWARF"    # DW_AT_producer
.ifndef CUEND
        .uleb128 0                      # End Of Children Mark
.endif
.Lcu_end0:

.ifdef SHORTINITLEN
        .byte    0x55                   # Too short Length of Unit
.endif
.ifdef BADTYPEUNIT
        .long    .Lcu_end1-.Lcu_start1  # Length of Unit
.Lcu_start1:    
        .short   5                      # DWARF version number
        .byte    2                      # DW_UT_type
        .byte    8                      # Address Size (in bytes)
        .long    .debug_abbrev          # Offset Into Abbrev. Section
        .quad    0xbaddefacedfacade     # Type Signature
        .long    BADTYPEUNIT            # Type DIE Offset
        .uleb128 0                      # End Of Children Mark
.Lcu_end1:
.endif
