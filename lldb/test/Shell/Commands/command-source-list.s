# REQUIRES: x86

# RUN: split-file %s %t
# RUN: llvm-mc --triple=x86_64-pc-linux -filetype=obj %t/a.s -o %t/a.o
# RUN: %lldb %t/a.o -o "settings set target.source-map . %t" \
# RUN:   -o "settings set interpreter.stop-command-source-on-error false" \
# RUN:   -s %t/commands -o exit 2>&1 | FileCheck %s

#--- commands
# CASE 0: function at the start of the file
source list -n func0
# CHECK-LABEL: source list -n func0
# CHECK-NEXT:  File: file0.c
# CHECK-NEXT:     1    content of file0.c:1
# CHECK-NEXT:     2    content of file0.c:2
# CHECK-NEXT:     3    content of file0.c:3
# CHECK-NEXT:     4    content of file0.c:4
# CHECK-NEXT:     5    content of file0.c:5
# CHECK-NEXT:     6    content of file0.c:6
# CHECK-NEXT:     7    content of file0.c:7
# CHECK-NEXT:     8    content of file0.c:8
# CHECK-NEXT:     9    content of file0.c:9
# CHECK-NEXT:     10   content of file0.c:10

# CASE 1: function in the middle of the file
source list -n func1
# CHECK-NEXT: source list -n func1
# CHECK-NEXT:  File: file0.c
# CHECK-NEXT:     5    content of file0.c:5
# CHECK-NEXT:     6    content of file0.c:6
# CHECK-NEXT:     7    content of file0.c:7
# CHECK-NEXT:     8    content of file0.c:8
# CHECK-NEXT:     9    content of file0.c:9
# CHECK-NEXT:     10   content of file0.c:10
# CHECK-NEXT:     11   content of file0.c:11
# CHECK-NEXT:     12   content of file0.c:12
# CHECK-NEXT:     13   content of file0.c:13
# CHECK-NEXT:     14   content of file0.c:14
# CHECK-NEXT:     15   content of file0.c:15
# CHECK-NEXT:     16   content of file0.c:16
# CHECK-NEXT:     17   content of file0.c:17

# CASE 2: function at the end of the file
source list -n func2
# CHECK-NEXT: source list -n func2
# CHECK-NEXT:  File: file0.c
# CHECK-NEXT:     20   content of file0.c:20
# CHECK-NEXT:     21   content of file0.c:21
# CHECK-NEXT:     22   content of file0.c:22
# CHECK-NEXT:     23   content of file0.c:23
# CHECK-NEXT:     24   content of file0.c:24
# CHECK-NEXT:     25   content of file0.c:25
# CHECK-NEXT:     26   content of file0.c:26
# CHECK-NEXT:     27   content of file0.c:27
# CHECK-NEXT:     28   content of file0.c:28
# CHECK-NEXT:     29   content of file0.c:29
# CHECK-NEXT:     30   content of file0.c:30

# CASE 3: function ends in a different file
source list -n func3
# CHECK-NEXT: source list -n func3
# CHECK-NEXT:  File: file0.c
# CHECK-NEXT:     1    content of file0.c:1
# CHECK-NEXT:     2    content of file0.c:2
# CHECK-NEXT:     3    content of file0.c:3
# CHECK-NEXT:     4    content of file0.c:4
# CHECK-NEXT:     5    content of file0.c:5
# CHECK-NEXT:     6    content of file0.c:6
# CHECK-NEXT:     7    content of file0.c:7
# CHECK-NEXT:     8    content of file0.c:8
# CHECK-NEXT:     9    content of file0.c:9
# CHECK-NEXT:     10   content of file0.c:10

# CASE 4: function has no line entry with line!=0
source list -n func4
# CHECK-LABEL: source list -n func4
# CHECK: error: Could not find line information for function "func4".

# CASE 5: discontinuous function
source list -n func5
# CHECK-LABEL: source list -n func5
# CHECK-NEXT:  File: file0.c
# CHECK-NEXT:     1    content of file0.c:1
# CHECK-NEXT:     2    content of file0.c:2
# CHECK-NEXT:     3    content of file0.c:3
# CHECK-NEXT:     4    content of file0.c:4
# CHECK-NEXT:     5    content of file0.c:5
# CHECK-NEXT:     6    content of file0.c:6
# CHECK-NEXT:     7    content of file0.c:7
# CHECK-NEXT:     8    content of file0.c:8
# CHECK-NEXT:     9    content of file0.c:9
# CHECK-NEXT:     10   content of file0.c:10


#--- a.s
        .file   0 "." "file0.c"
        .file   1 "." "file1.c"
        .text
func0:
        .loc    0 1
        nop
        .loc    0 5
        nop
.Lfunc0_end:

func1:
        .loc    0 10
        nop
        .loc    0 12
        nop
.Lfunc1_end:

func2:
        .loc    0 25
        nop
        .loc    0 30
        nop
.Lfunc2_end:

func3:
        .loc    0 1
        nop
        .loc    0 5
        nop
        .loc    1 5
        nop
.Lfunc3_end:

func4:
        .loc    0 0
        nop
.Lfunc4_end:

func5.__part.1:
        .loc    0 1
        nop
.Lfunc5.__part.1_end:

.Lpadding:
        nop

func5:
        .loc    0 5
        nop
.Lfunc5_end:

.Ltext_end:

        .section        .debug_abbrev,"",@progbits
        .byte   1                               # Abbreviation Code
        .byte   17                              # DW_TAG_compile_unit
        .byte   1                               # DW_CHILDREN_yes
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   37                              # DW_AT_producer
        .byte   8                               # DW_FORM_string
        .byte   19                              # DW_AT_language
        .byte   5                               # DW_FORM_data2
        .byte   17                              # DW_AT_low_pc
        .byte   1                               # DW_FORM_addr
        .byte   18                              # DW_AT_high_pc
        .byte   1                               # DW_FORM_addr
        .byte   16                              # DW_AT_stmt_list
        .byte   23                              # DW_FORM_sec_offset
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   2                               # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   0                               # DW_CHILDREN_no
        .byte   17                              # DW_AT_low_pc
        .byte   1                               # DW_FORM_addr
        .byte   18                              # DW_AT_high_pc
        .byte   1                               # DW_FORM_addr
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   3                               # Abbreviation Code
        .byte   46                              # DW_TAG_subprogram
        .byte   0                               # DW_CHILDREN_no
        .byte   85                              # DW_AT_ranges
        .byte   23                              # DW_FORM_sec_offset
        .byte   3                               # DW_AT_name
        .byte   8                               # DW_FORM_string
        .byte   0                               # EOM(1)
        .byte   0                               # EOM(2)
        .byte   0                               # EOM(3)
        .section        .debug_info,"",@progbits
.Lcu_begin0:
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  5                               # DWARF version number
        .byte   1                               # DWARF Unit Type
        .byte   8                               # Address Size (in bytes)
        .long   .debug_abbrev                   # Offset Into Abbrev. Section
        .byte   1                               # Abbrev DW_TAG_compile_unit
        .asciz  "file0.c"                       # DW_AT_producer
        .asciz  "Hand-written DWARF"            # DW_AT_producer
        .short  29                              # DW_AT_language
        .quad   .text                           # DW_AT_low_pc
        .quad   .Ltext_end                      # DW_AT_high_pc
        .long   .Lline_table_start0             # DW_AT_stmt_list
        .rept 5
        .byte   2                               # Abbrev DW_TAG_subprogram
        .quad   func\+                          # DW_AT_low_pc
        .quad   .Lfunc\+_end                    # DW_AT_high_pc
        .asciz  "func\+"                        # DW_AT_name
        .endr
        .byte   3                               # Abbrev DW_TAG_subprogram
        .long   .Ldebug_ranges0
        .asciz  "func5"                         # DW_AT_name
        .byte   0                               # End Of Children Mark
.Ldebug_info_end0:

        .section        .debug_rnglists,"",@progbits
        .long   .Ldebug_list_header_end0-.Ldebug_list_header_start0 # Length
.Ldebug_list_header_start0:
        .short  5                               # Version
        .byte   8                               # Address size
        .byte   0                               # Segment selector size
        .long   2                               # Offset entry count
.Lrnglists_table_base0:
        .long   .Ldebug_ranges0-.Lrnglists_table_base0
.Ldebug_ranges0:
        .byte   6                               # DW_RLE_start_end
        .quad   func5
        .quad   .Lfunc5_end
        .byte   6                               # DW_RLE_start_end
        .quad   func5.__part.1
        .quad   .Lfunc5.__part.1_end
        .byte   0                               # DW_RLE_end_of_list
.Ldebug_list_header_end0:
        .section        .debug_line,"",@progbits
.Lline_table_start0:

#--- file0.c
content of file0.c:1
content of file0.c:2
content of file0.c:3
content of file0.c:4
content of file0.c:5
content of file0.c:6
content of file0.c:7
content of file0.c:8
content of file0.c:9
content of file0.c:10
content of file0.c:11
content of file0.c:12
content of file0.c:13
content of file0.c:14
content of file0.c:15
content of file0.c:16
content of file0.c:17
content of file0.c:18
content of file0.c:19
content of file0.c:20
content of file0.c:21
content of file0.c:22
content of file0.c:23
content of file0.c:24
content of file0.c:25
content of file0.c:26
content of file0.c:27
content of file0.c:28
content of file0.c:29
content of file0.c:30
#--- file1.c
content of file1.c:1
content of file1.c:2
content of file1.c:3
content of file1.c:4
content of file1.c:5
content of file1.c:6
content of file1.c:7
content of file1.c:8
content of file1.c:9
content of file1.c:10
