# Test that exprloc-valued subrange bounds are evaluated instead of being
# treated as static constants.

# REQUIRES: lld, native, target-x86_64, system-linux

# RUN: llvm-mc -triple x86_64-unknown-linux-gnu %s -filetype=obj -o %t.o
# RUN: ld.lld %t.o -o %t -e main
# RUN: %lldb %t -b -o "breakpoint set -n after_init" -o run \
# RUN:   -o "frame variable --show-all-children array" -o exit | FileCheck %s

# CHECK-LABEL: frame variable --show-all-children array
# CHECK: (unsigned int[]) array = ([0] = 1, [1] = 3, [2] = 170, [3] = 187, [4] = 204, [5] = 221)

        .text
        .globl          main
        .type           main,@function
main:
.Lfunc_begin:
        .cfi_startproc
        pushq           %rbp
        .cfi_def_cfa_offset 16
        .cfi_offset     %rbp, -16
        movq            %rsp, %rbp
        .cfi_def_cfa_register %rbp
        subq            $64, %rsp
        movq            $5, -8(%rbp)
        movl            $1, -48(%rbp)
        movl            $3, -44(%rbp)
        movl            $170, -40(%rbp)
        movl            $187, -36(%rbp)
        movl            $204, -32(%rbp)
        movl            $221, -28(%rbp)
        .globl          after_init
after_init:
        nop
        xorl            %eax, %eax
        addq            $64, %rsp
        popq            %rbp
        .cfi_def_cfa    %rsp, 8
        retq
.Lfunc_end:
        .cfi_endproc
        .size           main, .Lfunc_end-main

        .section        .debug_info,"",@progbits
.Lcu_begin:
        .4byte          .Lcu_end-.Lcu_start
.Lcu_start:
        .2byte          4                       # DWARF version
        .4byte          .Labbrev_begin
        .byte           8                       # Address size

        .uleb128        1                       # DW_TAG_compile_unit
        .ascii          "exprloc-array.c\0"     # DW_AT_name
        .ascii          "hand-written DWARF\0"  # DW_AT_producer
        .byte           0x0c                    # DW_AT_language
        .quad           .Lfunc_begin            # DW_AT_low_pc
        .4byte          .Lfunc_end-.Lfunc_begin # DW_AT_high_pc

        .uleb128        2                       # DW_TAG_subprogram
        .ascii          "main\0"                # DW_AT_name
        .quad           .Lfunc_begin            # DW_AT_low_pc
        .4byte          .Lfunc_end-.Lfunc_begin # DW_AT_high_pc
        .byte           .Lframe_base_end-.Lframe_base_begin # DW_AT_frame_base exprloc size
.Lframe_base_begin:
        .byte           0x56                    # DW_OP_reg6
.Lframe_base_end:

        .uleb128        3                       # DW_TAG_variable
        .ascii          "array\0"               # DW_AT_name
        .4byte          .Larray_type-.Lcu_begin # DW_AT_type
        .uleb128        .Larray_loc_end-.Larray_loc_begin # DW_AT_location exprloc size
.Larray_loc_begin:
        .byte           0x91                    # DW_OP_fbreg
        .sleb128        -48                     # offset
.Larray_loc_end:

.Lu32_type:
        .uleb128        4                       # DW_TAG_base_type
        .ascii          "unsigned int\0"        # DW_AT_name
        .byte           4                       # DW_AT_byte_size
        .byte           7                       # DW_ATE_unsigned

.Larray_type:
        .uleb128        5                       # DW_TAG_array_type
        .4byte          .Lu32_type-.Lcu_begin   # DW_AT_type
        .uleb128        6                       # DW_TAG_subrange_type
        .4byte          .Lu32_type-.Lcu_begin   # DW_AT_type
        .byte           0                       # DW_AT_lower_bound
        .uleb128        .Lupper_bound_end-.Lupper_bound_begin # DW_AT_upper_bound exprloc size
.Lupper_bound_begin:
        .byte           0x91                    # DW_OP_fbreg
        .sleb128        -8                      # offset
        .byte           0x06                    # DW_OP_deref
.Lupper_bound_end:
        .byte           0                       # End of array type children

        .byte           0                       # End of subprogram children
        .byte           0                       # End of compile unit children
.Lcu_end:

        .section        .debug_abbrev,"",@progbits
.Labbrev_begin:
        .uleb128        1                       # Abbreviation code
        .uleb128        0x11                    # DW_TAG_compile_unit
        .byte           1                       # DW_CHILDREN_yes
        .uleb128        0x03                    # DW_AT_name
        .uleb128        0x08                    # DW_FORM_string
        .uleb128        0x25                    # DW_AT_producer
        .uleb128        0x08                    # DW_FORM_string
        .uleb128        0x13                    # DW_AT_language
        .uleb128        0x0b                    # DW_FORM_data1
        .uleb128        0x11                    # DW_AT_low_pc
        .uleb128        0x01                    # DW_FORM_addr
        .uleb128        0x12                    # DW_AT_high_pc
        .uleb128        0x06                    # DW_FORM_data4
        .byte           0
        .byte           0

        .uleb128        2                       # Abbreviation code
        .uleb128        0x2e                    # DW_TAG_subprogram
        .byte           1                       # DW_CHILDREN_yes
        .uleb128        0x03                    # DW_AT_name
        .uleb128        0x08                    # DW_FORM_string
        .uleb128        0x11                    # DW_AT_low_pc
        .uleb128        0x01                    # DW_FORM_addr
        .uleb128        0x12                    # DW_AT_high_pc
        .uleb128        0x06                    # DW_FORM_data4
        .uleb128        0x40                    # DW_AT_frame_base
        .uleb128        0x18                    # DW_FORM_exprloc
        .byte           0
        .byte           0

        .uleb128        3                       # Abbreviation code
        .uleb128        0x34                    # DW_TAG_variable
        .byte           0                       # DW_CHILDREN_no
        .uleb128        0x03                    # DW_AT_name
        .uleb128        0x08                    # DW_FORM_string
        .uleb128        0x49                    # DW_AT_type
        .uleb128        0x13                    # DW_FORM_ref4
        .uleb128        0x02                    # DW_AT_location
        .uleb128        0x18                    # DW_FORM_exprloc
        .byte           0
        .byte           0

        .uleb128        4                       # Abbreviation code
        .uleb128        0x24                    # DW_TAG_base_type
        .byte           0                       # DW_CHILDREN_no
        .uleb128        0x03                    # DW_AT_name
        .uleb128        0x08                    # DW_FORM_string
        .uleb128        0x0b                    # DW_AT_byte_size
        .uleb128        0x0b                    # DW_FORM_data1
        .uleb128        0x3e                    # DW_AT_encoding
        .uleb128        0x0b                    # DW_FORM_data1
        .byte           0
        .byte           0

        .uleb128        5                       # Abbreviation code
        .uleb128        0x01                    # DW_TAG_array_type
        .byte           1                       # DW_CHILDREN_yes
        .uleb128        0x49                    # DW_AT_type
        .uleb128        0x13                    # DW_FORM_ref4
        .byte           0
        .byte           0

        .uleb128        6                       # Abbreviation code
        .uleb128        0x21                    # DW_TAG_subrange_type
        .byte           0                       # DW_CHILDREN_no
        .uleb128        0x49                    # DW_AT_type
        .uleb128        0x13                    # DW_FORM_ref4
        .uleb128        0x22                    # DW_AT_lower_bound
        .uleb128        0x0b                    # DW_FORM_data1
        .uleb128        0x2f                    # DW_AT_upper_bound
        .uleb128        0x18                    # DW_FORM_exprloc
        .byte           0
        .byte           0

        .byte           0

        .section        .note.GNU-stack,"",@progbits
