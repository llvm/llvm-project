# REQUIRES: x86

# RUN: split-file %s %t
# RUN: llvm-mc -triple x86_64-pc-linux -filetype=obj %t/input.s -o %t/input.o
# RUN: %lldb %t/input.o -o "command script import %t/script.py" -o exit | FileCheck %s

# CHECK: Found 1 function(s).
# CHECK: foo: [input.o[0x0-0xe), input.o[0x14-0x1c)]
# CHECK-NEXT: input.o[0x0]: callq  0xe
# CHECK-NEXT: input.o[0x5]: jmp    0x1b
# CHECK-NEXT: input.o[0x7]: cmpl   $0x0, %edi
# CHECK-NEXT: input.o[0xa]: je     0x14
# CHECK-NEXT: input.o[0xc]: jmp    0x0
# CHECK-EMPTY:
# CHECK-NEXT: input.o[0x14]: callq  0x19
# CHECK-NEXT: input.o[0x19]: jmp    0x1b
# CHECK-NEXT: input.o[0x1b]: retq
## Testing the GetRangeIndexForBlockAddress API. "ffffffff" indicates that
## the address does not belong to any range.
# CHECK-NEXT: offset 0x00 => index 0
# CHECK-NEXT: offset 0x0c => index 0
# CHECK-NEXT: offset 0x0e => index ffffffff
# CHECK-NEXT: offset 0x13 => index ffffffff
# CHECK-NEXT: offset 0x14 => index 1
# CHECK-NEXT: offset 0x1b => index 1
# CHECK-NEXT: offset 0x1c => index ffffffff


#--- script.py
import lldb

def __lldb_init_module(debugger, internal_dict):
  target = debugger.GetSelectedTarget()
  sym_ctxs = target.FindFunctions("foo")
  print(f"Found {len(sym_ctxs)} function(s).")
  for ctx in sym_ctxs:
    fn = ctx.function
    print(f"{fn.name}: {fn.GetRanges()}")
    print(fn.GetInstructions(target))
    text = fn.addr.section
    for offset in [0x00, 0x0c, 0x0e, 0x13, 0x14, 0x1b, 0x1c]:
      idx = fn.block.GetRangeIndexForBlockAddress(lldb.SBAddress(text, offset))
      print(f"offset 0x{offset:02x} => index {idx:x}")

#--- input.s
# An example of a function which has been split into two parts. Roughly
# corresponds to this C code.
# int baz();
# int bar() { return 47; }
# int foo(int flag) { return flag ? bar() : baz(); }
# The function bar has been placed "in the middle" of foo.

        .text

        .type   foo,@function
foo.__part.1:
        .cfi_startproc
        callq   bar
        jmp     foo.__part.3
.Lfoo.__part.1_end:
        .size   foo.__part.1, .Lfoo.__part.1_end-foo.__part.1
        .cfi_endproc

foo:
        .cfi_startproc
        cmpl    $0, %edi
        je      foo.__part.2
        jmp     foo.__part.1
        .cfi_endproc
.Lfoo_end:
        .size   foo, .Lfoo_end-foo

bar:
        .cfi_startproc
        movl    $47, %eax
        retq
        .cfi_endproc
.Lbar_end:
        .size   bar, .Lbar_end-bar

foo.__part.2:
        .cfi_startproc
        callq   baz
        jmp     foo.__part.3
.Lfoo.__part.2_end:
        .size   foo.__part.2, .Lfoo.__part.2_end-foo.__part.2
        .cfi_endproc

foo.__part.3:
        .cfi_startproc
        retq
.Lfoo.__part.3_end:
        .size   foo.__part.3, .Lfoo.__part.3_end-foo.__part.3
        .cfi_endproc


        .section        .debug_abbrev,"",@progbits
        .byte   1                               # Abbreviation Code
        .byte   17                              # DW_TAG_compile_unit
        .byte   1                               # DW_CHILDREN_yes
        .byte   37                              # DW_AT_producer
        .byte   8                               # DW_FORM_string
        .byte   19                              # DW_AT_language
        .byte   5                               # DW_FORM_data2
        .byte   17                              # DW_AT_low_pc
        .byte   1                               # DW_FORM_addr
        .byte   85                              # DW_AT_ranges
        .byte   35                              # DW_FORM_rnglistx
        .byte   116                             # DW_AT_rnglists_base
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
        .byte   35                              # DW_FORM_rnglistx
        .byte   64                              # DW_AT_frame_base
        .byte   24                              # DW_FORM_exprloc
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
        .byte   1                               # Abbrev [1] DW_TAG_compile_unit
        .asciz  "Hand-written DWARF"            # DW_AT_producer
        .short  29                              # DW_AT_language
        .quad   0                               # DW_AT_low_pc
        .byte   1                               # DW_AT_ranges
        .long   .Lrnglists_table_base0          # DW_AT_rnglists_base
        .byte   2                               # Abbrev [2] DW_TAG_subprogram
        .quad   bar                             # DW_AT_low_pc
        .quad   .Lbar_end                       # DW_AT_high_pc
        .asciz  "bar"                           # DW_AT_name
        .byte   3                               # Abbrev [3] DW_TAG_subprogram
        .byte   0                               # DW_AT_ranges
        .byte   1                               # DW_AT_frame_base
        .byte   86
        .asciz  "foo"                           # DW_AT_name
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
        .long   .Ldebug_ranges1-.Lrnglists_table_base0
.Ldebug_ranges0:
        .byte   6                               # DW_RLE_start_end
        .quad   foo
        .quad   .Lfoo_end
        .byte   6                               # DW_RLE_start_end
        .quad   foo.__part.1
        .quad   .Lfoo.__part.1_end
        .byte   6                               # DW_RLE_start_end
        .quad   foo.__part.2
        .quad   .Lfoo.__part.2_end
        .byte   6                               # DW_RLE_start_end
        .quad   foo.__part.3
        .quad   .Lfoo.__part.3_end
        .byte   0                               # DW_RLE_end_of_list
.Ldebug_ranges1:
        .byte   6                               # DW_RLE_start_end
        .quad   bar
        .quad   .Lbar_end
        .byte   6                               # DW_RLE_start_end
        .quad   foo.__part.1
        .quad   .Lfoo.__part.1_end
        .byte   6                               # DW_RLE_start_end
        .quad   foo.__part.2
        .quad   .Lfoo.__part.2_end
        .byte   6                               # DW_RLE_start_end
        .quad   foo.__part.3
        .quad   .Lfoo.__part.3_end
        .byte   6                               # DW_RLE_start_end
        .quad   foo
        .quad   .Lfoo_end
        .byte   0                               # DW_RLE_end_of_list
.Ldebug_list_header_end0:

        .section        ".note.GNU-stack","",@progbits
