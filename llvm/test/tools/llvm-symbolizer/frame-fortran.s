// REQUIRES: x86-registered-target

// RUN: llvm-mc -filetype=obj -triple=x86_64-linux-gnu -o %t.o %s
// RUN: echo 'FRAME %t.o 0' | llvm-symbolizer | FileCheck %s

// Generated with:
//
// function foo(array)
//   integer, intent(in), dimension(2:3) :: array
// end function foo
//
// gcc -x f95 -g -S

// CHECK: foo
// CHECK-NEXT: array
// CHECK-NEXT: /home/ubuntu{{/|\\}}.{{/|\\}}example.cpp:1
// CHECK-NEXT: -24 8 ??

        .file   "example.cpp"
        .text
.Ltext0:
        .globl  foo_
        .type   foo_, @function
foo_:
.LFB0:
        .file 1 "./example.cpp"
        .loc 1 1 0
        .cfi_startproc
        pushq   %rbp
        .cfi_def_cfa_offset 16
        .cfi_offset 6, -16
        movq    %rsp, %rbp
        .cfi_def_cfa_register 6
        movq    %rdi, -8(%rbp)
        .loc 1 3 0
        nop
        popq    %rbp
        .cfi_def_cfa 7, 8
        ret
        .cfi_endproc
.LFE0:
        .size   foo_, .-foo_
.Letext0:
        .section        .debug_info,"",@progbits
.Ldebug_info0:
        .long   0x86
        .value  0x4
        .long   .Ldebug_abbrev0
        .byte   0x8
        .uleb128 0x1
        .long   .LASF3
        .byte   0xe
        .byte   0x2
        .long   .LASF4
        .long   .LASF5
        .quad   .Ltext0
        .quad   .Letext0-.Ltext0
        .long   .Ldebug_line0
        .uleb128 0x2
        .string "foo"
        .byte   0x1
        .byte   0x1
        .long   .LASF6
        .long   0x63
        .quad   .LFB0
        .quad   .LFE0-.LFB0
        .uleb128 0x1
        .byte   0x9c
        .long   0x63
        .uleb128 0x3
        .long   .LASF7
        .byte   0x1
        .byte   0x1
        .long   0x6a
        .uleb128 0x3
        .byte   0x91
        .sleb128 -24
        .byte   0x6
        .byte   0
        .uleb128 0x4
        .byte   0x4
        .byte   0x4
        .long   .LASF0
        .uleb128 0x5
        .long   0x82
        .long   0x7b
        .uleb128 0x6
        .long   0x7b
        .sleb128 2
        .sleb128 3
        .byte   0
        .uleb128 0x4
        .byte   0x8
        .byte   0x5
        .long   .LASF1
        .uleb128 0x4
        .byte   0x4
        .byte   0x5
        .long   .LASF2
        .byte   0
        .section        .debug_abbrev,"",@progbits
.Ldebug_abbrev0:
        .uleb128 0x1
        .uleb128 0x11
        .byte   0x1
        .uleb128 0x25
        .uleb128 0xe
        .uleb128 0x13
        .uleb128 0xb
        .uleb128 0x42
        .uleb128 0xb
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x1b
        .uleb128 0xe
        .uleb128 0x11
        .uleb128 0x1
        .uleb128 0x12
        .uleb128 0x7
        .uleb128 0x10
        .uleb128 0x17
        .byte   0
        .byte   0
        .uleb128 0x2
        .uleb128 0x2e
        .byte   0x1
        .uleb128 0x3f
        .uleb128 0x19
        .uleb128 0x3
        .uleb128 0x8
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x6e
        .uleb128 0xe
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x11
        .uleb128 0x1
        .uleb128 0x12
        .uleb128 0x7
        .uleb128 0x40
        .uleb128 0x18
        .uleb128 0x2117
        .uleb128 0x19
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x3
        .uleb128 0x5
        .byte   0
        .uleb128 0x3
        .uleb128 0xe
        .uleb128 0x3a
        .uleb128 0xb
        .uleb128 0x3b
        .uleb128 0xb
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x2
        .uleb128 0x18
        .byte   0
        .byte   0
        .uleb128 0x4
        .uleb128 0x24
        .byte   0
        .uleb128 0xb
        .uleb128 0xb
        .uleb128 0x3e
        .uleb128 0xb
        .uleb128 0x3
        .uleb128 0xe
        .byte   0
        .byte   0
        .uleb128 0x5
        .uleb128 0x1
        .byte   0x1
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x1
        .uleb128 0x13
        .byte   0
        .byte   0
        .uleb128 0x6
        .uleb128 0x21
        .byte   0
        .uleb128 0x49
        .uleb128 0x13
        .uleb128 0x22
        .uleb128 0xd
        .uleb128 0x2f
        .uleb128 0xd
        .byte   0
        .byte   0
        .byte   0
        .section        .debug_aranges,"",@progbits
        .long   0x2c
        .value  0x2
        .long   .Ldebug_info0
        .byte   0x8
        .byte   0
        .value  0
        .value  0
        .quad   .Ltext0
        .quad   .Letext0-.Ltext0
        .quad   0
        .quad   0
        .section        .debug_line,"",@progbits
.Ldebug_line0:
        .section        .debug_str,"MS",@progbits,1
.LASF5:
        .string "/home/ubuntu"
.LASF7:
        .string "array"
.LASF0:
        .string "real(kind=4)"
.LASF2:
        .string "integer(kind=4)"
.LASF6:
        .string "foo_"
.LASF1:
        .string "integer(kind=8)"
.LASF3:
        .string "GNU Fortran2008 9.1.0 -mtune=generic -march=x86-64 -g -g -fintrinsic-modules-path /opt/compiler-explorer/gcc-9.1.0/bin/../lib/gcc/x86_64-linux-gnu/9.1.0/finclude"
.LASF4:
        .string "./example.cpp"
        .ident  "GCC: (Compiler-Explorer-Build) 9.1.0"
        .section        .note.GNU-stack,"",@progbits
