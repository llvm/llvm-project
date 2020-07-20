! RUN: llvm-mc %s -arch=sparcv9 --position-independent -filetype=obj | llvm-readobj -r - | FileCheck --check-prefix=PIC %s
! RUN: llvm-mc %s -arch=sparcv9 -filetype=obj | llvm-readobj -r - | FileCheck --check-prefix=NOPIC %s


! PIC:      Relocations [
! PIC-NOT:    0x{{[0-9,A-F]+}} R_SPARC_WPLT30 .text 0xC
! PIC:        0x{{[0-9,A-F]+}} R_SPARC_PC22 _GLOBAL_OFFSET_TABLE_ 0x4
! PIC-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_PC10 _GLOBAL_OFFSET_TABLE_ 0x8
! PIC-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_PC22 _GLOBAL_OFFSET_TABLE_ 0x0
! PIC-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_PC10 _GLOBAL_OFFSET_TABLE_ 0x0
! PIC-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_GOT22 AGlobalVar 0x0
! PIC-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_GOT10 AGlobalVar 0x0
! PIC-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_GOT22 AGlobalVar 0x0
! PIC-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_GOT10 AGlobalVar 0x0
! PIC-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_GOT22 .LC0 0x0
! PIC-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_GOT10 .LC0 0x0
! PIC-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_WPLT30 bar 0x0
! PIC:        0x{{[0-9,A-F]+}} R_SPARC_GOT13 value 0x0
! PIC:      ]

! NOPIC:      Relocations [
! NOPIC-NOT:    0x{{[0-9,A-F]+}} R_SPARC_WPLT30 .text 0xC
! NOPIC:        0x{{[0-9,A-F]+}} R_SPARC_HI22 _GLOBAL_OFFSET_TABLE_ 0x4
! NOPIC-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_LO10 _GLOBAL_OFFSET_TABLE_ 0x8
! NOPIC-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_HI22 _GLOBAL_OFFSET_TABLE_ 0x0
! NOPIC-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_LO10 _GLOBAL_OFFSET_TABLE_ 0x0
! NOPIC-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_HI22 AGlobalVar 0x0
! NOPIC-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_LO10 AGlobalVar 0x0
! NOPIC-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_HI22 AGlobalVar 0x0
! NOPIC-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_LO10 AGlobalVar 0x0
! NOPIC-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_HI22 .rodata 0x0
! NOPIC-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_LO10 .rodata 0x0
! NOPIC-NEXT:   0x{{[0-9,A-F]+}} R_SPARC_WDISP30 bar 0x0
! NOPIC:        0x{{[0-9,A-F]+}} R_SPARC_13 value 0x0
! NOPIC:      ]

        .section        ".rodata"
        .align 8
.LC0:
        .asciz   "string"
        .section ".text"
        .text
        .globl  foo
        .align  4
        .type   foo,@function
foo:
        .cfi_startproc
        save %sp, -176, %sp
        .cfi_def_cfa_register %fp
        .cfi_window_save
        .cfi_register 15, 31
.Ltmp4:
        call .Ltmp5
.Ltmp6:
        sethi %hi(_GLOBAL_OFFSET_TABLE_+(.Ltmp6-.Ltmp4)), %i1
.Ltmp5:
        or %i1, %lo(_GLOBAL_OFFSET_TABLE_+(.Ltmp5-.Ltmp4)), %i1
        set _GLOBAL_OFFSET_TABLE_, %i1
        add %i1, %o7, %i1
        sethi %hi(AGlobalVar), %i2
        add %i2, %lo(AGlobalVar), %i2
        set AGlobalVar, %i2
        ldx [%i1+%i2], %i3
        ldx [%i3], %i3
        sethi %hi(.LC0), %i2
        add %i2, %lo(.LC0), %i2
        ldx [%i1+%i2], %i4
        call bar
        add %i0, %i1, %o0
        ret
        restore %g0, %o0, %o0
.Ltmp7:
        .size   foo, .Ltmp7-foo
        .cfi_endproc

        .type   AGlobalVar,@object      ! @AGlobalVar
        .section        .bss,#alloc,#write
        .globl  AGlobalVar
        .align  8
AGlobalVar:
        .xword  0                       ! 0x0
        .size   AGlobalVar, 8

        .section ".text"
        .text
        .globl  pic13
        .align  4
        .type   pic13,@function
pic13:
        save %sp, -128, %sp
.Ltmp0:
        call .Ltmp1
.Ltmp2:
        sethi %hi(_GLOBAL_OFFSET_TABLE_+(.Ltmp2-.Ltmp0)), %i0
.Ltmp1:
        or %i0, %lo(_GLOBAL_OFFSET_TABLE_+(.Ltmp1-.Ltmp0)), %i0
        add %i0, %o7, %i0
        ldx [%i0+value], %i0
        ld [%i0], %i0
        ret
        restore
.Lfunc_end0:
        .size pic13, .Lfunc_end0-pic13
