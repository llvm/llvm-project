; RUN: llvm-mc -triple riscv32-apple-macho -filetype=obj %s -o %t.o
; RUN: llvm-objdump -dr %t.o | FileCheck %s
; RUN: llvm-otool -Vtr %t.o | FileCheck %s --check-prefix=OTOOL

; hi20/low12 offsets of var are relative to the PC at Ltmp0
; (computed by auipc). Final value in a0:
;
;      a0 = PC_at_Ltmp0 + hi20_offset(var) + lo12_offset(var)
;
; The -0x4 offset of the auipc w.r.t the addi is inlined in the lw
; instruction.
;
; CHECK-LABEL: 00000000 <ltmp0>:
; CHECK-NEXT: 0: 00000517            auipc   a0, 0
; CHECK-NEXT:                        00000000:  RISCV_RELOC_HI20(pcrel)      var
; CHECK-NEXT: 4: ffc50513            addi    a0, a0, -0x4
; CHECK-NEXT:                        00000004:  RISCV_RELOC_LO12(pcrel)      var
Ltmp0:
        auipc a0, %pcrel_hi(var)
        addi a0, a0, %pcrel_lo(Ltmp0)

; hi20/low12 offsets of var+16 are relative to the PC at Ltmp1
; (computed by auipc). The relative offset is computed for an
; additional offset of 16 bytes from var.  Final value in a0:
;
;      a0 = PC_at_Ltmp1 + hi20_offset(var) + lo12_offset(var)
;
; The -0x4 offset of the auipc w.r.t the addi is inlined in the lw
; instruction.
;
; CHECK-NEXT: 8: 00000517            auipc   a0, 0
; CHECK-NEXT:                        00000008:  RISCV_RELOC_ADDEND   0x10
; CHECK-NEXT:                        00000008:  RISCV_RELOC_HI20(pcrel)      var
; CHECK-NEXT: c: ffc50513 	        addi a0, a0, -0x4
; CHECK-NEXT:                        0000000c: RISCV_RELOC_ADDEND 0x10
; CHECK-NEXT:                        0000000c: RISCV_RELOC_LO12(pcrel) var
Ltmp1:
        auipc a0, %pcrel_hi(var+16)
        addi a0, a0, %pcrel_lo(Ltmp1)

; This is a GOT-based address loading sequence, used for accessing
; global variables in shared libraries or position-independent
; executables (PIE). Uses indirection through the Global Offset
; Table (GOT).
;
; a0 = PC at Ltmp2 + high 20 bits of the offset of GOT entry for var (offset is relative to the PC)
;   auipc a0, %got_pcrel_hi(var)
; a0 = load from [a0 (computed previously with auipc) + the low 12 bits of offset of the GOT entry for var (relative to PC at Ltmp2)]
;   lw a0, %pcrel_lo(Ltmp2)(a0)
;
; The -0x4 offset of the auipc w.r.t the lw is inlined in the lw instruction.
; CHECK-NEXT: 10: 00000517           auipc   a0, 0
; CHECK-NEXT:                        00000010:  RISCV_RELOC_GOT_HI20(pcrel)  var
; CHECK-NEXT: 14: ffc52503           lw      a0, -0x4(a0)
; CHECK-NEXT:                        00000014:  RISCV_RELOC_GOT_LO12(pcrel)  var
Ltmp2:
        auipc a0, %got_pcrel_hi(var)
        lw a0, %pcrel_lo(Ltmp2)(a0)

; Same as the first example. This time, though, there is code in
; between the two relocations, which means that the low 12 bits
; needs to be shifted by the correct amount that takes the address
; computation relative to the PC at auipc (8 bytes before).
;
; CHECK-NEXT: 18: 00000517           auipc   a0, 0
; CHECK-NEXT:                        00000018:  RISCV_RELOC_HI20(pcrel)      var
; CHECK-NEXT:  1c: 00000013          nop
; CHECK-NEXT:  20: ff850513          addi    a0, a0, -0x8
; CHECK-NEXT:                        00000020:  RISCV_RELOC_LO12(pcrel)      var
Ltmp3:
        auipc a0, %pcrel_hi(var)
        nop
        addi a0, a0, %pcrel_lo(Ltmp3)

; Relocations for function calls. The jal instruction is patched
; with the relative offset of _bar to the PC (limited to 21 bits)

; CHECK-NEXT: [[PC_call:[0-9a-f]+]]:{{.*}}jal 0x[[PC_call]] <ltmp0+0x[[PC_call]]>
; CHECK-NEXT:                       RISCV_RELOC_BRANCH21 _bar
        call _bar

; CHECK-NEXT: [[PC_tail:[0-9a-f]+]]:{{.*}}j 0x[[PC_tail]] <ltmp0+0x[[PC_tail]]>
; CHECK-NEXT:                        RISCV_RELOC_BRANCH21 _bar
        tail _bar

; Non PIC/PIE code, used for baremetal, where absolute addresses are
; expected.
;
; a0 = (address_of_var[31:12] << 12) & 0xfffff000
; a0 = a0 + address_of_var[11:0] = address_of_var
;
; CHECK-NEXT: 2c: 00000537           lui     a0, 0
; CHECK-NEXT:                      0000002c: RISCV_RELOC_HI20 _var
; CHECK-NEXT: 30: 00050513           mv      a0, a0
; CHECK-NEXT:                      00000030:  RISCV_RELOC_LO12       _var
        lui a0, %hi(_var)
        addi a0, a0, %lo(_var)


; Data directives inside .text, with relocation checks:
;
        .data_region

; Absolute address. The linker will fill PC 32 and 34 with the 4
; bytes of the address of _bar.
;
; CHECK-NEXT: 34: 0000             <unknown>
; CHECK-NEXT:                      00000034:  RISCV_RELOC_UNSIGNED   _bar
; CHECK-NEXT: 36: 0000             <unknown>
        .word _bar

; Absolute address as before, but with an additional 42 bytes
; offset. The linker will see the offset stored @PC36 and use it as
; an addend to the address of _bar. The resulting value will be
; stored in the PCs at 36 and 38.
;
; CHECK-NEXT: 38: 002a             <unknown>
; CHECK-NEXT:                      00000038:  RISCV_RELOC_UNSIGNED   _bar
; CHECK-NEXT: 3a: 0000             <unknown>
        .word _bar + 42

; Subtraction expression. The 4 bytes at PCs 3a and 3c will be
; replaced with the subtraction of the address of _a and the
; address of _b.
;
; CHECK-NEXT: 3c: 0000             <unknown>
; CHECK-NEXT:                      0000003c:  RISCV_RELOC_SUBTRACTOR       _b
; CHECK-NEXT:                      0000003c:  RISCV_RELOC_UNSIGNED   _a
; CHECK-NEXT: 3e: 0000             <unknown>
        .word _a - _b

; Same as before, but the offset between _a and _b is shifted by 42
; bytes. The shift amount is stored in the 2 bytes at 3e.
;
; CHECK-NEXT: 40: 002a             <unknown>
; CHECK-NEXT:                      00000040:  RISCV_RELOC_SUBTRACTOR         _b
; CHECK-NEXT:                      00000040:  RISCV_RELOC_UNSIGNED   _a
; CHECK-NEXT: 42: 0000             <unknown>
        .word _a - _b + 42
        .end_data_region
; No more relocation directive past this line.
; CHECK-NOT: {{.}}

; OTOOL-LABEL: Relocation information (__TEXT,__text) 20 entries 
; OTOOL-NEXT:  address  pcrel length extern type    scattered symbolnum/value
; OTOOL-NEXT:  00000040 False long   True   1       False     _b
; OTOOL-NEXT:  00000040 False long   True   0       False     _a
; OTOOL-NEXT:  0000003c False long   True   1       False     _b
; OTOOL-NEXT:  0000003c False long   True   0       False     _a
; OTOOL-NEXT:  00000038 False long   True   0       False     _bar
; OTOOL-NEXT:  00000034 False long   True   0       False     _bar
; OTOOL-NEXT:  00000030 False long   True   4       False     _var
; OTOOL-NEXT:  0000002c False long   True   3       False     _var
; OTOOL-NEXT:  00000028 True  long   True   2       False     _bar
; OTOOL-NEXT:  00000024 True  long   True   2       False     _bar
; OTOOL-NEXT:  00000020 True  long   True   4       False     var
; OTOOL-NEXT:  00000018 True  long   True   3       False     var
; OTOOL-NEXT:  00000014 True  long   True   6       False     var
; OTOOL-NEXT:  00000010 True  long   True   5       False     var
; OTOOL-NEXT:  0000000c False long   False  8       False     addend = 0x000010
; OTOOL-NEXT:  0000000c True  long   True   4       False     var
; OTOOL-NEXT:  00000008 False long   False  8       False     addend = 0x000010
; OTOOL-NEXT:  00000008 True  long   True   3       False     var
; OTOOL-NEXT:  00000004 True  long   True   4       False     var
; OTOOL-NEXT:  00000000 True  long   True   3       False     var
; OTOOL-NEXT:  Contents of (__TEXT,__text) section
; OTOOL-NEXT:  00000000	auipc	a0, 0x0
; OTOOL-NEXT:  00000004	addi	a0, a0, -0x4
; OTOOL-NEXT:  00000008	auipc	a0, 0x0
; OTOOL-NEXT:  0000000c	addi	a0, a0, -0x4
; OTOOL-NEXT:  00000010	auipc	a0, 0x0
; OTOOL-NEXT:  00000014	lw	a0, -0x4(a0)
; OTOOL-NEXT:  00000018	auipc	a0, 0x0
; OTOOL-NEXT:  0000001c	nop
; OTOOL-NEXT:  00000020	addi	a0, a0, -0x8
; OTOOL-NEXT:  00000024	jal	0x0
; OTOOL-NEXT:  00000028	j	0x0
; OTOOL-NEXT:  0000002c	lui	a0, 0x0
; OTOOL-NEXT:  00000030	mv	a0, a0
; OTOOL-NEXT:  	.long 0	@ KIND_DATA
; OTOOL-NEXT:  	.long 42	@ KIND_DATA
; OTOOL-NEXT:  	.long 0	@ KIND_DATA
; OTOOL-NEXT:  	.long 42	@ KIND_DATA
; OTOOL-NOT: {{.}}
