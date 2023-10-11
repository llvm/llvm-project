# RUN: llvm-mc %s -triple=mips -mcpu=mips32r2 -filetype=asm | \
# RUN:   FileCheck %s -check-prefix=ASMOUT

# RUN: llvm-mc %s -triple=mips -mcpu=mips32r2 -filetype=obj -o - | \
# RUN:   llvm-readobj -S --section-data -r - | \
# RUN:     FileCheck %s -check-prefix=OBJOUT

# ASMOUT: .text
# ASMOUT:        .type _local_foo,@function
# ASMOUT:        .ent _local_foo
# ASMOUT:_local_foo:
# ASMOUT:        .frame $fp,16,$ra
# ASMOUT:        .mask 0x10101010,-4
# ASMOUT:        .fmask 0x01010101,-8
# ASMOUT:        .end _local_foo
# ASMOUT:        .size local_foo,

# OBJOUT: Section {
# OBJOUT:     Name: .pdr
# OBJOUT:     Type: SHT_PROGBITS (0x1)
# OBJOUT:     Flags [ (0x0)
# OBJOUT-NOT:   SHF_ALLOC (0x2)
# OBJOUT-NOT:   SHF_WRITE (0x1)
# OBJOUT:     ]
# OBJOUT:     Size: 64
# OBJOUT:     SectionData (
# OBJOUT:       0000: 00000000 10101010 FFFFFFFC 01010101
# OBJOUT:       0010: FFFFFFF8 00000010 0000001E 0000001F
# OBJOUT:       0020: 00000000 10101010 FFFFFFFC 01010101
# OBJOUT:       0030: FFFFFFF8 00000010 0000001E 0000001F
# OBJOUT:     )
# OBJOUT:   }

# We should also check if relocation information was correctly generated.
# OBJOUT:      Relocations [
# OBJOUT-NEXT:   Section ({{.*}}) .rel.pdr {
# OBJOUT-NEXT:     0x0 R_MIPS_32 .text
# OBJOUT-NEXT:     0x20 R_MIPS_32 _global_foo
# OBJOUT-NEXT:   }
# OBJOUT-NEXT: ]

.text
        .type _local_foo,@function
        .ent _local_foo
_local_foo:
        .frame $fp,16,$ra
        .mask 0x10101010,-4
        .fmask 0x01010101,-8
        .end _local_foo
        .size local_foo,.-_local_foo

        .globl _global_foo
        .type _global_foo,@function
        .ent _global_foo
_global_foo:
        .frame $fp,16,$ra
        .mask 0x10101010,-4
        .fmask 0x01010101,-8
        .end _global_foo
        .size global_foo,.-_global_foo
