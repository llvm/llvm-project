# RUN: llvm-mc -filetype=obj -triple=mipsel %s | llvm-readobj -r - | FileCheck %s

# Test the order of records in the relocation table.
#
# MIPS has a few relocations that have an AHL component in the expression used
# to evaluate them. This AHL component is an addend with the same number of
# bits as a symbol value but not all of our ABI's are able to supply a
# sufficiently sized addend in a single relocation.
#
# The O32 ABI for example, uses REL relocations which store the addend in the
# section data. All the relocations with AHL components affect 16-bit fields
# so the addend is limited to 16-bit. This ABI resolves the limitation by
# linking relocations (e.g. R_MIPS_HI16 and R_MIPS_LO16) and distributing the
# addend between the linked relocations. The ABI mandates that such relocations
# must be next to each other in a particular order (e.g. R_MIPS_HI16 must be
# followed by a matching R_MIPS_LO16) but the rule is less strict in practice.
#
# See MipsELFObjectWriter.cpp for a full description of the rules.
#
# TODO: Add mips16 and micromips tests.

# HILO 1: HI/LO already match
	.section .mips_hilo_1, "ax", @progbits
	lui $2, %hi(sym1)
	addiu $2, $2, %lo(sym1)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_hilo_1 {
# CHECK-NEXT:    0x0 R_MIPS_HI16 sym1
# CHECK-NEXT:    0x4 R_MIPS_LO16 sym1
# CHECK-NEXT:  }

# HILO 2: R_MIPS_HI16 must be followed by a matching R_MIPS_LO16.
	.section .mips_hilo_2, "ax", @progbits
	addiu $2, $2, %lo(sym1)
	lui $2, %hi(sym1)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_hilo_2 {
# CHECK-NEXT:    0x4 R_MIPS_HI16 sym1
# CHECK-NEXT:    0x0 R_MIPS_LO16 sym1
# CHECK-NEXT:  }

# HILO 3: R_MIPS_HI16 must be followed by a matching R_MIPS_LO16.
#         The second relocation matches if the symbol is the same.
	.section .mips_hilo_3, "ax", @progbits
	addiu $2, $2, %lo(sym1)
	lui $2, %hi(sym2)
	addiu $2, $2, %lo(sym2)
	lui $2, %hi(sym1)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_hilo_3 {
# CHECK-NEXT:    0xC R_MIPS_HI16 sym1
# CHECK-NEXT:    0x0 R_MIPS_LO16 sym1
# CHECK-NEXT:    0x4 R_MIPS_HI16 sym2
# CHECK-NEXT:    0x8 R_MIPS_LO16 sym2
# CHECK-NEXT:  }

# HILO 3b: Same as 3 but a different starting order.
	.section .mips_hilo_3b, "ax", @progbits
	addiu $2, $2, %lo(sym1)
	lui $2, %hi(sym1)
	addiu $2, $2, %lo(sym2)
	lui $2, %hi(sym2)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_hilo_3b {
# CHECK-NEXT:    0x4 R_MIPS_HI16 sym1
# CHECK-NEXT:    0x0 R_MIPS_LO16 sym1
# CHECK-NEXT:    0xC R_MIPS_HI16 sym2
# CHECK-NEXT:    0x8 R_MIPS_LO16 sym2
# CHECK-NEXT:  }

# HILO 4: R_MIPS_HI16 must be followed by a matching R_MIPS_LO16.
#         The second relocation matches if the symbol is the same and the
#         offset is the same.
	.section .mips_hilo_4, "ax", @progbits
	addiu $2, $2, %lo(sym1)
	addiu $2, $2, %lo(sym1+4)
	lui $2, %hi(sym1+4)
	lui $2, %hi(sym1)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_hilo_4 {
# CHECK-NEXT:    0xC R_MIPS_HI16 sym1
# CHECK-NEXT:    0x0 R_MIPS_LO16 sym1
# CHECK-NEXT:    0x8 R_MIPS_HI16 sym1
# CHECK-NEXT:    0x4 R_MIPS_LO16 sym1
# CHECK-NEXT:  }

# HILO 5: R_MIPS_HI16 must be followed by a matching R_MIPS_LO16.
#         The second relocation matches if the symbol is the same and the
#         offset is greater or equal. Exact matches are preferred so both
#         R_MIPS_HI16's match the same R_MIPS_LO16.
	.section .mips_hilo_5, "ax", @progbits
	lui $2, %hi(sym1)
	lui $2, %hi(sym1)
	addiu $2, $2, %lo(sym1+1)
	addiu $2, $2, %lo(sym1)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_hilo_5 {
# CHECK-NEXT:    0x8 R_MIPS_LO16 sym1
# CHECK-NEXT:    0x0 R_MIPS_HI16 sym1
# CHECK-NEXT:    0x4 R_MIPS_HI16 sym1
# CHECK-NEXT:    0xC R_MIPS_LO16 sym1
# CHECK-NEXT:  }

# HILO 6: R_MIPS_HI16 must be followed by a matching R_MIPS_LO16.
#         The second relocation matches if the symbol is the same and the
#         offset is greater or equal. Smaller offsets are preferred so both
#         R_MIPS_HI16's still match the same R_MIPS_LO16.
	.section .mips_hilo_6, "ax", @progbits
	lui $2, %hi(sym1)
	lui $2, %hi(sym1)
	addiu $2, $2, %lo(sym1+2)
	addiu $2, $2, %lo(sym1+1)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_hilo_6 {
# CHECK-NEXT:    0x8 R_MIPS_LO16 sym1
# CHECK-NEXT:    0x0 R_MIPS_HI16 sym1
# CHECK-NEXT:    0x4 R_MIPS_HI16 sym1
# CHECK-NEXT:    0xC R_MIPS_LO16 sym1
# CHECK-NEXT:  }

# HILO 7: R_MIPS_HI16 must be followed by a matching R_MIPS_LO16.
#         The second relocation matches if the symbol is the same and the
#         offset is greater or equal so that the carry bit is correct. The two
#         R_MIPS_HI16's therefore match different R_MIPS_LO16's.
	.section .mips_hilo_7, "ax", @progbits
	addiu $2, $2, %lo(sym1+1)
	addiu $2, $2, %lo(sym1+6)
	lui $2, %hi(sym1+4)
	lui $2, %hi(sym1)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_hilo_7 {
# CHECK-NEXT:    0xC R_MIPS_HI16 sym1
# CHECK-NEXT:    0x0 R_MIPS_LO16 sym1
# CHECK-NEXT:    0x8 R_MIPS_HI16 sym1
# CHECK-NEXT:    0x4 R_MIPS_LO16 sym1
# CHECK-NEXT:  }

# HILO 8: R_MIPS_LO16's may be orphaned.
	.section .mips_hilo_8, "ax", @progbits
	lw $2, %lo(sym1)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_hilo_8 {
# CHECK-NEXT:    0x0 R_MIPS_LO16 sym1
# CHECK-NEXT:  }

# HILO 8b: Another example of 8. The R_MIPS_LO16 at 0x4 is orphaned.
	.section .mips_hilo_8b, "ax", @progbits
	lw $2, %lo(sym1)
	lw $2, %lo(sym1)
	lui $2, %hi(sym1)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_hilo_8b {
# CHECK-NEXT:    0x0 R_MIPS_LO16 sym1
# CHECK-NEXT:    0x8 R_MIPS_HI16 sym1
# CHECK-NEXT:    0x4 R_MIPS_LO16 sym1
# CHECK-NEXT:  }

# HILO 9: R_MIPS_HI16's don't need a matching R_MIPS_LO16 to immediately follow
#         so long as there is one after the R_MIPS_HI16 somewhere. This isn't
#         permitted by the ABI specification but has been allowed in practice
#         for a very long time. The R_MIPS_HI16's should be ordered by the
#         address they affect for purely cosmetic reasons.
	.section .mips_hilo_9, "ax", @progbits
	lw $2, %lo(sym1)
	lui $2, %hi(sym1)
	lui $2, %hi(sym1)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_hilo_9 {
# CHECK-NEXT:    0x4 R_MIPS_HI16 sym1
# CHECK-NEXT:    0x8 R_MIPS_HI16 sym1
# CHECK-NEXT:    0x0 R_MIPS_LO16 sym1
# CHECK-NEXT:  }

# HILO 10: R_MIPS_HI16's must have a matching R_MIPS_LO16 somewhere though.
#          When this is impossible we have two possible bad behaviours
#          depending on the linker implementation:
#          * The linker silently computes the wrong value using a partially
#            matching R_MIPS_LO16.
#          * The linker rejects the relocation table as invalid.
#          The latter is preferable since it's far easier to detect and debug so
#          check that we encourage this behaviour by putting invalid
#          R_MIPS_HI16's at the end of the relocation table where the risk of a
#          partial match is very low.
	.section .mips_hilo_10, "ax", @progbits
	lui $2, %hi(sym1)
	lw $2, %lo(sym1)
	lui $2, %hi(sym2)
	lui $2, %hi(sym3)
	lw $2, %lo(sym3)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_hilo_10 {
# CHECK-NEXT:    0x0 R_MIPS_HI16 sym1
# CHECK-NEXT:    0x4 R_MIPS_LO16 sym1
# CHECK-NEXT:    0xC R_MIPS_HI16 sym3
# CHECK-NEXT:    0x10 R_MIPS_LO16 sym3
# CHECK-NEXT:    0x8 R_MIPS_HI16 sym2
# CHECK-NEXT:  }

# Now do the same tests for GOT/LO.
# The rules only apply to R_MIPS_GOT16 on local symbols which are also
# rewritten into section relative relocations.

# GOTLO 1: GOT/LO already match
	.section .mips_gotlo_1, "ax", @progbits
	lui $2, %got(local1)
	addiu $2, $2, %lo(local1)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_gotlo_1 {
# CHECK-NEXT:    0x0 R_MIPS_GOT16 .text
# CHECK-NEXT:    0x4 R_MIPS_LO16 .text
# CHECK-NEXT:  }

# GOTLO 2: R_MIPS_GOT16 must be followed by a matching R_MIPS_LO16.
	.section .mips_gotlo_2, "ax", @progbits
	addiu $2, $2, %lo(local1)
	lui $2, %got(local1)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_gotlo_2 {
# CHECK-NEXT:    0x4 R_MIPS_GOT16 .text
# CHECK-NEXT:    0x0 R_MIPS_LO16 .text
# CHECK-NEXT:  }

# GOTLO 3: R_MIPS_GOT16 must be followed by a matching R_MIPS_LO16.
#          The second relocation matches if the symbol is the same.
	.section .mips_gotlo_3, "ax", @progbits
	addiu $2, $2, %lo(local1)
	lui $2, %got(local2)
	addiu $2, $2, %lo(local2)
	lui $2, %got(local1)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_gotlo_3 {
# CHECK-NEXT:    0xC R_MIPS_GOT16 .text
# CHECK-NEXT:    0x0 R_MIPS_LO16 .text
# CHECK-NEXT:    0x4 R_MIPS_GOT16 .text
# CHECK-NEXT:    0x8 R_MIPS_LO16 .text
# CHECK-NEXT:  }

# GOTLO 3b: Same as 3 but a different starting order.
	.section .mips_gotlo_3b, "ax", @progbits
	addiu $2, $2, %lo(local1)
	lui $2, %got(local1)
	addiu $2, $2, %lo(local2)
	lui $2, %got(local2)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_gotlo_3b {
# CHECK-NEXT:    0x4 R_MIPS_GOT16 .text
# CHECK-NEXT:    0x0 R_MIPS_LO16 .text
# CHECK-NEXT:    0xC R_MIPS_GOT16 .text
# CHECK-NEXT:    0x8 R_MIPS_LO16 .text
# CHECK-NEXT:  }

# GOTLO 4: R_MIPS_GOT16 must be followed by a matching R_MIPS_LO16.
#          The second relocation matches if the symbol is the same and the
#          offset is the same.
	.section .mips_gotlo_4, "ax", @progbits
	addiu $2, $2, %lo(local1)
	addiu $2, $2, %lo(local1+4)
	lui $2, %got(local1+4)
	lui $2, %got(local1)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_gotlo_4 {
# CHECK-NEXT:    0xC R_MIPS_GOT16 .text
# CHECK-NEXT:    0x0 R_MIPS_LO16 .text
# CHECK-NEXT:    0x8 R_MIPS_GOT16 .text
# CHECK-NEXT:    0x4 R_MIPS_LO16 .text
# CHECK-NEXT:  }

# GOTLO 5: R_MIPS_GOT16 must be followed by a matching R_MIPS_LO16.
#          The second relocation matches if the symbol is the same and the
#          offset is greater or equal. Exact matches are preferred so both
#          R_MIPS_GOT16's match the same R_MIPS_LO16.
	.section .mips_gotlo_5, "ax", @progbits
	lui $2, %got(local1)
	lui $2, %got(local1)
	addiu $2, $2, %lo(local1+1)
	addiu $2, $2, %lo(local1)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_gotlo_5 {
# CHECK-NEXT:    0x8 R_MIPS_LO16 .text
# CHECK-NEXT:    0x0 R_MIPS_GOT16 .text
# CHECK-NEXT:    0x4 R_MIPS_GOT16 .text
# CHECK-NEXT:    0xC R_MIPS_LO16 .text
# CHECK-NEXT:  }

# GOTLO 6: R_MIPS_GOT16 must be followed by a matching R_MIPS_LO16.
#          The second relocation matches if the symbol is the same and the
#          offset is greater or equal. Smaller offsets are preferred so both
#          R_MIPS_GOT16's still match the same R_MIPS_LO16.
	.section .mips_gotlo_6, "ax", @progbits
	lui $2, %got(local1)
	lui $2, %got(local1)
	addiu $2, $2, %lo(local1+2)
	addiu $2, $2, %lo(local1+1)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_gotlo_6 {
# CHECK-NEXT:    0x8 R_MIPS_LO16 .text
# CHECK-NEXT:    0x0 R_MIPS_GOT16 .text
# CHECK-NEXT:    0x4 R_MIPS_GOT16 .text
# CHECK-NEXT:    0xC R_MIPS_LO16 .text
# CHECK-NEXT:  }

# GOTLO 7: R_MIPS_GOT16 must be followed by a matching R_MIPS_LO16.
#          The second relocation matches if the symbol is the same and the
#          offset is greater or equal so that the carry bit is correct. The two
#          R_MIPS_GOT16's therefore match different R_MIPS_LO16's.
	.section .mips_gotlo_7, "ax", @progbits
	addiu $2, $2, %lo(local1+1)
	addiu $2, $2, %lo(local1+6)
	lui $2, %got(local1+4)
	lui $2, %got(local1)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_gotlo_7 {
# CHECK-NEXT:    0xC R_MIPS_GOT16 .text
# CHECK-NEXT:    0x0 R_MIPS_LO16 .text
# CHECK-NEXT:    0x8 R_MIPS_GOT16 .text
# CHECK-NEXT:    0x4 R_MIPS_LO16 .text
# CHECK-NEXT:  }

# GOTLO 8: R_MIPS_LO16's may be orphaned.
	.section .mips_gotlo_8, "ax", @progbits
	lw $2, %lo(local1)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_gotlo_8 {
# CHECK-NEXT:    0x0 R_MIPS_LO16 .text
# CHECK-NEXT:  }

# GOTLO 8b: Another example of 8. The R_MIPS_LO16 at 0x4 is orphaned.
	.section .mips_gotlo_8b, "ax", @progbits
	lw $2, %lo(local1)
	lw $2, %lo(local1)
	lui $2, %got(local1)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_gotlo_8b {
# CHECK-NEXT:    0x0 R_MIPS_LO16 .text
# CHECK-NEXT:    0x8 R_MIPS_GOT16 .text
# CHECK-NEXT:    0x4 R_MIPS_LO16 .text
# CHECK-NEXT:  }

# GOTLO 9: R_MIPS_GOT16's don't need a matching R_MIPS_LO16 to immediately
#          follow so long as there is one after the R_MIPS_GOT16 somewhere.
#          This isn't permitted by the ABI specification but has been allowed
#          in practice for a very long time. The R_MIPS_GOT16's should be
#          ordered by the address they affect for purely cosmetic reasons.
	.section .mips_gotlo_9, "ax", @progbits
	lw $2, %lo(local1)
	lui $2, %got(local1)
	lui $2, %got(local1)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_gotlo_9 {
# CHECK-NEXT:    0x4 R_MIPS_GOT16 .text
# CHECK-NEXT:    0x8 R_MIPS_GOT16 .text
# CHECK-NEXT:    0x0 R_MIPS_LO16 .text
# CHECK-NEXT:  }

# GOTLO 10: R_MIPS_GOT16's must have a matching R_MIPS_LO16 somewhere though.
#           When this is impossible we have two possible bad behaviours
#           depending on the linker implementation:
#           * The linker silently computes the wrong value using a partially
#             matching R_MIPS_LO16.
#           * The linker rejects the relocation table as invalid.
#           The latter is preferable since it's far easier to detect and debug
#           so check that we encourage this behaviour by putting invalid
#           R_MIPS_GOT16's at the end of the relocation table where the risk of
#           a partial match is very low.
	.section .mips_gotlo_10, "ax", @progbits
	lui $2, %got(local1)
	lw $2, %lo(local1)
	lui $2, %got(local2)
	lui $2, %got(local3)
	lw $2, %lo(local3)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_gotlo_10 {
# CHECK-NEXT:    0x0 R_MIPS_GOT16 .text
# CHECK-NEXT:    0x4 R_MIPS_LO16 .text
# CHECK-NEXT:    0x8 R_MIPS_GOT16 .text
# CHECK-NEXT:    0xC R_MIPS_GOT16 .text
# CHECK-NEXT:    0x10 R_MIPS_LO16 .text
# CHECK-NEXT:  }

# Finally, do test 2 for R_MIPS_GOT16 on external symbols to prove they are
# exempt from the rules for local symbols.

# External GOTLO 2: R_MIPS_GOT16 must be followed by a matching R_MIPS_LO16.
	.section .mips_ext_gotlo_2, "ax", @progbits
	addiu $2, $2, %lo(sym1)
	lui $2, %got(sym1)

# CHECK-LABEL: Section ({{[0-9]+}}) .rel.mips_ext_gotlo_2 {
# CHECK-NEXT:    0x0 R_MIPS_LO16 sym1
# CHECK-NEXT:    0x4 R_MIPS_GOT16 sym1
# CHECK-NEXT:  }

# Define some local symbols.
        .text
        nop
local1: nop
local2: nop
local3: nop
