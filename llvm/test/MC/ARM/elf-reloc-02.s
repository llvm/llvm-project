// RUN: llvm-mc -triple=armv7-linux-gnueabi \
// RUN:    -mcpu=cortex-a8 -mattr=-neon -mattr=+vfp2  \
// RUN:    -filetype=obj %s -o - | \
// RUN:   llvm-readobj -r - | FileCheck -check-prefix=OBJ %s

// Ensure no regression on ARM/gcc compatibility for
// emitting explicit symbol relocs for nonexternal symbols
// versus section symbol relocs (with offset) -
//
// Default llvm behavior is to emit as section symbol relocs nearly
// everything that is not an undefined external. Unfortunately, this
// diverges from what codesourcery ARM/gcc does!
//
// Tests that reloc to .L.str* show up as explicit symbols

	movw	r1, :lower16:.L.str
	movt	r1, :upper16:.L.str

	.section	.rodata,"a",%progbits
.L.str:
	.asciz	"@null\n"

// OBJ:      Relocations [
// OBJ:        Section {{.*}} .rel.text {
// OBJ-NEXT:     0x{{[0-9,A-F]+}} R_ARM_MOVW_ABS_NC .L.str
// OBJ:        }
// OBJ:      ]
