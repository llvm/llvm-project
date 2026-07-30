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
// Tests that reloc to _MergedGlobals show up as explicit symbol reloc

	movw	r2, :lower16:_MergedGlobals

_MergedGlobals:
	.long	1


// OBJ: Relocations [
// OBJ:   Section {{.*}} .rel.text {
// OBJ:     0x{{[0-9,A-F]+}} R_ARM_MOVW_ABS_NC _MergedGlobals
// OBJ:   }
// OBJ: ]
