// REQUIRES: x86-registered-target
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t.o

// RUN: llvm-readobj --notes %t.o | FileCheck %s --check-prefix=LLVM
// RUN: llvm-readelf --notes %t.o | FileCheck %s --check-prefix=GNU

// GNU:      Displaying notes found at file offset 0x00000040 with length 0x00000010:
// GNU-NEXT:   Owner                 Data size       Description
// GNU-NEXT:   XYZ                  0x00000000       Unknown note type: (0x00000003)

// LLVM:      Notes [
// LLVM-NEXT:   NoteSection {
// LLVM-NEXT:     Offset: 0x40
// LLVM-NEXT:     Size: 0x10
// LLVM-NEXT:     Note {
// LLVM-NEXT:       Owner: XYZ
// LLVM-NEXT:       Data size: 0x0
// LLVM-NEXT:       Type: Unknown (0x00000003)
// LLVM-NEXT:     }
// LLVM-NEXT:   }
// LLVM-NEXT: ]

.section ".note.foo", "a"
	.align 4
	.long 4 /* namesz */
	.long 0 /* descsz */
	.long 3 /* type */
	.asciz "XYZ"
