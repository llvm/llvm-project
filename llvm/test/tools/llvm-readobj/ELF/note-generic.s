// REQUIRES: x86-registered-target
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t.o

// RUN: llvm-readobj --notes %t.o | FileCheck %s --check-prefix=LLVM
// RUN: llvm-readelf --notes %t.o | FileCheck %s --check-prefix=GNU

// GNU:      Displaying notes found in: .note.version
// GNU-NEXT:   Owner                 Data size       Description
// GNU-NEXT:   XYZ                  0x00000000       NT_VERSION (version)

// GNU:      Displaying notes found in: .note.arch
// GNU-NEXT:   Owner                 Data size       Description
// GNU-NEXT:   XYZ                  0x00000000       NT_ARCH (architecture)

// GNU:      Displaying notes found in: .note.open
// GNU-NEXT:   Owner                 Data size       Description
// GNU-NEXT:   XYZ                  0x00000000       OPEN

// GNU:      Displaying notes found in: .note.func
// GNU-NEXT:   Owner                 Data size       Description
// GNU-NEXT:   XYZ                  0x00000000       func

// LLVM:      Notes [
// LLVM-NEXT:   NoteSection {
// LLVM-NEXT:     Name: .note.version
// LLVM-NEXT:     Offset: 0x40
// LLVM-NEXT:     Size: 0x10
// LLVM-NEXT:     Note {
// LLVM-NEXT:       Owner: XYZ
// LLVM-NEXT:       Data size: 0x0
// LLVM-NEXT:       Type: NT_VERSION (version)
// LLVM-NEXT:     }
// LLVM-NEXT:   }
// LLVM-NEXT:   NoteSection {
// LLVM-NEXT:     Name: .note.arch
// LLVM-NEXT:     Offset: 0x50
// LLVM-NEXT:     Size: 0x10
// LLVM-NEXT:     Note {
// LLVM-NEXT:       Owner: XYZ
// LLVM-NEXT:       Data size: 0x0
// LLVM-NEXT:       Type: NT_ARCH (architecture)
// LLVM-NEXT:     }
// LLVM-NEXT:   }
// LLVM-NEXT:   NoteSection {
// LLVM-NEXT:     Name: .note.open
// LLVM-NEXT:     Offset: 0x60
// LLVM-NEXT:     Size: 0x10
// LLVM-NEXT:     Note {
// LLVM-NEXT:       Owner: XYZ
// LLVM-NEXT:       Data size: 0x0
// LLVM-NEXT:       Type: OPEN
// LLVM-NEXT:     }
// LLVM-NEXT:   }
// LLVM-NEXT:   NoteSection {
// LLVM-NEXT:     Name: .note.func
// LLVM-NEXT:     Offset: 0x70
// LLVM-NEXT:     Size: 0x10
// LLVM-NEXT:     Note {
// LLVM-NEXT:       Owner: XYZ
// LLVM-NEXT:       Data size: 0x0
// LLVM-NEXT:       Type: func
// LLVM-NEXT:     }
// LLVM-NEXT:   }
// LLVM-NEXT: ]

.section ".note.version", "a"
	.align 4
	.long 4 /* namesz */
	.long 0 /* descsz */
	.long 1 /* type = NT_VERSION */
	.asciz "XYZ"
.section ".note.arch", "a"
	.align 4
	.long 4 /* namesz */
	.long 0 /* descsz */
	.long 2 /* type = NT_ARCH*/
	.asciz "XYZ"
.section ".note.open", "a"
	.align 4
	.long 4 /* namesz */
	.long 0 /* descsz */
	.long 0x100 /* type = NT_GNU_BUILD_ATTRIBUTE_OPEN*/
	.asciz "XYZ"
.section ".note.func", "a"
	.align 4
	.long 4 /* namesz */
	.long 0 /* descsz */
	.long 0x101 /* type = NT_GNU_BUILD_ATTRIBUTE_FUNC*/
	.asciz "XYZ"
