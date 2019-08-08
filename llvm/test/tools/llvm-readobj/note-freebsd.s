// REQUIRES: x86-registered-target
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t.o

// RUN: llvm-readobj --notes %t.o | FileCheck %s --check-prefix=LLVM
// RUN: llvm-readelf --notes %t.o | FileCheck %s --check-prefix=GNU

// GNU:      Displaying notes found
// GNU-NEXT:   Owner                Data size        Description
// GNU-NEXT:   FreeBSD              0x00000000       NT_THRMISC (thrmisc structure)
// GNU-EMPTY:
// GNU-NEXT:   FreeBSD              0x00000000       NT_PROCSTAT_PROC (proc data)
// GNU-EMPTY:
// GNU-NEXT: Displaying notes found
// GNU-NEXT:   Owner                Data size        Description
// GNU-NEXT:   FreeBSD              0x00000000       NT_PROCSTAT_FILES (files data)

// LLVM:      Notes [
// LLVM-NEXT:   NoteSection {
// LLVM-NEXT:     Offset:
// LLVM-NEXT:     Size:
// LLVM-NEXT:     Note {
// LLVM-NEXT:       Owner: FreeBSD
// LLVM-NEXT:       Data size: 0x0
// LLVM-NEXT:       Type: NT_THRMISC (thrmisc structure)
// LLVM-NEXT:     }
// LLVM-NEXT:     Note {
// LLVM-NEXT:       Owner: FreeBSD
// LLVM-NEXT:       Data size: 0x0
// LLVM-NEXT:       Type: NT_PROCSTAT_PROC (proc data)
// LLVM-NEXT:     }
// LLVM-NEXT:   }
// LLVM-NEXT:   NoteSection {
// LLVM-NEXT:     Offset: 0x68
// LLVM-NEXT:     Size: 0x14
// LLVM-NEXT:     Note {
// LLVM-NEXT:       Owner: FreeBSD
// LLVM-NEXT:       Data size: 0x0
// LLVM-NEXT:       Type: NT_PROCSTAT_FILES (files data)
// LLVM-NEXT:     }
// LLVM-NEXT:   }
// LLVM-NEXT: ]

.section ".note.foo", "a"
	.align 4
	.long 8 /* namesz */
	.long 0 /* descsz */
	.long 7 /* type = NT_FREEBSD_THRMISC */
	.asciz "FreeBSD"
	.long 8 /* namesz */
	.long 0 /* descsz */
	.long 8 /* type = NT_FREEBSD_PROC */
	.asciz "FreeBSD"
.section ".note.bar", "a"
	.align 4
	.long 8 /* namesz */
	.long 0 /* descsz */
	.long 9 /* type = NT_FREEBSD_FILES */
	.asciz "FreeBSD"
