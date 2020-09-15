// REQUIRES: x86-registered-target
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t.o

// RUN: llvm-readobj --notes %t.o | FileCheck %s --check-prefix=LLVM
// RUN: llvm-readelf --notes %t.o | FileCheck %s --check-prefix=GNU

// GNU:      Displaying notes found in: .note.foo
// GNU-NEXT:   Owner                Data size        Description
// GNU-NEXT:   FreeBSD              0x00000000       NT_THRMISC (thrmisc structure)
// GNU-NEXT:   FreeBSD              0x00000000       NT_PROCSTAT_PROC (proc data)
// GNU-NEXT: Displaying notes found in: .note.bar
// GNU-NEXT:   Owner                Data size        Description
// GNU-NEXT:   FreeBSD              0x00000000       NT_PROCSTAT_FILES (files data)
// GNU-NEXT: Displaying notes found in: .note.baz
// GNU-NEXT:   Owner                Data size        Description
// GNU-NEXT:   FreeBSD              0x0000001c       Unknown note type: (0x00000003)
// GNU-NEXT:    description data: 4c 6f 72 65 6d 20 69 70 73 75 6d 20 64 6f 6c 6f 72 20 73 69 74 20 61 6d 65 74 00 00

// LLVM:      Notes [
// LLVM-NEXT:   NoteSection {
// LLVM-NEXT:     Name: .note.foo
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
// LLVM-NEXT:     Name: .note.bar
// LLVM-NEXT:     Offset: 0x68
// LLVM-NEXT:     Size: 0x14
// LLVM-NEXT:     Note {
// LLVM-NEXT:       Owner: FreeBSD
// LLVM-NEXT:       Data size: 0x0
// LLVM-NEXT:       Type: NT_PROCSTAT_FILES (files data)
// LLVM-NEXT:     }
// LLVM-NEXT:   }
// LLVM-NEXT:   NoteSection {
// LLVM-NEXT:     Name: .note.baz
// LLVM-NEXT:     Offset: 0x7C
// LLVM-NEXT:     Size: 0x30
// LLVM-NEXT:     Note {
// LLVM-NEXT:       Owner: FreeBSD
// LLVM-NEXT:       Data size: 0x1C
// LLVM-NEXT:       Type: Unknown (0x00000003)
// LLVM-NEXT:       Description data (
// LLVM-NEXT:         0000: 4C6F7265 6D206970 73756D20 646F6C6F  |Lorem ipsum dolo|
// LLVM-NEXT:         0010: 72207369 7420616D 65740000           |r sit amet..|
// LLVM-NEXT:       )
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
.section ".note.baz", "a"
       .align 4
       .long 8 /* namesz */
       .long end - begin /* descsz */
       .long 3 /* type */
       .asciz "FreeBSD"
begin:
       .asciz "Lorem ipsum dolor sit amet"
       .align 4
end:
