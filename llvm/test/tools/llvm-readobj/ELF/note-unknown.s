// REQUIRES: x86-registered-target
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t.o

// RUN: llvm-readobj --notes %t.o | FileCheck %s --check-prefix=LLVM
// RUN: llvm-readelf --notes %t.o | FileCheck %s --check-prefix=GNU

// GNU:      Displaying notes found in: .note.foo
// GNU-NEXT:   Owner                 Data size       Description
// GNU-NEXT:   XYZ                  0x00000000       Unknown note type: (0x00000003)
// GNU-EMPTY:
// GNU-NEXT: Displaying notes found in: .note.bar
// GNU-NEXT:   Owner                 Data size       Description
// GNU-NEXT:   XYZ                  0x0000001c       Unknown note type: (0x00000003)
// GNU-NEXT:     description data: 4c 6f 72 65 6d 20 69 70 73 75 6d 20 64 6f 6c 6f 72 20 73 69 74 20 61 6d 65 74 00 00
// GNU-EMPTY:

// LLVM:      Notes [
// LLVM-NEXT:   NoteSection {
// LLVM-NEXT:     Name: .note.foo
// LLVM-NEXT:     Offset: 0x40
// LLVM-NEXT:     Size: 0x10
// LLVM-NEXT:     Note {
// LLVM-NEXT:       Owner: XYZ
// LLVM-NEXT:       Data size: 0x0
// LLVM-NEXT:       Type: Unknown (0x00000003)
// LLVM-NEXT:     }
// LLVM-NEXT:   }
// LLVM-NEXT:   NoteSection {
// LLVM-NEXT:     Name: .note.bar
// LLVM-NEXT:     Offset: 0x50
// LLVM-NEXT:     Size: 0x2C
// LLVM-NEXT:     Note {
// LLVM-NEXT:       Owner: XYZ
// LLVM-NEXT:       Data size: 0x1C
// LLVM-NEXT:       Type: Unknown (0x00000003)
// LLVM-NEXT:       Description data (
// LLVM-NEXT:         0000: 4C6F7265 6D206970 73756D20 646F6C6F  |Lorem ipsum dolo|
// LLVM-NEXT:         0010: 72207369 7420616D 65740000           |r sit amet..|
// LLVM-NEXT:       )
// LLVM-NEXT:     }
// LLVM-NEXT:   }
// LLVM-NEXT:   NoteSection {
// LLVM-NEXT:     Name: .note.8
// LLVM-NEXT:     Offset: 0x80
// LLVM-NEXT:     Size: 0x40
// LLVM-NEXT:     Note {
// LLVM-NEXT:       Owner: WXYZ
// LLVM-NEXT:       Data size: 0x8
// LLVM-NEXT:       Type: Unknown (0x00000006)
// LLVM-NEXT:       Description data (
// LLVM-NEXT:         0000: 4C6F7265 6D000000                    |Lorem...|
// LLVM-NEXT:       )
// LLVM-NEXT:     }
// LLVM-NEXT:     Note {
// LLVM-NEXT:       Owner: VWXYZ
// LLVM-NEXT:       Data size: 0x8
// LLVM-NEXT:       Type: Unknown (0x00000006)
// LLVM-NEXT:       Description data (
// LLVM-NEXT:         0000: 78787800 00000000                    |xxx.....|
// LLVM-NEXT:       )
// LLVM-NEXT:     }
// LLVM-NEXT:   }
// LLVM-NEXT: ]

.section ".note.foo", "a"
	.long 4 /* namesz */
	.long 0 /* descsz */
	.long 3 /* type */
	.asciz "XYZ"
	.align 4
.section ".note.bar", "a"
       .long 4 /* namesz */
       .long end - begin /* descsz */
       .long 3 /* type */
       .asciz "XYZ"
       .align 4
begin:
       .asciz "Lorem ipsum dolor sit amet"
       .align 4
end:

.section ".note.8", "a"
       .long 5 /* namesz */
       .long 2f - 1f /* descsz */
       .long 6 /* type */
       .asciz "WXYZ"
       .align 8
1:
       .asciz "Lorem"
       .align 8
2:

       .long 6 /* namesz */
       .long 2f - 1f /* descsz */
       .long 6 /* type */
       .asciz "VWXYZ"
       .align 8
1:
       .asciz "xxx"
       .align 8
2:
