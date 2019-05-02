// REQUIRES: x86-registered-target
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t
// RUN: llvm-readelf --notes %t | FileCheck %s --check-prefix=GNU
// RUN: llvm-readobj --elf-output-style LLVM --notes %t | FileCheck %s --check-prefix=LLVM

// GNU:      Displaying notes found at file offset 0x00000040 with length 0x00000014:
// GNU-NEXT:   Owner                 Data size       Description
// GNU-NEXT:   GNU                   0x00000004      NT_GNU_PROPERTY_TYPE_0 (property note)
// GNU-NEXT:     Properties:  <corrupted GNU_PROPERTY_TYPE_0>

// LLVM:      Notes [
// LLVM-NEXT:   NoteSection {
// LLVM-NEXT:     Offset: 0x40
// LLVM-NEXT:     Size: 0x14
// LLVM-NEXT:     Note {
// LLVM-NEXT:       Owner: GNU
// LLVM-NEXT:       Data size: 0x4
// LLVM-NEXT:       Type: NT_GNU_PROPERTY_TYPE_0 (property note)
// LLVM-NEXT:       Property [
// LLVM-NEXT:         <corrupted GNU_PROPERTY_TYPE_0>
// LLVM-NEXT:       ]
// LLVM-NEXT:     }
// LLVM-NEXT:   }
// LLVM-NEXT: ]

// Section below is broken, check we report that.

.section ".note.gnu.property", "a"
.align 4 
  .long 4           /* Name length is always 4 ("GNU") */
  .long end - begin /* Data length */
  .long 5           /* Type: NT_GNU_PROPERTY_TYPE_0 */
  .asciz "GNU"      /* Name */
  .p2align 3
begin:
  .long 1           /* Type: GNU_PROPERTY_STACK_SIZE */
end:
