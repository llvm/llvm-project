// REQUIRES: x86-registered-target
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t
// RUN: llvm-readobj -elf-output-style GNU --notes %t | FileCheck %s

// CHECK:      Displaying notes found at file offset 0x00000040 with length 0x00000014:
// CHECK-NEXT:   Owner                 Data size       Description
// CHECK-NEXT:   GNU                   0x00000004      NT_GNU_PROPERTY_TYPE_0 (property note)
// CHECK-NEXT:     Properties:  <corrupted GNU_PROPERTY_TYPE_0>

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
