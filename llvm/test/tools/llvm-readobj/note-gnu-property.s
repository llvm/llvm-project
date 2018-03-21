// REQUIRES: x86-registered-target
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t
// RUN: llvm-readobj -elf-output-style GNU --notes %t | FileCheck %s

// CHECK:      Displaying notes found at file offset 0x00000040 with length 0x00000070:
// CHECK-NEXT:   Owner                 Data size       Description
// CHECK-NEXT:   GNU                   0x00000060      NT_GNU_PROPERTY_TYPE_0 (property note)
// CHECK-NEXT:     Properties:  stack size: 0x100
// CHECK-NEXT:     stack size: 0x100
// CHECK-NEXT:     no copy on protected
// CHECK-NEXT:     <application-specific type 0xfefefefe>
// CHECK-NEXT:     stack size: <corrupt length: 0x0>
// CHECK-NEXT:     stack size: <corrupt length: 0x4> 
// CHECK-NEXT:     no copy on protected <corrupt length: 0x1>
// CHECK-NEXT:     <corrupt type (0x2) datasz: 0x1>

.section ".note.gnu.property", "a"
.align 4 
  .long 4           /* Name length is always 4 ("GNU") */
  .long end - begin /* Data length */
  .long 5           /* Type: NT_GNU_PROPERTY_TYPE_0 */
  .asciz "GNU"      /* Name */
  .p2align 3
begin:
  .long 1           /* Type: GNU_PROPERTY_STACK_SIZE */
  .long 8           /* Data size */
  .quad 0x100       /* Data (stack size) */
  .p2align 3        /* Align to 8 byte for 64 bit */
  
  /* Test we handle alignment properly */
  .long 1           /* Type: GNU_PROPERTY_STACK_SIZE */
  .long 8           /* Data size */
  .long 0x100       /* Data (stack size) */
  .p2align 3        /* Align to 8 byte for 64 bit */
  
  .long 2           /* Type: GNU_PROPERTY_NO_COPY_ON_PROTECTED */
  .long 0           /* Data size */
  .p2align 3        /* Align to 8 byte for 64 bit */
  
  /* All notes below are broken. Test we are able to report them. */
  
  /* Broken note type */
  .long 0xfefefefe  /* Invalid type for testing */
  .long 0           /* Data size */
  .p2align 3        /* Align to 8 byte for 64 bit */
  
  /* GNU_PROPERTY_STACK_SIZE with zero stack size */
  .long 1           /* Type: GNU_PROPERTY_STACK_SIZE */
  .long 0           /* Data size */
  .p2align 3        /* Align to 8 byte for 64 bit */
  
  /* GNU_PROPERTY_STACK_SIZE with data size 4 (should be 8) */
  .long 1           /* Type: GNU_PROPERTY_STACK_SIZE */
  .long 4           /* Data size */
  .long 0x100       /* Data (stack size) */
  .p2align 3        /* Align to 8 byte for 64 bit */
  
  /* GNU_PROPERTY_NO_COPY_ON_PROTECTED with pr_datasz and some data */
  .long 2           /* Type: GNU_PROPERTY_NO_COPY_ON_PROTECTED */
  .long 1           /* Data size (corrupted) */
  .byte 1           /* Data */
  .p2align 3        /* Align to 8 byte for 64 bit */
  
  /* GNU_PROPERTY_NO_COPY_ON_PROTECTED with pr_datasz and without data */
  .long 2           /* Type: GNU_PROPERTY_NO_COPY_ON_PROTECTED */
  .long 1           /* Data size (corrupted) */
  .p2align 3        /* Align to 8 byte for 64 bit */
end:
