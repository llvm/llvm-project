## Check how the PointerToRawData field is set for empty sections.
##
## Some tools may leave it set to a nonzero value, while others
## set it to zero. Currently, the LLVM MC layer produces a .data
## section with a nonzero PointerToRawData value, even if no
## .data section needs to be emitted. Other tools, such as yaml2obj
## writes a zero field if there's no data.
##
## When llvm-objcopy copies object files, it either needs to
## update the value to a valid value (within the bounds of the
## file, even if the section is empty) or zero. Some tools
## (such as obj2yaml or llvm-objcopy) can error out if the value
## is out of bounds.
##
## Check that our input file has got a nonzero field, and that
## it is set to zero after a run with llvm-objcopy.

# REQUIRES: x86-registered-target

# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj %s -o %t.in.obj
# RUN: llvm-readobj --sections %t.in.obj | FileCheck %s --check-prefix=INPUT
# RUN: llvm-objcopy --remove-section=.bss %t.in.obj %t.out.obj
# RUN: llvm-readobj --sections %t.out.obj | FileCheck %s --check-prefix=OUTPUT

# INPUT:          Name: .data
# INPUT-NEXT:     VirtualSize: 0x0
# INPUT-NEXT:     VirtualAddress: 0x0
# INPUT-NEXT:     RawDataSize: 0
# INPUT-NEXT:     PointerToRawData: 0x8D

# OUTPUT:         Name: .data
# OUTPUT-NEXT:    VirtualSize: 0x0
# OUTPUT-NEXT:    VirtualAddress: 0x0
# OUTPUT-NEXT:    RawDataSize: 0
# OUTPUT-NEXT:    PointerToRawData: 0x0{{$}}

    .text
    .globl func
func:
    ret
