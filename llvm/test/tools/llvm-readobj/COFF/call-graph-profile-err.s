# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple x86_64-pc-win32 %s -o %t
# RUN: not llvm-readobj %t --cg-profile 2>&1 | FileCheck --check-prefix=ERR %s

## In order to use --cg-profile option, the section ".llvm.call-graph-profile" 
## should have two 4-byte fields representing the indexes of two symbols and
## one 8-byte fields representing the weight from first symbol to second 
## symbol.
## The section in this test case has 9 bytes of data, so it's malformed.

# ERR: error: '{{.*}}': Stream Error: The stream is too short to perform the requested operation.

.section .test
a:
b:
c:
d:
e:

.section ".llvm.call-graph-profile"
    .long 10    ## Symbol index of a.
    .long 11    ## Symbol index of b.
    .byte 32    ## Weight from a to b. It is an error, since it should have a length of 8 bytes.
