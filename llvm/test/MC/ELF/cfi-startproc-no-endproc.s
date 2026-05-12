# RUN: not llvm-mc %s -triple x86_64-linux -o /dev/null 2>&1 | FileCheck %s
# RUN: not llvm-mc %s -triple x86_64-linux -filetype=obj -o /dev/null 2>&1 | FileCheck %s

## https://github.com/llvm/llvm-project/issues/177852
## Check we don't crash when an inner .cfi_startproc in one section
## is left unfinished while a later frame in another section is
## properly closed (so the last DwarfFrameInfo entry has a valid End).

.pushsection .text.qux, "ax", @progbits
.type qux, @function
qux:
.cfi_startproc
.popsection

.type quux, @function
quux:
.cfi_startproc
.cfi_endproc

# CHECK: error: Unfinished frame!
