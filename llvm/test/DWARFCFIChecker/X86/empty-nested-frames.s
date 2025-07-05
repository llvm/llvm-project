# RUN: llvm-mc -triple x86_64-pc-linux-gnu %s --validate-cfi --filetype=null 2>&1 \
# RUN:   | FileCheck %s --allow-empty 
# CHECK-NOT: warning:

.pushsection A
f: 
.cfi_startproc
.pushsection B
g: 
.cfi_startproc
ret
.cfi_endproc
.popsection
ret
.cfi_endproc
.popsection
