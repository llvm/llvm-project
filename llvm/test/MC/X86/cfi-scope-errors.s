# RUN: not llvm-mc %s -triple x86_64-linux -o /dev/null 2>&1 | FileCheck %s
# RUN: not llvm-mc %s -triple x86_64-linux -filetype=obj -o /dev/null 2>&1 | FileCheck %s

.text
.cfi_def_cfa rsp, 8
# CHECK: error: this directive must appear between .cfi_startproc and .cfi_endproc directives

.cfi_startproc
nop

# TODO(kristina): As Reid suggested, this now supports source locations as a side effect
# of another patch aimed at fixing the crash that would occur here, however the other
# ones do not unfortunately. Will address it in a further patch propogating SMLoc down to
# other CFI directives at which point more LINE checks can be added to ensure proper source
# location reporting.

# This tests source location correctness as well as the error and it not crashing.
# CHECK: [[@LINE+2]]:1: error: starting new .cfi frame before finishing the previous one
.cfi_startproc

nop
.cfi_endproc

.cfi_def_cfa rsp, 8
# CHECK: error: this directive must appear between .cfi_startproc and .cfi_endproc directives
