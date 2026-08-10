# RUN: llvm-mc -preserve-comments -triple i386-unknown-unknown %s >%t-1.s
# RUN: llvm-mc -preserve-comments -triple i386-unknown-unknown %t-1.s >%t-2.s
# RUN: diff %t-1.s %t-2.s

# Test that various assembly round-trips when run through MC; the first
# transition from hand-written to "canonical" output may introduce some small
# differences, so we don't include the initial input in the comparison.

.text

# Some of these CFI instructions didn't consume the end of statement
# consistently, which led to extra empty lines in the output.
.cfi_sections .debug_frame
.cfi_startproc
.cfi_endproc
