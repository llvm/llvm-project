# Historically 'as' treats '.' as a reference to the current location in
# arbitrary contects. We don't support this in general.

# RUN: not llvm-mc -triple i386-unknown-unknown %s 2> %t
# RUN: FileCheck -input-file %t %s

# CHECK: assignment to pseudo-symbol '.' is unsupported (use '.space' or '.org').
. = . + 8

# CHECK: invalid use of pseudo-symbol '.' as a label
.:
        .long 0
