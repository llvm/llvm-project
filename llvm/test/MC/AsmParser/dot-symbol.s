# Historically 'as' treats '.' as a reference to the current location in
# arbitrary contexts. We don't support this in general.

# RUN: not llvm-mc -triple i386-unknown-unknown %s 2> %t
# RUN: FileCheck -input-file %t %s

# CHECK: invalid use of pseudo-symbol '.' as a label
.:
        .long 0
