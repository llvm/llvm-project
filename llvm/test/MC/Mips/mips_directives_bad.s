# Error checking for malformed directives
# RUN: not llvm-mc -triple mips-unknown-unknown %s 2>&1 | FileCheck %s

    .abicalls should have no operands
# CHECK:    :{{[0-9]+}}:{{[0-9]+}}: error: unexpected token, expected end of statement
# CHECK-NEXT:    .abicalls should have no operands
# CHECK-NEXT:              ^

# We don't know yet how to represent a list of options
# pic2 will eventually be legal so we will probably want
# to change it to something silly.

# Blank option operand
    .option 
# CHECK-NEXT:    :{{[0-9]+}}:{{[0-9]+}}: error: unexpected token, expected identifier
# CHECK-NEXT:    .option 
# CHECK-NEXT:            ^

# Numeric option operand
    .option 2
# CHECK-NEXT:    :{{[0-9]+}}:{{[0-9]+}}: error: unexpected token, expected identifier
# CHECK-NEXT:    .option 2
# CHECK-NEXT:            ^

# Register option operand
    .option $2
# CHECK-NEXT:    :{{[0-9]+}}:{{[0-9]+}}: error: unexpected token, expected identifier
# CHECK-NEXT:    .option $2
# CHECK-NEXT:            ^

    .option WithBadOption
# CHECK-NEXT:    :{{[0-9]+}}:{{[0-9]+}}: warning: unknown option, expected 'pic0' or 'pic2'
# CHECK-NEXT:    .option WithBadOption
# CHECK-NEXT:            ^

    .option pic0,
# CHECK-NEXT:    :{{[0-9]+}}:{{[0-9]+}}: error: unexpected token, expected end of statement
# CHECK-NEXT:    .option pic0,
# CHECK-NEXT:                ^

    .option pic0,pic2
# CHECK-NEXT:    :{{[0-9]+}}:{{[0-9]+}}: error: unexpected token, expected end of statement
# CHECK-NEXT:    .option pic0,pic2
# CHECK-NEXT:                ^

    .option pic0 pic2
# CHECK-NEXT:    :{{[0-9]+}}:{{[0-9]+}}: error: unexpected token, expected end of statement
# CHECK-NEXT:    .option pic0 pic2
# CHECK-NEXT:                 ^

    .option pic2,
# CHECK-NEXT:    :{{[0-9]+}}:{{[0-9]+}}: error: unexpected token, expected end of statement
# CHECK-NEXT:    .option pic2,
# CHECK-NEXT:                ^

    .option pic2 pic3
# CHECK-NEXT:    :{{[0-9]+}}:{{[0-9]+}}: error: unexpected token, expected end of statement
# CHECK-NEXT:    .option pic2 pic3
# CHECK-NEXT:                 ^
