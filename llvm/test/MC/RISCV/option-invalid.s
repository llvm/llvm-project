# RUN: not llvm-mc -triple riscv32 < %s 2>&1 \
# RUN: | FileCheck -check-prefixes=CHECK %s

# CHECK: error: unexpected token, expected identifier
.option

# CHECK: error: unexpected token, expected identifier
.option 123

# CHECK: error: unexpected token, expected identifier
.option "str"

# CHECK: error: unexpected token, expected end of statement
.option rvc foo

# CHECK: warning: unknown option, expected 'push', 'pop', 'rvc', 'norvc', 'relax' or 'norelax'
.option bar

# CHECK: error: .option pop with no .option push
.option pop

# CHECK: error: unexpected token, expected end of statement
.option push 123

# CHECK: error: unexpected token, expected end of statement
.option pop 123
