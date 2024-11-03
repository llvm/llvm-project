# RUN: not llvm-mc -triple riscv32 < %s 2>&1 \
# RUN:   | FileCheck -check-prefixes=CHECK %s

# CHECK: :[[#@LINE+1]]:8: error: expected identifier
.option

# CHECK: :[[#@LINE+1]]:9: error: expected identifier
.option 123

# CHECK: :[[#@LINE+1]]:9: error: expected identifier
.option "str"

# CHECK: :[[#@LINE+1]]:13: error: expected newline
.option rvc foo

# CHECK: :[[#@LINE+1]]:12: warning: unknown option, expected 'push', 'pop', 'rvc', 'norvc', 'relax' or 'norelax'
.option bar

# CHECK: :[[#@LINE+1]]:12: error: .option pop with no .option push
.option pop

# CHECK: :[[#@LINE+1]]:14: error: expected newline
.option push 123

# CHECK: :[[#@LINE+1]]:13: error: expected newline
.option pop 123
