# RUN: not llvm-mc -triple riscv32 < %s 2>&1 \
# RUN:   | FileCheck --implicit-check-not=error: %s

# CHECK: :[[#@LINE+1]]:8: error: expected identifier
.option

# CHECK: :[[#@LINE+1]]:9: error: expected identifier
.option 123

# CHECK: :[[#@LINE+1]]:9: error: expected identifier
.option "str"

# CHECK: :[[#@LINE+1]]:13: error: expected newline
.option rvc foo

# CHECK: :[[#@LINE+1]]:23: error: unexpected token, expected + or -
.option arch, +f, +d, rv32ifd, -d

# CHECK: :[[#@LINE+1]]:22: error: expected newline
.option arch, rv32ifd, +f, +d

# CHECK: :[[#@LINE+1]]:16: error: unexpected token, expected identifier
.option arch, +"c"

# CHECK: :[[#@LINE+1]]:16: error: unknown extension feature
.option arch, +x

# CHECK: :[[#@LINE+1]]:16: error: unknown extension feature
.option arch, +relax

# CHECK: :[[#@LINE+1]]:16: error: unexpected token, expected identifier
.option arch, +

# CHECK: :[[#@LINE+1]]:18: error: expected comma
.option arch, +c foo

# CHECK: :[[#@LINE+1]]:16: error: Extension version number parsing not currently implemented
.option arch, +c2p0

.option arch, +d
# CHECK: :[[#@LINE+1]]:16: error: Can't disable f extension, d extension requires f extension be enabled
.option arch, -f

# CHECK: :[[#@LINE+1]]:16: error: Can't disable zicsr extension, f extension requires zicsr extension be enabled
.option arch, -zicsr

# CHECK: :[[#@LINE+1]]:20: error: 'f' and 'zfinx' extensions are incompatible
.option arch, +f, +zfinx

## Make sure the above error isn't sticky
.option arch, +f

# CHECK: :[[#@LINE+1]]:13: error: expected newline
.option rvc foo

# CHECK: :[[#@LINE+1]]:12: warning: unknown option, expected 'push', 'pop', 'rvc', 'norvc', 'arch', 'relax' or 'norelax'
.option bar

# CHECK: :[[#@LINE+1]]:16: error: unknown extension feature
.option arch, -i

# CHECK: :[[#@LINE+1]]:12: error: .option pop with no .option push
.option pop

# CHECK: :[[#@LINE+1]]:14: error: expected newline
.option push 123

# CHECK: :[[#@LINE+1]]:13: error: expected newline
.option pop 123

# CHECK: :[[#@LINE+1]]:15: error: bad arch string switching from rv32 to rv64
.option arch, rv64gc
