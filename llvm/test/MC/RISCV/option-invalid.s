# RUN: llvm-mc --verify -triple riscv32 < %s

# expected-error@+1:8 {{expected identifier}}
.option

# expected-error@+1:9 {{expected identifier}}
.option 123

# expected-error@+1:9 {{expected identifier}}
.option "str"

# expected-error@+1:13 {{expected newline}}
.option rvc foo

# expected-error@+1:23 {{unexpected token, expected + or -}}
.option arch, +f, +d, rv32ifd, -d

# expected-error@+1:22 {{expected newline}}
.option arch, rv32ifd, +f, +d

# expected-error@+1:16 {{unexpected token, expected identifier}}
.option arch, +"c"

# expected-error@+1:16 {{unknown extension feature}}
.option arch, +x

# expected-error@+1:16 {{unknown extension feature}}
.option arch, +relax

# expected-error@+1:16 {{unexpected token, expected identifier}}
.option arch, +

# expected-error@+1:18 {{expected comma}}
.option arch, +c foo

# expected-error@+1:16 {{extension version number parsing not currently implemented}}
.option arch, +c2p0

.option arch, +d
# expected-error@+1:16 {{can't disable f extension; d extension requires f extension}}
.option arch, -f

# expected-error@+1:16 {{can't disable zicsr extension; f extension requires zicsr extension}}
.option arch, -zicsr

# expected-error@+1:20 {{'f' and 'zfinx' extensions are incompatible}}
.option arch, +f, +zfinx

## Make sure the above error isn't sticky
.option arch, +f

# expected-error@+1:13 {{expected newline}}
.option rvc foo

# expected-warning@+1:12 {{unknown option, expected 'push', 'pop', 'rvc', 'norvc', 'arch', 'relax', 'norelax', 'exact', or 'noexact'}}
.option bar

# expected-error@+1:12 {{.option pop with no .option push}}
.option pop

# expected-error@+1:14 {{expected newline}}
.option push 123

# expected-error@+1:13 {{expected newline}}
.option pop 123

# expected-error@+1:15 {{bad arch string switching from rv32 to rv64}}
.option arch, rv64gc

## Enabling +zca (or +zca,+zcb,+zcmp,+zcmt) followed by +f should not imply Zcf
## (which requires C or Zce).
.option arch, rv32i
.option arch, +zca
.option arch, +f
# expected-error@+1:1 {{instruction requires the following: 'C' (Compressed Instructions) and 'F' (Single-Precision Floating-Point) or 'Zcf' (Compressed Single-Precision Floating-Point Instructions)}}
c.flw fa0, 0(a0)

.option arch, rv32i
.option arch, +zca, +zcb, +zcmp, +zcmt
.option arch, +f
# expected-error@+1:1 {{instruction requires the following: 'C' (Compressed Instructions) and 'F' (Single-Precision Floating-Point) or 'Zcf' (Compressed Single-Precision Floating-Point Instructions)}}
c.flw fa0, 0(a0)
