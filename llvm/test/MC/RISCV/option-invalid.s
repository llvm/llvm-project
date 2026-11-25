# RUN: llvm-mc --verify -triple riscv32 < %s

.option
# expected-error@-1:8 {{expected identifier}}

.option 123 # expected-error:9 {{expected identifier}}

.option "str" # expected-error:9 {{expected identifier}}

.option rvc foo # expected-error:13 {{expected newline}}

.option arch, +f, +d, rv32ifd, -d # expected-error:23 {{unexpected token, expected + or -}}

.option arch, rv32ifd, +f, +d # expected-error:22 {{expected newline}}

.option arch, +"c" # expected-error:16 {{unexpected token, expected identifier}}

.option arch, +x # expected-error:16 {{unknown extension feature}}

.option arch, +relax # expected-error:16 {{unknown extension feature}}

.option arch, +
# expected-error@-1:16 {{unexpected token, expected identifier}}

.option arch, +c foo # expected-error:18 {{expected comma}}

.option arch, +c2p0 # expected-error:16 {{extension version number parsing not currently implemented}}

.option arch, +d
.option arch, -f # expected-error:16 {{can't disable f extension; d extension requires f extension}}

.option arch, -zicsr # expected-error:16 {{can't disable zicsr extension; f extension requires zicsr extension}}

.option arch, +f, +zfinx # expected-error:20 {{'f' and 'zfinx' extensions are incompatible}}

## Make sure the above error isn't sticky
.option arch, +f

.option rvc foo # expected-error:13 {{expected newline}}

.option bar
# expected-warning@-1:12 {{unknown option, expected 'push', 'pop', 'rvc', 'norvc', 'arch', 'relax', 'norelax', 'exact', or 'noexact'}}

.option pop
# expected-error@-1:12 {{.option pop with no .option push}}

.option push 123 # expected-error:14 {{expected newline}}

.option pop 123 # expected-error:13 {{expected newline}}

.option arch, rv64gc # expected-error:15 {{bad arch string switching from rv32 to rv64}}

## Enabling +zca (or +zca,+zcb,+zcmp,+zcmt) followed by +f should not imply Zcf
## (which requires C or Zce).
.option arch, rv32i
.option arch, +zca
.option arch, +f
c.flw fa0, 0(a0) # expected-error:1 {{instruction requires the following: 'C' (Compressed Instructions) and 'F' (Single-Precision Floating-Point) or 'Zcf' (Compressed Single-Precision Floating-Point Instructions)}}

.option arch, rv32i
.option arch, +zca, +zcb, +zcmp, +zcmt
.option arch, +f
c.flw fa0, 0(a0) # expected-error:1 {{instruction requires the following: 'C' (Compressed Instructions) and 'F' (Single-Precision Floating-Point) or 'Zcf' (Compressed Single-Precision Floating-Point Instructions)}}
