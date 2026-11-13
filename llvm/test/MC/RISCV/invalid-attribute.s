## Negative tests:
##  - Unknown attribute name.
##  - Feed integer value to string type attribute.
##  - Feed string value to integer type attribute.
##  - Invalid arch string.

# RUN: llvm-mc --verify %s -triple=riscv32 -filetype=asm
# RUN: llvm-mc --verify %s -triple=riscv64 -filetype=asm

.attribute unknown, "unknown" # expected-error:12 {{attribute name not recognised: unknown}}

.attribute arch, "foo" # expected-error:18 {{invalid arch name 'foo', string must begin with rv32{i,e,g,y}, rv64{i,e,g,y}, or a supported profile name}}

.attribute arch, "rv32i2p1_z2p0" # expected-error:18 {{invalid arch name 'rv32i2p1_z2p0', standard user-level extension name missing after 'z'}}

.attribute arch, "rv32i2p1_t2p0" # expected-error:18 {{invalid arch name 'rv32i2p1_t2p0', unsupported standard user-level extension 't'}}

.attribute stack_align, "16" # expected-error:25 {{expected numeric constant}}

.attribute unaligned_access, "0" # expected-error:30 {{expected numeric constant}}

.attribute priv_spec, "2" # expected-error:23 {{expected numeric constant}}

.attribute priv_spec_minor, "0" # expected-error:29 {{expected numeric constant}}

.attribute priv_spec_revision, "0" # expected-error:32 {{expected numeric constant}}

.attribute arch, 30 # expected-error:18 {{expected string constant}}

.attribute atomic_abi, "16" # expected-error:24 {{expected numeric constant}}
