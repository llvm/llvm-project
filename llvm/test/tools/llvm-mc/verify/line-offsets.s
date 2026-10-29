## Tests for '@+N' and '@-N' line offsets in --verify directives.

# REQUIRES: x86-registered-target
# RUN: rm -rf %t && split-file %s %t && cd %t

#--- octal.s
## '010' is decimal, not octal.
# RUN: llvm-mc --triple=x86_64 --verify octal.s
.foo_directive









# expected-error@-010 {{unknown directive}}

#--- leading-zero.s
## '08' is decimal, not an invalid octal number.
# RUN: llvm-mc --triple=x86_64 --verify leading-zero.s
.foo_directive







# expected-error@-08 {{unknown directive}}

#--- overflow.s
## Offsets that do not fit in an int are rejected.
# RUN: not llvm-mc --triple=x86_64 --verify overflow.s 2>&1 \
# RUN:   | FileCheck overflow.s --implicit-check-not=error:
# expected-error@+99999999999 {{overflow}}
# CHECK: overflow.s:[[#@LINE-1]]:3: error: invalid line offset '+99999999999'

#--- out-of-range.s
## Offsets before the start or past the end of the file are rejected.
# RUN: not llvm-mc --triple=x86_64 --verify out-of-range.s 2>&1 \
# RUN:   | FileCheck out-of-range.s --implicit-check-not=error:
## Line 0 is allowed since MLIR uses it for the implicit top-level module.
# expected-error@-5 {{line zero}}
# expected-error@-99 {{before start}}
# expected-error@+99 {{past end}}
# CHECK: out-of-range.s:6:3: error: line offset '-99' is before the start of the file
# CHECK: out-of-range.s:7:3: error: line offset '+99' is past the end of the file
# CHECK: out-of-range.s:5:3: error: expected error "line zero" was not produced
