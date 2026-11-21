## Tests for '@+N' and '@-N' line offsets in --verify directives.

# REQUIRES: x86-registered-target
# RUN: rm -rf %t && split-file %s %t && cd %t

#--- octal.s
## FIXME: '010' is parsed as octal (8), so the directive points at line 6.
# RUN: not llvm-mc --triple=x86_64 --verify octal.s 2>&1 \
# RUN:   | FileCheck octal.s --implicit-check-not=error:
.foo_directive
# CHECK: octal.s:[[#@LINE-1]]:1: error: unexpected error: unknown directive








# expected-error@-010 {{unknown directive}}
# CHECK: octal.s:[[#@LINE-1]]:3: error: expected error "unknown directive" was not produced

#--- leading-zero.s
## FIXME: '08' is not valid octal, so the offset is read uninitialized.
# DISABLED: llvm-mc --triple=x86_64 --verify leading-zero.s
.foo_directive







# expected-error@-08 {{unknown directive}}

#--- overflow.s
## FIXME: The offset does not fit in an int, so it is read uninitialized.
# DISABLED: not llvm-mc --triple=x86_64 --verify overflow.s 2>&1 \
# DISABLED:   | FileCheck overflow.s --implicit-check-not=error:
# expected-error@+99999999999 {{overflow}}
# CHECK: overflow.s:[[#@LINE-1]]:3: error: invalid line offset '+99999999999'

#--- out-of-range.s
## FIXME: Offsets before the start or past the end of the file are accepted.
# RUN: not llvm-mc --triple=x86_64 --verify out-of-range.s 2>&1 \
# RUN:   | FileCheck out-of-range.s --implicit-check-not=error:
## Line 0 is allowed since MLIR uses it for the implicit top-level module.
# expected-error@-5 {{line zero}}
# expected-error@-99 {{before start}}
# expected-error@+99 {{past end}}
# CHECK: out-of-range.s:5:3: error: expected error "line zero" was not produced
# CHECK: out-of-range.s:6:3: error: expected error "before start" was not produced
# CHECK: out-of-range.s:7:3: error: expected error "past end" was not produced
