## Verifier errors highlight the directive.

# REQUIRES: x86-registered-target
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: not llvm-mc --triple=x86_64 --verify input.s 2>&1 \
# RUN:   | FileCheck %s --strict-whitespace

## FIXME: The highlight ends in the middle of the directive.
# CHECK:      input.s:1:7: error: expected error "never produced" was not produced
# CHECK-NEXT: {{^}}nop # expected-error {{[{][{]}}never produced{{[}][}]}}{{$}}
# CHECK-NEXT: {{^}}      ^~~~~~~~~~~~~~{{$}}

#--- input.s
nop # expected-error {{never produced}}
