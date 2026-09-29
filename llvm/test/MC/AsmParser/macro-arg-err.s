# RUN: not llvm-mc -triple=x86_64 %s -o /dev/null 2>&1 | FileCheck %s --match-full-lines --strict-whitespace

#      CHECK:{{.*}}.s:[[#@LINE+3]]:22: error: macro 'dup_params' has multiple parameters named 'a'
# CHECK-NEXT:.macro dup_params a a
# CHECK-NEXT:                     ^
.macro dup_params a a
#      CHECK:{{.*}}.s:[[#@LINE+3]]:6: error: unexpected '.endm' in file, no current macro definition
# CHECK-NEXT:.endm
# CHECK-NEXT:     ^
.endm

# CHECK-NEXT:{{.*}}.s:[[#@LINE+6]]:14: error: too many positional arguments
# CHECK-NEXT:one_arg 42,  42
# CHECK-NEXT:             ^
.macro one_arg bar
        .long \bar
.endm
one_arg 42,  42

# CHECK-NEXT:{{.*}}.s:[[#@LINE+6]]:10: error: Wrong number of arguments
# CHECK-NEXT:no_arg 42
# CHECK-NEXT:         ^
.macro no_arg
.ascii "$20"
.endm
no_arg 42

.macro double first = -1, second = -1
.long \first
.long \second
.endm

# CHECK-NEXT:{{.*}}.s:[[#@LINE+3]]:14: error: too many positional arguments
# CHECK-NEXT:double 0, 1, 2
# CHECK-NEXT:             ^
double 0, 1, 2

# CHECK-NEXT:{{.*}}.s:[[#@LINE+3]]:20: error: cannot mix positional and keyword arguments
# CHECK-NEXT:double second = 1, 2
# CHECK-NEXT:                   ^
double second = 1, 2

# CHECK-NEXT:{{.*}}.s:[[#@LINE+3]]:8: error: parameter named 'third' does not exist for macro 'double'
# CHECK-NEXT:double third = 0
# CHECK-NEXT:       ^
double third = 0

# CHECK-NEXT:{{.*}}.s:[[#@LINE+3]]:8: error: expected identifier in '.macro' directive
# CHECK-NEXT:.macro 23
# CHECK-NEXT:       ^
.macro 23

# CHECK-NEXT:{{.*}}.s:[[#@LINE+3]]:10: error: expected identifier in '.macro' directive
# CHECK-NEXT:.macro a 23
# CHECK-NEXT:         ^
.macro a 23
