# RUN: not llvm-mc -triple x86_64 %s -o /dev/null 2>&1 | FileCheck %s --match-full-lines --strict-whitespace
# RUN: not llvm-mc -triple x86_64-apple-darwin10 %s -o /dev/null 2>&1 | FileCheck %s --match-full-lines --strict-whitespace

.macro .test0
.endmacro

.macros_off
#      CHECK:{{.*}}.s:[[#@LINE+3]]:1: error: unknown directive
# CHECK-NEXT:.test0
# CHECK-NEXT:^
.test0
.macros_on

.test0

# CHECK-NEXT:{{.*}}.s:[[#@LINE+3]]:1: error: macro '.test0' is already defined
# CHECK-NEXT:.macro .test0
# CHECK-NEXT:^
.macro .test0
.endmacro

# CHECK-NEXT:{{.*}}.s:[[#@LINE+3]]:10: error: unexpected '.endmacro' in file, no current macro definition
# CHECK-NEXT:.endmacro
# CHECK-NEXT:         ^
.endmacro

# CHECK-NEXT:{{.*}}.s:[[#@LINE+3]]:1: error: no matching '.endmacro' in definition
# CHECK-NEXT:.macro dummy
# CHECK-NEXT:^
.macro dummy
