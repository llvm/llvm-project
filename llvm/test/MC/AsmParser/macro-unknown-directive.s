# RUN: not llvm-mc -triple x86_64 %s -o /dev/null 2>&1 | FileCheck %s --match-full-lines --strict-whitespace

#      CHECK:{{.*}}macro-unknown-directive.s:13:1: error: unknown directive
# CHECK-NEXT:.macrobody0
# CHECK-NEXT:^
# CHECK-NEXT:{{.*}}macro-unknown-directive.s:16:1: note: while in macro instantiation
# CHECK-NEXT:.test0
# CHECK-NEXT:^
# CHECK-NEXT:{{.*}}macro-unknown-directive.s:19:1: note: while in macro instantiation
# CHECK-NEXT:.test1
# CHECK-NEXT:^
.macro .test0
.macrobody0
.endm
.macro .test1
.test0
.endm

.test1
