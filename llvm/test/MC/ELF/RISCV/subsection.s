# RUN: not llvm-mc -filetype=obj -triple=riscv64 -mattr=-relax %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:
# RUN: not llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR --implicit-check-not=error:

a:
  nop
b:
  call foo@plt
c:
  nop
d:

.data
## Positive subsection numbers
.subsection c-b
.subsection d-b
.subsection c-a

.subsection b-a
.subsection d-c

## Negative subsection numbers
# ERR: :[[#@LINE+1]]:14: error: subsection number -8 is not within [0,2147483647]
.subsection b-c
# ERR: :[[#@LINE+1]]:14: error: subsection number -12 is not within [0,2147483647]
.subsection b-d
# ERR: :[[#@LINE+1]]:14: error: subsection number -12 is not within [0,2147483647]
.subsection a-c
# ERR: :[[#@LINE+1]]:14: error: subsection number -4 is not within [0,2147483647]
.subsection a-b
# ERR: :[[#@LINE+1]]:14: error: subsection number -4 is not within [0,2147483647]
.subsection c-d
