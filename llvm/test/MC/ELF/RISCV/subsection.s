# RUN: not llvm-mc -filetype=obj -triple=riscv64 -mattr=-relax %s -o /dev/null 2>&1 | FileCheck %s --check-prefixes=ERR,NORELAX --implicit-check-not=error:
# RUN: not llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax %s -o /dev/null 2>&1 | FileCheck %s --check-prefixes=ERR,RELAX --implicit-check-not=error:

a:
  nop
b:
  call foo@plt
c:
  nop
d:

.data
## Positive subsection numbers
## With relaxation, report an error as c-b is not an assemble-time constant.
# RELAX: :[[#@LINE+1]]:14: error: cannot evaluate subsection number
.subsection c-b
# RELAX: :[[#@LINE+1]]:14: error: cannot evaluate subsection number
.subsection d-b
# RELAX: :[[#@LINE+1]]:14: error: cannot evaluate subsection number
.subsection c-a

.subsection b-a
.subsection d-c

## Negative subsection numbers
# NORELAX: :[[#@LINE+2]]:14: error: subsection number -8 is not within [0,2147483647]
# RELAX:   :[[#@LINE+1]]:14: error: cannot evaluate subsection number
.subsection b-c
# NORELAX: :[[#@LINE+2]]:14: error: subsection number -12 is not within [0,2147483647]
# RELAX:   :[[#@LINE+1]]:14: error: cannot evaluate subsection number
.subsection b-d
# NORELAX: :[[#@LINE+2]]:14: error: subsection number -12 is not within [0,2147483647]
# RELAX:   :[[#@LINE+1]]:14: error: cannot evaluate subsection number
.subsection a-c
# ERR:     :[[#@LINE+1]]:14: error: subsection number -4 is not within [0,2147483647]
.subsection a-b
# ERR:     :[[#@LINE+1]]:14: error: subsection number -4 is not within [0,2147483647]
.subsection c-d
