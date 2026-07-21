# RUN: not llvm-mc -triple x86_64 %s -o /dev/null 2>&1 | FileCheck %s --match-full-lines --strict-whitespace

## This test verifies that:
## * Macro instantiation notes point directly to the `.macro` signature line itself
##   (e.g. `.macro .test1` instead of the first statement of the macro body).
## * Nested active macro instantiation stacks unwind cleanly.
## * Column numbers and caret alignments inside `<instantiation>` virtual buffers
##   are reported correctly (e.g. matching `.macrobody0` at column 7 on line 2).


#      CHECK:<instantiation>:2:7: error: unknown directive
# CHECK-NEXT:{{^      }}.macrobody0
# CHECK-NEXT:{{^      }}^
# CHECK-NEXT:{{.*}}macro-unknown-directive.s:24:1: note: while in macro instantiation
# CHECK-NEXT:{{^}}.macro .test1
# CHECK-NEXT:{{^}}^
# CHECK-NEXT:{{.*}}macro-unknown-directive.s:29:1: note: while in macro instantiation
# CHECK-NEXT:{{^}}.test1
# CHECK-NEXT:{{^}}^
.macro .test0
  # comment inside test0
      .macrobody0
.endm
.macro .test1
  # comment inside test1
  .test0
.endm

.test1

#      CHECK:<instantiation>:1:35: error: literal value out of range for directive
# CHECK-NEXT:{{^}}mov extremely_long_register_name, 9999999999999999999999999999999999
# CHECK-NEXT:{{^                                  }}^
# CHECK-NEXT:{{.*}}macro-unknown-directive.s:42:1: note: while in macro instantiation
# CHECK-NEXT:{{^}}test_long_arg extremely_long_register_name
# CHECK-NEXT:{{^}}^

.macro test_long_arg reg
  mov \reg, 9999999999999999999999999999999999
.endm

test_long_arg extremely_long_register_name
