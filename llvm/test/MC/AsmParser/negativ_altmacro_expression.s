# RUN: not llvm-mc -triple i386-linux-gnu %s 2>&1 | FileCheck %s

# This test is a negative test for the altmacro expression.
# In this test we check the '.noaltmacro' directive.
# We expect that '.altmacro' and '.noaltmacro' will act as a switch on/off directives to the alternate macro mode.
# .noaltmacro returns the format into a regular macro handling.
# The default mode is ".noaltmacro" as first test checks.

# CHECK:  error: unknown token in expression
# CHECK-NEXT: addl $%(1%4), %eax
.macro inner_percent arg
    addl $\arg, %eax
.endm

inner_percent %(1%4)

.altmacro
.noaltmacro

# CHECK: multi_args_macro %(1+4-5) 1 %2+1
# CHECK: error: unknown token in expression
# CHECK-NEXT: addl $%(1+4-5), %eax


# CHECK: multi_args_macro %(1+4-5),1,%4%10
# CHECK: error: unknown token in expression
# CHECK-NEXT: addl $%(1+4-5), %eax
.macro multi_args_macro arg1 arg2 arg3
  label\arg1\arg2\arg3:
  addl $\arg1, %eax
.endm

multi_args_macro %(1+4-5) 1 %2+1
multi_args_macro %(1+4-5),1,%4%10
