# RUN: not llvm-mc -triple i386-linux-gnu %s 2>&1 | FileCheck %s

# This test checks the altmacro string delimiter '<' and '>'.
# In this test we check the '.noaltmacro' directive.
# We expect that '.altmacro' and '.noaltmacro' will act as a switch on/off directives to the alternate macro mode.
# .noaltmacro returns the format into a regular macro handling.
# The default mode is ".noaltmacro". 

# Test #1: default mode
# CHECK:  error: unexpected token at start of statement
# CHECK-NEXT: <simpleCheck>:
.macro simple_check_0 name
    \name:
.endm

simple_check_0 <simpleCheck>


.altmacro
.noaltmacro

# Test #2: Switching from alternate mode to default mode
# CHECK:  error: unexpected token at start of statement
# CHECK-NEXT: <simpleCheck1>:
.macro simple_check_1 name
    \name:
.endm

simple_check_1 <simpleCheck1>
