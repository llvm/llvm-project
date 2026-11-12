# RUN: llvm-mc -triple i386-linux-gnu  %s| FileCheck %s

# This test checks the altmacro string delimiter '<' and '>'.

.altmacro

# Test #1:
# You can delimit strings with matching angle brackets '<' '>'.
# If an argument begins with '<' and ends with '>'.
# The argument is considered as a string.

# CHECK: simpleCheck:
.macro simple_check_0 name
    \name:
   addl $5,%eax
.endm

simple_check_0 <simpleCheck>

# Test #2:
# Except adding new string marks '<..>', a regular macro behavior is expected.

# CHECK:  simpleCheck0:
# CHECK: addl    $0, %eax
.macro concat string1 string2 string3
   \string1\string2\string3:
        addl $\string3, %eax
.endm

concat <simple>,<Check>,<0>

# Test #3:
# The altmacro cannot affect the regular less/greater behavior.

# CHECK: addl $-1, %eax
# CHECK: addl $0, %eax

.macro fun3 arg1 arg2
   addl $\arg1,%eax
   addl $\arg2,%eax
.endm

fun3 5<6 , 5>8

# Test #4:
# If a comma is present inside an angle brackets,
# the comma considered as a character and not as a separator.
# This check checks the ability to split the string to different
# arguments according to the use of the comma.
# Fun2 sees the comma as a character.
# Fun3 sees the comma as a separator.

# CHECK: addl $5, %eax
# CHECK: addl $6, %eax
.macro fun2 arg
   fun3 \arg
.endm

fun2 <5,6>

# Test #5:
# If argument begin with '<' and there is no '>' to close it.
# A regular macro behavior is expected.

# CHECK: addl $4, %eax
.macro fun4 arg1 arg2
   .if \arg2\arg1
   addl $\arg2,%eax
   .endif
.endm

fun4 <5,4
.noaltmacro
