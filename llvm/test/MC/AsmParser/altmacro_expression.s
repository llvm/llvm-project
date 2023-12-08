# RUN: llvm-mc -triple i386-linux-gnu %s | FileCheck %s

# Checking that the '%' was evaluated as a string first
# In a fail scenario: The asmprint will print: addl $%(1+4), %eax

# CHECK:  addl $5, %eax
.altmacro
.macro percent_expr arg
    addl $\arg, %eax
.endm

percent_expr %(1+4)


# Checking that the second '%' acts as modulo operator
# The altmacro percent '%' must be located before the first argument
# If a percent is located in the middle of the estimated argument without any
# '%' in the beginning , error will be generated.
# The second percent '%' after the first altmacro percent '%' is a regular operator.

# CHECK:  addl $1, %eax
.macro inner_percent arg
    addl $\arg, %eax
.endm

inner_percent %(1%4)


# Checking for nested macro
# The first argument use is for the calling function and the second use is for the evaluation.

# CHECK:  addl    $1, %eax
.macro macro_call_0 number
    addl $\number, %eax
.endm

.macro macro_call_1 number
    macro_call_\number %(\number + 1)
.endm

macro_call_1 %(1-1)


# Checking the ability to pass a number of arguments.
# The arguments can be separated by ',' or not.

# CHECK: label013:
# CHECK:  addl $0, %eax
# CHECK:  addl $1, %eax
# CHECK:  addl $3, %eax

# CHECK: label014:
# CHECK:  addl $0, %eax
# CHECK:  addl $1, %eax
# CHECK:  addl $4, %eax

.macro multi_args_macro arg1 arg2 arg3
    label\arg1\arg2\arg3:
	addl $\arg1, %eax
	addl $\arg2, %eax
	addl $\arg3, %eax
.endm

multi_args_macro %(1+4-5) 1 %2+1
multi_args_macro %(1+4-5),1,%4%10
