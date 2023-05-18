# RUN: export LSAN_OPTIONS=detect_leaks=0
# RUN: not llvm-mc -triple x86_64 %s -o /dev/null 2>&1 | FileCheck %s

## This also tests that we don't assert due to an active macro instantiation.
# CHECK: <instantiation>:4:1: error: unmatched .ifs or .elses

	.macro macro parameter=0
		.if \parameter
		.else
	.endm

	macro 1

