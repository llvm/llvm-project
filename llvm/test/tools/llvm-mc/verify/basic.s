# Tests for llvm-mc's -verify=<prefixes> diagnostic-verification mode.

# REQUIRES: riscv-registered-target
# RUN: rm -rf %t && split-file %s %t

## Matching expected-error directives: a plain substring match, a '-re'
## regex match (note the nested {{...}} for the regex portion itself), an
## '@+1' forward offset, a bare ':<col>' column check, and a combined
## '@offset:col' all pass.
# RUN: llvm-mc -triple riscv32 -verify %t/match.s

## A directive with a column that doesn't match the actual diagnostic's
## column fails verification, same as a line mismatch would.
# RUN: not llvm-mc -triple riscv32 -verify %t/wrong-column.s 2>&1 \
# RUN:   | FileCheck %t/wrong-column.s --check-prefix=WRONG-COLUMN

## A custom '-verify=<prefix>' prefix is recognized instead of 'expected'.
# RUN: llvm-mc -triple riscv32 -verify=check %t/match-custom-prefix.s

## A diagnostic whose text matches an expected-* comment but whose kind
## doesn't (e.g. expected-warning for an actual error) is a "near miss": it
## still counts as not produced, since it wasn't a full match.
# RUN: not llvm-mc -triple riscv32 -verify %t/near-miss.s 2>&1 \
# RUN:   | FileCheck %t/near-miss.s --check-prefix=NEAR-MISS

## A diagnostic with no matching expected-* comment fails verification.
# RUN: not llvm-mc -triple riscv32 -verify %t/unexpected.s 2>&1 \
# RUN:   | FileCheck %t/unexpected.s --check-prefix=UNEXPECTED

## An expected-error that is never produced fails verification.
# RUN: not llvm-mc -triple riscv32 -verify %t/missing.s 2>&1 \
# RUN:   | FileCheck %t/missing.s --check-prefix=MISSING

## An expected-error in a '.include'd file that's never produced also fails
## verification: the included file's buffer isn't known to SourceMgr until
## it's actually .include'd during parsing, so process() never has an actual
## diagnostic in it to trigger scanning it. This must be caught by verify()
## re-scanning all of SourceMgr's buffers, not just the ones process() saw.
# RUN: not llvm-mc -triple riscv32 -verify -I %t %t/main-include.s 2>&1 \
# RUN:   | FileCheck %t/included-missing.s --check-prefix=INCLUDE-MISSING

## A directive-looking magic string outside of a comment is not treated as a
## directive, so the resulting diagnostic is still unexpected.
# RUN: not llvm-mc -triple riscv32 -verify %t/outside-comment.s 2>&1 \
# RUN:   | FileCheck %t/outside-comment.s --check-prefix=UNEXPECTED

## A file with no diagnostics and no expected-* comments passes. Unlike
## clang's -verify, there is no dedicated 'expected-no-diagnostics' marker:
## none is needed, since if nothing is expected and nothing happens,
## verification trivially succeeds, and any diagnostic that *does* occur is
## still caught as unexpected (see the 'unexpected' case above).
# RUN: llvm-mc -triple riscv32 -verify %t/no-diagnostics.s

## Diagnostics reported through MCContext (e.g. by the DWARF CFI checker,
## which doesn't go through SourceMgr's diagnostic handler) are also
## checked, not just diagnostics from the assembler/parser.
# RUN: llvm-mc -triple riscv32 -verify -validate-cfi -filetype=null %t/cfi.s

## -verify is rejected together with -as-lex, since -as-lex doesn't report
## its errors as diagnostics with a message/location at all.
# RUN: not llvm-mc -triple riscv32 -verify -as-lex %t/no-diagnostics.s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=AS-LEX-ERR
# AS-LEX-ERR: -verify is not supported with -as-lex

#--- match.s
.foo_directive
# expected-error@-1 {{unknown directive}}

.bar_directive
# expected-error-re@-1 {{unknown {{.*}}directive}}

# expected-error@+1 {{unknown directive}}
.baz_directive

.qux_directive # expected-error:1 {{unknown directive}}

.quux_directive
# expected-error@-1:1 {{unknown directive}}

#--- wrong-column.s
.foo_directive # expected-error:99 {{unknown directive}}
# WRONG-COLUMN: error: unknown directive
# WRONG-COLUMN: expected error "unknown directive" was not produced

#--- match-custom-prefix.s
.foo_directive
# check-error@-1 {{unknown directive}}

#--- near-miss.s
.foo_directive
# expected-warning@-1 {{unknown directive}}
# NEAR-MISS: 'error' diagnostic emitted when expecting a 'warning'
# NEAR-MISS: expected warning "unknown directive" was not produced

#--- unexpected.s
.foo_directive
# UNEXPECTED: error: unknown directive

#--- missing.s
addi a0, a0, 1
# expected-error@-1 {{this will never happen}}
# MISSING: expected error "this will never happen" was not produced

#--- main-include.s
.include "included-missing.s"

#--- included-missing.s
addi a0, a0, 1
# expected-error@-1 {{this will never happen}}
# INCLUDE-MISSING: expected error "this will never happen" was not produced

#--- outside-comment.s
.foo_directive  expected-error {{unknown directive}}
# UNEXPECTED: error: unknown directive

#--- no-diagnostics.s
addi a0, a0, 1

#--- cfi.s
	.text
	.globl	f
	.type	f,@function
f:
	.cfi_startproc
	.cfi_same_value ra
	li a0, 10
	# expected-error@-1 {{changed register X10, that register X10's unwinding rule uses, but there is no CFI directives about it}}
	.cfi_endproc
