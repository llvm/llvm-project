## Tests for llvm-mc's -verify=<prefixes> diagnostic-verification mode.

# REQUIRES: riscv-registered-target
# RUN: rm -rf %t && split-file %s %t

## Matching expected-error directives: a plain substring match, a '-re'
## regex match (note the nested {{...}} for the regex portion itself), an
## '@+1' forward offset, and '@above'/'@below' offsets all pass.
# RUN: llvm-mc -triple riscv32 -verify %t/match.s

## Malformed '-re' regex directives (missing closing '}}' or invalid regex
## syntax) and out-of-bounds line offsets fail verification with an error at
## the directive.
# RUN: not llvm-mc -triple riscv32 -verify %t/invalid-regex.s 2>&1 \
# RUN:   | FileCheck %t/invalid-regex.s --check-prefix=INVALID-REGEX
# RUN: not llvm-mc -triple riscv32 -verify %t/invalid-offset.s 2>&1 \
# RUN:   | FileCheck %t/invalid-offset.s --check-prefix=INVALID-OFFSET

## Custom '-verify=<prefixes>' prefixes (including a comma-separated list)
## are recognized instead of 'expected', and prefixes that merely end with
## '<prefix>' (such as 'other-check' or 'uncheck') are ignored.
# RUN: llvm-mc -triple riscv32 -verify=check,alt %t/match-custom-prefix.s

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

## An expected-error in a '.include'd file that emits no diagnostics is still
## checked by verify().
# RUN: not llvm-mc -triple riscv32 -verify -I %t %t/main-include.s 2>&1 \
# RUN:   | FileCheck %t/included-missing.s --check-prefix=INCLUDE-MISSING

## A directive-looking magic string outside of a comment is not treated as a
## directive, so the resulting diagnostic is still unexpected.
# RUN: not llvm-mc -triple riscv32 -verify %t/outside-comment.s 2>&1 \
# RUN:   | FileCheck %t/outside-comment.s --check-prefix=UNEXPECTED

## A file with no diagnostics and no expected-* directives passes.
# RUN: llvm-mc -triple riscv32 -verify %t/no-diagnostics.s

## Diagnostics reported through MCContext (e.g. by the DWARF CFI checker,
## which doesn't go through SourceMgr's diagnostic handler) are also
## checked, not just diagnostics from the assembler/parser.
# RUN: llvm-mc -triple riscv32 -verify -validate-cfi -filetype=null %t/cfi.s

## Disassembler diagnostics (e.g. invalid instruction encodings) are also
## checked under -disassemble -verify.
# RUN: llvm-mc -triple riscv32 -disassemble -verify %t/disassemble.txt

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

.above_directive
# expected-error@above {{unknown directive}}

# expected-error@below {{unknown directive}}
.below_directive

## Multiple expected-note directives on the same line whose substrings both
## match the first emitted note ("unexpected extra operand for instruction")
## still match both notes because unmatched directives are preferred over
## reusing an already-matched one.
sfence.vma a0, 0x10
# expected-error@-1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2 {{r}}
# expected-note@-3 {{register must be a GPR}}

#--- invalid-regex.s
addi a0, a0, 1
# expected-error-re@-1 {{unclosed {{regex}}
# expected-error-re@-2 {{invalid {{[}} regex}}
# INVALID-REGEX: error: found start of regex with no end '}}'
# INVALID-REGEX: error: invalid regex:

#--- invalid-offset.s
addi a0, a0, 1
# expected-error@-99 {{before start}}
# expected-error@+99 {{past end}}
# INVALID-OFFSET: error: line offset '-99' is before the start of the file
# INVALID-OFFSET: error: line offset '+99' is past the end of the file

#--- match-custom-prefix.s
.foo_directive
# check-error@-1 {{unknown directive}}
# other-check-error@-2 {{not a match for -verify=check,alt}}
# uncheck-error@-3 {{not a match for -verify=check,alt}}

.bar_directive
# alt-error@-1 {{unknown directive}}

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
	.cfi_same_value a0
	li a0, 10
	# expected-error@-1 {{changed register X10, that register X10's unwinding rule uses, but there is no CFI directives about it}}
	.cfi_endproc

#--- disassemble.txt
0x00 0x00 0x00 0x00 # expected-warning {{invalid instruction encoding}}
