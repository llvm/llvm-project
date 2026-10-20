## Tests for llvm-mc's --verify diagnostic-verification mode.

# REQUIRES: riscv-registered-target
# RUN: rm -rf %t && split-file %s %t && cd %t

## Matching directives: a substring match, a '-re' regex match, '@+N' and
## '@-N' offsets, '@above'/'@below', ':<col>' columns, and directives in '#'
## and '//' comments.
# RUN: llvm-mc --triple=riscv32 --verify match.s

## A directive with the wrong column, or an invalid column, fails.
# RUN: not llvm-mc --triple=riscv32 --verify wrong-column.s 2>&1 \
# RUN:   | FileCheck wrong-column.s
# RUN: not llvm-mc --triple=riscv32 --verify invalid-column.s 2>&1 \
# RUN:   | FileCheck invalid-column.s

## Custom prefixes are recognized instead of 'expected', but not as the suffix
## of a longer word such as 'uncheck'.
# RUN: llvm-mc --triple=riscv32 --verify=check,alt custom-prefix.s

## Malformed '-re' directives and out-of-range line offsets fail verification.
# RUN: not llvm-mc --triple=riscv32 --verify invalid-regex.s 2>&1 \
# RUN:   | FileCheck invalid-regex.s
# RUN: not llvm-mc --triple=riscv32 --verify invalid-offset.s 2>&1 \
# RUN:   | FileCheck invalid-offset.s

## A diagnostic that matches the text but not the kind of a directive.
# RUN: not llvm-mc --triple=riscv32 --verify near-miss.s 2>&1 \
# RUN:   | FileCheck near-miss.s

## Diagnostics without a matching directive fail verification, including when
## the directive-like text is outside of a comment.
# RUN: not llvm-mc --triple=riscv32 --verify unexpected.s 2>&1 \
# RUN:   | FileCheck unexpected.s
# RUN: not llvm-mc --triple=riscv32 --verify outside-comment.s 2>&1 \
# RUN:   | FileCheck outside-comment.s

## A directive that is never matched fails verification.
# RUN: not llvm-mc --triple=riscv32 --verify missing.s 2>&1 \
# RUN:   | FileCheck missing.s

## Each directive matches only one diagnostic.
# RUN: not llvm-mc --triple=riscv32 --verify duplicate.s 2>&1 \
# RUN:   | FileCheck duplicate.s --implicit-check-not=error:

## Directives in included files without diagnostics are checked too.
# RUN: not llvm-mc --triple=riscv32 --verify include.s 2>&1 \
# RUN:   | FileCheck included.s

## A file without diagnostics or directives passes.
# RUN: llvm-mc --triple=riscv32 --verify no-diagnostics.s

## Diagnostics reported through MCContext rather than SourceMgr (e.g. by the
## CFI checker) are checked too.
# RUN: llvm-mc --triple=riscv32 --verify --validate-cfi --filetype=null cfi.s

## Disassembler diagnostics are checked too.
# RUN: llvm-mc --triple=riscv32 --disassemble --verify disassemble.txt

## Output files are not kept if assembling failed, even if all errors were
## expected.
# RUN: llvm-mc --triple=riscv32 --verify --filetype=obj -o match.o match.s
# RUN: not ls match.o

## Errors that are not reported as diagnostics cannot be expected.
# RUN: not llvm-mc --triple=riscv32 --verify --defsym=bad no-diagnostics.s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=DEFSYM
# DEFSYM: error: defsym must be of the form: sym=value: bad

# RUN: not llvm-mc --triple=riscv32 --verify=1x no-diagnostics.s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=BAD-PREFIX
# BAD-PREFIX: error: invalid prefix '1x' for --verify: prefixes must start with a letter and contain only alphanumeric characters, hyphens and underscores

# RUN: not llvm-mc --triple=riscv32 --verify --as-lex no-diagnostics.s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=AS-LEX
# AS-LEX: error: --verify is not supported with --as-lex

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

.qux_directive # expected-error {{unknown directive}}
.quux_directive // expected-error {{unknown directive}}
.col_directive # expected-error:1 {{unknown directive}}

## A diagnostic that is reported twice on a line needs two directives.
.dup_directive; .dup_directive
# expected-error@-1 {{unknown directive}}
# expected-error@-2 {{unknown directive}}

sfence.vma a0, 0x10
# expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
# expected-note@-2:16 {{unexpected extra operand for instruction}}
# expected-note@-3:16 {{register must be a GPR}}

#--- wrong-column.s
.foo_directive # expected-error:2 {{unknown directive}}
# CHECK: error: unknown directive
# CHECK: error: expected error "unknown directive" was not produced

#--- invalid-column.s
addi a0, a0, 1
# expected-error@-1:0 {{never happens}}
# expected-error@unknown:1 {{never happens}}
# CHECK: error: invalid column ':0'
# CHECK: error: cannot specify a column with '@unknown'

#--- custom-prefix.s
.foo_directive
# check-error@-1 {{unknown directive}}
# uncheck-error@-2 {{not a match for --verify=check,alt}}
# expected-error@-3 {{not a match for --verify=check,alt}}

.bar_directive
# alt-error@-1 {{unknown directive}}

#--- invalid-regex.s
addi a0, a0, 1
# expected-error-re@-1 {{unclosed {{regex}}
# expected-error-re@-2 {{invalid {{[}} regex}}
# CHECK: error: found start of regex with no end '}}'
# CHECK: error: invalid regex:

#--- invalid-offset.s
addi a0, a0, 1
# expected-error@-99 {{before start}}
# expected-error@+99 {{past end}}
# CHECK: error: line offset '-99' is before the start of the file
# CHECK: error: line offset '+99' is past the end of the file

#--- near-miss.s
.foo_directive
# expected-warning@-1 {{unknown directive}}
# CHECK: error: 'error' diagnostic emitted when expecting a 'warning'
# CHECK: error: expected warning "unknown directive" was not produced

#--- unexpected.s
.foo_directive
# CHECK: error: unknown directive

#--- outside-comment.s
.foo_directive expected-error {{unknown directive}}
# CHECK: error: unknown directive

#--- missing.s
addi a0, a0, 1
# expected-error@-1 {{this will never happen}}
# CHECK: error: expected error "this will never happen" was not produced

#--- duplicate.s
.dup_directive; .dup_directive
# expected-error@-1 {{unknown directive}}
# CHECK: duplicate.s:1:17: error: unknown directive

#--- include.s
.include "included.s"

#--- included.s
addi a0, a0, 1
# expected-error@-1 {{this will never happen}}
# CHECK: error: expected error "this will never happen" was not produced

#--- no-diagnostics.s
addi a0, a0, 1

#--- cfi.s
  .text
  .globl f
  .type f,@function
f:
  .cfi_startproc
  .cfi_same_value a0
  li a0, 10
  # expected-error@-1 {{changed register X10, that register X10's unwinding rule uses, but there is no CFI directives about it}}
  .cfi_endproc

#--- disassemble.txt
0x00 0x00 # expected-warning {{invalid instruction encoding}}
