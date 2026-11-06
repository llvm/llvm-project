## Tests for llvm-mc's --verify diagnostic-verification mode.

# REQUIRES: x86-registered-target
# RUN: rm -rf %t && split-file --leading-lines %s %t && cd %t

#--- match.s
## Matching directives: a substring match, a '-re' regex match, '@+N' and
## '@-N' offsets, '@above'/'@below', ':<col>' columns, and directives in '#'
## and '//' comments.
# RUN: llvm-mc --triple=x86_64 --verify match.s
## Output files are not kept if assembling failed.
# RUN: llvm-mc --triple=x86_64 --verify --filetype=obj -o match.o match.s
# RUN: not ls match.o

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

## A diagnostic that is reported twice on a line needs two directives or a count.
.dup_directive; .dup_directive
# expected-error@-1 {{unknown directive}}
# expected-error@-2 {{unknown directive}}
.dup_directive; .dup_directive
# expected-error@-1 2 {{unknown directive}}

## A directive that was already matched does not steal a later diagnostic from
## a second directive on the same line.
movl $1, %eax, %ebx; movl $1, %eax, %ebx
# expected-error@-1 {{operand}}
# expected-error@-2 {{invalid operand}}

movl $1, %eax, %ebx
# expected-error@-1:16 {{invalid operand for instruction}}

#--- wrong-column.s
## A directive with the wrong column fails.
# RUN: not llvm-mc --triple=x86_64 --verify wrong-column.s 2>&1 \
# RUN:   | FileCheck wrong-column.s
.foo_directive # expected-error:2 {{unknown directive}}
# CHECK: wrong-column.s:[[#@LINE-1]]:1: error: unexpected error: unknown directive
# CHECK: error: expected error "unknown directive" was not produced

#--- invalid-column.s
## Invalid columns are rejected.
# RUN: not llvm-mc --triple=x86_64 --verify invalid-column.s 2>&1 \
# RUN:   | FileCheck invalid-column.s
nop
# expected-error@-1:0 {{never happens}}
# expected-error@unknown:1 {{never happens}}
# CHECK: error: invalid column ':0'
# CHECK: error: cannot specify a column with '@unknown'

#--- custom-prefix.s
## Custom prefixes are recognized instead of 'expected', but not as the suffix
## of a longer word such as 'uncheck'.
# RUN: llvm-mc --triple=x86_64 --verify=check,alt custom-prefix.s
# RUN: not llvm-mc --triple=x86_64 --verify=1x custom-prefix.s 2>&1 \
# RUN:   | FileCheck custom-prefix.s --check-prefix=BAD-PREFIX
# BAD-PREFIX: error: invalid prefix '1x' for --verify: prefixes must start with a letter and contain only alphanumeric characters, hyphens and underscores
.foo_directive
# check-error@-1 {{unknown directive}}
# uncheck-error@-2 {{not a match for --verify=check,alt}}
# expected-error@-3 {{not a match for --verify=check,alt}}

.bar_directive
# alt-error@-1 {{unknown directive}}

#--- invalid-regex.s
## Malformed '-re' directives fail verification.
# RUN: not llvm-mc --triple=x86_64 --verify invalid-regex.s 2>&1 \
# RUN:   | FileCheck invalid-regex.s
nop
# expected-error-re@-1 {{unclosed {{regex}}
# expected-error-re@-2 {{invalid {{[}} regex}}
# CHECK: error: found start of regex with no end '}}'
# CHECK: error: invalid regex:

#--- near-miss.s
## A diagnostic that matches the text but not the kind of a directive.
# RUN: not llvm-mc --triple=x86_64 --verify near-miss.s 2>&1 \
# RUN:   | FileCheck near-miss.s
.foo_directive
# expected-warning@-1 {{unknown directive}}
# CHECK: error: 'error' diagnostic emitted when expecting a 'warning'
# CHECK: error: expected warning "unknown directive" was not produced

#--- unexpected.s
## Diagnostics without a matching directive fail verification.
# RUN: not llvm-mc --triple=x86_64 --verify unexpected.s 2>&1 \
# RUN:   | FileCheck unexpected.s
.foo_directive
# CHECK: unexpected.s:[[#@LINE-1]]:1: error: unexpected error: unknown directive

#--- missing.s
## A directive that is never matched fails verification, and the output file
## is not kept.
# RUN: not llvm-mc --triple=x86_64 --verify missing.s 2>&1 \
# RUN:   | FileCheck missing.s
# RUN: not llvm-mc --triple=x86_64 --verify --filetype=obj -o missing.o missing.s 2>&1 \
# RUN:   | FileCheck missing.s
# RUN: not ls missing.o
nop
# expected-error@-1 {{this will never happen}}
# CHECK: error: expected error "this will never happen" was not produced

#--- duplicate.s
## Each directive matches only one diagnostic, and an already-matched
## directive is not reported as a near miss for a later diagnostic.
# RUN: not llvm-mc --triple=x86_64 --verify duplicate.s 2>&1 \
# RUN:   | FileCheck duplicate.s --implicit-check-not=error:
.dup_directive; .dup_directive
# expected-error@-1 {{unknown directive}}
# CHECK: duplicate.s:[[#@LINE-2]]:17: error: unexpected error: unknown directive
.foo_directive; .warning "unknown directive"
# expected-error@-1 {{unknown directive}}
# CHECK: duplicate.s:[[#@LINE-2]]:17: error: unexpected warning: unknown directive

#--- too-few.s
## Directives with an invalid count or a count that is matched fewer times fail.
# RUN: not llvm-mc --triple=x86_64 --verify too-few.s 2>&1 \
# RUN:   | FileCheck too-few.s --implicit-check-not=error:
.foo_directive
# expected-error@-1 2 {{unknown directive}}
# expected-warning@-2 0 {{zero count}}
# CHECK: too-few.s:[[#@LINE-1]]:3: error: invalid count '0'
# CHECK: too-few.s:[[#@LINE-3]]:3: error: expected error "unknown directive" was produced 1 of 2 times

#--- no-diagnostics.s
## A file without diagnostics or directives passes.
# RUN: llvm-mc --triple=x86_64 --verify no-diagnostics.s
## --as-lex reports no diagnostics, so --verify is rejected.
# RUN: not llvm-mc --triple=x86_64 --verify --as-lex no-diagnostics.s 2>&1 \
# RUN:   | FileCheck no-diagnostics.s --check-prefix=AS-LEX
# AS-LEX: error: --verify is not supported with --as-lex
nop

#--- cfi.s
## Diagnostics reported through MCContext rather than SourceMgr (e.g. by the
## CFI checker) are checked too.
# RUN: llvm-mc --triple=x86_64 --verify --validate-cfi --filetype=null cfi.s
  .text
  .globl f
  .type f,@function
f:
  .cfi_startproc
  .cfi_same_value %rax
  movq $10, %rax
  # expected-error@-1 {{changed register RAX, that register RAX's unwinding rule uses, but there is no CFI directives about it}}
  .cfi_endproc

#--- disassemble.txt
## Disassembler diagnostics are checked too.
# RUN: llvm-mc --triple=x86_64 --disassemble --verify disassemble.txt
0x0f 0xff # expected-warning {{invalid instruction encoding}}
[0x0f 0xff] # expected-warning {{invalid instruction encoding}}

#--- unknown-loc.s
## Locationless errors (such as an invalid --defsym) fail unless matched with
## '@unknown'.
# RUN: llvm-mc --triple=x86_64 --verify --defsym=bad unknown-loc.s
# RUN: not llvm-mc --triple=x86_64 --verify --defsym=bad no-diagnostics.s 2>&1 \
# RUN:   | FileCheck unknown-loc.s --check-prefix=DEFSYM
# DEFSYM: error: unexpected error: defsym must be of the form: sym=value: bad
# expected-error@unknown {{defsym must be of the form: sym=value: bad}}

#--- unknown-loc-mismatch.s
## '@unknown' does not match diagnostics that have a location.
# RUN: not llvm-mc --triple=x86_64 --verify unknown-loc-mismatch.s 2>&1 \
# RUN:   | FileCheck unknown-loc-mismatch.s --implicit-check-not=error:
.foo_directive
# expected-error@unknown {{unknown directive}}
# CHECK: unknown-loc-mismatch.s:[[#@LINE-2]]:1: error: unexpected error: unknown directive
# CHECK: error: expected error "unknown directive" was not produced
