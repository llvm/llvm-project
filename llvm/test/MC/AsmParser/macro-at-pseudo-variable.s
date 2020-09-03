# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

.macro A
  add  $1\@, %eax
.endm

.macro B
  sub  $1\@, %eax
.endm

  A
# CHECK: addl  $10, %eax
  A
# CHECK: addl  $11, %eax
  B
# CHECK: subl  $12, %eax
  B
# CHECK: subl  $13, %eax

# The following uses of \@ are undocumented, but valid:
.irpc foo,234
  add  $\foo\@, %eax
.endr
# CHECK: addl  $24, %eax
# CHECK: addl  $34, %eax
# CHECK: addl  $44, %eax

.irp reg,%eax,%ebx
  sub  $2\@, \reg
.endr
# CHECK: subl  $24, %eax
# CHECK: subl  $24, %ebx

# Test that .irp(c) and .rep(t) do not increase \@.
# Only the use of A should increase \@, so we can test that it increases by 1
# each time.

.irpc foo,123
  sub  $\foo, %eax
.endr

  A
# CHECK: addl  $14, %eax

.irp reg,%eax,%ebx
  sub  $4, \reg
.endr

  A
# CHECK: addl  $15, %eax

.rept 2
  sub  $5, %eax
.endr

  A
# CHECK: addl  $16, %eax

.rep 3
  sub  $6, %eax
.endr

  A
# CHECK: addl  $17, %eax
