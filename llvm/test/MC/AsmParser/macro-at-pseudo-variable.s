# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -triple i386 a.s | FileCheck %s
# RUN: llvm-mc -triple i386 b.s | FileCheck %s --check-prefix=CHECK2

#--- a.s
.macro A
  add  $1\@, %eax
  add  $2\+, %eax
.endm

.macro B
  sub  $1\@, %eax
  sub  $2\+, %eax
.endm

  A
# CHECK:      addl  $10, %eax
# CHECK-NEXT: addl  $20, %eax
  A
# CHECK:      addl  $11, %eax
# CHECK-NEXT: addl  $21, %eax
  B
# CHECK:      subl  $12, %eax
# CHECK-NEXT: subl  $20, %eax
  B
# CHECK:      subl  $13, %eax
# CHECK-NEXT: subl  $21, %eax

# The following uses of \@ are undocumented, but valid:
.irpc foo,234
  add  $\foo\@, %eax
.endr
# CHECK:      addl  $24, %eax
# CHECK-NEXT: addl  $34, %eax
# CHECK-NEXT: addl  $44, %eax

.irp reg,%eax,%ebx
  sub  $2\@, \reg
.endr
# CHECK:      subl  $24, %eax
# CHECK-NEXT: subl  $24, %ebx

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

#--- b.s
.rept 2
  .print "r\+"
.endr
.irpc foo,12
  .print "\+i"
.endr
# CHECK2:      r0
# CHECK2-NEXT: r1
# CHECK2-NEXT: 0i
# CHECK2-NEXT: 1i

.rept 2
  .rept 2
    .print "n\+"
  .endr
.endr
# CHECK2:      n0
# CHECK2-NEXT: n0
# CHECK2-NEXT: n1
# CHECK2-NEXT: n1
