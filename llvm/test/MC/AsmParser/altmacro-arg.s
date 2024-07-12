# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -triple=x86_64 a.s | FileCheck %s
# RUN: llvm-mc -triple=x86_64 b.s | FileCheck %s --check-prefix=CHECK1

#--- a.s
.altmacro
# CHECK:      ja .Ltmp0
# CHECK-NEXT: xorq %rax, %rax
# CHECK:      .data
# CHECK-NEXT: .ascii "b cc rbx"
# CHECK-NEXT: .ascii "ara rax rax raxx"
.macro gen a, ra, rax
  ja 1f
  xorq %rax, %rax
1:
.data
  .ascii "\a \ra \rax"
  .ascii "a\()ra ra\()x rax raxx"
.endm
gen b, cc, rbx

#--- b.s
.altmacro
# CHECK1:      1 1 ._a&a
# CHECK1-NEXT: 1 2 ._a&a $b&b
# CHECK1-NEXT: \$b \$b
.irp ._a,1
  .print "\._a \._a& ._a&a"
  .irp $b,2
    .print "\._a \$b ._a&a $b&b"
  .endr
  .print "\$b \$b&"
.endr

# CHECK1:      1 1& ._a&a
# CHECK1-NEXT: \$b \$b&
.noaltmacro
.irp ._a,1
  .print "\._a \._a& ._a&a"
  .print "\$b \$b&"
.endr
.altmacro
