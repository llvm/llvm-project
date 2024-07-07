## Arguments can be expanded even if they are not preceded by \
# RUN: llvm-mc -triple=x86_64 %s | FileCheck %s

# CHECK:      1 1 1a
# CHECK-NEXT: 1 2 1a 2b
# CHECK-NEXT: \$b \$b
.altmacro
.irp ._a,1
  .print "\._a \._a& ._a&a"
  .irp $b,2
    .print "\._a \$b ._a&a $b&b"
  .endr
  .print "\$b \$b&"
.endr

# CHECK:      1 1& ._a&a
# CHECK-NEXT: \$b \$b&
.noaltmacro
.irp ._a,1
  .print "\._a \._a& ._a&a"
  .print "\$b \$b&"
.endr
