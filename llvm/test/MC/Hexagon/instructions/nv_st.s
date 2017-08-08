# RUN: llvm-mc -triple=hexagon -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s
# Hexagon Programmer's Reference Manual 11.7.2 NV/ST

# Store new-value byte
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 82 f5 b1 3b
{ r31 = r31
  memb(r17 + r21<<#3) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 11 c2 a0 48
{ r31 = r31
  memb(gp+#17) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 15 c2 b1 a1
{ r31 = r31
  memb(r17+#21) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 02 e2 b1 a9
{ r31 = r31
  memb(r17 ++ I:circ(m1)) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 28 e2 b1 a9
{ r31 = r31
  memb(r17 ++ #5:circ(m1)) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 28 c2 b1 ab
{ r31 = r31
  memb(r17++#5) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 00 e2 b1 ad
{ r31 = r31
  memb(r17++m1) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 00 e2 b1 af
{ r31 = r31
  memb(r17 ++ m1:brev) = r31.new }

# Store new-value byte conditionally
# CHECK: 1f 40 7f 70
# CHECK-NEXT: e2 f5 b1 34
{ r31 = r31
  if (p3) memb(r17+r21<<#3) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: e2 f5 b1 35
{ r31 = r31
  if (!p3) memb(r17+r21<<#3) = r31.new }
# CHECK: 03 40 45 85
# CHECK-NEXT: 1f 40 7f 70
# CHECK-NEXT: e2 f5 b1 36
{ p3 = r5
  r31 = r31
  if (p3.new) memb(r17+r21<<#3) = r31.new }
# CHECK: 03 40 45 85
# CHECK-NEXT: 1f 40 7f 70
# CHECK-NEXT: e2 f5 b1 37
{ p3 = r5
  r31 = r31
  if (!p3.new) memb(r17+r21<<#3) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: ab c2 b1 40
{ r31 = r31
  if (p3) memb(r17+#21) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: ab c2 b1 44
{ r31 = r31
  if (!p3) memb(r17+#21) = r31.new }
# CHECK: 03 40 45 85
# CHECK-NEXT: 1f 40 7f 70
# CHECK-NEXT: ab c2 b1 42
{ p3 = r5
  r31 = r31
  if (p3.new) memb(r17+#21) = r31.new }
# CHECK: 03 40 45 85
# CHECK-NEXT: 1f 40 7f 70
# CHECK-NEXT: ab c2 b1 46
{ p3 = r5
  r31 = r31
  if (!p3.new) memb(r17+#21) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 2b e2 b1 ab
{ r31 = r31
  if (p3) memb(r17++#5) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 2f e2 b1 ab
{ r31 = r31
  if (!p3) memb(r17++#5) = r31.new }
# CHECK: 03 40 45 85
# CHECK-NEXT: 1f 40 7f 70
# CHECK-NEXT: ab e2 b1 ab
{ p3 = r5
  r31 = r31
  if (p3.new) memb(r17++#5) = r31.new }
# CHECK: 03 40 45 85
# CHECK-NEXT: 1f 40 7f 70
# CHECK-NEXT: af e2 b1 ab
{ p3 = r5
  r31 = r31
  if (!p3.new) memb(r17++#5) = r31.new }

# Store new-value halfword
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 8a f5 b1 3b
{ r31 = r31
  memh(r17 + r21<<#3) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 15 ca a0 48
{ r31 = r31
  memh(gp+#42) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 15 ca b1 a1
{ r31 = r31
  memh(r17+#42) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 02 ea b1 a9
{ r31 = r31
  memh(r17 ++ I:circ(m1)) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 28 ea b1 a9
{ r31 = r31
  memh(r17 ++ #10:circ(m1)) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 28 ca b1 ab
{ r31 = r31
  memh(r17++#10) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 00 ea b1 ad
{ r31 = r31
  memh(r17++m1) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 00 ea b1 af
{ r31 = r31
  memh(r17 ++ m1:brev) = r31.new }

# Store new-value halfword conditionally
# CHECK: 1f 40 7f 70
# CHECK-NEXT: ea f5 b1 34
{ r31 = r31
  if (p3) memh(r17+r21<<#3) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: ea f5 b1 35
{ r31 = r31
  if (!p3) memh(r17+r21<<#3) = r31.new }
# CHECK: 03 40 45 85
# CHECK-NEXT: 1f 40 7f 70
# CHECK-NEXT: ea f5 b1 36
{ p3 = r5
  r31 = r31
  if (p3.new) memh(r17+r21<<#3) = r31.new }
# CHECK: 03 40 45 85
# CHECK-NEXT: 1f 40 7f 70
# CHECK-NEXT: ea f5 b1 37
{ p3 = r5
  r31 = r31
  if (!p3.new) memh(r17+r21<<#3) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: ab ca b1 40
{ r31 = r31
  if (p3) memh(r17+#42) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: ab ca b1 44
{ r31 = r31
  if (!p3) memh(r17+#42) = r31.new }
# CHECK: 03 40 45 85
# CHECK-NEXT: 1f 40 7f 70
# CHECK-NEXT: ab ca b1 42
{ p3 = r5
  r31 = r31
  if (p3.new) memh(r17+#42) = r31.new }
# CHECK: 03 40 45 85
# CHECK-NEXT: 1f 40 7f 70
# CHECK-NEXT: ab ca b1 46
{ p3 = r5
  r31 = r31
  if (!p3.new) memh(r17+#42) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 2b ea b1 ab
{ r31 = r31
  if (p3) memh(r17++#10) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 2f ea b1 ab
{ r31 = r31
  if (!p3) memh(r17++#10) = r31.new }
# CHECK: 03 40 45 85
# CHECK-NEXT: 1f 40 7f 70
# CHECK-NEXT: ab ea b1 ab
{ p3 = r5
  r31 = r31
  if (p3.new) memh(r17++#10) = r31.new }
# CHECK: 03 40 45 85
# CHECK-NEXT: 1f 40 7f 70
# CHECK-NEXT: af ea b1 ab
{ p3 = r5
  r31 = r31
  if (!p3.new) memh(r17++#10) = r31.new }

# Store new-value word
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 92 f5 b1 3b
{ r31 = r31
  memw(r17 + r21<<#3) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 15 d2 a0 48
{ r31 = r31
  memw(gp+#84) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 15 d2 b1 a1
{ r31 = r31
  memw(r17+#84) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 02 f2 b1 a9
{ r31 = r31
  memw(r17 ++ I:circ(m1)) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 28 f2 b1 a9
{ r31 = r31
  memw(r17 ++ #20:circ(m1)) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 28 d2 b1 ab
{ r31 = r31
  memw(r17++#20) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 00 f2 b1 ad
{ r31 = r31
  memw(r17++m1) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 00 f2 b1 af
{ r31 = r31
  memw(r17 ++ m1:brev) = r31.new }

# Store new-value word conditionally
# CHECK: 1f 40 7f 70
# CHECK-NEXT: f2 f5 b1 34
{ r31 = r31
  if (p3) memw(r17+r21<<#3) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: f2 f5 b1 35
{ r31 = r31
  if (!p3) memw(r17+r21<<#3) = r31.new }
# CHECK: 03 40 45 85
# CHECK-NEXT: 1f 40 7f 70
# CHECK-NEXT: f2 f5 b1 36
{ p3 = r5
  r31 = r31
  if (p3.new) memw(r17+r21<<#3) = r31.new }
# CHECK: 03 40 45 85
# CHECK-NEXT: 1f 40 7f 70
# CHECK-NEXT: f2 f5 b1 37
{ p3 = r5
  r31 = r31
  if (!p3.new) memw(r17+r21<<#3) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: ab d2 b1 40
{ r31 = r31
  if (p3) memw(r17+#84) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: ab d2 b1 44
{ r31 = r31
  if (!p3) memw(r17+#84) = r31.new }
# CHECK: 03 40 45 85
# CHECK-NEXT: 1f 40 7f 70
# CHECK-NEXT: ab d2 b1 42
{ p3 = r5
  r31 = r31
  if (p3.new) memw(r17+#84) = r31.new }
# CHECK: 03 40 45 85
# CHECK-NEXT: 1f 40 7f 70
# CHECK-NEXT: ab d2 b1 46
{ p3 = r5
  r31 = r31
  if (!p3.new) memw(r17+#84) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 2b f2 b1 ab
{ r31 = r31
  if (p3) memw(r17++#20) = r31.new }
# CHECK: 1f 40 7f 70
# CHECK-NEXT: 2f f2 b1 ab
{ r31 = r31
  if (!p3) memw(r17++#20) = r31.new }
# CHECK: 03 40 45 85
# CHECK-NEXT: 1f 40 7f 70
# CHECK-NEXT: ab f2 b1 ab
{ p3 = r5
  r31 = r31
  if (p3.new) memw(r17++#20) = r31.new }
# CHECK: 03 40 45 85
# CHECK-NEXT: 1f 40 7f 70
# CHECK-NEXT: af f2 b1 ab
{ p3 = r5
  r31 = r31
  if (!p3.new) memw(r17++#20) = r31.new }
