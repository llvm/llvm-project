# RUN: llvm-mc -triple=hexagon -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s
# Hexagon Programmer's Reference Manual 11.8 ST

# Store doubleword
# CHECK: 9e f5 d1 3b
memd(r17 + r21<<#3) = r31:30
# CHECK: 28 d4 c0 48
memd(gp+#320) = r21:20
# CHECK: 02 40 00 00
# CHECK-NEXT: 28 d4 c0 48
memd(##168) = r21:20
memd(r17+#168) = r21:20
# CHECK: 02 f4 d1 a9
memd(r17 ++ I:circ(m1)) = r21:20
# CHECK: 28 f4 d1 a9
memd(r17 ++ #40:circ(m1)) = r21:20
# CHECK: 28 d4 d1 ab
memd(r17++#40) = r21:20
# CHECK: 00 40 00 00
# CHECK-NEXT: d5 fe d1 ad
memd(r17<<#3 + ##21) = r31:30
memd(r17++m1) = r21:20
# CHECK: 00 f4 d1 af
memd(r17 ++ m1:brev) = r21:20

# Store doubleword conditionally
# CHECK: fe f5 d1 34
if (p3) memd(r17+r21<<#3) = r31:30
# CHECK: fe f5 d1 35
if (!p3) memd(r17+r21<<#3) = r31:30
# CHECK: 03 40 45 85
# CHECK-NEXT: fe f5 d1 36
{ p3 = r5
  if (p3.new) memd(r17+r21<<#3) = r31:30 }
# CHECK: 03 40 45 85
# CHECK-NEXT: fe f5 d1 37
{ p3 = r5
  if (!p3.new) memd(r17+r21<<#3) = r31:30 }
# CHECK: ab de d1 40
if (p3) memd(r17+#168) = r31:30
# CHECK: ab de d1 44
if (!p3) memd(r17+#168) = r31:30
# CHECK: 03 40 45 85
# CHECK-NEXT: ab de d1 42
{ p3 = r5
  if (p3.new) memd(r17+#168) = r31:30 }
# CHECK: 03 40 45 85
# CHECK-NEXT: ab de d1 46
{ p3 = r5
  if (!p3.new) memd(r17+#168) = r31:30 }
# CHECK: 2b f4 d1 ab
if (p3) memd(r17++#40) = r21:20
# CHECK: 2f f4 d1 ab
if (!p3) memd(r17++#40) = r21:20
# CHECK: 03 40 45 85
# CHECK-NEXT: ab f4 d1 ab
{ p3 = r5
  if (p3.new) memd(r17++#40) = r21:20 }
# CHECK: 03 40 45 85
# CHECK-NEXT: af f4 d1 ab
{ p3 = r5
  if (!p3.new) memd(r17++#40) = r21:20 }
# CHECK: 02 40 00 00
# CHECK-NEXT: c3 d4 c2 af
if (p3) memd(##168) = r21:20
# CHECK: 02 40 00 00
# CHECK-NEXT: c7 d4 c2 af
if (!p3) memd(##168) = r21:20
# CHECK: 03 40 45 85
# CHECK-NEXT: 02 40 00 00
# CHECK-NEXT: c3 f4 c2 af
{ p3 = r5
  if (p3.new) memd(##168) = r21:20 }
# CHECK: 03 40 45 85
# CHECK-NEXT: 02 40 00 00
# CHECK-NEXT: c7 f4 c2 af
{ p3 = r5
  if (!p3.new) memd(##168) = r21:20 }

# Store byte
# CHECK: 9f f5 11 3b
memb(r17 + r21<<#3) = r31
# CHECK: 9f ca 11 3c
memb(r17+#21)=#31
# CHECK: 15 d5 00 48
memb(gp+#21) = r21
# CHECK: 00 40 00 00
# CHECK-NEXT: 15 d5 00 48
memb(##21) = r21
# CHECK: 15 d5 11 a1
memb(r17+#21) = r21
# CHECK: 02 f5 11 a9
memb(r17 ++ I:circ(m1)) = r21
# CHECK: 28 f5 11 a9
memb(r17 ++ #5:circ(m1)) = r21
# CHECK: 28 d5 11 ab
memb(r17++#5) = r21
# CHECK: 00 40 00 00
# CHECK-NEXT: d5 ff 11 ad
memb(r17<<#3 + ##21) = r31
# CHECK: 00 f5 11 ad
memb(r17++m1) = r21
# CHECK: 00 f5 11 af
memb(r17 ++ m1:brev) = r21

# Store byte conditionally
# CHECK: ff f5 11 34
if (p3) memb(r17+r21<<#3) = r31
# CHECK: ff f5 11 35
if (!p3) memb(r17+r21<<#3) = r31
# CHECK: 03 40 45 85
# CHECK-NEXT: ff f5 11 36
{ p3 = r5
  if (p3.new) memb(r17+r21<<#3) = r31 }
# CHECK: 03 40 45 85
# CHECK-NEXT: ff f5 11 37
{ p3 = r5
  if (!p3.new) memb(r17+r21<<#3) = r31 }
# CHECK: ff ca 11 38
if (p3) memb(r17+#21)=#31
# CHECK: ff ca 91 38
if (!p3) memb(r17+#21)=#31
# CHECK: 03 40 45 85
# CHECK-NEXT: ff ca 11 39
{ p3 = r5
  if (p3.new) memb(r17+#21)=#31 }
# CHECK: 03 40 45 85
# CHECK-NEXT: ff ca 91 39
{ p3 = r5
  if (!p3.new) memb(r17+#21)=#31 }
# CHECK: ab df 11 40
if (p3) memb(r17+#21) = r31
# CHECK: ab df 11 44
if (!p3) memb(r17+#21) = r31
# CHECK: 03 40 45 85
# CHECK-NEXT: ab df 11 42
{ p3 = r5
  if (p3.new) memb(r17+#21) = r31 }
# CHECK: 03 40 45 85
# CHECK-NEXT: ab df 11 46
{ p3 = r5
  if (!p3.new) memb(r17+#21) = r31 }
# CHECK: 2b f5 11 ab
if (p3) memb(r17++#5) = r21
# CHECK: 2f f5 11 ab
if (!p3) memb(r17++#5) = r21
# CHECK: 03 40 45 85
# CHECK-NEXT: ab f5 11 ab
{ p3 = r5
  if (p3.new) memb(r17++#5) = r21 }
# CHECK: 03 40 45 85
# CHECK-NEXT: af f5 11 ab
{ p3 = r5
  if (!p3.new) memb(r17++#5) = r21 }
# CHECK: 00 40 00 00
# CHECK-NEXT: ab d5 01 af
if (p3) memb(##21) = r21
# CHECK: 00 40 00 00
# CHECK-NEXT: af d5 01 af
if (!p3) memb(##21) = r21
# CHECK: 03 40 45 85
# CHECK-NEXT: 00 40 00 00
# CHECK-NEXT: ab f5 01 af
{ p3 = r5
  if (p3.new) memb(##21) = r21 }
# CHECK: 03 40 45 85
# CHECK-NEXT: 00 40 00 00
# CHECK-NEXT: af f5 01 af
{ p3 = r5
  if (!p3.new) memb(##21) = r21 }

# Store halfword
# CHECK: 9f f5 51 3b
memh(r17 + r21<<#3) = r31
# CHECK: 9f f5 71 3b
memh(r17 + r21<<#3) = r31.h
# CHECK: 95 cf 31 3c
memh(r17+#62)=#21
# CHECK: 00 40 00 00
# CHECK-NEXT: 2a d5 40 48
memh(##42) = r21
# CHECK: 00 40 00 00
# CHECK-NEXT: 2a d5 60 48
memh(##42) = r21.h
# CHECK: 2a d5 40 48
memh(gp+#84) = r21
# CHECK: 2a d5 60 48
memh(gp+#84) = r21.h
# CHECK: 15 df 51 a1
memh(r17+#42) = r31
# CHECK: 15 df 71 a1
memh(r17+#42) = r31.h
# CHECK: 02 f5 51 a9
memh(r17 ++ I:circ(m1)) = r21
# CHECK: 28 f5 51 a9
memh(r17 ++ #10:circ(m1)) = r21
# CHECK: 02 f5 71 a9
memh(r17 ++ I:circ(m1)) = r21.h
# CHECK: 28 f5 71 a9
memh(r17 ++ #10:circ(m1)) = r21.h
# CHECK: 28 d5 51 ab
memh(r17++#10) = r21
# CHECK: 00 40 00 00
# CHECK-NEXT: d5 ff 51 ad
memh(r17<<#3 + ##21) = r31
# CHECK: 28 d5 71 ab
memh(r17++#10) = r21.h
# CHECK: 00 40 00 00
# CHECK-NEXT: d5 ff 71 ad
memh(r17<<#3 + ##21) = r31.h
# CHECK: 00 f5 51 ad
memh(r17++m1) = r21
# CHECK: 00 f5 71 ad
memh(r17++m1) = r21.h
# CHECK: 00 f5 51 af
memh(r17 ++ m1:brev) = r21
# CHECK: 00 f5 71 af
memh(r17 ++ m1:brev) = r21.h

# Store halfword conditionally
# CHECK: ff f5 51 34
if (p3) memh(r17+r21<<#3) = r31
# CHECK: ff f5 71 34
if (p3) memh(r17+r21<<#3) = r31.h
# CHECK: ff f5 51 35
if (!p3) memh(r17+r21<<#3) = r31
# CHECK: ff f5 71 35
if (!p3) memh(r17+r21<<#3) = r31.h
# CHECK: 03 40 45 85
# CHECK-NEXT: ff f5 51 36
{ p3 = r5
  if (p3.new) memh(r17+r21<<#3) = r31 }
# CHECK: 03 40 45 85
# CHECK-NEXT: ff f5 71 36
{ p3 = r5
  if (p3.new) memh(r17+r21<<#3) = r31.h }
# CHECK: 03 40 45 85
# CHECK-NEXT: ff f5 51 37
{ p3 = r5
  if (!p3.new) memh(r17+r21<<#3) = r31 }
# CHECK: 03 40 45 85
# CHECK-NEXT: ff f5 71 37
{ p3 = r5
  if (!p3.new) memh(r17+r21<<#3) = r31.h }
# CHECK: f5 cf 31 38
if (p3) memh(r17+#62)=#21
# CHECK: f5 cf b1 38
if (!p3) memh(r17+#62)=#21
# CHECK: 03 40 45 85
# CHECK-NEXT: f5 cf 31 39
{ p3 = r5
  if (p3.new) memh(r17+#62)=#21 }
# CHECK: 03 40 45 85
# CHECK-NEXT: f5 cf b1 39
{ p3 = r5
  if (!p3.new) memh(r17+#62)=#21 }
# CHECK: fb d5 51 40
if (p3) memh(r17+#62) = r21
# CHECK: fb d5 71 40
if (p3) memh(r17+#62) = r21.h
# CHECK: fb d5 51 44
if (!p3) memh(r17+#62) = r21
# CHECK: fb d5 71 44
if (!p3) memh(r17+#62) = r21.h
# CHECK: 03 40 45 85
# CHECK-NEXT: fb d5 51 42
{ p3 = r5
  if (p3.new) memh(r17+#62) = r21 }
# CHECK: 03 40 45 85
# CHECK-NEXT: fb d5 71 42
{ p3 = r5
  if (p3.new) memh(r17+#62) = r21.h }
# CHECK: 03 40 45 85
# CHECK-NEXT: fb d5 51 46
{ p3 = r5
  if (!p3.new) memh(r17+#62) = r21 }
# CHECK: 03 40 45 85
# CHECK-NEXT: fb d5 71 46
{ p3 = r5
  if (!p3.new) memh(r17+#62) = r21.h }
# CHECK: 2b f5 51 ab
if (p3) memh(r17++#10) = r21
# CHECK: 2f f5 51 ab
if (!p3) memh(r17++#10) = r21
# CHECK: 03 40 45 85
# CHECK-NEXT: ab f5 51 ab
{ p3 = r5
  if (p3.new) memh(r17++#10) = r21 }
# CHECK: 03 40 45 85
# CHECK-NEXT: af f5 51 ab
{ p3 = r5
  if (!p3.new) memh(r17++#10) = r21 }
# CHECK: 2b f5 71 ab
if (p3) memh(r17++#10) = r21.h
# CHECK: 2f f5 71 ab
if (!p3) memh(r17++#10) = r21.h
# CHECK: 03 40 45 85
# CHECK-NEXT: ab f5 71 ab
{ p3 = r5
  if (p3.new) memh(r17++#10) = r21.h }
# CHECK: 03 40 45 85
# CHECK-NEXT: af f5 71 ab
{ p3 = r5
  if (!p3.new) memh(r17++#10) = r21.h }
# CHECK: 00 40 00 00
# CHECK-NEXT: d3 d5 42 af
if (p3) memh(##42) = r21
# CHECK: 00 40 00 00
# CHECK-NEXT: d3 d5 62 af
if (p3) memh(##42) = r21.h
# CHECK: 00 40 00 00
# CHECK-NEXT: d7 d5 42 af
if (!p3) memh(##42) = r21
# CHECK: 00 40 00 00
# CHECK-NEXT: d7 d5 62 af
if (!p3) memh(##42) = r21.h
# CHECK: 03 40 45 85
# CHECK-NEXT: 00 40 00 00
# CHECK-NEXT: d3 f5 42 af
{ p3 = r5
  if (p3.new) memh(##42) = r21 }
# CHECK: 03 40 45 85
# CHECK-NEXT: 00 40 00 00
# CHECK-NEXT: d3 f5 62 af
{ p3 = r5
  if (p3.new) memh(##42) = r21.h }
# CHECK: 03 40 45 85
# CHECK-NEXT: 00 40 00 00
# CHECK-NEXT: d7 f5 42 af
{ p3 = r5
  if (!p3.new) memh(##42) = r21 }
# CHECK: 03 40 45 85
# CHECK-NEXT: 00 40 00 00
# CHECK-NEXT: d7 f5 62 af
{ p3 = r5
  if (!p3.new) memh(##42) = r21.h }

# Store word
# CHECK: 9f f5 91 3b
memw(r17 + r21<<#3) = r31
# CHECK: 9f ca 51 3c
memw(r17+#84)=#31
# CHECK: 15 df 80 48
memw(gp+#84) = r31
# CHECK: 01 40 00 00
# CHECK-NEXT: 14 d5 80 48
memw(##84) = r21
# CHECK: 9f ca 51 3c
memw(r17+#84)=#31
# CHECK: 15 df 91 a1
memw(r17+#84) = r31
# CHECK: 02 f5 91 a9
memw(r17 ++ I:circ(m1)) = r21
# CHECK: 28 f5 91 a9
memw(r17 ++ #20:circ(m1)) = r21
# CHECK: 28 d5 91 ab
memw(r17++#20) = r21
# CHECK: 00 40 00 00
# CHECK-NEXT: d5 ff 91 ad
memw(r17<<#3 + ##21) = r31
# CHECK: 00 f5 91 ad
memw(r17++m1) = r21
# CHECK: 00 f5 91 af
memw(r17 ++ m1:brev) = r21

# Store word conditionally
# CHECK: ff f5 91 34
if (p3) memw(r17+r21<<#3) = r31
# CHECK: ff f5 91 35
if (!p3) memw(r17+r21<<#3) = r31
# CHECK: 03 40 45 85
# CHECK-NEXT: ff f5 91 36
{ p3 = r5
  if (p3.new) memw(r17+r21<<#3) = r31 }
# CHECK: 03 40 45 85
# CHECK-NEXT: ff f5 91 37
{ p3 = r5
  if (!p3.new) memw(r17+r21<<#3) = r31 }
# CHECK: ff ca 51 38
if (p3) memw(r17+#84)=#31
# CHECK: ff ca d1 38
if (!p3) memw(r17+#84)=#31
# CHECK: 03 40 45 85
# CHECK-NEXT: ff ca 51 39
{ p3 = r5
  if (p3.new) memw(r17+#84)=#31 }
# CHECK: 03 40 45 85
# CHECK-NEXT: ff ca d1 39
{ p3 = r5
  if (!p3.new) memw(r17+#84)=#31 }
# CHECK: ab df 91 40
if (p3) memw(r17+#84) = r31
# CHECK: ab df 91 44
if (!p3) memw(r17+#84) = r31
# CHECK: 03 40 45 85
# CHECK-NEXT: ab df 91 42
{ p3 = r5
  if (p3.new) memw(r17+#84) = r31 }
# CHECK: 03 40 45 85
# CHECK-NEXT: ab df 91 46
{ p3 = r5
  if (!p3.new) memw(r17+#84) = r31 }
# CHECK: 2b f5 91 ab
if (p3) memw(r17++#20) = r21
# CHECK: 2f f5 91 ab
if (!p3) memw(r17++#20) = r21
# CHECK: 03 40 45 85
# CHECK-NEXT: af f5 91 ab
{ p3 = r5
  if (!p3.new) memw(r17++#20) = r21 }
# CHECK: 03 40 45 85
# CHECK-NEXT: ab f5 91 ab
{ p3 = r5
  if (p3.new) memw(r17++#20) = r21 }
# CHECK: 01 40 00 00
# CHECK-NEXT: a3 d5 81 af
if (p3) memw(##84) = r21
# CHECK: 01 40 00 00
# CHECK-NEXT: a7 d5 81 af
if (!p3) memw(##84) = r21
# CHECK: 03 40 45 85
# CHECK-NEXT: 01 40 00 00
# CHECK-NEXT: a3 f5 81 af
{ p3 = r5
  if (p3.new) memw(##84) = r21 }
# CHECK: 03 40 45 85
# CHECK-NEXT: 01 40 00 00
# CHECK-NEXT: a7 f5 81 af
{ p3 = r5
  if (!p3.new) memw(##84) = r21 }

# Allocate stack frame
# CHECK: 1f c0 9d a0
allocframe(#248)
