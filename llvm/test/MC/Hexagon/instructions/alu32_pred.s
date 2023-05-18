# RUN: llvm-mc -triple hexagon -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s
# Hexagon Programmer's Reference Manual 11.1.3 ALU32/PRED

# Conditional add
# CHECK: f1 c3 75 74
if (p3) r17 = add(r21, #31)
# CHECK: 03 40 45 85
# CHECK-NEXT: f1 e3 75 74
{ p3 = r5
  if (p3.new) r17 = add(r21, #31) }
# CHECK: f1 c3 f5 74
if (!p3) r17 = add(r21, #31)
# CHECK: 03 40 45 85
# CHECK-NEXT: f1 e3 f5 74
{ p3 = r5
  if (!p3.new) r17 = add(r21, #31) }
# CHECK: 71 df 15 fb
if (p3) r17 = add(r21, r31)
# CHECK: 03 40 45 85
# CHECK-NEXT: 71 ff 15 fb
{ p3 = r5
  if (p3.new) r17 = add(r21, r31) }
# CHECK: f1 df 15 fb
if (!p3) r17 = add(r21, r31)
# CHECK: 03 40 45 85
# CHECK-NEXT: f1 ff 15 fb
{ p3 = r5
  if (!p3.new) r17 = add(r21, r31) }

# Conditional shift halfword
# CHECK: 11 e3 15 70
if (p3) r17 = aslh(r21)
# CHECK: 03 40 45 85
# CHECK-NEXT: 11 e7 15 70
{ p3 = r5
  if (p3.new) r17 = aslh(r21) }
# CHECK: 11 eb 15 70
if (!p3) r17 = aslh(r21)
# CHECK: 03 40 45 85
# CHECK-NEXT: 11 ef 15 70
{ p3 = r5
  if (!p3.new) r17 = aslh(r21) }
# CHECK: 11 e3 35 70
if (p3) r17 = asrh(r21)
# CHECK: 03 40 45 85
# CHECK-NEXT: 11 e7 35 70
{ p3 = r5
  if (p3.new) r17 = asrh(r21) }
# CHECK: 11 eb 35 70
if (!p3) r17 = asrh(r21)
# CHECK: 03 40 45 85
# CHECK-NEXT: 11 ef 35 70
{ p3 = r5
  if (!p3.new) r17 = asrh(r21) }

# Conditional combine
# CHECK: 70 df 15 fd
if (p3) r17:16 = combine(r21, r31)
# CHECK: f0 df 15 fd
if (!p3) r17:16 = combine(r21, r31)
# CHECK: 03 40 45 85
# CHECK-NEXT: 70 ff 15 fd
{ p3 = r5
  if (p3.new) r17:16 = combine(r21, r31) }
# CHECK: 03 40 45 85
# CHECK-NEXT: f0 ff 15 fd
{ p3 = r5
  if (!p3.new) r17:16 = combine(r21, r31) }

# Conditional logical operations
# CHECK: 71 df 15 f9
if (p3) r17 = and(r21, r31)
# CHECK: f1 df 15 f9
if (!p3) r17 = and(r21, r31)
# CHECK: 03 40 45 85
# CHECK-NEXT: 71 ff 15 f9
{ p3 = r5
  if (p3.new) r17 = and(r21, r31) }
# CHECK: 03 40 45 85
# CHECK-NEXT: f1 ff 15 f9
{ p3 = r5
  if (!p3.new) r17 = and(r21, r31) }
# CHECK: 71 df 35 f9
if (p3) r17 = or(r21, r31)
# CHECK: f1 df 35 f9
if (!p3) r17 = or(r21, r31)
# CHECK: 03 40 45 85
# CHECK-NEXT: 71 ff 35 f9
{ p3 = r5
  if (p3.new) r17 = or(r21, r31) }
# CHECK: 03 40 45 85
# CHECK-NEXT: f1 ff 35 f9
{ p3 = r5
  if (!p3.new) r17 = or(r21, r31) }
# CHECK: 71 df 75 f9
if (p3) r17 = xor(r21, r31)
# CHECK: f1 df 75 f9
if (!p3) r17 = xor(r21, r31)
# CHECK: 03 40 45 85
# CHECK-NEXT: 71 ff 75 f9
{ p3 = r5
  if (p3.new) r17 = xor(r21, r31) }
# CHECK: 03 40 45 85
# CHECK-NEXT: f1 ff 75 f9
{ p3 = r5
  if (!p3.new) r17 = xor(r21, r31) }

# Conditional subtract
# CHECK: 71 df 35 fb
if (p3) r17 = sub(r31, r21)
# CHECK: f1 df 35 fb
if (!p3) r17 = sub(r31, r21)
# CHECK: 03 40 45 85
# CHECK-NEXT: 71 ff 35 fb
{ p3 = r5
  if (p3.new) r17 = sub(r31, r21) }
# CHECK: 03 40 45 85
# CHECK-NEXT: f1 ff 35 fb
{ p3 = r5
  if (!p3.new) r17 = sub(r31, r21) }

# Conditional sign extend
# CHECK: 11 e3 b5 70
if (p3) r17 = sxtb(r21)
# CHECK: 11 eb b5 70
if (!p3) r17 = sxtb(r21)
# CHECK: 03 40 45 85
# CHECK-NEXT: 11 e7 b5 70
{ p3 = r5
  if (p3.new) r17 = sxtb(r21) }
# CHECK: 03 40 45 85
# CHECK-NEXT: 11 ef b5 70
{ p3 = r5
  if (!p3.new) r17 = sxtb(r21) }
# CHECK: 11 e3 f5 70
if (p3) r17 = sxth(r21)
# CHECK: 11 eb f5 70
if (!p3) r17 = sxth(r21)
# CHECK: 03 40 45 85
# CHECK-NEXT: 11 e7 f5 70
{ p3 = r5
  if (p3.new) r17 = sxth(r21) }
# CHECK: 03 40 45 85
# CHECK-NEXT: 11 ef f5 70
{ p3 = r5
  if (!p3.new) r17 = sxth(r21) }

# Conditional transfer
# CHECK: b1 c2 60 7e
if (p3) r17 = #21
# CHECK: b1 c2 e0 7e
if (!p3) r17 = #21
# CHECK: 03 40 45 85
# CHECK-NEXT: b1 e2 60 7e
{ p3 = r5
  if (p3.new) r17 = #21 }
# CHECK: 03 40 45 85
# CHECK-NEXT: b1 e2 e0 7e
{ p3 = r5
  if (!p3.new) r17 = #21 }

# Conditional zero extend
# CHECK: 11 e3 95 70
if (p3) r17 = zxtb(r21)
# CHECK: 11 eb 95 70
if (!p3) r17 = zxtb(r21)
# CHECK: 03 40 45 85
# CHECK-NEXT: 11 e7 95 70
{ p3 = r5
  if (p3.new) r17 = zxtb(r21) }
# CHECK: 03 40 45 85
# CHECK-NEXT: 11 ef 95 70
{ p3 = r5
  if (!p3.new) r17 = zxtb(r21) }
# CHECK: 11 e3 d5 70
if (p3) r17 = zxth(r21)
# CHECK: 11 eb d5 70
if (!p3) r17 = zxth(r21)
# CHECK: 03 40 45 85
# CHECK-NEXT: 11 e7 d5 70
{ p3 = r5
  if (p3.new) r17 = zxth(r21) }
# CHECK: 03 40 45 85
# CHECK-NEXT: 11 ef d5 70
{ p3 = r5
  if (!p3.new) r17 = zxth(r21) }

# Compare
# CHECK: e3 c3 15 75
p3 = cmp.eq(r21, #31)
# CHECK: f3 c3 15 75
p3 = !cmp.eq(r21, #31)
# CHECK: e3 c3 55 75
p3 = cmp.gt(r21, #31)
# CHECK: f3 c3 55 75
p3 = !cmp.gt(r21, #31)
# CHECK: e3 c3 95 75
p3 = cmp.gtu(r21, #31)
# CHECK: f3 c3 95 75
p3 = !cmp.gtu(r21, #31)
# CHECK: 03 df 15 f2
p3 = cmp.eq(r21, r31)
# CHECK: 13 df 15 f2
p3 = !cmp.eq(r21, r31)
# CHECK: 03 df 55 f2
p3 = cmp.gt(r21, r31)
# CHECK: 13 df 55 f2
p3 = !cmp.gt(r21, r31)
# CHECK: 03 df 75 f2
p3 = cmp.gtu(r21, r31)
# CHECK: 13 df 75 f2
p3 = !cmp.gtu(r21, r31)

# Compare to general register
# CHECK: f1 e3 55 73
r17 = cmp.eq(r21, #31)
# CHECK: f1 e3 75 73
r17 = !cmp.eq(r21, #31)
# CHECK: 11 df 55 f3
r17 = cmp.eq(r21, r31)
# CHECK: 11 df 75 f3
r17 = !cmp.eq(r21, r31)
