# RUN: llvm-mc -triple hexagon -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s
# Hexagon Programmer's Reference Manual 11.3 JR

# Call subroutine from register
# CHECK: 00 c0 b5 50
callr r21
# CHECK: 00 c1 15 51
if (p1) callr r21
# CHECK: 00 c3 35 51
if (!p3) callr r21

# Hint an indirect jump address
# CHECK: 00 c0 b5 52
hintjr(r21)

# Jump to address from register
# CHECK: 00 c0 95 52
jumpr r21
# CHECK: 00 c1 55 53
if (p1) jumpr r21
# CHECK: 03 40 45 85
# CHECK-NEXT: 00 cb 55 53
{ p3 = r5
  if (p3.new) jumpr:nt r21 }
# CHECK: 03 40 45 85
# CHECK-NEXT: 00 db 55 53
{ p3 = r5
  if (p3.new) jumpr:t r21 }
# CHECK: 00 c3 75 53
if (!p3) jumpr r21
# CHECK: 03 40 45 85
# CHECK-NEXT: 00 cb 75 53
{ p3 = r5
  if (!p3.new) jumpr:nt r21 }
# CHECK: 03 40 45 85
# CHECK-NEXT: 00 db 75 53
{ p3 = r5
  if (!p3.new) jumpr:t r21 }
