# RUN: not llvm-mc -triple=hexagon -mv75 -mhvx %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple=hexagon -mv79 -mhvx %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple=hexagon -mv81 -mhvx %s 2>&1 | FileCheck %s

### 3-source and non-unary ALU, not valid in any version.
{
    V1.uh = vasr(V3:2.w,V0.uh):sat
    V5 = vand(v4, V6)
}
# CHECK: :[[@LINE-1]]:1: error: invalid instruction packet: HVX resource use violation
# CHECK: :[[@LINE-4]]:5: note: This instruction cannot be paired with an HVX non-unary instruction
# CHECK-NEXT: V1.uh = vasr(V3:2.w,V0.uh):sat
# CHECK: :[[@LINE-5]]:5: note: HVX non-unary instruction
# CHECK-NEXT: V5 = vand(v4, V6)

### 3-source and VP, not valid in any version.
{
v1.uh = vasr(v1:0.w,v0.uh):sat
v0 = vror(v0,R0)
}
# CHECK: :[[@LINE-1]]:1: error: invalid instruction packet: HVX resource use violation
# CHECK: :[[@LINE-4]]:1: note: This instruction cannot be paired with an HVX permute instruction
# CHECK-NEXT: v1.uh = vasr(v1:0.w,v0.uh):sat
# CHECK: :[[@LINE-5]]:1: note: HVX permute instruction
# CHECK-NEXT: v0 = vror(v0,R0)

### 3SRC and VS, not valid.
{
v1.uh = vasr(v1:0.w,v0.uh):sat
v0.w = vasr(v0.w,r0)
}
# CHECK: :[[@LINE-1]]:1: error: invalid instruction packet: HVX resource use violation
# CHECK: :[[@LINE-4]]:1: note: This instruction cannot be paired with another HVX shift instruction
# CHECK-NEXT: v1.uh = vasr(v1:0.w,v0.uh):sat
# CHECK: :[[@LINE-5]]:1: note: another HVX shift instruction
# CHECK-NEXT: v0.w = vasr(v0.w,r0)

### Two 3SRC instructions, not valid.
{
v1.uh = vasr(v1:0.w,v0.uh):sat
v0.uh = vasr(v1:0.w,v0.uh):sat
}
# CHECK: :[[@LINE-1]]:1: error: invalid instruction packet: HVX resource use violation
# CHECK: :[[@LINE-4]]:1: note: This instruction cannot be paired with another HVX shift instruction
# CHECK-NEXT: v1.uh = vasr(v1:0.w,v0.uh):sat
# CHECK: :[[@LINE-5]]:1: note: another HVX shift instruction
# CHECK-NEXT: v0.uh = vasr(v1:0.w,v0.uh):sat

### 3SRC and prefixsum, not valid.
{
v1.uh = vasr(v1:0.w,v0.uh):sat
v4.b = prefixsum(q0)
}
# CHECK: :[[@LINE-1]]:1: error: invalid instruction packet: HVX resource use violation
# CHECK: :[[@LINE-4]]:1: note: This instruction cannot be paired with another HVX shift instruction
# CHECK-NEXT: v1.uh = vasr(v1:0.w,v0.uh):sat
# CHECK: :[[@LINE-5]]:1: note: another HVX shift instruction
# CHECK-NEXT: v4.b = prefixsum(q0)
