# RUN: llvm-mc -triple=hexagon -mv79 -mhvx --filetype=obj %s \
# RUN: | llvm-objdump --mcpu=hexagonv79 --mattr=+hvx -d - | FileCheck %s

# RUN: llvm-mc -triple=hexagon -mv81 -mhvx --filetype=obj %s \
# RUN: | llvm-objdump --mcpu=hexagonv81 --mattr=+hvx -d - | FileCheck %s

# RUN: not llvm-mc -triple=hexagon -mv75 -mhvx %s 2>&1 \
# RUN: | FileCheck --check-prefix=V75 --implicit-check-not=error %s

# RUN: not llvm-mc -triple=hexagon -mv75 -mhvx --defsym ERR=1 %s 2>&1 \
# RUN: | FileCheck --check-prefix=ERR %s
# RUN: not llvm-mc -triple=hexagon -mv79 -mhvx --defsym ERR=1 %s 2>&1 \
# RUN: | FileCheck --check-prefix=ERR %s
# RUN: not llvm-mc -triple=hexagon -mv81 -mhvx --defsym ERR=1 %s 2>&1 \
# RUN: | FileCheck --check-prefix=ERR %s

### Two shifts in one packet are valid in v79 and later: the second shift
### uses the shift/permute resource. Before v79 only one shift resource is
### available.
{
v1.w = vlsr(v2.w,v3.w)
v4.w = vlsr(v5.w,v6.w)
}
# CHECK: { v1.w = vlsr(v2.w,v3.w)
# CHECK-NEXT: v4.w = vlsr(v5.w,v6.w) }
# V75: :[[@LINE-3]]:1: error: invalid instruction packet: slot error

### A shift-resource-only instruction (prefixsum) and a regular shift: the
### regular shift uses the shift/permute resource in v79 and later.
{
v1.b = prefixsum(q0)
v4.w = vlsr(v5.w,v6.w)
}
# CHECK: { v1.b = prefixsum(q0)
# CHECK-NEXT: v4.w = vlsr(v5.w,v6.w) }
# V75: :[[@LINE-3]]:1: error: invalid instruction packet: slot error

### Two shifts and an ALU instruction: the ALU instruction uses a multiply
### resource.
{
v1.w = vlsr(v2.w,v3.w)
v7.w = vlsr(v8.w,v9.w)
v4.w = vadd(v5.w,v6.w)
}
# CHECK: { v1.w = vlsr(v2.w,v3.w)
# CHECK-NEXT: v7.w = vlsr(v8.w,v9.w)
# CHECK-NEXT: v4.w = vadd(v5.w,v6.w) }
# V75: :[[@LINE-4]]:1: error: invalid instruction packet: slot error

### A shift and a permute are valid in all versions.
{
v1.w = vlsr(v2.w,v3.w)
v4 = vdelta(v5,v6)
}
# CHECK: { v4 = vdelta(v5,v6)
# CHECK-NEXT: v1.w = vlsr(v2.w,v3.w) }

.ifdef ERR
### The packets below are invalid in all versions: there are only two
### resources that can execute shifts, and prefixsum can only use one of them.

### Three shifts.
{
v1.w = vlsr(v2.w,v3.w)
v7.w = vlsr(v8.w,v9.w)
v4.w = vlsr(v5.w,v6.w)
}
# ERR: :[[@LINE-1]]:1: error: invalid instruction packet: slot error

### Two shifts and a permute.
{
v1.w = vlsr(v2.w,v3.w)
v7.w = vlsr(v8.w,v9.w)
v4 = vdelta(v5,v6)
}
# ERR: :[[@LINE-1]]:1: error: invalid instruction packet: slot error

### Two prefixsum instructions.
{
v1.b = prefixsum(q0)
v4.h = prefixsum(q1)
}
# ERR: :[[@LINE-1]]:1: error: invalid instruction packet: slot error

### A prefixsum, a shift, and a permute.
{
v1.b = prefixsum(q0)
v7.w = vlsr(v8.w,v9.w)
v4 = vdelta(v5,v6)
}
# ERR: :[[@LINE-1]]:1: error: invalid instruction packet: slot error
.endif
