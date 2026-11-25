# RUN: llvm-mc -triple=hexagon -mv75 -mhvx --filetype=obj %s \
# RUN: | llvm-objdump --mcpu=hexagonv75 --mattr=+hvx -d - | FileCheck %s

# RUN: llvm-mc -triple=hexagon -mv79 -mhvx --filetype=obj %s \
# RUN: | llvm-objdump --mcpu=hexagonv79 --mattr=+hvx -d - | FileCheck %s

# RUN: llvm-mc -triple=hexagon -mv81 -mhvx --filetype=obj %s \
# RUN: | llvm-objdump --mcpu=hexagonv81 --mattr=+hvx -d - | FileCheck %s

### 3SRC and unary ALU, valid
{
v1.uh = vasr(v1:0.w,v0.uh):sat
v2 = v3
v3 = v4
v4 = v5
}
# CHECK: { v1.uh = vasr(v1:0.w,v0.uh):sat
# CHECK-NEXT:v2 = v3
# CHECK-NEXT:v3 = v4
# CHECK-NEXT:v4 = v5 }
