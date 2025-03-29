# RUN: llvm-mc -triple=hexagon -mv69 -mhvx -filetype=obj %s | \
# RUN:   llvm-objdump --triple=hexagon --mcpu=hexagonv69 --mattr=+hvx -d - | \
# RUN:   FileCheck %s
# CHECK: 00000000 <.text>:

{
  V4:5.w = vadd(V1:0.w, V3:2.w)
  vmem(r0+#0) = v4.new
}
# CHECK-NEXT: 1c6240c5 { 	v4:5.w = vadd(v1:0.w,v3:2.w)
# CHECK-NEXT: 2820c023   	vmem(r0+#0x0) = v4.new }

{
  V4:5.w = vadd(V1:0.w, V3:2.w)
  vmem(r0+#0) = v5.new
}
# CHECK-NEXT: 1c6240c5 { 	v4:5.w = vadd(v1:0.w,v3:2.w)
# CHECK-NEXT: 2820c022   	vmem(r0+#0x0) = v5.new }
