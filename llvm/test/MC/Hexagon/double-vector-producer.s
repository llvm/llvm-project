# RUN: llvm-mc -triple=hexagon -mcpu=hexagonv60 -mhvx -filetype=obj %s | llvm-objdump -d - | FileCheck %s
{
  v1:0 = vshuff(v1,v0,r7)
  v2.w = vadd(v13.w,v15.w)
  v3.w = vadd(v8.w,v14.w)
  vmem(r2+#-2) = v0.new
}

# CHECK: 60 61 07 1b
# CHECK: 02 4d 4f 1c
# CHECK: 03 48 4e 1c
# CHECK: 26 e6 22 28
