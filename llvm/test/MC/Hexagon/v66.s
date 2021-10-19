# RUN: llvm-mc -arch=hexagon -mcpu=hexagonv66 -mhvx -filetype=obj %s | llvm-objdump --mcpu=hexagonv66 --mattr=+hvx -d - | FileCheck %s

# CHECK: 1d8362e4 { v4.w = vsatdw(v2.w,v3.w)
{
  v4.w = vsatdw(v2.w, v3.w)
  vmem(r16+#0) = v4.new
}

# CHECK: 1aaae5e0 { v1:0.w = vasrinto(v5.w,v10.w) }
  v1:0.w = vasrinto(v5.w, v10.w)

# CHECK: 1aaae5e0 { v1:0.w = vasrinto(v5.w,v10.w) }
  v1:0 = vasrinto(v5, v10)

# CHECK: 1d89ef14 { v20.w = vadd(v15.w,v9.w,q0):carry:sat }
  v20.w = vadd(v15.w, v9.w, q0):carry:sat

