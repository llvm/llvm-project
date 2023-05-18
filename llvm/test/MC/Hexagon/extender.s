# RUN: llvm-mc -triple=hexagon -filetype=obj %s | llvm-objdump --no-print-imm-hex -d - | FileCheck %s
#

# STrib_abs_V4
{
  memb(##1024056) = r0
}

# CHECK: immext(#1024000)
# CHECK: memb(##1024056) = r0

# S2_storerbgp
{
  memb(GP + #56) = r0
}

# CHECK: memb(gp+#56) = r0

# STrih_abs_V4
{
  memh(##1024056) = r0
}

# CHECK: immext(#1024000)
# CHECK: memh(##1024056) = r0

# S2_storerhgp
{
  memh(GP + #56) = r0
}

# CHECK: memh(gp+#56) = r0

# STriw_abs_V4
{
  memw(##1024056) = r0
}

# CHECK: immext(#1024000)
# CHECK: memw(##1024056) = r0

# S2_storerigp
{
  memw(GP + #56) = r0
}

# CHECK: memw(gp+#56) = r0

# STrib_abs_nv_V4
{
  r0 = #1
  memb(##1024056) = r0.new
}

# CHECK: r0 = #1
# CHECK: immext(#1024000)
# CHECK: memb(##1024056) = r0.new

# S2_storerbnewgp
{
  r0 = #1
  memb(GP + #56) = r0.new
}

# CHECK: r0 = #1
# CHECK: memb(gp+#56) = r0.new

# STrih_abs_nv_V4
{
  r0 = #1
  memh(##1024056) = r0.new
}

# CHECK: r0 = #1
# CHECK: immext(#1024000)
# CHECK: memh(##1024056) = r0.new

# S2_storerhnewgp
{
  r0 = #1
  memh(GP + #56) = r0.new
}

# CHECK: r0 = #1
# CHECK: memh(gp+#56) = r0.new

# STriw_abs_nv_V4
{
  r0 = #1
  memw(##1024056) = r0.new
}

# CHECK: r0 = #1
# CHECK: immext(#1024000)
# CHECK: memw(##1024056) = r0.new

# S2_storerinewgp
{
  r0 = #1
  memw(GP + #56) = r0.new
}

# CHECK: r0 = #1
# CHECK: memw(gp+#56) = r0.new

# STrid_abs_V4
{
  memd(##1024056) = r1:0
}

# CHECK: immext(#1024000)
# CHECK: memd(##1024056) = r1:0

# S2_storerdgp
{
  memd(GP + #56) = r1:0
}

# CHECK: memd(gp+#56) = r1:0

# LDrib_abs_V4
{
  r0 = memb(##1024056)
}

# CHECK: immext(#1024000)
# CHECK: r0 = memb(##1024056)

# LDb_GP_V4
{
  r0 = memb(GP + #56)
}

# CHECK: r0 = memb(gp+#56)

# LDriub_abs_V4
{
  r0 = memub(##1024056)
}

# CHECK: immext(#1024000)
# CHECK: r0 = memub(##1024056)

# LDub_GP_V4
{
  r0 = memub(GP + #56)
}

# CHECK: r0 = memub(gp+#56)

# LDrih_abs_V4
{
  r0 = memh(##1024056)
}

# CHECK: immext(#1024000)
# CHECK: r0 = memh(##1024056)

# LDh_GP_V4
{
  r0 = memh(GP + #56)
}

# CHECK: r0 = memh(gp+#56)

# LDriuh_abs_V4
{
  r0 = memuh(##1024056)
}

# CHECK: immext(#1024000)
# CHECK: r0 = memuh(##1024056)

# LDuh_GP_V4
{
  r0 = memuh(GP + #56)
}

# CHECK: r0 = memuh(gp+#56)

# LDriw_abs_V4
{
  r0 = memw(##1024056)
}

# CHECK: immext(#1024000)
# CHECK: r0 = memw(##1024056)

# LDw_GP_V4
{
  r0 = memw(GP + #56)
}

# CHECK: r0 = memw(gp+#56)

# LDrid_abs_V4
{
  r1:0 = memd(##1024056)
}

# CHECK: immext(#1024000)
# CHECK: r1:0 = memd(##1024056)

# LDd_GP_V4
{
  r1:0 = memd(GP + #56)
}

# CHECK: r1:0 = memd(gp+#56)


{
  r0 = add(r0, ##123456)
  r1 = add(r1, #-64)
}

# CHECK: r0 = add(r0,##123456); r1 = add(r1,#-64)
