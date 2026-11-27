# RUN: llvm-mc -triple=hexagon --mcpu=hexagonv85 --mattr=+hlx -filetype=obj %s | \
# RUN:   llvm-objdump --mcpu=hexagonv85 --mattr=+hlx -d - | FileCheck %s

# Check that the HLX (Hexagon eLementwise eXtension) instructions assemble
# and disassemble.

# LX_add_h
l0.h = add(l1.h,l2.h)
# CHECK: e504ec40

# LX_add_w
l1:0.w = add(l3:2.w,l5:4.w)
# CHECK: ef08e080

# LX_sub_h
l0.h = sub(l1.h,l2.h)
# CHECK: e504f040

# LX_and
l0 = and(l1,l2)
# CHECK: e504f440

# LX_or
l0 = or(l1,l2)
# CHECK: e504f840

# LX_xor
l0 = xor(l1,l2)
# CHECK: e504f461

# LX_mux_h
l0.h = mux(l0.h,l1.h,l2.h)
# CHECK: e504e040

# LX_assign_s
l0.h = #5
# CHECK: e525e821

# LX_absdiff_h
l0.uh = absdiff(l1.h,l2.h)
# CHECK: ed04e440

# LX_absdiff_uw
l1:0.uw = absdiff(l3:2.uw,l5:4.uw)
# CHECK: ed08e4a1

# LX_max_h
l0.h = max(l1.h,l2.h)
# CHECK: e504e440

# LX_min_h
l0.h = min(l1.h,l2.h)
# CHECK: e504e441

# LX_abs_uh
l0.uh = abs(l1.h)
# CHECK: e501e860

# LX_S16b_new
# CHECK: e5206821
# CHECK: a480c000
{ l0.h = #0
  mem(r0) = l0.new }
