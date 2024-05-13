# RUN: not llvm-mc -triple=hexagon -mcpu=hexagonv62 -filetype=obj -o - %s
# Check that a duplex involving dealloc_return is correctly checked
# dealloc_return cannot be involved in a double jump packet

{ r0=add(r0,#-1)
  p0=cmp.eq(r0,r0); if (p0.new) jump:nt 0
  if (p0) dealloc_return }
