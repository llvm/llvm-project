# RUN: not llvm-mc -triple=hexagon -mv60 -mhvx -filetype=asm %s 2>%t; FileCheck %s --check-prefix=CHECK-V60-ERROR <%t
# RUN:     llvm-mc -triple=hexagon -mv62 -mhvx -filetype=asm %s | FileCheck %s

// for this a v60+/hvx instruction sequence, make sure fails with v60
// but passes with v62.  this is because this instruction uses different
// itinerary between v60 and v62
{
  v0.h=vsat(v5.w,v9.w)
  v16.h=vsat(v6.w,v26.w)
}
# CHECK-V60-ERROR: rror: invalid instruction packet: slot error
# CHECK: v0.h = vsat(v5.w,v9.w)
# CHECK: v16.h = vsat(v6.w,v26.w)
