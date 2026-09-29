# RUN: not llvm-mc -triple=hexagon %s 2>%t; FileCheck --implicit-check-not=error: %s <%t

.Lfoo:
{ p3:0 = r0
  if (!p0.new) jump:t .Lfoo }

# CHECK: error: register `P0' used with `.new' but not validly modified in the same packet

{ c4 = r0
  if (!p0.new) jump:t .Lfoo }

# CHECK: error: register `P0' used with `.new' but not validly modified in the same packet

{ c4 = r0
  p0 = r0
  if (!p0.new) jump:t .Lfoo }

# CHECK: error: register `P0' used with `.new' but not validly modified in the same packet
# CHECK: error: register `P3_0' modified more than once
