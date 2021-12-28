# RUN: not llvm-mc -arch=hexagon -filetype=obj -mhvx -mcpu=hexagonv66 %s 2> %t; FileCheck --implicit-check-not=error %s <%t

{
  if (p0) memb(r14+#8)=r4.new
  if (p0) z=vmem(r4++#0)
}

# CHECK: error: Instruction does not have a valid new register producer
