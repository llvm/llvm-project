# RUN: not llvm-mc -arch=hexagon -mhvx -filetype=asm %s 2>%t; FileCheck %s <%t

{ r0=memw(r1=##0)
  memw(r0)=r1.new }
# CHECK: 3:3: note: Absolute-set registers cannot be a new-value producer
# CHECK: 4:3: error: Instruction does not have a valid new register producer

{ r1:0=r1:0
  memw(r0)=r0.new }
# CHECK: 8:3: note: Double registers cannot be new-value producers
# CHECK: 9:3: error: Instruction does not have a valid new register producer

{ r1=memw(r0++m0)
  memw(r0)=r0.new }
# CHECK: 13:3: note: Auto-increment registers cannot be a new-value producer
# CHECK: 14:3: error: Instruction does not have a valid new register producer

{ r0=sfadd(r0,r0)
  if (cmp.eq(r0.new,r0)) jump:t 0x0 }
# CHECK: 18:3: note: FPU instructions cannot be new-value producers for jumps
# CHECK: 19:3: error: Instruction does not have a valid new register producer

{ v0=vmem(r0++m0)
  memw(r0)=r0.new }
# CHECK: 23:3: note: Auto-increment registers cannot be a new-value producer
# CHECK: 24:3: error: Instruction does not have a valid new register producer

{ if (p0) r0=r0
  if (!p0) memw(r0)=r0.new }
# CHECK: 28:3: note: Register producer has the opposite predicate sense as consumer
# CHECK: 29:3: error: Instruction does not have a valid new register producer

{ if (p0) r0=r0
  memw(r0)=r0.new }
# CHECK: 33:3: note: Register producer is predicated and consumer is unconditional
# CHECK: 34:3: error: Instruction does not have a valid new register producer

{ if (p0) r0=r0
  if (cmp.eq(r0.new,r0)) jump:t 0x0 }
# CHECK: 38:3: note: Register producer is predicated and consumer is unconditional
# CHECK: 39:3: error: Instruction does not have a valid new register producer

{ r0=memw(r1=##0)
  if (p0) memw(r0)=r1.new }
# CHECK: 43:3: note: Absolute-set registers cannot be a new-value producer
# CHECK: 44:3: error: Instruction does not have a valid new register producer
