# RUN: not llvm-mc -triple=hexagon < %s 2>&1 | \
# RUN:     FileCheck %s --check-prefix=CHECK-STRICT
# RUN: not llvm-mc -triple=hexagon -relax-nv-checks < %s 2>&1 | \
# RUN:     FileCheck %s --check-prefix=CHECK-RELAXED

# CHECK-STRICT: :10:3: note: Register producer has the opposite predicate sense as consumer
# CHECK-RELAXED: :10:3: note: Register producer has the opposite predicate sense as consumer
{
  # invalid: r0 definition predicated on the opposite condition
  if (p3) r0 = add(r1, r2)
  if (!p3) memb(r20) = r0.new
}

# CHECK-STRICT: :18:3: note: FPU instructions cannot be new-value producers for jumps
# CHECK-RELAXED: :18:3: note: FPU instructions cannot be new-value producers for jumps
# CHECK-RELAXED: :19:3: error: Instruction does not have a valid new register producer
{ # invalid: new-value compare-and-jump cannot use floating point value
  r0 = sfadd(r1, r2)
  if (cmp.eq(r0.new, #0)) jump:nt .
}

# No errors from this point on with the relaxed checks.
# CHECK-RELAXED-NOT: error

# CHECK-STRICT: :28:3: note: Register producer is predicated and consumer is unconditional
{
  # valid in relaxed, p0 could always be true
  if (p0) r0 = r1
  if (cmp.eq(r0.new, #0)) jump:nt .
}

# CHECK-STRICT: :36:3: note: Register producer does not use the same predicate register as the consumer
{
  # valid (relaxed): p2 and p3 cannot be proven to violate the new-value
  # requirements
  if (p3) r0 = add(r1, r2)
  if (p2) memb(r20) = r0.new
}

# CHECK-STRICT: :43:3: note: Register producer is predicated and consumer is unconditional
{
  # valid (relaxed): p3 could be always true
  if (p3) r0 = add(r1, r2)
  memb(r20) = r0.new
}


# No errors from this point on with the strict checks.
# CHECK-RELAXED-NOT: error

{
  # valid: r0 defined unconditionally
  r0 = add(r1, r2)
  if (p2) memb(r20) = r0.new
}

{
  # valid: r0 definition and use identically predicated
  if (p3) r0 = add(r1, r2)
  if (p3) memb(r20) = r0.new
}

{
  # valid: r0 defined regardless of p0
  if (p0) r0 = #0
  if (!p0) r0 = #1
  if (p0) memb(r20) = r0.new
}

