# RUN: llvm-mc -triple=hexagon < %s 2>%t ; \
# RUN:     FileCheck %s < %t --check-prefix=CHECK-STRICT
# RUN: llvm-mc -triple=hexagon -relax-nv-checks < %s 2>%t ; \
# RUN:     FileCheck %s < %t --check-prefix=CHECK-RELAXED

# CHECK-STRICT: :12:1: error: register `R0' used with `.new' but not validly modified in the same packet
# CHECK-RELAXED: :12:1: error: register `R0' used with `.new' but not validly modified in the same packet
{
  # invalid: r0 definition predicated on the opposite condition
  if (p3) r0 = add(r1, r2)
  if (!p3) memb(r20) = r0.new
}

# CHECK-STRICT: :20:1: error: register `R0' used with `.new' but not validly modified in the same packet
# CHECK-RELAXED: :20:1: error: register `R0' used with `.new' but not validly modified in the same packet
{
  # invalid: new-value compare-and-jump cannot use floating point value
  r0 = sfadd(r1, r2)
  if (cmp.eq(r0.new, #0)) jump:nt .
}

# CHECK-STRICT: :29:1: error: register `R0' used with `.new' but not validly modified in the same packet
# CHECK-RELAXED: :29:1: error: register `R0' used with `.new' but not validly modified in the same packet
{
  # invalid: definition of r0 should be unconditional (not explicitly docu-
  # mented)
  if (p0) r0 = r1
  if (cmp.eq(r0.new, #0)) jump:nt .
}


# No errors from this point on with the relaxed checks.
# CHECK-RELAXED-NOT: error

# CHECK-STRICT: :41:1: error: register `R0' used with `.new' but not validly modified in the same packet
{
  # valid (relaxed): p2 and p3 cannot be proven to violate the new-value
  # requirements
  if (p3) r0 = add(r1, r2)
  if (p2) memb(r20) = r0.new
}

# CHECK-STRICT: :48:1: error: register `R0' used with `.new' but not validly modified in the same packet
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

