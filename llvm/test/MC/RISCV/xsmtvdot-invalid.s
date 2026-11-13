# RUN: llvm-mc --verify -triple riscv64 -mattr=+xsmtvdot < %s

# NoSlide
smt.vmadot   v1, v2, v4 // expected-error:14 {{invalid operand for instruction}}

smt.vmadot   v2, v2, v4 // expected-error:14 {{the destination vector register group cannot overlap the source vector register group}}

smt.vmadot   v4, v2, v4 // expected-error:14 {{the destination vector register group cannot overlap the source vector register group}}

smt.vmadot   v2, v3, v4 // expected-error:14 {{the destination vector register group cannot overlap the source vector register group}}

smt.vmadot   v2, v4, v3 // expected-error:14 {{the destination vector register group cannot overlap the source vector register group}}

# slide
smt.vmadot1   v1, v2, v2 // expected-error:15 {{invalid operand for instruction}}

smt.vmadot1   v2, v5, v4 // expected-error:19 {{invalid operand for instruction}}

smt.vmadot1   v2, v2, v4 // expected-error:15 {{the destination vector register group cannot overlap the source vector register group}}

smt.vmadot1   v2, v4, v2 // expected-error:15 {{the destination vector register group cannot overlap the source vector register group}}

smt.vmadot1   v0, v1, v2 // expected-error:19 {{invalid operand for instruction}}

smt.vmadot1   v0, v2, v1 // expected-error:15 {{the destination vector register group cannot overlap the source vector register group}}
