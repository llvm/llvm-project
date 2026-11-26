# RUN: llvm-mc --verify -triple riscv64 -mattr=+xtheadmempair < %s

th.ldd t0, t1, (t2), 5, 4   # expected-error:22 {{immediate must be an integer in the range [0, 3]}}
th.ldd t0, t1, (t2)         # expected-error:29 {{too few operands for instruction}}
th.ldd t0, t1, (t2), 3, 5   # expected-error:25 {{operand must be constant 4}}
th.sdd a0, a1, (a2), 5, 4   # expected-error:22 {{immediate must be an integer in the range [0, 3]}}
th.sdd a0, a1, (a2)         # expected-error:29 {{too few operands for instruction}}
th.sdd a0, a1, (a2), 3, 5   # expected-error:25 {{operand must be constant 4}}
th.lwud t0, t1, (t2), 5, 4  # expected-error:23 {{immediate must be an integer in the range [0, 3]}}
th.lwud t0, t1, (t2)        # expected-error:29 {{too few operands for instruction}}
th.lwud t0, t1, (t2), 3, 5  # expected-error:26 {{operand must be constant 3}}
th.lwd a3, a4, (a5), 5, 4   # expected-error:22 {{immediate must be an integer in the range [0, 3]}}
th.lwd a3, a4, (a5)         # expected-error:29 {{too few operands for instruction}}
th.lwd a3, a4, (a5), 3, 5   # expected-error:25 {{operand must be constant 3}}
th.swd t3, t4, (t5), 5, 4   # expected-error:22 {{immediate must be an integer in the range [0, 3]}}
th.swd t3, t4, (t5)         # expected-error:29 {{too few operands for instruction}}
th.swd t3, t4, (t5), 3, 5   # expected-error:25 {{operand must be constant 3}}
th.ldd x6, x6, (x7), 2, 4   # expected-error:8 {{rs1, rd1, and rd2 cannot overlap}}
th.lwud x6, x6, (x6), 2, 3  # expected-error:9 {{rs1, rd1, and rd2 cannot overlap}}
