# RUN: llvm-mc --verify -triple riscv32 -mattr=+xtheadcmo < %s
# RUN: llvm-mc --verify -triple riscv64 -mattr=+xtheadcmo < %s

th.dcache.csw # expected-error:1 {{too few operands for instruction}}
th.dcache.isw # expected-error:1 {{too few operands for instruction}}
th.dcache.cisw # expected-error:1 {{too few operands for instruction}}
th.dcache.cval1 # expected-error:1 {{too few operands for instruction}}
th.dcache.cva # expected-error:1 {{too few operands for instruction}}
th.dcache.iva # expected-error:1 {{too few operands for instruction}}
th.dcache.civa # expected-error:1 {{too few operands for instruction}}
th.dcache.cpal1 # expected-error:1 {{too few operands for instruction}}
th.dcache.cpa # expected-error:1 {{too few operands for instruction}}
th.dcache.ipa # expected-error:1 {{too few operands for instruction}}
th.dcache.cipa # expected-error:1 {{too few operands for instruction}}
th.icache.iva # expected-error:1 {{too few operands for instruction}}
th.icache.ipa # expected-error:1 {{too few operands for instruction}}

th.dcache.call t0 # expected-error:16 {{unexpected extra operand for instruction}}
th.dcache.iall t0 # expected-error:16 {{unexpected extra operand for instruction}}
th.dcache.ciall t0 # expected-error:17 {{unexpected extra operand for instruction}}
th.icache.iall t0 # expected-error:16 {{unexpected extra operand for instruction}}
th.icache.ialls t0 # expected-error:17 {{unexpected extra operand for instruction}}
th.l2cache.call t0 # expected-error:17 {{unexpected extra operand for instruction}}
th.l2cache.iall t0 # expected-error:17 {{unexpected extra operand for instruction}}
th.l2cache.ciall t0 # expected-error:18 {{unexpected extra operand for instruction}}
