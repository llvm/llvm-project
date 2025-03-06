# RUN: llvm-mc %s -triple=xtensa -mattr=+mac16 -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s

.align	4
LBL0:

# CHECK-INST: umul.aa.ll	a2, a3
# CHECK: encoding: [0x34,0x02,0x70]
	umul.aa.ll	a2, a3
# CHECK-INST: umul.aa.lh	a2, a3
# CHECK: encoding: [0x34,0x02,0x72]
	umul.aa.lh	a2, a3
# CHECK-INST: umul.aa.hl	a2, a3
# CHECK: encoding: [0x34,0x02,0x71]
	umul.aa.hl	a2, a3
# CHECK-INST: umul.aa.hh	a2, a3
# CHECK: encoding: [0x34,0x02,0x73]
	umul.aa.hh	a2, a3

# CHECK-INST: mul.aa.ll	a2, a3
# CHECK: encoding: [0x34,0x02,0x74]
	mul.aa.ll	a2, a3
# CHECK-INST: mul.aa.lh	a2, a3
# CHECK: encoding: [0x34,0x02,0x76]
	mul.aa.lh	a2, a3
# CHECK-INST: mul.aa.hl	a2, a3
# CHECK: encoding: [0x34,0x02,0x75]
	mul.aa.hl	a2, a3
# CHECK-INST: mul.aa.hh	a2, a3
# CHECK: encoding: [0x34,0x02,0x77]
	mul.aa.hh	a2, a3

# CHECK-INST: mul.ad.ll	a2, m2
# CHECK: encoding: [0x04,0x02,0x34]
	mul.ad.ll	a2, m2
# CHECK-INST: mul.ad.lh	a2, m2
# CHECK: encoding: [0x04,0x02,0x36]
	mul.ad.lh	a2, m2
# CHECK-INST: mul.ad.hl	a2, m2
# CHECK: encoding: [0x04,0x02,0x35]
	mul.ad.hl	a2, m2
# CHECK-INST: mul.ad.hh	a2, m2
# CHECK: encoding: [0x04,0x02,0x37]
	mul.ad.hh	a2, m2

# CHECK-INST: mul.da.ll	m1, a3
# CHECK: encoding: [0x34,0x40,0x64]
	mul.da.ll	m1, a3
# CHECK-INST: mul.da.lh	m1, a3
# CHECK: encoding: [0x34,0x40,0x66]
	mul.da.lh	m1, a3
# CHECK-INST: mul.da.hl	m1, a3
# CHECK: encoding: [0x34,0x40,0x65]
	mul.da.hl	m1, a3
# CHECK-INST: mul.da.hh	m1, a3
# CHECK: encoding: [0x34,0x40,0x67]
	mul.da.hh	m1, a3

# CHECK-INST: mul.dd.ll	m1, m2
# CHECK: encoding: [0x04,0x40,0x24]
	mul.dd.ll	m1, m2
# CHECK-INST: mul.dd.lh	m1, m2
# CHECK: encoding: [0x04,0x40,0x26]
	mul.dd.lh	m1, m2
# CHECK-INST: mul.dd.hl	m1, m2
# CHECK: encoding: [0x04,0x40,0x25]
	mul.dd.hl	m1, m2
# CHECK-INST: mul.dd.hh	m1, m2
# CHECK: encoding: [0x04,0x40,0x27]
	mul.dd.hh	m1, m2

# CHECK-INST: mula.aa.ll	a2, a3
# CHECK: encoding: [0x34,0x02,0x78]
	mula.aa.ll	a2, a3
# CHECK-INST: mula.aa.lh	a2, a3
# CHECK: encoding: [0x34,0x02,0x7a]
	mula.aa.lh	a2, a3
# CHECK-INST: mula.aa.hl	a2, a3
# CHECK: encoding: [0x34,0x02,0x79]
	mula.aa.hl	a2, a3
# CHECK-INST: mula.aa.hh	a2, a3
# CHECK: encoding: [0x34,0x02,0x7b]
	mula.aa.hh	a2, a3

# CHECK-INST: mula.ad.ll	a2, m2
# CHECK: encoding: [0x04,0x02,0x38]
	mula.ad.ll	a2, m2
# CHECK-INST: mula.ad.lh	a2, m2
# CHECK: encoding: [0x04,0x02,0x3a]
	mula.ad.lh	a2, m2
# CHECK-INST: mula.ad.hl	a2, m2
# CHECK: encoding: [0x04,0x02,0x39]
	mula.ad.hl	a2, m2
# CHECK-INST: mula.ad.hh	a2, m2
# CHECK: encoding: [0x04,0x02,0x3b]
	mula.ad.hh	a2, m2

# CHECK-INST: mula.da.ll	m1, a3
# CHECK: encoding: [0x34,0x40,0x68]
	mula.da.ll	m1, a3
# CHECK-INST: mula.da.lh	m1, a3
# CHECK: encoding: [0x34,0x40,0x6a]
	mula.da.lh	m1, a3
# CHECK-INST: mula.da.hl	m1, a3
# CHECK: encoding: [0x34,0x40,0x69]
	mula.da.hl	m1, a3
# CHECK-INST: mula.da.hh	m1, a3
# CHECK: encoding: [0x34,0x40,0x6b]
	mula.da.hh	m1, a3

# CHECK-INST: mula.dd.ll	m1, m2
# CHECK: encoding: [0x04,0x40,0x28]
	mula.dd.ll	m1, m2
# CHECK-INST: mula.dd.lh	m1, m2
# CHECK: encoding: [0x04,0x40,0x2a]
	mula.dd.lh	m1, m2
# CHECK-INST: mula.dd.hl	m1, m2
# CHECK: encoding: [0x04,0x40,0x29]
	mula.dd.hl	m1, m2
# CHECK-INST: mula.dd.hh	m1, m2
# CHECK: encoding: [0x04,0x40,0x2b]
	mula.dd.hh	m1, m2

# CHECK-INST: muls.aa.ll	a2, a3
# CHECK: encoding: [0x34,0x02,0x7c]
	muls.aa.ll	a2, a3
# CHECK-INST: muls.aa.lh	a2, a3
# CHECK: encoding: [0x34,0x02,0x7e]
	muls.aa.lh	a2, a3
# CHECK-INST: muls.aa.hl	a2, a3
# CHECK: encoding: [0x34,0x02,0x7d]
	muls.aa.hl	a2, a3
# CHECK-INST: muls.aa.hh	a2, a3
# CHECK: encoding: [0x34,0x02,0x7f]
	muls.aa.hh	a2, a3

# CHECK-INST: muls.ad.ll	a2, m2
# CHECK: encoding: [0x04,0x02,0x3c]
	muls.ad.ll	a2, m2
# CHECK-INST: muls.ad.lh	a2, m2
# CHECK: encoding: [0x04,0x02,0x3e]
	muls.ad.lh	a2, m2
# CHECK-INST: muls.ad.hl	a2, m2
# CHECK: encoding: [0x04,0x02,0x3d]
	muls.ad.hl	a2, m2
# CHECK-INST: muls.ad.hh	a2, m2
# CHECK: encoding: [0x04,0x02,0x3f]
	muls.ad.hh	a2, m2

# CHECK-INST: muls.da.ll	m1, a3
# CHECK: encoding: [0x34,0x40,0x6c]
	muls.da.ll	m1, a3
# CHECK-INST: muls.da.lh	m1, a3
# CHECK: encoding: [0x34,0x40,0x6e]
	muls.da.lh	m1, a3
# CHECK-INST: muls.da.hl	m1, a3
# CHECK: encoding: [0x34,0x40,0x6d]
	muls.da.hl	m1, a3
# CHECK-INST: muls.da.hh	m1, a3
# CHECK: encoding: [0x34,0x40,0x6f]
	muls.da.hh	m1, a3

# CHECK-INST: muls.dd.ll	m1, m2
# CHECK: encoding: [0x04,0x40,0x2c]
	muls.dd.ll	m1, m2
# CHECK-INST: muls.dd.lh	m1, m2
# CHECK: encoding: [0x04,0x40,0x2e]
	muls.dd.lh	m1, m2
# CHECK-INST: muls.dd.hl	m1, m2
# CHECK: encoding: [0x04,0x40,0x2d]
	muls.dd.hl	m1, m2
# CHECK-INST: muls.dd.hh	m1, m2
# CHECK: encoding: [0x04,0x40,0x2f]
	muls.dd.hh	m1, m2

# CHECK-INST: mula.da.ll.lddec	 m1, a8, m0, a3
# CHECK: encoding: [0x34,0x18,0x58]
	mula.da.ll.lddec	 m1, a8, m0, a3
# CHECK-INST: mula.da.hl.lddec	 m1, a8, m0, a3
# CHECK: encoding: [0x34,0x18,0x59]
	mula.da.hl.lddec	 m1, a8, m0, a3
# CHECK-INST: mula.da.lh.lddec	 m1, a8, m0, a3
# CHECK: encoding: [0x34,0x18,0x5a]
	mula.da.lh.lddec	 m1, a8, m0, a3
# CHECK-INST: mula.da.hh.lddec	 m1, a8, m0, a3
# CHECK: encoding: [0x34,0x18,0x5b]
	mula.da.hh.lddec	 m1, a8, m0, a3

# CHECK-INST: mula.dd.ll.lddec	 m1, a8, m0, m2
# CHECK: encoding: [0x04,0x18,0x18]
	mula.dd.ll.lddec	 m1, a8, m0, m2
# CHECK-INST: mula.dd.hl.lddec	 m1, a8, m0, m2
# CHECK: encoding: [0x04,0x18,0x19]
	mula.dd.hl.lddec	 m1, a8, m0, m2
# CHECK-INST: mula.dd.lh.lddec	 m1, a8, m0, m2
# CHECK: encoding: [0x04,0x18,0x1a]
	mula.dd.lh.lddec	 m1, a8, m0, m2
# CHECK-INST: mula.dd.hh.lddec	 m1, a8, m0, m2
# CHECK: encoding: [0x04,0x18,0x1b]
	mula.dd.hh.lddec	 m1, a8, m0, m2

# CHECK-INST: mula.da.ll.ldinc	 m1, a8, m0, a3
# CHECK: encoding: [0x34,0x18,0x48]
	mula.da.ll.ldinc	 m1, a8, m0, a3
# CHECK-INST: mula.da.hl.ldinc	 m1, a8, m0, a3
# CHECK: encoding: [0x34,0x18,0x49]
	mula.da.hl.ldinc	 m1, a8, m0, a3
# CHECK-INST: mula.da.lh.ldinc	 m1, a8, m0, a3
# CHECK: encoding: [0x34,0x18,0x4a]
	mula.da.lh.ldinc	 m1, a8, m0, a3
# CHECK-INST: mula.da.hh.ldinc	 m1, a8, m0, a3
# CHECK: encoding: [0x34,0x18,0x4b]
	mula.da.hh.ldinc	 m1, a8, m0, a3

# CHECK-INST: mula.dd.ll.ldinc	 m1, a8, m0, m2
# CHECK: encoding: [0x04,0x18,0x08]
	mula.dd.ll.ldinc	 m1, a8, m0, m2
# CHECK-INST: mula.dd.hl.ldinc	 m1, a8, m0, m2
# CHECK: encoding: [0x04,0x18,0x09]
	mula.dd.hl.ldinc	 m1, a8, m0, m2
# CHECK-INST: mula.dd.lh.ldinc	 m1, a8, m0, m2
# CHECK: encoding: [0x04,0x18,0x0a]
	mula.dd.lh.ldinc	 m1, a8, m0, m2
# CHECK-INST: mula.dd.hh.ldinc	 m1, a8, m0, m2
# CHECK: encoding: [0x04,0x18,0x0b]
	mula.dd.hh.ldinc	 m1, a8, m0, m2

# CHECK-INST: lddec	 m0, a8
# CHECK: encoding: [0x04,0x08,0x90]
	lddec	 m0, a8
# CHECK-INST: ldinc	 m0, a8
# CHECK: encoding: [0x04,0x08,0x80]
	ldinc	 m0, a8
