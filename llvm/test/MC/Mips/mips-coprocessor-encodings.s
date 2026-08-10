# RUN: llvm-mc %s -triple=mips64-unknown-freebsd -show-encoding \
# RUN:| FileCheck --check-prefix=MIPS64 %s

# MIPS64:	dmtc0	$12, $16, 2             # encoding: [0x40,0xac,0x80,0x02]
# MIPS64:	dmtc0	$12, $16, 0             # encoding: [0x40,0xac,0x80,0x00]
# MIPS64:	mtc0	$12, $16, 2             # encoding: [0x40,0x8c,0x80,0x02]
# MIPS64:	mtc0	$12, $16, 0             # encoding: [0x40,0x8c,0x80,0x00]
# MIPS64:	dmfc0	$12, $16, 2             # encoding: [0x40,0x2c,0x80,0x02]
# MIPS64:	dmfc0	$12, $16, 0             # encoding: [0x40,0x2c,0x80,0x00]
# MIPS64:	mfc0	$12, $16, 2             # encoding: [0x40,0x0c,0x80,0x02]
# MIPS64:	mfc0	$12, $16, 0             # encoding: [0x40,0x0c,0x80,0x00]

	dmtc0	$12, $16, 2
	dmtc0	$12, $16
	mtc0	$12, $16, 2
	mtc0	$12, $16
	dmfc0	$12, $16, 2
	dmfc0	$12, $16
	mfc0	$12, $16, 2
	mfc0	$12, $16

# MIPS64:	dmtc2	$12, $16, 2             # encoding: [0x48,0xac,0x80,0x02]
# MIPS64:	dmtc2	$12, $16, 0             # encoding: [0x48,0xac,0x80,0x00]
# MIPS64:	mtc2	$12, $16, 2             # encoding: [0x48,0x8c,0x80,0x02]
# MIPS64:	mtc2	$12, $16, 0             # encoding: [0x48,0x8c,0x80,0x00]
# MIPS64:	dmfc2	$12, $16, 2             # encoding: [0x48,0x2c,0x80,0x02]
# MIPS64:	dmfc2	$12, $16, 0             # encoding: [0x48,0x2c,0x80,0x00]
# MIPS64:	mfc2	$12, $16, 2             # encoding: [0x48,0x0c,0x80,0x02]
# MIPS64:	mfc2	$12, $16, 0             # encoding: [0x48,0x0c,0x80,0x00]

	dmtc2	$12, $16, 2
	dmtc2	$12, $16
	mtc2	$12, $16, 2
	mtc2	$12, $16
	dmfc2	$12, $16, 2
	dmfc2	$12, $16
	mfc2	$12, $16, 2
	mfc2	$12, $16
