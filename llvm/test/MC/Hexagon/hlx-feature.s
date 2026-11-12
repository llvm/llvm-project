/// Check that HLX instructions are not parsed by default.
// RUN: not llvm-mc --mcpu=hexagonv85 -triple=hexagon -filetype=obj %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ERROR --implicit-check-not=error

// ERROR: error: invalid instruction
// ERROR-NEXT: l0.uh=max(l0.uh,l0.uh)

/// Check the +hlx attribute in llvm-mc.
// RUN: llvm-mc --mcpu=hexagonv85 -triple=hexagon --mattr=+hlx -filetype=obj %s 2>&1 \
// RUN:   | llvm-objdump -d --no-print-imm-hex --mcpu=hexagonv85 --mattr=+hlx - \
// RUN:   | FileCheck %s --implicit-check-not=error

/// Check the -mhlx option in llvm-mc.
// RUN: llvm-mc --mcpu=hexagonv85 -triple=hexagon -mhlx -filetype=obj %s 2>&1 \
// RUN:   | llvm-objdump -d --no-print-imm-hex --mcpu=hexagonv85 --mattr=+hlx - \
// RUN:   | FileCheck %s --implicit-check-not=error

// CHECK: L0.uh = max(L0.uh,L0.uh)

/// Check that llvm-objdump does not disassemble HLX by default.
// RUN: llvm-mc -triple=hexagon --mcpu=hexagonv85 --mattr=+hlx -filetype=obj %s 2>&1 \
// RUN:   | llvm-objdump -d --no-print-imm-hex - \
// RUN:   | FileCheck %s --check-prefix=UNKNOWN --match-full-lines

// UNKNOWN:      00000000 <.text>:
// UNKNOWN-NEXT:        0:	20 e4 00 e5	e500e420	<unknown>

l0.uh=max(l0.uh,l0.uh)
