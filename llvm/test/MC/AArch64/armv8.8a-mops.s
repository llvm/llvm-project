// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+mops,+mte < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+v8.8a,+mte < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+mops,+mte < %s \
// RUN:        | llvm-objdump -d --mattr=+mops,+mte - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+mops,+mte < %s \
// RUN:   | llvm-objdump -d --mattr=-mops,-mte - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+mops,+mte < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+mops,+mte -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST



cpyfp [x0]!, [x1]!, x2!
// CHECK-INST: cpyfp [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x04,0x01,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19010440      <unknown>

cpyfpwn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfpwn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x44,0x01,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19014440      <unknown>

cpyfprn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfprn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x84,0x01,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19018440      <unknown>

cpyfpn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfpn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xc4,0x01,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1901c440      <unknown>

cpyfpwt [x0]!, [x1]!, x2!
// CHECK-INST: cpyfpwt [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x14,0x01,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19011440      <unknown>

cpyfpwtwn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfpwtwn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x54,0x01,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19015440      <unknown>

cpyfpwtrn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfpwtrn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x94,0x01,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19019440      <unknown>

cpyfpwtn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfpwtn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xd4,0x01,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1901d440      <unknown>

cpyfprt [x0]!, [x1]!, x2!
// CHECK-INST: cpyfprt [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x24,0x01,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19012440      <unknown>

cpyfprtwn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfprtwn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x64,0x01,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19016440      <unknown>

cpyfprtrn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfprtrn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xa4,0x01,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1901a440      <unknown>

cpyfprtn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfprtn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xe4,0x01,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1901e440      <unknown>

cpyfpt [x0]!, [x1]!, x2!
// CHECK-INST: cpyfpt [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x34,0x01,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19013440      <unknown>

cpyfptwn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfptwn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x74,0x01,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19017440      <unknown>

cpyfptrn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfptrn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xb4,0x01,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1901b440      <unknown>

cpyfptn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfptn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xf4,0x01,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1901f440      <unknown>

cpyfm [x0]!, [x1]!, x2!
// CHECK-INST: cpyfm [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x04,0x41,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19410440      <unknown>

cpyfmwn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfmwn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x44,0x41,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19414440      <unknown>

cpyfmrn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfmrn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x84,0x41,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19418440      <unknown>

cpyfmn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfmn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xc4,0x41,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1941c440      <unknown>

cpyfmwt [x0]!, [x1]!, x2!
// CHECK-INST: cpyfmwt [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x14,0x41,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19411440      <unknown>

cpyfmwtwn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfmwtwn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x54,0x41,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19415440      <unknown>

cpyfmwtrn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfmwtrn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x94,0x41,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19419440      <unknown>

cpyfmwtn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfmwtn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xd4,0x41,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1941d440      <unknown>

cpyfmrt [x0]!, [x1]!, x2!
// CHECK-INST: cpyfmrt [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x24,0x41,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19412440      <unknown>

cpyfmrtwn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfmrtwn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x64,0x41,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19416440      <unknown>

cpyfmrtrn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfmrtrn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xa4,0x41,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1941a440      <unknown>

cpyfmrtn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfmrtn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xe4,0x41,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1941e440      <unknown>

cpyfmt [x0]!, [x1]!, x2!
// CHECK-INST: cpyfmt [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x34,0x41,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19413440      <unknown>

cpyfmtwn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfmtwn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x74,0x41,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19417440      <unknown>

cpyfmtrn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfmtrn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xb4,0x41,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1941b440      <unknown>

cpyfmtn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfmtn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xf4,0x41,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1941f440      <unknown>

cpyfe [x0]!, [x1]!, x2!
// CHECK-INST: cpyfe [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x04,0x81,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19810440      <unknown>

cpyfewn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfewn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x44,0x81,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19814440      <unknown>

cpyfern [x0]!, [x1]!, x2!
// CHECK-INST: cpyfern [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x84,0x81,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19818440      <unknown>

cpyfen [x0]!, [x1]!, x2!
// CHECK-INST: cpyfen [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xc4,0x81,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1981c440      <unknown>

cpyfewt [x0]!, [x1]!, x2!
// CHECK-INST: cpyfewt [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x14,0x81,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19811440      <unknown>

cpyfewtwn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfewtwn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x54,0x81,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19815440      <unknown>

cpyfewtrn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfewtrn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x94,0x81,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19819440      <unknown>

cpyfewtn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfewtn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xd4,0x81,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1981d440      <unknown>

cpyfert [x0]!, [x1]!, x2!
// CHECK-INST: cpyfert [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x24,0x81,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19812440      <unknown>

cpyfertwn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfertwn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x64,0x81,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19816440      <unknown>

cpyfertrn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfertrn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xa4,0x81,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1981a440      <unknown>

cpyfertn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfertn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xe4,0x81,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1981e440      <unknown>

cpyfet [x0]!, [x1]!, x2!
// CHECK-INST: cpyfet [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x34,0x81,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19813440      <unknown>

cpyfetwn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfetwn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x74,0x81,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19817440      <unknown>

cpyfetrn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfetrn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xb4,0x81,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1981b440      <unknown>

cpyfetn [x0]!, [x1]!, x2!
// CHECK-INST: cpyfetn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xf4,0x81,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1981f440      <unknown>

cpyp [x0]!, [x1]!, x2!
// CHECK-INST: cpyp [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x04,0x01,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d010440      <unknown>

cpypwn [x0]!, [x1]!, x2!
// CHECK-INST: cpypwn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x44,0x01,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d014440      <unknown>

cpyprn [x0]!, [x1]!, x2!
// CHECK-INST: cpyprn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x84,0x01,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d018440      <unknown>

cpypn [x0]!, [x1]!, x2!
// CHECK-INST: cpypn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xc4,0x01,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d01c440      <unknown>

cpypwt [x0]!, [x1]!, x2!
// CHECK-INST: cpypwt [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x14,0x01,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d011440      <unknown>

cpypwtwn [x0]!, [x1]!, x2!
// CHECK-INST: cpypwtwn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x54,0x01,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d015440      <unknown>

cpypwtrn [x0]!, [x1]!, x2!
// CHECK-INST: cpypwtrn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x94,0x01,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d019440      <unknown>

cpypwtn [x0]!, [x1]!, x2!
// CHECK-INST: cpypwtn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xd4,0x01,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d01d440      <unknown>

cpyprt [x0]!, [x1]!, x2!
// CHECK-INST: cpyprt [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x24,0x01,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d012440      <unknown>

cpyprtwn [x0]!, [x1]!, x2!
// CHECK-INST: cpyprtwn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x64,0x01,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d016440      <unknown>

cpyprtrn [x0]!, [x1]!, x2!
// CHECK-INST: cpyprtrn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xa4,0x01,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d01a440      <unknown>

cpyprtn [x0]!, [x1]!, x2!
// CHECK-INST: cpyprtn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xe4,0x01,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d01e440      <unknown>

cpypt [x0]!, [x1]!, x2!
// CHECK-INST: cpypt [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x34,0x01,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d013440      <unknown>

cpyptwn [x0]!, [x1]!, x2!
// CHECK-INST: cpyptwn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x74,0x01,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d017440      <unknown>

cpyptrn [x0]!, [x1]!, x2!
// CHECK-INST: cpyptrn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xb4,0x01,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d01b440      <unknown>

cpyptn [x0]!, [x1]!, x2!
// CHECK-INST: cpyptn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xf4,0x01,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d01f440      <unknown>

cpym [x0]!, [x1]!, x2!
// CHECK-INST: cpym [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x04,0x41,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d410440      <unknown>

cpymwn [x0]!, [x1]!, x2!
// CHECK-INST: cpymwn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x44,0x41,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d414440      <unknown>

cpymrn [x0]!, [x1]!, x2!
// CHECK-INST: cpymrn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x84,0x41,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d418440      <unknown>

cpymn [x0]!, [x1]!, x2!
// CHECK-INST: cpymn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xc4,0x41,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d41c440      <unknown>

cpymwt [x0]!, [x1]!, x2!
// CHECK-INST: cpymwt [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x14,0x41,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d411440      <unknown>

cpymwtwn [x0]!, [x1]!, x2!
// CHECK-INST: cpymwtwn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x54,0x41,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d415440      <unknown>

cpymwtrn [x0]!, [x1]!, x2!
// CHECK-INST: cpymwtrn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x94,0x41,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d419440      <unknown>

cpymwtn [x0]!, [x1]!, x2!
// CHECK-INST: cpymwtn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xd4,0x41,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d41d440      <unknown>

cpymrt [x0]!, [x1]!, x2!
// CHECK-INST: cpymrt [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x24,0x41,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d412440      <unknown>

cpymrtwn [x0]!, [x1]!, x2!
// CHECK-INST: cpymrtwn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x64,0x41,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d416440      <unknown>

cpymrtrn [x0]!, [x1]!, x2!
// CHECK-INST: cpymrtrn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xa4,0x41,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d41a440      <unknown>

cpymrtn [x0]!, [x1]!, x2!
// CHECK-INST: cpymrtn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xe4,0x41,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d41e440      <unknown>

cpymt [x0]!, [x1]!, x2!
// CHECK-INST: cpymt [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x34,0x41,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d413440      <unknown>

cpymtwn [x0]!, [x1]!, x2!
// CHECK-INST: cpymtwn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x74,0x41,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d417440      <unknown>

cpymtrn [x0]!, [x1]!, x2!
// CHECK-INST: cpymtrn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xb4,0x41,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d41b440      <unknown>

cpymtn [x0]!, [x1]!, x2!
// CHECK-INST: cpymtn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xf4,0x41,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d41f440      <unknown>

cpye [x0]!, [x1]!, x2!
// CHECK-INST: cpye [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x04,0x81,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d810440      <unknown>

cpyewn [x0]!, [x1]!, x2!
// CHECK-INST: cpyewn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x44,0x81,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d814440      <unknown>

cpyern [x0]!, [x1]!, x2!
// CHECK-INST: cpyern [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x84,0x81,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d818440      <unknown>

cpyen [x0]!, [x1]!, x2!
// CHECK-INST: cpyen [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xc4,0x81,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d81c440      <unknown>

cpyewt [x0]!, [x1]!, x2!
// CHECK-INST: cpyewt [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x14,0x81,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d811440      <unknown>

cpyewtwn [x0]!, [x1]!, x2!
// CHECK-INST: cpyewtwn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x54,0x81,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d815440      <unknown>

cpyewtrn [x0]!, [x1]!, x2!
// CHECK-INST: cpyewtrn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x94,0x81,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d819440      <unknown>

cpyewtn [x0]!, [x1]!, x2!
// CHECK-INST: cpyewtn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xd4,0x81,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d81d440      <unknown>

cpyert [x0]!, [x1]!, x2!
// CHECK-INST: cpyert [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x24,0x81,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d812440      <unknown>

cpyertwn [x0]!, [x1]!, x2!
// CHECK-INST: cpyertwn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x64,0x81,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d816440      <unknown>

cpyertrn [x0]!, [x1]!, x2!
// CHECK-INST: cpyertrn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xa4,0x81,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d81a440      <unknown>

cpyertn [x0]!, [x1]!, x2!
// CHECK-INST: cpyertn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xe4,0x81,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d81e440      <unknown>

cpyet [x0]!, [x1]!, x2!
// CHECK-INST: cpyet [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x34,0x81,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d813440      <unknown>

cpyetwn [x0]!, [x1]!, x2!
// CHECK-INST: cpyetwn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0x74,0x81,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d817440      <unknown>

cpyetrn [x0]!, [x1]!, x2!
// CHECK-INST: cpyetrn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xb4,0x81,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d81b440      <unknown>

cpyetn [x0]!, [x1]!, x2!
// CHECK-INST: cpyetn [x0]!, [x1]!, x2!
// CHECK-ENCODING: encoding: [0x40,0xf4,0x81,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d81f440      <unknown>

setp [x0]!, x1!, x2
// CHECK-INST: setp [x0]!, x1!, x2
// CHECK-ENCODING: encoding: [0x20,0x04,0xc2,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19c20420      <unknown>

setpt [x0]!, x1!, x2
// CHECK-INST: setpt [x0]!, x1!, x2
// CHECK-ENCODING: encoding: [0x20,0x14,0xc2,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19c21420      <unknown>

setpn [x0]!, x1!, x2
// CHECK-INST: setpn [x0]!, x1!, x2
// CHECK-ENCODING: encoding: [0x20,0x24,0xc2,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19c22420      <unknown>

setptn [x0]!, x1!, x2
// CHECK-INST: setptn [x0]!, x1!, x2
// CHECK-ENCODING: encoding: [0x20,0x34,0xc2,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19c23420      <unknown>

setm [x0]!, x1!, x2
// CHECK-INST: setm [x0]!, x1!, x2
// CHECK-ENCODING: encoding: [0x20,0x44,0xc2,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19c24420      <unknown>

setmt [x0]!, x1!, x2
// CHECK-INST: setmt [x0]!, x1!, x2
// CHECK-ENCODING: encoding: [0x20,0x54,0xc2,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19c25420      <unknown>

setmn [x0]!, x1!, x2
// CHECK-INST: setmn [x0]!, x1!, x2
// CHECK-ENCODING: encoding: [0x20,0x64,0xc2,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19c26420      <unknown>

setmtn [x0]!, x1!, x2
// CHECK-INST: setmtn [x0]!, x1!, x2
// CHECK-ENCODING: encoding: [0x20,0x74,0xc2,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19c27420      <unknown>

sete [x0]!, x1!, x2
// CHECK-INST: sete [x0]!, x1!, x2
// CHECK-ENCODING: encoding: [0x20,0x84,0xc2,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19c28420      <unknown>

setet [x0]!, x1!, x2
// CHECK-INST: setet [x0]!, x1!, x2
// CHECK-ENCODING: encoding: [0x20,0x94,0xc2,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19c29420      <unknown>

seten [x0]!, x1!, x2
// CHECK-INST: seten [x0]!, x1!, x2
// CHECK-ENCODING: encoding: [0x20,0xa4,0xc2,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19c2a420      <unknown>

setetn [x0]!, x1!, x2
// CHECK-INST: setetn [x0]!, x1!, x2
// CHECK-ENCODING: encoding: [0x20,0xb4,0xc2,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19c2b420      <unknown>

setgp [x0]!, x1!, x2
// CHECK-INST: setgp [x0]!, x1!, x2
// CHECK-ENCODING: encoding: [0x20,0x04,0xc2,0x1d]
// CHECK-ERROR: error: instruction requires: mops mte
// CHECK-UNKNOWN:  1dc20420      <unknown>

setgpt [x0]!, x1!, x2
// CHECK-INST: setgpt [x0]!, x1!, x2
// CHECK-ENCODING: encoding: [0x20,0x14,0xc2,0x1d]
// CHECK-ERROR: error: instruction requires: mops mte
// CHECK-UNKNOWN:  1dc21420      <unknown>

setgpn [x0]!, x1!, x2
// CHECK-INST: setgpn [x0]!, x1!, x2
// CHECK-ENCODING: encoding: [0x20,0x24,0xc2,0x1d]
// CHECK-ERROR: error: instruction requires: mops mte
// CHECK-UNKNOWN:  1dc22420      <unknown>

setgptn [x0]!, x1!, x2
// CHECK-INST: setgptn [x0]!, x1!, x2
// CHECK-ENCODING: encoding: [0x20,0x34,0xc2,0x1d]
// CHECK-ERROR: error: instruction requires: mops mte
// CHECK-UNKNOWN:  1dc23420      <unknown>

setgm [x0]!, x1!, x2
// CHECK-INST: setgm [x0]!, x1!, x2
// CHECK-ENCODING: encoding: [0x20,0x44,0xc2,0x1d]
// CHECK-ERROR: error: instruction requires: mops mte
// CHECK-UNKNOWN:  1dc24420      <unknown>

setgmt [x0]!, x1!, x2
// CHECK-INST: setgmt [x0]!, x1!, x2
// CHECK-ENCODING: encoding: [0x20,0x54,0xc2,0x1d]
// CHECK-ERROR: error: instruction requires: mops mte
// CHECK-UNKNOWN:  1dc25420      <unknown>

setgmn [x0]!, x1!, x2
// CHECK-INST: setgmn [x0]!, x1!, x2
// CHECK-ENCODING: encoding: [0x20,0x64,0xc2,0x1d]
// CHECK-ERROR: error: instruction requires: mops mte
// CHECK-UNKNOWN:  1dc26420      <unknown>

setgmtn [x0]!, x1!, x2
// CHECK-INST: setgmtn [x0]!, x1!, x2
// CHECK-ENCODING: encoding: [0x20,0x74,0xc2,0x1d]
// CHECK-ERROR: error: instruction requires: mops mte
// CHECK-UNKNOWN:  1dc27420      <unknown>

setge [x0]!, x1!, x2
// CHECK-INST: setge [x0]!, x1!, x2
// CHECK-ENCODING: encoding: [0x20,0x84,0xc2,0x1d]
// CHECK-ERROR: error: instruction requires: mops mte
// CHECK-UNKNOWN:  1dc28420      <unknown>

setget [x0]!, x1!, x2
// CHECK-INST: setget [x0]!, x1!, x2
// CHECK-ENCODING: encoding: [0x20,0x94,0xc2,0x1d]
// CHECK-ERROR: error: instruction requires: mops mte
// CHECK-UNKNOWN:  1dc29420      <unknown>

setgen [x0]!, x1!, x2
// CHECK-INST: setgen [x0]!, x1!, x2
// CHECK-ENCODING: encoding: [0x20,0xa4,0xc2,0x1d]
// CHECK-ERROR: error: instruction requires: mops mte
// CHECK-UNKNOWN:  1dc2a420      <unknown>

setgetn [x0]!, x1!, x2
// CHECK-INST: setgetn [x0]!, x1!, x2
// CHECK-ENCODING: encoding: [0x20,0xb4,0xc2,0x1d]
// CHECK-ERROR: error: instruction requires: mops mte
// CHECK-UNKNOWN:  1dc2b420      <unknown>

// XZR can only be used at:
//  - the size operand in CPY.
//  - the size or source operands in SET.

cpyfp [x0]!, [x1]!, xzr!
// CHECK-INST: cpyfp [x0]!, [x1]!, xzr!
// CHECK-ENCODING: encoding: [0xe0,0x07,0x01,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  190107e0      <unknown>

cpyfm [x0]!, [x1]!, xzr!
// CHECK-INST: cpyfm [x0]!, [x1]!, xzr!
// CHECK-ENCODING: encoding: [0xe0,0x07,0x41,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  194107e0      <unknown>

cpyfe [x0]!, [x1]!, xzr!
// CHECK-INST: cpyfe [x0]!, [x1]!, xzr!
// CHECK-ENCODING: encoding: [0xe0,0x07,0x81,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  198107e0      <unknown>

cpyp [x0]!, [x1]!, xzr!
// CHECK-INST: cpyp [x0]!, [x1]!, xzr!
// CHECK-ENCODING: encoding: [0xe0,0x07,0x01,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d0107e0      <unknown>

cpym [x0]!, [x1]!, xzr!
// CHECK-INST: cpym [x0]!, [x1]!, xzr!
// CHECK-ENCODING: encoding: [0xe0,0x07,0x41,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d4107e0      <unknown>

cpye [x0]!, [x1]!, xzr!
// CHECK-INST: cpye [x0]!, [x1]!, xzr!
// CHECK-ENCODING: encoding: [0xe0,0x07,0x81,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1d8107e0      <unknown>

setp [x0]!, xzr!, x2
// CHECK-INST: setp [x0]!, xzr!, x2
// CHECK-ENCODING: encoding: [0xe0,0x07,0xc2,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19c207e0      <unknown>

setp [x0]!, x1!, xzr
// CHECK-INST: setp [x0]!, x1!, xzr
// CHECK-ENCODING: encoding: [0x20,0x04,0xdf,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19df0420      <unknown>

setm [x0]!, xzr!, x2
// CHECK-INST: setm [x0]!, xzr!, x2
// CHECK-ENCODING: encoding: [0xe0,0x47,0xc2,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19c247e0      <unknown>

setm [x0]!, x1!, xzr
// CHECK-INST: setm [x0]!, x1!, xzr
// CHECK-ENCODING: encoding: [0x20,0x44,0xdf,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19df4420      <unknown>

sete [x0]!, xzr!, x2
// CHECK-INST: sete [x0]!, xzr!, x2
// CHECK-ENCODING: encoding: [0xe0,0x87,0xc2,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19c287e0      <unknown>

sete [x0]!, x1!, xzr
// CHECK-INST: sete [x0]!, x1!, xzr
// CHECK-ENCODING: encoding: [0x20,0x84,0xdf,0x19]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  19df8420      <unknown>

setgp [x0]!, xzr!, x2
// CHECK-INST: setgp [x0]!, xzr!, x2
// CHECK-ENCODING: encoding: [0xe0,0x07,0xc2,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1dc207e0      <unknown>

setgp [x0]!, x1!, xzr
// CHECK-INST: setgp [x0]!, x1!, xzr
// CHECK-ENCODING: encoding: [0x20,0x04,0xdf,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1ddf0420      <unknown>

setgm [x0]!, xzr!, x2
// CHECK-INST: setgm [x0]!, xzr!, x2
// CHECK-ENCODING: encoding: [0xe0,0x47,0xc2,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1dc247e0      <unknown>

setgm [x0]!, x1!, xzr
// CHECK-INST: setgm [x0]!, x1!, xzr
// CHECK-ENCODING: encoding: [0x20,0x44,0xdf,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1ddf4420      <unknown>

setge [x0]!, xzr!, x2
// CHECK-INST: setge [x0]!, xzr!, x2
// CHECK-ENCODING: encoding: [0xe0,0x87,0xc2,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1dc287e0      <unknown>

setge [x0]!, x1!, xzr
// CHECK-INST: setge [x0]!, x1!, xzr
// CHECK-ENCODING: encoding: [0x20,0x84,0xdf,0x1d]
// CHECK-ERROR: error: instruction requires: mops
// CHECK-UNKNOWN:  1ddf8420      <unknown>
