// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+mops-go,+mte < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+all < %s \
// RUN:        | llvm-objdump -d --mattr=+mops-go,+mte --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+all < %s \
// RUN:        | llvm-objdump -d --mattr=-mops-go,-mte --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+mops-go,+mte < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+mops-go,+mte -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

//------------------------------------------------------------------------------
// FEAT_MOPS_GO Extension instructions
//------------------------------------------------------------------------------

setgop [x3]!, x2!, x1
// CHECK-INST:    setgop [x3]!, x2!, x1
// CHECK-ENCODING: [0x43,0x00,0xc1,0x1d]
// CHECK-UNKNOWN: 1dc10043
// CHECK-ERROR: instruction requires: mops-go mte

setgom [x3]!, x2!, x1
// CHECK-INST:    setgom [x3]!, x2!, x1
// CHECK-ENCODING: [0x43,0x40,0xc1,0x1d]
// CHECK-UNKNOWN: 1dc14043
// CHECK-ERROR: instruction requires: mops-go mte

setgoe [x3]!, x2!, x1
// CHECK-INST:    setgoe [x3]!, x2!, x1
// CHECK-ENCODING: [0x43,0x80,0xc1,0x1d]
// CHECK-UNKNOWN: 1dc18043
// CHECK-ERROR: instruction requires: mops-go mte

setgopn [x3]!, x2!, x1
// CHECK-INST:    setgopn [x3]!, x2!, x1
// CHECK-ENCODING: [0x43,0x20,0xc1,0x1d]
// CHECK-UNKNOWN: 1dc12043
// CHECK-ERROR: instruction requires: mops-go mte

setgomn [x3]!, x2!, x1
// CHECK-INST:    setgomn [x3]!, x2!, x1
// CHECK-ENCODING: [0x43,0x60,0xc1,0x1d]
// CHECK-UNKNOWN: 1dc16043
// CHECK-ERROR: instruction requires: mops-go mte

setgoen [x3]!, x2!, x1
// CHECK-INST:    setgoen [x3]!, x2!, x1
// CHECK-ENCODING: [0x43,0xa0,0xc1,0x1d]
// CHECK-UNKNOWN: 1dc1a043
// CHECK-ERROR: instruction requires: mops-go mte

setgopt [x3]!, x2!, x1
// CHECK-INST:    setgopt [x3]!, x2!, x1
// CHECK-ENCODING: [0x43,0x10,0xc1,0x1d]
// CHECK-UNKNOWN: 1dc11043
// CHECK-ERROR: instruction requires: mops-go mte

setgomt [x3]!, x2!, x1
// CHECK-INST:    setgomt [x3]!, x2!, x1
// CHECK-ENCODING: [0x43,0x50,0xc1,0x1d]
// CHECK-UNKNOWN: 1dc15043
// CHECK-ERROR: instruction requires: mops-go mte

setgoet [x3]!, x2!, x1
// CHECK-INST:    setgoet [x3]!, x2!, x1
// CHECK-ENCODING: [0x43,0x90,0xc1,0x1d]
// CHECK-UNKNOWN: 1dc19043
// CHECK-ERROR: instruction requires: mops-go mte

setgoptn [x3]!, x2!, x1
// CHECK-INST:    setgoptn [x3]!, x2!, x1
// CHECK-ENCODING: [0x43,0x30,0xc1,0x1d]
// CHECK-UNKNOWN: 1dc13043
// CHECK-ERROR: instruction requires: mops-go mte

setgomtn [x3]!, x2!, x1
// CHECK-INST:    setgomtn [x3]!, x2!, x1
// CHECK-ENCODING: [0x43,0x70,0xc1,0x1d]
// CHECK-UNKNOWN: 1dc17043
// CHECK-ERROR: instruction requires: mops-go mte

setgoetn [x3]!, x2!, x1
// CHECK-INST:    setgoetn [x3]!, x2!, x1
// CHECK-ENCODING: [0x43,0xb0,0xc1,0x1d]
// CHECK-UNKNOWN: 1dc1b043
// CHECK-ERROR: instruction requires: mops-go mte
