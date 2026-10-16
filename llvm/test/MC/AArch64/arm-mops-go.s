// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+mops-go,+mte < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+mops-go,+mte < %s \
// RUN:        | llvm-objdump -d --mattr=+mops-go,+mte --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+mops-go,+mte < %s \
// RUN:        | llvm-objdump -d --mattr=-mops-go,-mte --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+mops-go,+mte < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+mops-go,+mte -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

//------------------------------------------------------------------------------
// FEAT_MOPS_GO Extension instructions
//------------------------------------------------------------------------------

setgop [x3]!, x2!
// CHECK-INST:    setgop [x3]!, x2!
// CHECK-ENCODING: [0x43,0x00,0xdf,0x1d]
// CHECK-UNKNOWN: 1ddf0043
// CHECK-ERROR: instruction requires: mops-go mte

setgom [x3]!, x2!
// CHECK-INST:    setgom [x3]!, x2!
// CHECK-ENCODING: [0x43,0x40,0xdf,0x1d]
// CHECK-UNKNOWN: 1ddf4043
// CHECK-ERROR: instruction requires: mops-go mte

setgoe [x3]!, x2!
// CHECK-INST:    setgoe [x3]!, x2!
// CHECK-ENCODING: [0x43,0x80,0xdf,0x1d]
// CHECK-UNKNOWN: 1ddf8043
// CHECK-ERROR: instruction requires: mops-go mte

setgopn [x3]!, x2!
// CHECK-INST:    setgopn [x3]!, x2!
// CHECK-ENCODING: [0x43,0x20,0xdf,0x1d]
// CHECK-UNKNOWN: 1ddf2043
// CHECK-ERROR: instruction requires: mops-go mte

setgomn [x3]!, x2!
// CHECK-INST:    setgomn [x3]!, x2!
// CHECK-ENCODING: [0x43,0x60,0xdf,0x1d]
// CHECK-UNKNOWN: 1ddf6043
// CHECK-ERROR: instruction requires: mops-go mte

setgoen [x3]!, x2!
// CHECK-INST:    setgoen [x3]!, x2!
// CHECK-ENCODING: [0x43,0xa0,0xdf,0x1d]
// CHECK-UNKNOWN: 1ddfa043
// CHECK-ERROR: instruction requires: mops-go mte

setgopt [x3]!, x2!
// CHECK-INST:    setgopt [x3]!, x2!
// CHECK-ENCODING: [0x43,0x10,0xdf,0x1d]
// CHECK-UNKNOWN: 1ddf1043
// CHECK-ERROR: instruction requires: mops-go mte

setgomt [x3]!, x2!
// CHECK-INST:    setgomt [x3]!, x2!
// CHECK-ENCODING: [0x43,0x50,0xdf,0x1d]
// CHECK-UNKNOWN: 1ddf5043
// CHECK-ERROR: instruction requires: mops-go mte

setgoet [x3]!, x2!
// CHECK-INST:    setgoet [x3]!, x2!
// CHECK-ENCODING: [0x43,0x90,0xdf,0x1d]
// CHECK-UNKNOWN: 1ddf9043
// CHECK-ERROR: instruction requires: mops-go mte

setgoptn [x3]!, x2!
// CHECK-INST:    setgoptn [x3]!, x2!
// CHECK-ENCODING: [0x43,0x30,0xdf,0x1d]
// CHECK-UNKNOWN: 1ddf3043
// CHECK-ERROR: instruction requires: mops-go mte

setgomtn [x3]!, x2!
// CHECK-INST:    setgomtn [x3]!, x2!
// CHECK-ENCODING: [0x43,0x70,0xdf,0x1d]
// CHECK-UNKNOWN: 1ddf7043
// CHECK-ERROR: instruction requires: mops-go mte

setgoetn [x3]!, x2!
// CHECK-INST:    setgoetn [x3]!, x2!
// CHECK-ENCODING: [0x43,0xb0,0xdf,0x1d]
// CHECK-UNKNOWN: 1ddfb043
// CHECK-ERROR: instruction requires: mops-go mte
