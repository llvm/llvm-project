//===- SDNodeProperties.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_BASIC_SDNODEPROPERTIES_H
#define LLVM_UTILS_TABLEGEN_BASIC_SDNODEPROPERTIES_H

namespace llvm {

class Record;

// SelectionDAG node properties.
//  SDNPMemOperand: indicates that a node touches memory and therefore must
//                  have an associated memory operand that describes the access.
enum SDNP {
  SDNPCommutative,
  SDNPAssociative,
  SDNPHasChain,
  SDNPOutGlue,
  SDNPInGlue,
  SDNPOptInGlue,
  SDNPMayLoad,
  SDNPMayStore,
  SDNPSideEffect,
  SDNPMemOperand,
  SDNPVariadic,
};

unsigned parseSDPatternOperatorProperties(const Record *R);

} // namespace llvm

#endif // LLVM_UTILS_TABLEGEN_BASIC_SDNODEPROPERTIES_H
