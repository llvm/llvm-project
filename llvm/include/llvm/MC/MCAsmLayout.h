//===- MCAsmLayout.h - Assembly Layout Object -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCASMLAYOUT_H
#define LLVM_MC_MCASMLAYOUT_H

namespace llvm {
class MCAssembler;

class MCAsmLayout {
public:
  MCAsmLayout(MCAssembler &) {}
};

} // end namespace llvm

#endif
