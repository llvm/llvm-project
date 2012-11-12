//===--------- llvm/AddressingMode.h - Addressing Mode    -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//  This file contains addressing mode data structures which are shared
//  between LSR and a number of places in the codegen.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADDRESSING_MODE_H
#define LLVM_ADDRESSING_MODE_H

#include "llvm/Support/DataTypes.h"

namespace llvm {

class GlobalValue;

/// AddrMode - This represents an addressing mode of:
///    BaseGV + BaseOffs + BaseReg + Scale*ScaleReg
/// If BaseGV is null,  there is no BaseGV.
/// If BaseOffs is zero, there is no base offset.
/// If HasBaseReg is false, there is no base register.
/// If Scale is zero, there is no ScaleReg.  Scale of 1 indicates a reg with
/// no scale.
///
struct AddrMode {
  GlobalValue *BaseGV;
  int64_t      BaseOffs;
  bool         HasBaseReg;
  int64_t      Scale;
  AddrMode() : BaseGV(0), BaseOffs(0), HasBaseReg(false), Scale(0) {}
};

} // End llvm namespace

#endif
