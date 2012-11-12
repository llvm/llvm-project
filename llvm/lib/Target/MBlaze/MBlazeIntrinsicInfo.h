//===-- MBlazeIntrinsicInfo.h - MBlaze Intrinsic Information ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MBlaze implementation of TargetIntrinsicInfo.
//
//===----------------------------------------------------------------------===//
#ifndef MBLAZEINTRINSICS_H
#define MBLAZEINTRINSICS_H

#include "llvm/Target/TargetIntrinsicInfo.h"

namespace llvm {

  class MBlazeIntrinsicInfo : public TargetIntrinsicInfo {
  public:
    std::string getName(unsigned IntrID, Type **Tys = 0,
                        unsigned numTys = 0) const;
    unsigned lookupName(const char *Name, unsigned Len) const;
    unsigned lookupGCCName(const char *Name) const;
    bool isOverloaded(unsigned IID) const;
    Function *getDeclaration(Module *M, unsigned ID, Type **Tys = 0,
                             unsigned numTys = 0) const;
  };

}

#endif
