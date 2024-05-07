//===- DXIL.cpp - Abstractions for DXIL constructs ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/DXIL.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

using namespace llvm;
using namespace llvm::dxil;

static Error errInvalid(const char *Msg) {
  return createStringError(std::errc::invalid_argument, Msg);
}

template <typename... Ts>
static Error errInvalid(const char *Fmt, const Ts &...Vals) {
  return createStringError(std::errc::invalid_argument, Fmt, Vals...);
}

Expected<DXILVersion> DXILVersion::get(Module &M) {
  Triple TT(Triple::normalize(M.getTargetTriple()));

  if (!TT.isDXIL())
    return errInvalid("Cannot get DXIL version for arch '%s'",
                      TT.getArchName().str().c_str());

  switch (TT.getSubArch()) {
  case Triple::NoSubArch:
  case Triple::DXILSubArch_v1_0:
    return DXILVersion(1, 0);
  case Triple::DXILSubArch_v1_1:
    return DXILVersion(1, 1);
  case Triple::DXILSubArch_v1_2:
    return DXILVersion(1, 2);
  case Triple::DXILSubArch_v1_3:
    return DXILVersion(1, 3);
  case Triple::DXILSubArch_v1_4:
    return DXILVersion(1, 4);
  case Triple::DXILSubArch_v1_5:
    return DXILVersion(1, 5);
  case Triple::DXILSubArch_v1_6:
    return DXILVersion(1, 6);
  case Triple::DXILSubArch_v1_7:
    return DXILVersion(1, 7);
  case Triple::DXILSubArch_v1_8:
    return DXILVersion(1, 8);
  default:
    return errInvalid("Cannot get DXIL version for arch '%s'",
                      TT.getArchName().str().c_str());
  }
}

Expected<DXILVersion> DXILVersion::readDXIL(Module &M) {
  NamedMDNode *DXILVersionMD = M.getNamedMetadata("dx.version");
  if (!DXILVersionMD)
    return DXILVersion();

  if (DXILVersionMD->getNumOperands() != 1)
    return errInvalid("dx.version must have one operand");

  MDNode *N = DXILVersionMD->getOperand(0);
  if (N->getNumOperands() != 2)
    return errInvalid("dx.version must have 2 components, not %d",
                      N->getNumOperands());

  const auto *MajorOp = mdconst::dyn_extract<ConstantInt>(N->getOperand(0));
  const auto *MinorOp = mdconst::dyn_extract<ConstantInt>(N->getOperand(1));
  if (!MajorOp)
    return errInvalid("dx.version major version must be an integer");
  if (!MinorOp)
    return errInvalid("dx.version minor version must be an integer");

  return DXILVersion(MajorOp->getZExtValue(), MinorOp->getZExtValue());
}

void DXILVersion::strip(Module &M) { M.setTargetTriple("dxil-ms-dx"); }

void DXILVersion::embed(Module &M) {
  Triple TT(Triple::normalize(M.getTargetTriple()));
  SmallString<64> Triple;
  raw_svector_ostream OS(Triple);
  print(OS);
  OS << "-" << TT.getVendorName() << "-" << TT.getOSAndEnvironmentName();
  M.setTargetTriple(OS.str());
}

void DXILVersion::stripDXIL(Module &M) {
  if (NamedMDNode *V = M.getNamedMetadata("dx.version")) {
    V->dropAllReferences();
    V->eraseFromParent();
  }
}

void DXILVersion::embedDXIL(Module &M) {
  LLVMContext &Ctx = M.getContext();
  IRBuilder<> B(Ctx);

  Metadata *Vals[2];
  Vals[0] = ConstantAsMetadata::get(B.getInt32(Major));
  Vals[1] = ConstantAsMetadata::get(B.getInt32(Minor));
  MDNode *MD = MDNode::get(Ctx, Vals);

  NamedMDNode *V = M.getOrInsertNamedMetadata("dx.version");
  if (V->getNumOperands())
    V->setOperand(0, MD);
  else
    V->addOperand(MD);
}
