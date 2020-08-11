//===- ELFStub.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===/

#include "llvm/InterfaceStub/ELFStub.h"

using namespace llvm;
using namespace llvm::elfabi;

ELFStub::ELFStub(ELFStub const &Stub) {
  TbeVersion = Stub.TbeVersion;
  Arch = Stub.Arch;
  SoName = Stub.SoName;
  NeededLibs = Stub.NeededLibs;
  Symbols = Stub.Symbols;
}

ELFStub::ELFStub(ELFStub &&Stub) {
  TbeVersion = std::move(Stub.TbeVersion);
  Arch = std::move(Stub.Arch);
  SoName = std::move(Stub.SoName);
  NeededLibs = std::move(Stub.NeededLibs);
  Symbols = std::move(Stub.Symbols);
}
