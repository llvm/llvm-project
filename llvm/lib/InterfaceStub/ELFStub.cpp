//===- ELFStub.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===/

#include "llvm/InterfaceStub/ELFStub.h"
#include "llvm/Support/Error.h"

using namespace llvm;
using namespace llvm::elfabi;

ELFStub::ELFStub(ELFStub const &Stub) {
  TbeVersion = Stub.TbeVersion;
  Target = Stub.Target;
  SoName = Stub.SoName;
  NeededLibs = Stub.NeededLibs;
  Symbols = Stub.Symbols;
}

ELFStub::ELFStub(ELFStub &&Stub) {
  TbeVersion = std::move(Stub.TbeVersion);
  Target = std::move(Stub.Target);
  SoName = std::move(Stub.SoName);
  NeededLibs = std::move(Stub.NeededLibs);
  Symbols = std::move(Stub.Symbols);
}

ELFStubTriple::ELFStubTriple(ELFStubTriple const &Stub) {
  TbeVersion = Stub.TbeVersion;
  Target = Stub.Target;
  SoName = Stub.SoName;
  NeededLibs = Stub.NeededLibs;
  Symbols = Stub.Symbols;
}

ELFStubTriple::ELFStubTriple(ELFStub const &Stub) {
  TbeVersion = Stub.TbeVersion;
  Target = Stub.Target;
  SoName = Stub.SoName;
  NeededLibs = Stub.NeededLibs;
  Symbols = Stub.Symbols;
}

ELFStubTriple::ELFStubTriple(ELFStubTriple &&Stub) {
  TbeVersion = std::move(Stub.TbeVersion);
  Target = std::move(Stub.Target);
  SoName = std::move(Stub.SoName);
  NeededLibs = std::move(Stub.NeededLibs);
  Symbols = std::move(Stub.Symbols);
}
