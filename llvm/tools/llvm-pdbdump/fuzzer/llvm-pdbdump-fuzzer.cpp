//===-- llvm-pdbdump-fuzzer.cpp - Fuzz the llvm-pdbdump tool --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements a function that runs llvm-pdbdump
///  on a single input. This function is then linked into the Fuzzer library.
///
//===----------------------------------------------------------------------===//
#include "llvm/DebugInfo/CodeView/SymbolDumper.h"
#include "llvm/DebugInfo/CodeView/TypeDumper.h"
#include "llvm/DebugInfo/PDB/Raw/DbiStream.h"
#include "llvm/DebugInfo/PDB/Raw/ModStream.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/RawSession.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace llvm;

extern "C" void LLVMFuzzerTestOneInput(uint8_t *data, size_t size) {
  std::unique_ptr<MemoryBuffer> Buff = MemoryBuffer::getMemBuffer(
      StringRef((const char *)data, size), "", false);

  ScopedPrinter P(nulls());
  codeview::CVTypeDumper TD(P, false);

  std::unique_ptr<pdb::PDBFile> File(new pdb::PDBFile(std::move(Buff)));
  if (auto E = File->parseFileHeaders()) {
    consumeError(std::move(E));
    return;
  }
  if (auto E = File->parseStreamData()) {
    consumeError(std::move(E));
    return;
  }

  auto DbiS = File->getPDBDbiStream();
  if (auto E = DbiS.takeError()) {
    consumeError(std::move(E));
    return;
  }
  auto TpiS = File->getPDBTpiStream();
  if (auto E = TpiS.takeError()) {
    consumeError(std::move(E));
    return;
  }
  auto IpiS = File->getPDBIpiStream();
  if (auto E = IpiS.takeError()) {
    consumeError(std::move(E));
    return;
  }
  auto InfoS = File->getPDBInfoStream();
  if (auto E = InfoS.takeError()) {
    consumeError(std::move(E));
    return;
  }
  pdb::DbiStream &DS = DbiS.get();

  for (auto &Modi : DS.modules()) {
    pdb::ModStream ModS(*File, Modi.Info);
    if (auto E = ModS.reload()) {
      consumeError(std::move(E));
      return;
    }
    codeview::CVSymbolDumper SD(P, TD, nullptr, false);
    bool HadError = false;
    for (auto &S : ModS.symbols(&HadError)) {
      SD.dump(S);
    }
  }
}
