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
#include "llvm/ADT/STLExtras.h"
#include "llvm/DebugInfo/CodeView/BinaryByteStream.h"
#include "llvm/DebugInfo/CodeView/SymbolDumper.h"
#include "llvm/DebugInfo/CodeView/TypeDumper.h"
#include "llvm/DebugInfo/PDB/Raw/DbiStream.h"
#include "llvm/DebugInfo/PDB/Raw/IPDBStreamData.h"
#include "llvm/DebugInfo/PDB/Raw/MappedBlockStream.h"
#include "llvm/DebugInfo/PDB/Raw/ModuleDebugStream.h"
#include "llvm/DebugInfo/PDB/Raw/PDBFile.h"
#include "llvm/DebugInfo/PDB/Raw/RawSession.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace llvm;

namespace {
// We need a class which behaves like an immutable BinaryByteStream, but whose
// data
// is backed by an llvm::MemoryBuffer.  It also needs to own the underlying
// MemoryBuffer, so this simple adapter is a good way to achieve that.
class InputByteStream : public codeview::BinaryByteStream<false> {
public:
  explicit InputByteStream(std::unique_ptr<MemoryBuffer> Buffer)
      : BinaryByteStream(ArrayRef<uint8_t>(Buffer->getBuffer().bytes_begin(),
                                           Buffer->getBuffer().bytes_end())),
        MemBuffer(std::move(Buffer)) {}

  std::unique_ptr<MemoryBuffer> MemBuffer;
};
}

extern "C" int LLVMFuzzerTestOneInput(uint8_t *data, size_t size) {
  std::unique_ptr<MemoryBuffer> Buff = MemoryBuffer::getMemBuffer(
      StringRef((const char *)data, size), "", false);

  ScopedPrinter P(nulls());
  codeview::CVTypeDumper TD(&P, false);

  auto InputStream = llvm::make_unique<InputByteStream>(std::move(Buff));
  std::unique_ptr<pdb::PDBFile> File(new pdb::PDBFile(std::move(InputStream)));
  if (auto E = File->parseFileHeaders()) {
    consumeError(std::move(E));
    return 0;
  }
  if (auto E = File->parseStreamData()) {
    consumeError(std::move(E));
    return 0;
  }

  auto DbiS = File->getPDBDbiStream();
  if (auto E = DbiS.takeError()) {
    consumeError(std::move(E));
    return 0;
  }
  auto TpiS = File->getPDBTpiStream();
  if (auto E = TpiS.takeError()) {
    consumeError(std::move(E));
    return 0;
  }
  auto IpiS = File->getPDBIpiStream();
  if (auto E = IpiS.takeError()) {
    consumeError(std::move(E));
    return 0;
  }
  auto InfoS = File->getPDBInfoStream();
  if (auto E = InfoS.takeError()) {
    consumeError(std::move(E));
    return 0;
  }
  pdb::DbiStream &DS = DbiS.get();

  for (auto &Modi : DS.modules()) {
    auto ModStreamData = pdb::MappedBlockStream::createIndexedStream(
      Modi.Info.getModuleStreamIndex(), *File);
    if (!ModStreamData) {
      consumeError(ModStreamData.takeError());
      return 0;
    }
    pdb::ModuleDebugStream ModS(Modi.Info, std::move(*ModStreamData));
    if (auto E = ModS.reload()) {
      consumeError(std::move(E));
      return 0;
    }
    codeview::CVSymbolDumper SD(P, TD, nullptr, false);
    bool HadError = false;
    for (auto &S : ModS.symbols(&HadError)) {
      SD.dump(S);
    }
  }
  return 0;
}
