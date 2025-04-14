#include "llvm/Object/ObjectFile.h"
#include "llvm/ExecutionEngine/Orc/ELFStaticSymtabGenerator.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/DebugUtils.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Error.h"

#define DEBUG_TYPE "orc"

namespace llvm {
namespace orc {

ELFStaticSymtabGenerator::ELFStaticSymtabGenerator(ExecutionSession &ES) : ES(ES) {
  auto MBOrErr = MemoryBuffer::getFile("/proc/self/exe");
  if (!MBOrErr) {
    return;
  }
  std::unique_ptr<MemoryBuffer> MB = std::move(MBOrErr.get());
  auto ObjOrErr = object::ObjectFile::createObjectFile(MB->getMemBufferRef());
  if (!ObjOrErr) {
    consumeError(ObjOrErr.takeError());
    return;
  }
  auto *Obj = ObjOrErr->get();

  for (const auto &Sym : Obj->symbols()) {
    Expected<StringRef> NameOrErr = Sym.getName();
    if (!NameOrErr) {
      consumeError(NameOrErr.takeError());
      continue;
    }
    StringRef Name = *NameOrErr;

    // Retrieve the symbol flags.
    Expected<uint32_t> FlagsOrErr = Sym.getFlags();
    if (!FlagsOrErr) {
      consumeError(FlagsOrErr.takeError());
      continue;
    }
    uint32_t Flags = *FlagsOrErr;

    // Only add symbols that are defined.
    if (!(Flags & object::BasicSymbolRef::SF_Undefined)) {
      Expected<uint64_t> AddrOrErr = Sym.getAddress();
      if (!AddrOrErr) {
        consumeError(AddrOrErr.takeError());
        continue;
      }
      JITTargetAddress Addr = *AddrOrErr;
      // Intern the symbol name using the ExecutionSession's symbol pool.
      SymbolStringPtr SSP = ES.getSymbolStringPool()->intern(Name);
      StaticSymbols[SSP] = Addr;
    }
  }
}

Error ELFStaticSymtabGenerator::tryToGenerate(LookupState &LS, LookupKind K,
                                              JITDylib &JD,
                                              JITDylibLookupFlags JDLookupFlags,
                                              const SymbolLookupSet &Symbols) {

  SymbolMap NewSymbols;
  for (auto &KV : Symbols) {
    SymbolStringPtr Name = KV.first;
    if (auto It = StaticSymbols.find(Name); It != StaticSymbols.end()) {
      JITEvaluatedSymbol Sym(It->second, JITSymbolFlags::Exported);
      NewSymbols[Name] = ExecutorSymbolDef(ExecutorAddr(It->second), JITSymbolFlags::Exported);
    }
  }
  // If any symbols were found, add them to the JITDylib.
  if (!NewSymbols.empty())
    return JD.define(absoluteSymbols(std::move(NewSymbols)));
  return Error::success();
}
  
} // end namespace orc
} // end namespace llvm

