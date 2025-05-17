#ifndef LLVM_EXECUTIONENGINE_ORC_ELFSTATICSYMTABGENERATOR_H
#define LLVM_EXECUTIONENGINE_ORC_ELFSTATICSYMTABGENERATOR_H

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"

namespace llvm {
namespace orc {

class ELFStaticSymtabGenerator : public DefinitionGenerator {
public:
  ELFStaticSymtabGenerator(ExecutionSession &ES);
  
  Error tryToGenerate(LookupState &LS, LookupKind K, JITDylib &JD,
                      JITDylibLookupFlags JDLookupFlags,
                      const SymbolLookupSet &Symbols) override;
private:
  ExecutionSession &ES;
  std::map<SymbolStringPtr, JITTargetAddress> StaticSymbols;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_ELFSTATICSYMTABGENERATOR_H
