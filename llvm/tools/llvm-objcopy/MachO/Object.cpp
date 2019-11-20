#include "Object.h"
#include "../llvm-objcopy.h"

namespace llvm {
namespace objcopy {
namespace macho {

const SymbolEntry *SymbolTable::getSymbolByIndex(uint32_t Index) const {
  assert(Index < Symbols.size() && "invalid symbol index");
  return Symbols[Index].get();
}

SymbolEntry *SymbolTable::getSymbolByIndex(uint32_t Index) {
  return const_cast<SymbolEntry *>(
      static_cast<const SymbolTable *>(this)->getSymbolByIndex(Index));
}

void SymbolTable::removeSymbols(
    function_ref<bool(const std::unique_ptr<SymbolEntry> &)> ToRemove) {
  Symbols.erase(
      std::remove_if(std::begin(Symbols), std::end(Symbols), ToRemove),
      std::end(Symbols));
}

void Object::removeSections(function_ref<bool(const Section &)> ToRemove) {
  for (LoadCommand &LC : LoadCommands)
    LC.Sections.erase(std::remove_if(std::begin(LC.Sections),
                                     std::end(LC.Sections), ToRemove),
                      std::end(LC.Sections));
}

void Object::addLoadCommand(LoadCommand LC) {
  LoadCommands.push_back(std::move(LC));
}

} // end namespace macho
} // end namespace objcopy
} // end namespace llvm
