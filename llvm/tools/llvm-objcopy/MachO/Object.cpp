#include "Object.h"
#include "../llvm-objcopy.h"

namespace llvm {
namespace objcopy {
namespace macho {

const SymbolEntry *SymbolTable::getSymbolByIndex(uint32_t Index) const {
  assert(Index < Symbols.size() && "invalid symbol index");
  return Symbols[Index].get();
}

void Object::removeSections(function_ref<bool(const Section &)> ToRemove) {
  for (LoadCommand &LC : LoadCommands)
    LC.Sections.erase(std::remove_if(std::begin(LC.Sections),
                                     std::end(LC.Sections), ToRemove),
                      std::end(LC.Sections));
}

} // end namespace macho
} // end namespace objcopy
} // end namespace llvm
