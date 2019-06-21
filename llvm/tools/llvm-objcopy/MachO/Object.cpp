#include "Object.h"
#include "../llvm-objcopy.h"

namespace llvm {
namespace objcopy {
namespace macho {

const SymbolEntry *SymbolTable::getSymbolByIndex(uint32_t Index) const {
  assert(Index < Symbols.size() && "invalid symbol index");
  return Symbols[Index].get();
}

} // end namespace macho
} // end namespace objcopy
} // end namespace llvm
