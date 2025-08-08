
#include "llvm/Support/DXILABI.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace llvm;
namespace llvm {
namespace dxil {
StringRef getResourceClassName(dxil::ResourceClass RC) {
  return enumToStringRef(RC, dxbc::getResourceClasses());
}
} // namespace dxil
} // namespace llvm