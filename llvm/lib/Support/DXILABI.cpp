
#include "llvm/Support/DXILABI.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
namespace llvm {
namespace dxil {
StringRef getResourceClassName(dxil::ResourceClass RC) {
  switch (RC) {
  case dxil::ResourceClass::SRV:
    return "SRV";
  case dxil::ResourceClass::UAV:
    return "UAV";
  case dxil::ResourceClass::CBuffer:
    return "CBuffer";
  case dxil::ResourceClass::Sampler:
    return "Sampler";
  }
  llvm_unreachable("Unhandled ResourceClass");
}
} // namespace dxil
} // namespace llvm