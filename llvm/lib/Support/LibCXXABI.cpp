#include "llvm/Support/LibCXXABI.h"

namespace llvm {

std::unique_ptr<CXXABI> CXXABI::Create(Triple &TT) {
  if (TT.getOS() == Triple::Linux)
    return std::make_unique<Itanium>();

  return nullptr;
}

std::string CXXABI::getTypeNameFromTypeInfo(StringRef TypeInfo) {
  assert(TypeInfo.starts_with(getTypeInfoPrefix()) &&
         "TypeInfo is not starts with the correct type infor prefix");
  TypeInfo.consume_front(getTypeInfoPrefix());
  return getTypeNamePrefix() + TypeInfo.str();
}

std::string CXXABI::getTypeInfoFromVTable(StringRef VTable) {
  assert(VTable.starts_with(getVTablePrefix()) &&
         "TypeInfo is not starts with the correct type infor prefix");
  VTable.consume_front(getVTablePrefix());
  return getTypeInfoPrefix() + VTable.str();
}
} // namespace llvm
