#ifndef LLVM_SUPPORT_LIBCXXABI_H
#define LLVM_SUPPORT_LIBCXXABI_H

#include "llvm/IR/DataLayout.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {

class CXXABI {

  virtual const char * getVTablePrefix() = 0;
  virtual const char * getTypeNamePrefix() = 0;
  virtual const char * getTypeInfoPrefix() = 0;

public:
  static std::unique_ptr<CXXABI> Create(Triple &TT);
  virtual ~CXXABI() {}
  virtual int64_t
  getOffsetFromTypeInfoSlotToAddressPoint(const DataLayout &DT) = 0;

  bool isVTable(StringRef Name) { return Name.starts_with(getVTablePrefix()); }
  bool isTypeName(StringRef Name) {
    return Name.starts_with(getTypeNamePrefix());
  }
  bool isTypeInfo(StringRef Name) {
    return Name.starts_with(getTypeInfoPrefix());
  }

  std::string getTypeNameFromTypeInfo(StringRef TypeInfo);
  std::string getTypeInfoFromVTable(StringRef VTable);
};

class Itanium final : public CXXABI {

  const char * getVTablePrefix() override { return "_ZTV"; }
  const char * getTypeNamePrefix() override { return "_ZTS"; }
  const char * getTypeInfoPrefix() override { return "_ZTI"; }

public:
  virtual ~Itanium() {}

  int64_t
  getOffsetFromTypeInfoSlotToAddressPoint(const DataLayout &DL) override {
    return -2 * static_cast<int64_t>(DL.getPointerSize());
  }
};

} // namespace llvm
#endif
