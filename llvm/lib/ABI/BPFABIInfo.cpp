#include "llvm/ABI/ABIInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"

/*using namespace llvm;*/

namespace {

class BPFABIInfo final : public llvm::abi::ABIInfo {
public:
  BPFABIInfo() {}

  llvm::abi::ABIArgInfo
  classifyArgumentType(const llvm::abi::ABIType *Ty) override;
  llvm::abi::ABIArgInfo
  classifyReturnType(const llvm::abi::ABIType *Ty) override;
  llvm::Type *getLLVMType(const llvm::abi::ABIType *Ty,
                          llvm::LLVMContext &Context) override;

private:
  llvm::abi::ABIArgInfo classifyType(const llvm::abi::ABIType *Ty,
                                     bool IsReturn);
  bool shouldPassIndirect(const llvm::abi::ABIType *Ty, bool IsReturn);
};

bool BPFABIInfo::shouldPassIndirect(const llvm::abi::ABIType *Ty,
                                    bool IsReturn) {
  if (llvm::isa<llvm::abi::RecordType>(Ty) ||
      llvm::isa<llvm::abi::UnionType>(Ty))
    return true;

  if (llvm::isa<llvm::abi::ArrayType>(Ty))
    return true;

  if (const auto *ST = llvm::dyn_cast<llvm::abi::ScalarType>(Ty)) {
    uint64_t Size = ST->getSize();
    return Size > 8;
  }

  return false;
}

llvm::abi::ABIArgInfo BPFABIInfo::classifyType(const llvm::abi::ABIType *Ty,
                                               bool IsReturn) {
  if (Ty->getKind() == llvm::abi::ABITypeKind::Void)
    return llvm::abi::ABIArgInfo::getIgnore();

  if (shouldPassIndirect(Ty, IsReturn))
    return llvm::abi::ABIArgInfo::getIndirect(Ty);

  return llvm::abi::ABIArgInfo::getDirect();
}

llvm::abi::ABIArgInfo
BPFABIInfo::classifyArgumentType(const llvm::abi::ABIType *Ty) {
  return classifyType(Ty, /*IsReturn=*/false);
}

llvm::abi::ABIArgInfo
BPFABIInfo::classifyReturnType(const llvm::abi::ABIType *Ty) {
  return classifyType(Ty, /*IsReturn=*/true);
}

llvm::Type *BPFABIInfo::getLLVMType(const llvm::abi::ABIType *Ty,
                                    llvm::LLVMContext &Context) {
  switch (Ty->getKind()) {
  case llvm::abi::ABITypeKind::Void:
    return llvm::Type::getVoidTy(Context);
  case llvm::abi::ABITypeKind::Bool:
    return llvm::Type::getInt1Ty(Context);
  case llvm::abi::ABITypeKind::Char:
  case llvm::abi::ABITypeKind::SChar:
  case llvm::abi::ABITypeKind::UChar:
    return llvm::Type::getInt8Ty(Context);
  case llvm::abi::ABITypeKind::Short:
  case llvm::abi::ABITypeKind::UShort:
    return llvm::Type::getInt16Ty(Context);
  case llvm::abi::ABITypeKind::Int:
  case llvm::abi::ABITypeKind::UInt:
    return llvm::Type::getInt32Ty(Context);
  case llvm::abi::ABITypeKind::Long:
  case llvm::abi::ABITypeKind::ULong:
  case llvm::abi::ABITypeKind::LongLong:
  case llvm::abi::ABITypeKind::ULongLong:
    return llvm::Type::getInt64Ty(Context);
  case llvm::abi::ABITypeKind::Float:
    return llvm::Type::getFloatTy(Context);
  case llvm::abi::ABITypeKind::Double:
    return llvm::Type::getDoubleTy(Context);
  case llvm::abi::ABITypeKind::LongDouble:
    return llvm::Type::getFP128Ty(Context);
  case llvm::abi::ABITypeKind::Pointer: {
    const llvm::abi::PointerType *PT = llvm::cast<llvm::abi::PointerType>(Ty);
    return llvm::PointerType::get(getLLVMType(PT->getPointeeType(), Context),
                                  0);
  }
  case llvm::abi::ABITypeKind::Array: {
    const llvm::abi::ArrayType *AT = llvm::cast<llvm::abi::ArrayType>(Ty);
    return llvm::ArrayType::get(getLLVMType(AT->getElementType(), Context),
                                AT->getNumElements());
  }
  case llvm::abi::ABITypeKind::Record: {
    const llvm::abi::RecordType *RT = llvm::cast<llvm::abi::RecordType>(Ty);
    llvm::SmallVector<llvm::Type *, 8> Elements;
    for (const llvm::abi::RecordField &Field : RT->getFields()) {
      llvm::Type *FieldTy = getLLVMType(Field.Type, Context);
      Elements.push_back(FieldTy);
    }
    return llvm::StructType::get(Context, Elements, RT->isPacked());
  }
  case llvm::abi::ABITypeKind::Union: {
    const llvm::abi::UnionType *UT = llvm::cast<llvm::abi::UnionType>(Ty);
    llvm::Type *LargestType = nullptr;

    for (const llvm::abi::RecordField &Field : UT->getFields()) {
      llvm::Type *FieldTy = getLLVMType(Field.Type, Context);
      if (!LargestType ||
          (FieldTy->isStructTy() && !LargestType->isStructTy()) ||
          (FieldTy->isArrayTy() && !LargestType->isArrayTy())) {
        LargestType = FieldTy;
      }
    }

    if (!LargestType)
      return llvm::StructType::get(Context);

    return llvm::StructType::get(Context, LargestType);
  }
  }
  llvm_unreachable("Unknown ABIType kind");
}

} // end anonymous namespace

std::unique_ptr<llvm::abi::ABIInfo>
llvm::abi::createABIInfo(StringRef TargetTriple) {
  if (TargetTriple.starts_with("bpf"))
    return std::make_unique<BPFABIInfo>();

  llvm_unreachable("Unsupported target triple");
}
