#ifndef LLVM_ABI_QUALTYPE_MAPPER_H
#define LLVM_ABI_QUALTYPE_MAPPER_H

#include "llvm/Support/Allocator.h"
#include <clang/AST/ASTContext.h>
#include <clang/AST/Type.h>
#include <llvm/ABI/Types.h>
#include <llvm/ADT/DenseMap.h>

namespace llvm {
namespace abi {

class QualTypeMapper {
private:
  clang::ASTContext &ASTCtx;
  TypeBuilder Builder;

  // llvm::DenseMap<clang::QualType	, const Type*> TypeCache;

  const Type *convertBuiltinType(const clang::BuiltinType *BT);
  const Type *convertPointerType(const clang::PointerType *PT);
  const Type *convertArrayType(const clang::ArrayType *AT);
  const Type *convertVectorType(const clang::VectorType *VT);
  const Type *convertRecordType(const clang::RecordType *RT);
  const Type *convertFunctionType(const clang::FunctionProtoType *FT);
  const Type *convertEnumType(const clang::EnumType *ET);

  void computeRecordLayout(const clang::RecordDecl *RD,
                           llvm::SmallVectorImpl<FieldInfo> &Fields,
                           uint64_t &TotalSize, uint64_t &Alignment,
                           StructPacking &Packing);

  uint64_t getTypeSize(clang::QualType QT) const;
  uint64_t getTypeAlign(clang::QualType QT) const;
  uint64_t getPointerSize() const;
  uint64_t getPointerAlign() const;

public:
  explicit QualTypeMapper(clang::ASTContext &Ctx, BumpPtrAllocator &Alloc)
      : ASTCtx(Ctx), Builder(Alloc) {}

  const Type *convertType(clang::QualType QT);

  // void clearCache() {TypeCache.clear();}

  TypeBuilder getTypeBuilder() { return Builder; }
};

} // namespace abi
} // namespace llvm

#endif // !LLVM_ABI_QUALTYPE_MAPPER_H
