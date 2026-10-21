//===- MemRefInterfaces.cpp - OpenACC external models ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/Extensions/MemRefInterfaces.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUBase.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
using namespace mlir::acc;

namespace {
static bool isScalarLikeType(Type type) {
  return type.isIntOrIndexOrFloat() || isa<ComplexType>(type);
}

/// Helper function to attach the `VarName` attribute to an operation
/// if a variable name is provided.
static void attachVarNameAttr(Operation *op, OpBuilder &builder,
                              StringRef varName) {
  if (!varName.empty()) {
    auto varNameAttr = acc::VarNameAttr::get(builder.getContext(), varName);
    op->setDiscardableAttr(acc::getVarNameAttrName(), varNameAttr);
  }
}

template <typename T>
struct MemRefPointerLikeModel
    : public PointerLikeType::ExternalModel<MemRefPointerLikeModel<T>, T> {
  Type getElementType(Type pointer) const {
    return cast<T>(pointer).getElementType();
  }

  mlir::acc::VariableTypeCategory
  getPointeeTypeCategory(Type pointer, TypedValue<PointerLikeType> varPtr,
                         Type varType) const {
    if (auto mappableTy = dyn_cast<MappableType>(varType)) {
      return mappableTy.getTypeCategory(varPtr);
    }
    auto memrefTy = cast<T>(pointer);
    if (!memrefTy.hasRank()) {
      // This memref is unranked - aka it could have any rank, including a
      // rank of 0 which could mean scalar. For now, return uncategorized.
      return mlir::acc::VariableTypeCategory::uncategorized;
    }

    if (memrefTy.getRank() == 0) {
      if (isScalarLikeType(memrefTy.getElementType())) {
        return mlir::acc::VariableTypeCategory::scalar;
      }
      // Zero-rank non-scalar - need further analysis to determine the type
      // category. For now, return uncategorized.
      return mlir::acc::VariableTypeCategory::uncategorized;
    }

    // It has a rank - must be an array.
    assert(memrefTy.getRank() > 0 && "rank expected to be positive");
    return mlir::acc::VariableTypeCategory::array;
  }

  mlir::Value genAllocate(Type pointer, OpBuilder &builder, Location loc,
                          StringRef varName, Type varType, Value originalVar,
                          bool &needsFree) const {
    auto memrefTy = cast<MemRefType>(pointer);

    // Check if this is a static memref (all dimensions are known) - if yes
    // then we can generate an alloca operation.
    if (memrefTy.hasStaticShape()) {
      needsFree = false; // alloca doesn't need deallocation
      auto allocaOp = memref::AllocaOp::create(builder, loc, memrefTy);
      attachVarNameAttr(allocaOp, builder, varName);
      return allocaOp.getResult();
    }

    // For dynamic memrefs, extract sizes from the original variable if
    // provided. Otherwise they cannot be handled.
    if (originalVar && originalVar.getType() == memrefTy &&
        memrefTy.hasRank()) {
      SmallVector<Value> dynamicSizes;
      for (int64_t i = 0; i < memrefTy.getRank(); ++i) {
        if (memrefTy.isDynamicDim(i)) {
          // Extract the size of dimension i from the original variable
          auto indexValue = arith::ConstantIndexOp::create(builder, loc, i);
          auto dimSize =
              memref::DimOp::create(builder, loc, originalVar, indexValue);
          dynamicSizes.push_back(dimSize);
        }
        // Note: We only add dynamic sizes to the dynamicSizes array
        // Static dimensions are handled automatically by AllocOp
      }
      needsFree = true; // alloc needs deallocation
      auto allocOp =
          memref::AllocOp::create(builder, loc, memrefTy, dynamicSizes);
      attachVarNameAttr(allocOp, builder, varName);
      return allocOp.getResult();
    }

    // TODO: Unranked not yet supported.
    return {};
  }

  bool genFree(Type pointer, OpBuilder &builder, Location loc,
               TypedValue<PointerLikeType> varToFree, Value allocRes,
               Type varType) const {
    if (auto memrefValue = dyn_cast<TypedValue<MemRefType>>(varToFree)) {
      // Use allocRes if provided to determine the allocation type
      Value valueToInspect = allocRes ? allocRes : memrefValue;

      // Walk through casts to find the original allocation
      Value currentValue = valueToInspect;
      Operation *originalAlloc = nullptr;

      // Follow the chain of operations to find the original allocation
      // even if a casted result is provided.
      while (currentValue) {
        if (auto *definingOp = currentValue.getDefiningOp()) {
          // Check if this is an allocation operation
          if (isa<memref::AllocOp, memref::AllocaOp>(definingOp)) {
            originalAlloc = definingOp;
            break;
          }

          // Check if this is a cast operation we can look through
          if (auto castOp = dyn_cast<memref::CastOp>(definingOp)) {
            currentValue = castOp.getSource();
            continue;
          }

          // Check for other cast-like operations
          if (auto reinterpretCastOp =
                  dyn_cast<memref::ReinterpretCastOp>(definingOp)) {
            currentValue = reinterpretCastOp.getSource();
            continue;
          }

          // If we can't look through this operation, stop
          break;
        }
        // This is a block argument or similar - can't trace further.
        break;
      }

      if (originalAlloc) {
        if (isa<memref::AllocaOp>(originalAlloc)) {
          // This is an alloca - no dealloc needed, but return true (success)
          return true;
        }
        if (isa<memref::AllocOp>(originalAlloc)) {
          // This is an alloc - generate dealloc on varToFree
          memref::DeallocOp::create(builder, loc, memrefValue);
          return true;
        }
      }
    }

    return false;
  }

  bool genCopy(Type pointer, OpBuilder &builder, Location loc,
               TypedValue<PointerLikeType> destination,
               TypedValue<PointerLikeType> source, Type varType) const {
    // Generate a copy operation between two memrefs
    auto destMemref = dyn_cast_if_present<TypedValue<MemRefType>>(destination);
    auto srcMemref = dyn_cast_if_present<TypedValue<MemRefType>>(source);

    // As per memref documentation, source and destination must have same
    // element type and shape in order to be compatible. We do not want to fail
    // with an IR verification error - thus check that before generating the
    // copy operation.
    if (destMemref && srcMemref &&
        destMemref.getType().getElementType() ==
            srcMemref.getType().getElementType() &&
        destMemref.getType().getShape() == srcMemref.getType().getShape()) {
      memref::CopyOp::create(builder, loc, srcMemref, destMemref);
      return true;
    }

    return false;
  }

  mlir::Value genLoad(Type pointer, OpBuilder &builder, Location loc,
                      TypedValue<PointerLikeType> srcPtr,
                      Type valueType) const {
    // Load from a memref - only valid for scalar memrefs (rank 0).
    // This is because the address computation for memrefs is part of the load
    // (and not computed separately), but the API does not have arguments for
    // indexing.
    auto memrefValue = dyn_cast_if_present<TypedValue<MemRefType>>(srcPtr);
    if (!memrefValue)
      return {};

    auto memrefTy = memrefValue.getType();

    // Only load from scalar memrefs (rank 0)
    if (memrefTy.getRank() != 0)
      return {};

    return memref::LoadOp::create(builder, loc, memrefValue, ValueRange{});
  }

  bool genStore(Type pointer, OpBuilder &builder, Location loc,
                Value valueToStore, TypedValue<PointerLikeType> destPtr) const {
    // Store to a memref - only valid for scalar memrefs (rank 0)
    // This is because the address computation for memrefs is part of the store
    // (and not computed separately), but the API does not have arguments for
    // indexing.
    auto memrefValue = dyn_cast_if_present<TypedValue<MemRefType>>(destPtr);
    if (!memrefValue)
      return false;

    auto memrefTy = memrefValue.getType();

    // Only store to scalar memrefs (rank 0)
    if (memrefTy.getRank() != 0)
      return false;

    memref::StoreOp::create(builder, loc, valueToStore, memrefValue);
    return true;
  }

  Value genCast(Type, OpBuilder &builder, Location loc, Value value,
                Type resultType) const {
    if (value.getType() == resultType)
      return value;

    if (isa<BaseMemRefType>(value.getType()) &&
        isa<BaseMemRefType>(resultType)) {
      if (memref::CastOp::areCastCompatible(TypeRange(value.getType()),
                                            TypeRange(resultType)))
        return memref::CastOp::create(builder, loc, resultType, value);
      if (memref::MemorySpaceCastOp::areCastCompatible(
              TypeRange(value.getType()), TypeRange(resultType)))
        return memref::MemorySpaceCastOp::create(builder, loc, resultType,
                                                 value);
    }

    // If one side is not a memref, try the other type's `PointerLikeType`
    // implementation (since it may be an out-of-tree reference type that
    // we cannot generate here).
    if (auto resPtrLike = dyn_cast<PointerLikeType>(resultType))
      if (!isa<BaseMemRefType>(resPtrLike))
        if (Value v = resPtrLike.genCast(builder, loc, value, resultType))
          return v;
    if (auto valPtrLike = dyn_cast<PointerLikeType>(value.getType()))
      if (!isa<BaseMemRefType>(valPtrLike))
        if (Value v = valPtrLike.genCast(builder, loc, value, resultType))
          return v;

    return {};
  }

  bool isDeviceAccessible(Type pointer, Value var) const {
    auto memrefTy = cast<T>(pointer);
    Attribute memSpace = memrefTy.getMemorySpace();
    return isa_and_nonnull<gpu::AddressSpaceAttr>(memSpace);
  }

  MemRefType getAsMemRefType(Type pointer, ModuleOp module) const {
    (void)module;
    return dyn_cast<MemRefType>(pointer);
  }
};

struct MemrefAddressOfGlobalModel
    : public AddressOfGlobalOpInterface::ExternalModel<
          MemrefAddressOfGlobalModel, memref::GetGlobalOp> {
  SymbolRefAttr getSymbol(Operation *op) const {
    auto getGlobalOp = cast<memref::GetGlobalOp>(op);
    return getGlobalOp.getNameAttr();
  }
};

struct MemrefGlobalVariableModel
    : public GlobalVariableOpInterface::ExternalModel<MemrefGlobalVariableModel,
                                                      memref::GlobalOp> {
  bool isConstant(Operation *op) const {
    auto globalOp = cast<memref::GlobalOp>(op);
    return globalOp.getConstant();
  }

  bool hasInitializer(Operation *op) const {
    auto globalOp = cast<memref::GlobalOp>(op);
    return globalOp.getInitialValue().has_value();
  }

  Region *getInitRegion(Operation *op) const {
    // GlobalOp uses attributes for initialization, not regions
    return nullptr;
  }

  bool isDeviceAccessible(Operation *op) const {
    auto globalOp = cast<memref::GlobalOp>(op);
    Attribute memSpace = globalOp.getType().getMemorySpace();
    return isa_and_nonnull<gpu::AddressSpaceAttr>(memSpace);
  }

  bool isInDeviceMemory(Operation *op) const {
    // A memref address space models storage that is physically resident on the
    // device, so a device-accessible global is also in device memory. (There
    // is no host-shared/migratable address space to exclude here.)
    return isDeviceAccessible(op);
  }

  bool isCompilerGenerated(Operation *op) const { return false; }
};
} // namespace

void mlir::acc::registerMemRefInterfaces(DialectRegistry &registry) {
  // MemRef types belong to BuiltinDialect, so IR can use them without any
  // memref operations. Install their models as soon as OpenACC is loaded.
  registry.addExtension(+[](MLIRContext *ctx, OpenACCDialect *) {
    ctx->getOrLoadDialect<memref::MemRefDialect>();
    ctx->getOrLoadDialect<arith::ArithDialect>();
    MemRefType::attachInterface<MemRefPointerLikeModel<MemRefType>>(*ctx);
    UnrankedMemRefType::attachInterface<
        MemRefPointerLikeModel<UnrankedMemRefType>>(*ctx);
    memref::GetGlobalOp::attachInterface<MemrefAddressOfGlobalModel>(*ctx);
    memref::GlobalOp::attachInterface<MemrefGlobalVariableModel>(*ctx);
  });
}
