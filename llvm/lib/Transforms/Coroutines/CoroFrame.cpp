//===- CoroFrame.cpp - Builds and manipulates coroutine frame -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file contains classes used to discover if for a particular value
// its definition precedes and its uses follow a suspend block. This is
// referred to as a suspend crossing value.
//
// Using the information discovered we form a Coroutine Frame structure to
// contain those values. All uses of those values are replaced with appropriate
// GEP + load from the coroutine frame. At the point of the definition we spill
// the value into the coroutine frame.
//===----------------------------------------------------------------------===//

#include "CoroInternal.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/StackLifetime.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/OptimizedStructLayout.h"
#include "llvm/Transforms/Coroutines/ABI.h"
#include "llvm/Transforms/Coroutines/CoroInstr.h"
#include "llvm/Transforms/Coroutines/MaterializationUtils.h"
#include "llvm/Transforms/Coroutines/SpillUtils.h"
#include "llvm/Transforms/Coroutines/SuspendCrossingInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include <algorithm>
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "coro-frame"

namespace {
class FrameTypeBuilder;
// Mapping from the to-be-spilled value to all the users that need reload.
struct FrameDataInfo {
  // All the values (that are not allocas) that needs to be spilled to the
  // frame.
  coro::SpillInfo &Spills;
  // Allocas contains all values defined as allocas that need to live in the
  // frame.
  SmallVectorImpl<coro::AllocaInfo> &Allocas;

  FrameDataInfo(coro::SpillInfo &Spills,
                SmallVectorImpl<coro::AllocaInfo> &Allocas)
      : Spills(Spills), Allocas(Allocas) {}

  SmallVector<Value *, 8> getAllDefs() const {
    SmallVector<Value *, 8> Defs;
    for (const auto &P : Spills)
      Defs.push_back(P.first);
    for (const auto &A : Allocas)
      Defs.push_back(A.Alloca);
    return Defs;
  }

  uint32_t getFieldIndex(Value *V) const {
    auto Itr = FieldIndexMap.find(V);
    assert(Itr != FieldIndexMap.end() &&
           "Value does not have a frame field index");
    return Itr->second;
  }

  void setFieldIndex(Value *V, uint32_t Index) {
    assert((LayoutIndexUpdateStarted || FieldIndexMap.count(V) == 0) &&
           "Cannot set the index for the same field twice.");
    FieldIndexMap[V] = Index;
  }

  Align getAlign(Value *V) const {
    auto Iter = FieldAlignMap.find(V);
    assert(Iter != FieldAlignMap.end());
    return Iter->second;
  }

  void setAlign(Value *V, Align AL) {
    assert(FieldAlignMap.count(V) == 0);
    FieldAlignMap.insert({V, AL});
  }

  uint64_t getDynamicAlign(Value *V) const {
    auto Iter = FieldDynamicAlignMap.find(V);
    assert(Iter != FieldDynamicAlignMap.end());
    return Iter->second;
  }

  void setDynamicAlign(Value *V, uint64_t Align) {
    assert(FieldDynamicAlignMap.count(V) == 0);
    FieldDynamicAlignMap.insert({V, Align});
  }

  uint64_t getOffset(Value *V) const {
    auto Iter = FieldOffsetMap.find(V);
    assert(Iter != FieldOffsetMap.end());
    return Iter->second;
  }

  void setOffset(Value *V, uint64_t Offset) {
    assert(FieldOffsetMap.count(V) == 0);
    FieldOffsetMap.insert({V, Offset});
  }

  // Remap the index of every field in the frame, using the final layout index.
  void updateLayoutIndex(FrameTypeBuilder &B);

private:
  // LayoutIndexUpdateStarted is used to avoid updating the index of any field
  // twice by mistake.
  bool LayoutIndexUpdateStarted = false;
  // Map from values to their slot indexes on the frame. They will be first set
  // with their original insertion field index. After the frame is built, their
  // indexes will be updated into the final layout index.
  DenseMap<Value *, uint32_t> FieldIndexMap;
  // Map from values to their alignment on the frame. They would be set after
  // the frame is built.
  DenseMap<Value *, Align> FieldAlignMap;
  DenseMap<Value *, uint64_t> FieldDynamicAlignMap;
  // Map from values to their offset on the frame. They would be set after
  // the frame is built.
  DenseMap<Value *, uint64_t> FieldOffsetMap;
};
} // namespace

#ifndef NDEBUG
static void dumpSpills(StringRef Title, const coro::SpillInfo &Spills) {
  dbgs() << "------------- " << Title << " --------------\n";
  for (const auto &E : Spills) {
    E.first->dump();
    dbgs() << "   user: ";
    for (auto *I : E.second)
      I->dump();
  }
}

static void dumpAllocas(const SmallVectorImpl<coro::AllocaInfo> &Allocas) {
  dbgs() << "------------- Allocas --------------\n";
  for (const auto &A : Allocas) {
    A.Alloca->dump();
  }
}
#endif

namespace {
using FieldIDType = size_t;
// We cannot rely solely on natural alignment of a type when building a
// coroutine frame and if the alignment specified on the Alloca instruction
// differs from the natural alignment of the alloca type we will need to insert
// padding.
class FrameTypeBuilder {
private:
  struct Field {
    uint64_t Size;
    uint64_t Offset;
    Type *Ty;
    FieldIDType LayoutFieldIndex;
    Align Alignment;
    Align TyAlignment;
    uint64_t DynamicAlignBuffer;
  };

  const DataLayout &DL;
  LLVMContext &Context;
  uint64_t StructSize = 0;
  Align StructAlign;
  bool IsFinished = false;

  std::optional<Align> MaxFrameAlignment;

  SmallVector<Field, 8> Fields;
  DenseMap<Value*, unsigned> FieldIndexByKey;

public:
  FrameTypeBuilder(LLVMContext &Context, const DataLayout &DL,
                   std::optional<Align> MaxFrameAlignment)
      : DL(DL), Context(Context), MaxFrameAlignment(MaxFrameAlignment) {}

  /// Add a field to this structure for the storage of an `alloca`
  /// instruction.
  [[nodiscard]] FieldIDType addFieldForAlloca(AllocaInst *AI,
                                              bool IsHeader = false) {
    Type *Ty = AI->getAllocatedType();

    // Make an array type if this is a static array allocation.
    if (AI->isArrayAllocation()) {
      if (auto *CI = dyn_cast<ConstantInt>(AI->getArraySize()))
        Ty = ArrayType::get(Ty, CI->getValue().getZExtValue());
      else
        report_fatal_error("Coroutines cannot handle non static allocas yet");
    }

    return addField(Ty, AI->getAlign(), IsHeader);
  }

  /// We want to put the allocas whose lifetime-ranges are not overlapped
  /// into one slot of coroutine frame.
  /// Consider the example at:https://bugs.llvm.org/show_bug.cgi?id=45566
  ///
  ///     cppcoro::task<void> alternative_paths(bool cond) {
  ///         if (cond) {
  ///             big_structure a;
  ///             process(a);
  ///             co_await something();
  ///         } else {
  ///             big_structure b;
  ///             process2(b);
  ///             co_await something();
  ///         }
  ///     }
  ///
  /// We want to put variable a and variable b in the same slot to
  /// reduce the size of coroutine frame.
  ///
  /// This function use StackLifetime algorithm to partition the AllocaInsts in
  /// Spills to non-overlapped sets in order to put Alloca in the same
  /// non-overlapped set into the same slot in the Coroutine Frame. Then add
  /// field for the allocas in the same non-overlapped set by using the largest
  /// type as the field type.
  ///
  /// Side Effects: Because We sort the allocas, the order of allocas in the
  /// frame may be different with the order in the source code.
  void addFieldForAllocas(const Function &F, FrameDataInfo &FrameData,
                          coro::Shape &Shape, bool OptimizeFrame);

  /// Add a field to this structure.
  [[nodiscard]] FieldIDType addField(Type *Ty, MaybeAlign MaybeFieldAlignment,
                                     bool IsHeader = false,
                                     bool IsSpillOfValue = false) {
    assert(!IsFinished && "adding fields to a finished builder");
    assert(Ty && "must provide a type for a field");

    // The field size is always the alloc size of the type.
    uint64_t FieldSize = DL.getTypeAllocSize(Ty);

    // For an alloca with size=0, we don't need to add a field and they
    // can just point to any index in the frame. Use index 0.
    if (FieldSize == 0) {
      return 0;
    }

    // The field alignment might not be the type alignment, but we need
    // to remember the type alignment anyway to build the type.
    // If we are spilling values we don't need to worry about ABI alignment
    // concerns.
    Align ABIAlign = DL.getABITypeAlign(Ty);
    Align TyAlignment = ABIAlign;
    if (IsSpillOfValue && MaxFrameAlignment && *MaxFrameAlignment < ABIAlign)
      TyAlignment = *MaxFrameAlignment;
    Align FieldAlignment = MaybeFieldAlignment.value_or(TyAlignment);

    // The field alignment could be bigger than the max frame case, in that case
    // we request additional storage to be able to dynamically align the
    // pointer.
    uint64_t DynamicAlignBuffer = 0;
    if (MaxFrameAlignment && (FieldAlignment > *MaxFrameAlignment)) {
      DynamicAlignBuffer =
          offsetToAlignment(MaxFrameAlignment->value(), FieldAlignment);
      FieldAlignment = *MaxFrameAlignment;
      FieldSize = FieldSize + DynamicAlignBuffer;
    }

    // Lay out header fields immediately.
    uint64_t Offset;
    if (IsHeader) {
      Offset = alignTo(StructSize, FieldAlignment);
      StructSize = Offset + FieldSize;

      // Everything else has a flexible offset.
    } else {
      Offset = OptimizedStructLayoutField::FlexibleOffset;
    }

    Fields.push_back({FieldSize, Offset, Ty, 0, FieldAlignment, TyAlignment,
                      DynamicAlignBuffer});
    return Fields.size() - 1;
  }

  /// Finish the layout and create the struct type with the given name.
  StructType *finish(StringRef Name);

  uint64_t getStructSize() const {
    assert(IsFinished && "not yet finished!");
    return StructSize;
  }

  Align getStructAlign() const {
    assert(IsFinished && "not yet finished!");
    return StructAlign;
  }

  FieldIDType getLayoutFieldIndex(FieldIDType Id) const {
    assert(IsFinished && "not yet finished!");
    return Fields[Id].LayoutFieldIndex;
  }

  Field getLayoutField(FieldIDType Id) const {
    assert(IsFinished && "not yet finished!");
    return Fields[Id];
  }
};
} // namespace

void FrameDataInfo::updateLayoutIndex(FrameTypeBuilder &B) {
  auto Updater = [&](Value *I) {
    auto Field = B.getLayoutField(getFieldIndex(I));
    setFieldIndex(I, Field.LayoutFieldIndex);
    setAlign(I, Field.Alignment);
    uint64_t dynamicAlign =
        Field.DynamicAlignBuffer
            ? Field.DynamicAlignBuffer + Field.Alignment.value()
            : 0;
    setDynamicAlign(I, dynamicAlign);
    setOffset(I, Field.Offset);
  };
  LayoutIndexUpdateStarted = true;
  for (auto &S : Spills)
    Updater(S.first);
  for (const auto &A : Allocas)
    Updater(A.Alloca);
  LayoutIndexUpdateStarted = false;
}

void FrameTypeBuilder::addFieldForAllocas(const Function &F,
                                          FrameDataInfo &FrameData,
                                          coro::Shape &Shape,
                                          bool OptimizeFrame) {
  using AllocaSetType = SmallVector<AllocaInst *, 4>;
  SmallVector<AllocaSetType, 4> NonOverlapedAllocas;

  // We need to add field for allocas at the end of this function.
  auto AddFieldForAllocasAtExit = make_scope_exit([&]() {
    for (auto AllocaList : NonOverlapedAllocas) {
      auto *LargestAI = *AllocaList.begin();
      FieldIDType Id = addFieldForAlloca(LargestAI);
      for (auto *Alloca : AllocaList)
        FrameData.setFieldIndex(Alloca, Id);
    }
  });

  if (!OptimizeFrame) {
    for (const auto &A : FrameData.Allocas) {
      AllocaInst *Alloca = A.Alloca;
      NonOverlapedAllocas.emplace_back(AllocaSetType(1, Alloca));
    }
    return;
  }

  // Because there are paths from the lifetime.start to coro.end
  // for each alloca, the liferanges for every alloca is overlaped
  // in the blocks who contain coro.end and the successor blocks.
  // So we choose to skip there blocks when we calculate the liferange
  // for each alloca. It should be reasonable since there shouldn't be uses
  // in these blocks and the coroutine frame shouldn't be used outside the
  // coroutine body.
  //
  // Note that the user of coro.suspend may not be SwitchInst. However, this
  // case seems too complex to handle. And it is harmless to skip these
  // patterns since it just prevend putting the allocas to live in the same
  // slot.
  DenseMap<SwitchInst *, BasicBlock *> DefaultSuspendDest;
  for (auto *CoroSuspendInst : Shape.CoroSuspends) {
    for (auto *U : CoroSuspendInst->users()) {
      if (auto *ConstSWI = dyn_cast<SwitchInst>(U)) {
        auto *SWI = const_cast<SwitchInst *>(ConstSWI);
        DefaultSuspendDest[SWI] = SWI->getDefaultDest();
        SWI->setDefaultDest(SWI->getSuccessor(1));
      }
    }
  }

  auto ExtractAllocas = [&]() {
    AllocaSetType Allocas;
    Allocas.reserve(FrameData.Allocas.size());
    for (const auto &A : FrameData.Allocas)
      Allocas.push_back(A.Alloca);
    return Allocas;
  };
  StackLifetime StackLifetimeAnalyzer(F, ExtractAllocas(),
                                      StackLifetime::LivenessType::May);
  StackLifetimeAnalyzer.run();
  auto DoAllocasInterfere = [&](const AllocaInst *AI1, const AllocaInst *AI2) {
    return StackLifetimeAnalyzer.getLiveRange(AI1).overlaps(
        StackLifetimeAnalyzer.getLiveRange(AI2));
  };
  auto GetAllocaSize = [&](const coro::AllocaInfo &A) {
    std::optional<TypeSize> RetSize = A.Alloca->getAllocationSize(DL);
    assert(RetSize && "Variable Length Arrays (VLA) are not supported.\n");
    assert(!RetSize->isScalable() && "Scalable vectors are not yet supported");
    return RetSize->getFixedValue();
  };
  // Put larger allocas in the front. So the larger allocas have higher
  // priority to merge, which can save more space potentially. Also each
  // AllocaSet would be ordered. So we can get the largest Alloca in one
  // AllocaSet easily.
  sort(FrameData.Allocas, [&](const auto &Iter1, const auto &Iter2) {
    return GetAllocaSize(Iter1) > GetAllocaSize(Iter2);
  });
  for (const auto &A : FrameData.Allocas) {
    AllocaInst *Alloca = A.Alloca;
    bool Merged = false;
    // Try to find if the Alloca does not interfere with any existing
    // NonOverlappedAllocaSet. If it is true, insert the alloca to that
    // NonOverlappedAllocaSet.
    for (auto &AllocaSet : NonOverlapedAllocas) {
      assert(!AllocaSet.empty() && "Processing Alloca Set is not empty.\n");
      bool NoInterference = none_of(AllocaSet, [&](auto Iter) {
        return DoAllocasInterfere(Alloca, Iter);
      });
      // If the alignment of A is multiple of the alignment of B, the address
      // of A should satisfy the requirement for aligning for B.
      //
      // There may be other more fine-grained strategies to handle the alignment
      // infomation during the merging process. But it seems hard to handle
      // these strategies and benefit little.
      bool Alignable = [&]() -> bool {
        auto *LargestAlloca = *AllocaSet.begin();
        return LargestAlloca->getAlign().value() % Alloca->getAlign().value() ==
               0;
      }();
      bool CouldMerge = NoInterference && Alignable;
      if (!CouldMerge)
        continue;
      AllocaSet.push_back(Alloca);
      Merged = true;
      break;
    }
    if (!Merged) {
      NonOverlapedAllocas.emplace_back(AllocaSetType(1, Alloca));
    }
  }
  // Recover the default target destination for each Switch statement
  // reserved.
  for (auto SwitchAndDefaultDest : DefaultSuspendDest) {
    SwitchInst *SWI = SwitchAndDefaultDest.first;
    BasicBlock *DestBB = SwitchAndDefaultDest.second;
    SWI->setDefaultDest(DestBB);
  }
  // This Debug Info could tell us which allocas are merged into one slot.
  LLVM_DEBUG(for (auto &AllocaSet
                  : NonOverlapedAllocas) {
    if (AllocaSet.size() > 1) {
      dbgs() << "In Function:" << F.getName() << "\n";
      dbgs() << "Find Union Set "
             << "\n";
      dbgs() << "\tAllocas are \n";
      for (auto Alloca : AllocaSet)
        dbgs() << "\t\t" << *Alloca << "\n";
    }
  });
}

StructType *FrameTypeBuilder::finish(StringRef Name) {
  assert(!IsFinished && "already finished!");

  // Prepare the optimal-layout field array.
  // The Id in the layout field is a pointer to our Field for it.
  SmallVector<OptimizedStructLayoutField, 8> LayoutFields;
  LayoutFields.reserve(Fields.size());
  for (auto &Field : Fields) {
    LayoutFields.emplace_back(&Field, Field.Size, Field.Alignment,
                              Field.Offset);
  }

  // Perform layout.
  auto SizeAndAlign = performOptimizedStructLayout(LayoutFields);
  StructSize = SizeAndAlign.first;
  StructAlign = SizeAndAlign.second;

  auto getField = [](const OptimizedStructLayoutField &LayoutField) -> Field & {
    return *static_cast<Field *>(const_cast<void*>(LayoutField.Id));
  };

  // We need to produce a packed struct type if there's a field whose
  // assigned offset isn't a multiple of its natural type alignment.
  bool Packed = [&] {
    for (auto &LayoutField : LayoutFields) {
      auto &F = getField(LayoutField);
      if (!isAligned(F.TyAlignment, LayoutField.Offset))
        return true;
    }
    return false;
  }();

  // Build the struct body.
  SmallVector<Type*, 16> FieldTypes;
  FieldTypes.reserve(LayoutFields.size() * 3 / 2);
  uint64_t LastOffset = 0;
  for (auto &LayoutField : LayoutFields) {
    auto &F = getField(LayoutField);

    auto Offset = LayoutField.Offset;

    // Add a padding field if there's a padding gap and we're either
    // building a packed struct or the padding gap is more than we'd
    // get from aligning to the field type's natural alignment.
    assert(Offset >= LastOffset);
    if (Offset != LastOffset) {
      if (Packed || alignTo(LastOffset, F.TyAlignment) != Offset)
        FieldTypes.push_back(ArrayType::get(Type::getInt8Ty(Context),
                                            Offset - LastOffset));
    }

    F.Offset = Offset;
    F.LayoutFieldIndex = FieldTypes.size();

    FieldTypes.push_back(F.Ty);
    if (F.DynamicAlignBuffer) {
      FieldTypes.push_back(
          ArrayType::get(Type::getInt8Ty(Context), F.DynamicAlignBuffer));
    }
    LastOffset = Offset + F.Size;
  }

  StructType *Ty = StructType::create(Context, FieldTypes, Name, Packed);

#ifndef NDEBUG
  // Check that the IR layout matches the offsets we expect.
  auto Layout = DL.getStructLayout(Ty);
  for (auto &F : Fields) {
    assert(Ty->getElementType(F.LayoutFieldIndex) == F.Ty);
    assert(Layout->getElementOffset(F.LayoutFieldIndex) == F.Offset);
  }
#endif

  IsFinished = true;

  return Ty;
}

static void cacheDIVar(FrameDataInfo &FrameData,
                       DenseMap<Value *, DILocalVariable *> &DIVarCache) {
  for (auto *V : FrameData.getAllDefs()) {
    if (DIVarCache.contains(V))
      continue;

    auto CacheIt = [&DIVarCache, V](const auto &Container) {
      auto *I = llvm::find_if(Container, [](auto *DDI) {
        return DDI->getExpression()->getNumElements() == 0;
      });
      if (I != Container.end())
        DIVarCache.insert({V, (*I)->getVariable()});
    };
    CacheIt(findDVRDeclares(V));
  }
}

/// Create name for Type. It uses MDString to store new created string to
/// avoid memory leak.
static StringRef solveTypeName(Type *Ty) {
  if (Ty->isIntegerTy()) {
    // The longest name in common may be '__int_128', which has 9 bits.
    SmallString<16> Buffer;
    raw_svector_ostream OS(Buffer);
    OS << "__int_" << cast<IntegerType>(Ty)->getBitWidth();
    auto *MDName = MDString::get(Ty->getContext(), OS.str());
    return MDName->getString();
  }

  if (Ty->isFloatingPointTy()) {
    if (Ty->isFloatTy())
      return "__float_";
    if (Ty->isDoubleTy())
      return "__double_";
    return "__floating_type_";
  }

  if (Ty->isPointerTy())
    return "PointerType";

  if (Ty->isStructTy()) {
    if (!cast<StructType>(Ty)->hasName())
      return "__LiteralStructType_";

    auto Name = Ty->getStructName();

    SmallString<16> Buffer(Name);
    for (auto &Iter : Buffer)
      if (Iter == '.' || Iter == ':')
        Iter = '_';
    auto *MDName = MDString::get(Ty->getContext(), Buffer.str());
    return MDName->getString();
  }

  return "UnknownType";
}

static DIType *solveDIType(DIBuilder &Builder, Type *Ty,
                           const DataLayout &Layout, DIScope *Scope,
                           unsigned LineNum,
                           DenseMap<Type *, DIType *> &DITypeCache) {
  if (DIType *DT = DITypeCache.lookup(Ty))
    return DT;

  StringRef Name = solveTypeName(Ty);

  DIType *RetType = nullptr;

  if (Ty->isIntegerTy()) {
    auto BitWidth = cast<IntegerType>(Ty)->getBitWidth();
    RetType = Builder.createBasicType(Name, BitWidth, dwarf::DW_ATE_signed,
                                      llvm::DINode::FlagArtificial);
  } else if (Ty->isFloatingPointTy()) {
    RetType = Builder.createBasicType(Name, Layout.getTypeSizeInBits(Ty),
                                      dwarf::DW_ATE_float,
                                      llvm::DINode::FlagArtificial);
  } else if (Ty->isPointerTy()) {
    // Construct PointerType points to null (aka void *) instead of exploring
    // pointee type to avoid infinite search problem. For example, we would be
    // in trouble if we traverse recursively:
    //
    //  struct Node {
    //      Node* ptr;
    //  };
    RetType =
        Builder.createPointerType(nullptr, Layout.getTypeSizeInBits(Ty),
                                  Layout.getABITypeAlign(Ty).value() * CHAR_BIT,
                                  /*DWARFAddressSpace=*/std::nullopt, Name);
  } else if (Ty->isStructTy()) {
    auto *DIStruct = Builder.createStructType(
        Scope, Name, Scope->getFile(), LineNum, Layout.getTypeSizeInBits(Ty),
        Layout.getPrefTypeAlign(Ty).value() * CHAR_BIT,
        llvm::DINode::FlagArtificial, nullptr, llvm::DINodeArray());

    auto *StructTy = cast<StructType>(Ty);
    SmallVector<Metadata *, 16> Elements;
    for (unsigned I = 0; I < StructTy->getNumElements(); I++) {
      DIType *DITy = solveDIType(Builder, StructTy->getElementType(I), Layout,
                                 DIStruct, LineNum, DITypeCache);
      assert(DITy);
      Elements.push_back(Builder.createMemberType(
          DIStruct, DITy->getName(), DIStruct->getFile(), LineNum,
          DITy->getSizeInBits(), DITy->getAlignInBits(),
          Layout.getStructLayout(StructTy)->getElementOffsetInBits(I),
          llvm::DINode::FlagArtificial, DITy));
    }

    Builder.replaceArrays(DIStruct, Builder.getOrCreateArray(Elements));

    RetType = DIStruct;
  } else {
    LLVM_DEBUG(dbgs() << "Unresolved Type: " << *Ty << "\n");
    TypeSize Size = Layout.getTypeSizeInBits(Ty);
    auto *CharSizeType = Builder.createBasicType(
        Name, 8, dwarf::DW_ATE_unsigned_char, llvm::DINode::FlagArtificial);

    if (Size <= 8)
      RetType = CharSizeType;
    else {
      if (Size % 8 != 0)
        Size = TypeSize::getFixed(Size + 8 - (Size % 8));

      RetType = Builder.createArrayType(
          Size, Layout.getPrefTypeAlign(Ty).value(), CharSizeType,
          Builder.getOrCreateArray(Builder.getOrCreateSubrange(0, Size / 8)));
    }
  }

  DITypeCache.insert({Ty, RetType});
  return RetType;
}

/// Build artificial debug info for C++ coroutine frames to allow users to
/// inspect the contents of the frame directly
///
/// Create Debug information for coroutine frame with debug name "__coro_frame".
/// The debug information for the fields of coroutine frame is constructed from
/// the following way:
/// 1. For all the value in the Frame, we search the use of dbg.declare to find
///    the corresponding debug variables for the value. If we can find the
///    debug variable, we can get full and accurate debug information.
/// 2. If we can't get debug information in step 1 and 2, we could only try to
///    build the DIType by Type. We did this in solveDIType. We only handle
///    integer, float, double, integer type and struct type for now.
static void buildFrameDebugInfo(Function &F, coro::Shape &Shape,
                                FrameDataInfo &FrameData) {
  DISubprogram *DIS = F.getSubprogram();
  // If there is no DISubprogram for F, it implies the function is compiled
  // without debug info. So we also don't generate debug info for the frame.
  if (!DIS || !DIS->getUnit() ||
      !dwarf::isCPlusPlus(
          (dwarf::SourceLanguage)DIS->getUnit()->getSourceLanguage()) ||
      DIS->getUnit()->getEmissionKind() != DICompileUnit::DebugEmissionKind::FullDebug)
    return;

  assert(Shape.ABI == coro::ABI::Switch &&
         "We could only build debug infomation for C++ coroutine now.\n");

  DIBuilder DBuilder(*F.getParent(), /*AllowUnresolved*/ false);

  DIFile *DFile = DIS->getFile();
  unsigned LineNum = DIS->getLine();

  DICompositeType *FrameDITy = DBuilder.createStructType(
      DIS->getUnit(), Twine(F.getName() + ".coro_frame_ty").str(),
      DFile, LineNum, Shape.FrameSize * 8,
      Shape.FrameAlign.value() * 8, llvm::DINode::FlagArtificial, nullptr,
      llvm::DINodeArray());
  StructType *FrameTy = Shape.FrameTy;
  SmallVector<Metadata *, 16> Elements;
  DataLayout Layout = F.getDataLayout();

  DenseMap<Value *, DILocalVariable *> DIVarCache;
  cacheDIVar(FrameData, DIVarCache);

  unsigned ResumeIndex = coro::Shape::SwitchFieldIndex::Resume;
  unsigned DestroyIndex = coro::Shape::SwitchFieldIndex::Destroy;
  unsigned IndexIndex = Shape.SwitchLowering.IndexField;

  DenseMap<unsigned, StringRef> NameCache;
  NameCache.insert({ResumeIndex, "__resume_fn"});
  NameCache.insert({DestroyIndex, "__destroy_fn"});
  NameCache.insert({IndexIndex, "__coro_index"});

  Type *ResumeFnTy = FrameTy->getElementType(ResumeIndex),
       *DestroyFnTy = FrameTy->getElementType(DestroyIndex),
       *IndexTy = FrameTy->getElementType(IndexIndex);

  DenseMap<unsigned, DIType *> TyCache;
  TyCache.insert(
      {ResumeIndex, DBuilder.createPointerType(
                        nullptr, Layout.getTypeSizeInBits(ResumeFnTy))});
  TyCache.insert(
      {DestroyIndex, DBuilder.createPointerType(
                         nullptr, Layout.getTypeSizeInBits(DestroyFnTy))});

  /// FIXME: If we fill the field `SizeInBits` with the actual size of
  /// __coro_index in bits, then __coro_index wouldn't show in the debugger.
  TyCache.insert({IndexIndex, DBuilder.createBasicType(
                                  "__coro_index",
                                  (Layout.getTypeSizeInBits(IndexTy) < 8)
                                      ? 8
                                      : Layout.getTypeSizeInBits(IndexTy),
                                  dwarf::DW_ATE_unsigned_char)});

  for (auto *V : FrameData.getAllDefs()) {
    auto It = DIVarCache.find(V);
    if (It == DIVarCache.end())
      continue;

    auto Index = FrameData.getFieldIndex(V);

    NameCache.insert({Index, It->second->getName()});
    TyCache.insert({Index, It->second->getType()});
  }

  // Cache from index to (Align, Offset Pair)
  DenseMap<unsigned, std::pair<unsigned, unsigned>> OffsetCache;
  // The Align and Offset of Resume function and Destroy function are fixed.
  OffsetCache.insert({ResumeIndex, {8, 0}});
  OffsetCache.insert({DestroyIndex, {8, 8}});
  OffsetCache.insert(
      {IndexIndex,
       {Shape.SwitchLowering.IndexAlign, Shape.SwitchLowering.IndexOffset}});

  for (auto *V : FrameData.getAllDefs()) {
    auto Index = FrameData.getFieldIndex(V);

    OffsetCache.insert(
        {Index, {FrameData.getAlign(V).value(), FrameData.getOffset(V)}});
  }

  DenseMap<Type *, DIType *> DITypeCache;
  // This counter is used to avoid same type names. e.g., there would be
  // many i32 and i64 types in one coroutine. And we would use i32_0 and
  // i32_1 to avoid the same type. Since it makes no sense the name of the
  // fields confilicts with each other.
  unsigned UnknownTypeNum = 0;
  for (unsigned Index = 0; Index < FrameTy->getNumElements(); Index++) {
    auto OCIt = OffsetCache.find(Index);
    if (OCIt == OffsetCache.end())
      continue;

    std::string Name;
    uint64_t SizeInBits;
    uint32_t AlignInBits;
    uint64_t OffsetInBits;
    DIType *DITy = nullptr;

    Type *Ty = FrameTy->getElementType(Index);
    assert(Ty->isSized() && "We can't handle type which is not sized.\n");
    SizeInBits = Layout.getTypeSizeInBits(Ty).getFixedValue();
    AlignInBits = OCIt->second.first * 8;
    OffsetInBits = OCIt->second.second * 8;

    if (auto It = NameCache.find(Index); It != NameCache.end()) {
      Name = It->second.str();
      DITy = TyCache[Index];
    } else {
      DITy = solveDIType(DBuilder, Ty, Layout, FrameDITy, LineNum, DITypeCache);
      assert(DITy && "SolveDIType shouldn't return nullptr.\n");
      Name = DITy->getName().str();
      Name += "_" + std::to_string(UnknownTypeNum);
      UnknownTypeNum++;
    }

    Elements.push_back(DBuilder.createMemberType(
        FrameDITy, Name, DFile, LineNum, SizeInBits, AlignInBits, OffsetInBits,
        llvm::DINode::FlagArtificial, DITy));
  }

  DBuilder.replaceArrays(FrameDITy, DBuilder.getOrCreateArray(Elements));

  auto *FrameDIVar =
      DBuilder.createAutoVariable(DIS, "__coro_frame", DFile, LineNum,
                                  FrameDITy, true, DINode::FlagArtificial);

  // Subprogram would have ContainedNodes field which records the debug
  // variables it contained. So we need to add __coro_frame to the
  // ContainedNodes of it.
  //
  // If we don't add __coro_frame to the RetainedNodes, user may get
  // `no symbol __coro_frame in context` rather than `__coro_frame`
  // is optimized out, which is more precise.
  auto RetainedNodes = DIS->getRetainedNodes();
  SmallVector<Metadata *, 32> RetainedNodesVec(RetainedNodes.begin(),
                                               RetainedNodes.end());
  RetainedNodesVec.push_back(FrameDIVar);
  DIS->replaceOperandWith(7, (MDTuple::get(F.getContext(), RetainedNodesVec)));

  // Construct the location for the frame debug variable. The column number
  // is fake but it should be fine.
  DILocation *DILoc =
      DILocation::get(DIS->getContext(), LineNum, /*Column=*/1, DIS);
  assert(FrameDIVar->isValidLocationForIntrinsic(DILoc));

  DbgVariableRecord *NewDVR =
      new DbgVariableRecord(ValueAsMetadata::get(Shape.FramePtr), FrameDIVar,
                            DBuilder.createExpression(), DILoc,
                            DbgVariableRecord::LocationType::Declare);
  BasicBlock::iterator It = Shape.getInsertPtAfterFramePtr();
  It->getParent()->insertDbgRecordBefore(NewDVR, It);
}

// Build a struct that will keep state for an active coroutine.
//   struct f.frame {
//     ResumeFnTy ResumeFnAddr;
//     ResumeFnTy DestroyFnAddr;
//     ... promise (if present) ...
//     int ResumeIndex;
//     ... spills ...
//   };
static StructType *buildFrameType(Function &F, coro::Shape &Shape,
                                  FrameDataInfo &FrameData,
                                  bool OptimizeFrame) {
  LLVMContext &C = F.getContext();
  const DataLayout &DL = F.getDataLayout();

  // We will use this value to cap the alignment of spilled values.
  std::optional<Align> MaxFrameAlignment;
  if (Shape.ABI == coro::ABI::Async)
    MaxFrameAlignment = Shape.AsyncLowering.getContextAlignment();
  FrameTypeBuilder B(C, DL, MaxFrameAlignment);

  AllocaInst *PromiseAlloca = Shape.getPromiseAlloca();
  std::optional<FieldIDType> SwitchIndexFieldId;

  if (Shape.ABI == coro::ABI::Switch) {
    auto *FnPtrTy = PointerType::getUnqual(C);

    // Add header fields for the resume and destroy functions.
    // We can rely on these being perfectly packed.
    (void)B.addField(FnPtrTy, std::nullopt, /*header*/ true);
    (void)B.addField(FnPtrTy, std::nullopt, /*header*/ true);

    // PromiseAlloca field needs to be explicitly added here because it's
    // a header field with a fixed offset based on its alignment. Hence it
    // needs special handling and cannot be added to FrameData.Allocas.
    if (PromiseAlloca)
      FrameData.setFieldIndex(
          PromiseAlloca, B.addFieldForAlloca(PromiseAlloca, /*header*/ true));

    // Add a field to store the suspend index.  This doesn't need to
    // be in the header.
    unsigned IndexBits = std::max(1U, Log2_64_Ceil(Shape.CoroSuspends.size()));
    Type *IndexType = Type::getIntNTy(C, IndexBits);

    SwitchIndexFieldId = B.addField(IndexType, std::nullopt);
  } else {
    assert(PromiseAlloca == nullptr && "lowering doesn't support promises");
  }

  // Because multiple allocas may own the same field slot,
  // we add allocas to field here.
  B.addFieldForAllocas(F, FrameData, Shape, OptimizeFrame);
  // Add PromiseAlloca to Allocas list so that
  // 1. updateLayoutIndex could update its index after
  // `performOptimizedStructLayout`
  // 2. it is processed in insertSpills.
  if (Shape.ABI == coro::ABI::Switch && PromiseAlloca)
    // We assume that the promise alloca won't be modified before
    // CoroBegin and no alias will be create before CoroBegin.
    FrameData.Allocas.emplace_back(
        PromiseAlloca, DenseMap<Instruction *, std::optional<APInt>>{}, false);
  // Create an entry for every spilled value.
  for (auto &S : FrameData.Spills) {
    Type *FieldType = S.first->getType();
    // For byval arguments, we need to store the pointed value in the frame,
    // instead of the pointer itself.
    if (const Argument *A = dyn_cast<Argument>(S.first))
      if (A->hasByValAttr())
        FieldType = A->getParamByValType();
    FieldIDType Id = B.addField(FieldType, std::nullopt, false /*header*/,
                                true /*IsSpillOfValue*/);
    FrameData.setFieldIndex(S.first, Id);
  }

  StructType *FrameTy = [&] {
    SmallString<32> Name(F.getName());
    Name.append(".Frame");
    return B.finish(Name);
  }();

  FrameData.updateLayoutIndex(B);
  Shape.FrameAlign = B.getStructAlign();
  Shape.FrameSize = B.getStructSize();

  switch (Shape.ABI) {
  case coro::ABI::Switch: {
    // In the switch ABI, remember the switch-index field.
    auto IndexField = B.getLayoutField(*SwitchIndexFieldId);
    Shape.SwitchLowering.IndexField = IndexField.LayoutFieldIndex;
    Shape.SwitchLowering.IndexAlign = IndexField.Alignment.value();
    Shape.SwitchLowering.IndexOffset = IndexField.Offset;

    // Also round the frame size up to a multiple of its alignment, as is
    // generally expected in C/C++.
    Shape.FrameSize = alignTo(Shape.FrameSize, Shape.FrameAlign);
    break;
  }

  // In the retcon ABI, remember whether the frame is inline in the storage.
  case coro::ABI::Retcon:
  case coro::ABI::RetconOnce: {
    auto Id = Shape.getRetconCoroId();
    Shape.RetconLowering.IsFrameInlineInStorage
      = (B.getStructSize() <= Id->getStorageSize() &&
         B.getStructAlign() <= Id->getStorageAlignment());
    break;
  }
  case coro::ABI::Async: {
    Shape.AsyncLowering.FrameOffset =
        alignTo(Shape.AsyncLowering.ContextHeaderSize, Shape.FrameAlign);
    // Also make the final context size a multiple of the context alignment to
    // make allocation easier for allocators.
    Shape.AsyncLowering.ContextSize =
        alignTo(Shape.AsyncLowering.FrameOffset + Shape.FrameSize,
                Shape.AsyncLowering.getContextAlignment());
    if (Shape.AsyncLowering.getContextAlignment() < Shape.FrameAlign) {
      report_fatal_error(
          "The alignment requirment of frame variables cannot be higher than "
          "the alignment of the async function context");
    }
    break;
  }
  }

  return FrameTy;
}

// Replace all alloca and SSA values that are accessed across suspend points
// with GetElementPointer from coroutine frame + loads and stores. Create an
// AllocaSpillBB that will become the new entry block for the resume parts of
// the coroutine:
//
//    %hdl = coro.begin(...)
//    whatever
//
// becomes:
//
//    %hdl = coro.begin(...)
//    br label %AllocaSpillBB
//
//  AllocaSpillBB:
//    ; geps corresponding to allocas that were moved to coroutine frame
//    br label PostSpill
//
//  PostSpill:
//    whatever
//
//
static void insertSpills(const FrameDataInfo &FrameData, coro::Shape &Shape) {
  LLVMContext &C = Shape.CoroBegin->getContext();
  Function *F = Shape.CoroBegin->getFunction();
  IRBuilder<> Builder(C);
  StructType *FrameTy = Shape.FrameTy;
  Value *FramePtr = Shape.FramePtr;
  DominatorTree DT(*F);
  SmallDenseMap<Argument *, AllocaInst *, 4> ArgToAllocaMap;

  // Create a GEP with the given index into the coroutine frame for the original
  // value Orig. Appends an extra 0 index for array-allocas, preserving the
  // original type.
  auto GetFramePointer = [&](Value *Orig) -> Value * {
    FieldIDType Index = FrameData.getFieldIndex(Orig);
    SmallVector<Value *, 3> Indices = {
        ConstantInt::get(Type::getInt32Ty(C), 0),
        ConstantInt::get(Type::getInt32Ty(C), Index),
    };

    if (auto *AI = dyn_cast<AllocaInst>(Orig)) {
      if (auto *CI = dyn_cast<ConstantInt>(AI->getArraySize())) {
        auto Count = CI->getValue().getZExtValue();
        if (Count > 1) {
          Indices.push_back(ConstantInt::get(Type::getInt32Ty(C), 0));
        }
      } else {
        report_fatal_error("Coroutines cannot handle non static allocas yet");
      }
    }

    auto GEP = cast<GetElementPtrInst>(
        Builder.CreateInBoundsGEP(FrameTy, FramePtr, Indices));
    if (auto *AI = dyn_cast<AllocaInst>(Orig)) {
      if (FrameData.getDynamicAlign(Orig) != 0) {
        assert(FrameData.getDynamicAlign(Orig) == AI->getAlign().value());
        auto *M = AI->getModule();
        auto *IntPtrTy = M->getDataLayout().getIntPtrType(AI->getType());
        auto *PtrValue = Builder.CreatePtrToInt(GEP, IntPtrTy);
        auto *AlignMask =
            ConstantInt::get(IntPtrTy, AI->getAlign().value() - 1);
        PtrValue = Builder.CreateAdd(PtrValue, AlignMask);
        PtrValue = Builder.CreateAnd(PtrValue, Builder.CreateNot(AlignMask));
        return Builder.CreateIntToPtr(PtrValue, AI->getType());
      }
      // If the type of GEP is not equal to the type of AllocaInst, it implies
      // that the AllocaInst may be reused in the Frame slot of other
      // AllocaInst. So We cast GEP to the AllocaInst here to re-use
      // the Frame storage.
      //
      // Note: If we change the strategy dealing with alignment, we need to refine
      // this casting.
      if (GEP->getType() != Orig->getType())
        return Builder.CreateAddrSpaceCast(GEP, Orig->getType(),
                                           Orig->getName() + Twine(".cast"));
    }
    return GEP;
  };

  for (auto const &E : FrameData.Spills) {
    Value *Def = E.first;
    auto SpillAlignment = Align(FrameData.getAlign(Def));
    // Create a store instruction storing the value into the
    // coroutine frame.
    BasicBlock::iterator InsertPt = coro::getSpillInsertionPt(Shape, Def, DT);

    Type *ByValTy = nullptr;
    if (auto *Arg = dyn_cast<Argument>(Def)) {
      // If we're spilling an Argument, make sure we clear 'captures'
      // from the coroutine function.
      Arg->getParent()->removeParamAttr(Arg->getArgNo(), Attribute::Captures);

      if (Arg->hasByValAttr())
        ByValTy = Arg->getParamByValType();
    }

    auto Index = FrameData.getFieldIndex(Def);
    Builder.SetInsertPoint(InsertPt->getParent(), InsertPt);
    auto *G = Builder.CreateConstInBoundsGEP2_32(
        FrameTy, FramePtr, 0, Index, Def->getName() + Twine(".spill.addr"));
    if (ByValTy) {
      // For byval arguments, we need to store the pointed value in the frame,
      // instead of the pointer itself.
      auto *Value = Builder.CreateLoad(ByValTy, Def);
      Builder.CreateAlignedStore(Value, G, SpillAlignment);
    } else {
      Builder.CreateAlignedStore(Def, G, SpillAlignment);
    }

    BasicBlock *CurrentBlock = nullptr;
    Value *CurrentReload = nullptr;
    for (auto *U : E.second) {
      // If we have not seen the use block, create a load instruction to reload
      // the spilled value from the coroutine frame. Populates the Value pointer
      // reference provided with the frame GEP.
      if (CurrentBlock != U->getParent()) {
        CurrentBlock = U->getParent();
        Builder.SetInsertPoint(CurrentBlock,
                               CurrentBlock->getFirstInsertionPt());

        auto *GEP = GetFramePointer(E.first);
        GEP->setName(E.first->getName() + Twine(".reload.addr"));
        if (ByValTy)
          CurrentReload = GEP;
        else
          CurrentReload = Builder.CreateAlignedLoad(
              FrameTy->getElementType(FrameData.getFieldIndex(E.first)), GEP,
              SpillAlignment, E.first->getName() + Twine(".reload"));

        TinyPtrVector<DbgVariableRecord *> DVRs = findDVRDeclares(Def);
        // Try best to find dbg.declare. If the spill is a temp, there may not
        // be a direct dbg.declare. Walk up the load chain to find one from an
        // alias.
        if (F->getSubprogram()) {
          auto *CurDef = Def;
          while (DVRs.empty() && isa<LoadInst>(CurDef)) {
            auto *LdInst = cast<LoadInst>(CurDef);
            // Only consider ptr to ptr same type load.
            if (LdInst->getPointerOperandType() != LdInst->getType())
              break;
            CurDef = LdInst->getPointerOperand();
            if (!isa<AllocaInst, LoadInst>(CurDef))
              break;
            DVRs = findDVRDeclares(CurDef);
          }
        }

        auto SalvageOne = [&](DbgVariableRecord *DDI) {
          // This dbg.declare is preserved for all coro-split function
          // fragments. It will be unreachable in the main function, and
          // processed by coro::salvageDebugInfo() by the Cloner.
          DbgVariableRecord *NewDVR = new DbgVariableRecord(
              ValueAsMetadata::get(CurrentReload), DDI->getVariable(),
              DDI->getExpression(), DDI->getDebugLoc(),
              DbgVariableRecord::LocationType::Declare);
          Builder.GetInsertPoint()->getParent()->insertDbgRecordBefore(
              NewDVR, Builder.GetInsertPoint());
          // This dbg.declare is for the main function entry point.  It
          // will be deleted in all coro-split functions.
          coro::salvageDebugInfo(ArgToAllocaMap, *DDI, false /*UseEntryValue*/);
        };
        for_each(DVRs, SalvageOne);
      }

      // If we have a single edge PHINode, remove it and replace it with a
      // reload from the coroutine frame. (We already took care of multi edge
      // PHINodes by normalizing them in the rewritePHIs function).
      if (auto *PN = dyn_cast<PHINode>(U)) {
        assert(PN->getNumIncomingValues() == 1 &&
               "unexpected number of incoming "
               "values in the PHINode");
        PN->replaceAllUsesWith(CurrentReload);
        PN->eraseFromParent();
        continue;
      }

      // Replace all uses of CurrentValue in the current instruction with
      // reload.
      U->replaceUsesOfWith(Def, CurrentReload);
      // Instructions are added to Def's user list if the attached
      // debug records use Def. Update those now.
      for (DbgVariableRecord &DVR : filterDbgVars(U->getDbgRecordRange()))
        DVR.replaceVariableLocationOp(Def, CurrentReload, true);
    }
  }

  BasicBlock *FramePtrBB = Shape.getInsertPtAfterFramePtr()->getParent();

  auto SpillBlock = FramePtrBB->splitBasicBlock(
      Shape.getInsertPtAfterFramePtr(), "AllocaSpillBB");
  SpillBlock->splitBasicBlock(&SpillBlock->front(), "PostSpill");
  Shape.AllocaSpillBlock = SpillBlock;

  // retcon and retcon.once lowering assumes all uses have been sunk.
  if (Shape.ABI == coro::ABI::Retcon || Shape.ABI == coro::ABI::RetconOnce ||
      Shape.ABI == coro::ABI::Async) {
    // If we found any allocas, replace all of their remaining uses with Geps.
    Builder.SetInsertPoint(SpillBlock, SpillBlock->begin());
    for (const auto &P : FrameData.Allocas) {
      AllocaInst *Alloca = P.Alloca;
      auto *G = GetFramePointer(Alloca);

      // Remove any lifetime intrinsics, now that these are no longer allocas.
      for (User *U : make_early_inc_range(Alloca->users())) {
        auto *I = cast<Instruction>(U);
        if (I->isLifetimeStartOrEnd())
          I->eraseFromParent();
      }

      // We are not using ReplaceInstWithInst(P.first, cast<Instruction>(G))
      // here, as we are changing location of the instruction.
      G->takeName(Alloca);
      Alloca->replaceAllUsesWith(G);
      Alloca->eraseFromParent();
    }
    return;
  }

  // If we found any alloca, replace all of their remaining uses with GEP
  // instructions. To remain debugbility, we replace the uses of allocas for
  // dbg.declares and dbg.values with the reload from the frame.
  // Note: We cannot replace the alloca with GEP instructions indiscriminately,
  // as some of the uses may not be dominated by CoroBegin.
  Builder.SetInsertPoint(Shape.AllocaSpillBlock,
                         Shape.AllocaSpillBlock->begin());
  SmallVector<Instruction *, 4> UsersToUpdate;
  for (const auto &A : FrameData.Allocas) {
    AllocaInst *Alloca = A.Alloca;
    UsersToUpdate.clear();
    for (User *U : make_early_inc_range(Alloca->users())) {
      auto *I = cast<Instruction>(U);
      // It is meaningless to retain the lifetime intrinsics refer for the
      // member of coroutine frames and the meaningless lifetime intrinsics
      // are possible to block further optimizations.
      if (I->isLifetimeStartOrEnd())
        I->eraseFromParent();
      else if (DT.dominates(Shape.CoroBegin, I))
        UsersToUpdate.push_back(I);
    }

    if (UsersToUpdate.empty())
      continue;
    auto *G = GetFramePointer(Alloca);
    G->setName(Alloca->getName() + Twine(".reload.addr"));

    SmallVector<DbgVariableRecord *> DbgVariableRecords;
    findDbgUsers(Alloca, DbgVariableRecords);
    for (auto *DVR : DbgVariableRecords)
      DVR->replaceVariableLocationOp(Alloca, G);

    for (Instruction *I : UsersToUpdate)
      I->replaceUsesOfWith(Alloca, G);
  }
  Builder.SetInsertPoint(&*Shape.getInsertPtAfterFramePtr());
  for (const auto &A : FrameData.Allocas) {
    AllocaInst *Alloca = A.Alloca;
    if (A.MayWriteBeforeCoroBegin) {
      // isEscaped really means potentially modified before CoroBegin.
      if (Alloca->isArrayAllocation())
        report_fatal_error(
            "Coroutines cannot handle copying of array allocas yet");

      auto *G = GetFramePointer(Alloca);
      auto *Value = Builder.CreateLoad(Alloca->getAllocatedType(), Alloca);
      Builder.CreateStore(Value, G);
    }
    // For each alias to Alloca created before CoroBegin but used after
    // CoroBegin, we recreate them after CoroBegin by applying the offset
    // to the pointer in the frame.
    for (const auto &Alias : A.Aliases) {
      auto *FramePtr = GetFramePointer(Alloca);
      auto &Value = *Alias.second;
      auto ITy = IntegerType::get(C, Value.getBitWidth());
      auto *AliasPtr =
          Builder.CreatePtrAdd(FramePtr, ConstantInt::get(ITy, Value));
      Alias.first->replaceUsesWithIf(
          AliasPtr, [&](Use &U) { return DT.dominates(Shape.CoroBegin, U); });
    }
  }

  // PromiseAlloca is not collected in FrameData.Allocas. So we don't handle
  // the case that the PromiseAlloca may have writes before CoroBegin in the
  // above codes. And it may be problematic in edge cases. See
  // https://github.com/llvm/llvm-project/issues/57861 for an example.
  if (Shape.ABI == coro::ABI::Switch && Shape.SwitchLowering.PromiseAlloca) {
    AllocaInst *PA = Shape.SwitchLowering.PromiseAlloca;
    // If there is memory accessing to promise alloca before CoroBegin;
    bool HasAccessingPromiseBeforeCB = llvm::any_of(PA->uses(), [&](Use &U) {
      auto *Inst = dyn_cast<Instruction>(U.getUser());
      if (!Inst || DT.dominates(Shape.CoroBegin, Inst))
        return false;

      if (auto *CI = dyn_cast<CallInst>(Inst)) {
        // It is fine if the call wouldn't write to the Promise.
        // This is possible for @llvm.coro.id intrinsics, which
        // would take the promise as the second argument as a
        // marker.
        if (CI->onlyReadsMemory() ||
            CI->onlyReadsMemory(CI->getArgOperandNo(&U)))
          return false;
        return true;
      }

      return isa<StoreInst>(Inst) ||
             // It may take too much time to track the uses.
             // Be conservative about the case the use may escape.
             isa<GetElementPtrInst>(Inst) ||
             // There would always be a bitcast for the promise alloca
             // before we enabled Opaque pointers. And now given
             // opaque pointers are enabled by default. This should be
             // fine.
             isa<BitCastInst>(Inst);
    });
    if (HasAccessingPromiseBeforeCB) {
      Builder.SetInsertPoint(&*Shape.getInsertPtAfterFramePtr());
      auto *G = GetFramePointer(PA);
      auto *Value = Builder.CreateLoad(PA->getAllocatedType(), PA);
      Builder.CreateStore(Value, G);
    }
  }
}

// Moves the values in the PHIs in SuccBB that correspong to PredBB into a new
// PHI in InsertedBB.
static void movePHIValuesToInsertedBlock(BasicBlock *SuccBB,
                                         BasicBlock *InsertedBB,
                                         BasicBlock *PredBB,
                                         PHINode *UntilPHI = nullptr) {
  auto *PN = cast<PHINode>(&SuccBB->front());
  do {
    int Index = PN->getBasicBlockIndex(InsertedBB);
    Value *V = PN->getIncomingValue(Index);
    PHINode *InputV = PHINode::Create(
        V->getType(), 1, V->getName() + Twine(".") + SuccBB->getName());
    InputV->insertBefore(InsertedBB->begin());
    InputV->addIncoming(V, PredBB);
    PN->setIncomingValue(Index, InputV);
    PN = dyn_cast<PHINode>(PN->getNextNode());
  } while (PN != UntilPHI);
}

// Rewrites the PHI Nodes in a cleanuppad.
static void rewritePHIsForCleanupPad(BasicBlock *CleanupPadBB,
                                     CleanupPadInst *CleanupPad) {
  // For every incoming edge to a CleanupPad we will create a new block holding
  // all incoming values in single-value PHI nodes. We will then create another
  // block to act as a dispather (as all unwind edges for related EH blocks
  // must be the same).
  //
  // cleanuppad:
  //    %2 = phi i32[%0, %catchswitch], [%1, %catch.1]
  //    %3 = cleanuppad within none []
  //
  // It will create:
  //
  // cleanuppad.corodispatch
  //    %2 = phi i8[0, %catchswitch], [1, %catch.1]
  //    %3 = cleanuppad within none []
  //    switch i8 % 2, label %unreachable
  //            [i8 0, label %cleanuppad.from.catchswitch
  //             i8 1, label %cleanuppad.from.catch.1]
  // cleanuppad.from.catchswitch:
  //    %4 = phi i32 [%0, %catchswitch]
  //    br %label cleanuppad
  // cleanuppad.from.catch.1:
  //    %6 = phi i32 [%1, %catch.1]
  //    br %label cleanuppad
  // cleanuppad:
  //    %8 = phi i32 [%4, %cleanuppad.from.catchswitch],
  //                 [%6, %cleanuppad.from.catch.1]

  // Unreachable BB, in case switching on an invalid value in the dispatcher.
  auto *UnreachBB = BasicBlock::Create(
      CleanupPadBB->getContext(), "unreachable", CleanupPadBB->getParent());
  IRBuilder<> Builder(UnreachBB);
  Builder.CreateUnreachable();

  // Create a new cleanuppad which will be the dispatcher.
  auto *NewCleanupPadBB =
      BasicBlock::Create(CleanupPadBB->getContext(),
                         CleanupPadBB->getName() + Twine(".corodispatch"),
                         CleanupPadBB->getParent(), CleanupPadBB);
  Builder.SetInsertPoint(NewCleanupPadBB);
  auto *SwitchType = Builder.getInt8Ty();
  auto *SetDispatchValuePN =
      Builder.CreatePHI(SwitchType, pred_size(CleanupPadBB));
  CleanupPad->removeFromParent();
  CleanupPad->insertAfter(SetDispatchValuePN->getIterator());
  auto *SwitchOnDispatch = Builder.CreateSwitch(SetDispatchValuePN, UnreachBB,
                                                pred_size(CleanupPadBB));

  int SwitchIndex = 0;
  SmallVector<BasicBlock *, 8> Preds(predecessors(CleanupPadBB));
  for (BasicBlock *Pred : Preds) {
    // Create a new cleanuppad and move the PHI values to there.
    auto *CaseBB = BasicBlock::Create(CleanupPadBB->getContext(),
                                      CleanupPadBB->getName() +
                                          Twine(".from.") + Pred->getName(),
                                      CleanupPadBB->getParent(), CleanupPadBB);
    updatePhiNodes(CleanupPadBB, Pred, CaseBB);
    CaseBB->setName(CleanupPadBB->getName() + Twine(".from.") +
                    Pred->getName());
    Builder.SetInsertPoint(CaseBB);
    Builder.CreateBr(CleanupPadBB);
    movePHIValuesToInsertedBlock(CleanupPadBB, CaseBB, NewCleanupPadBB);

    // Update this Pred to the new unwind point.
    setUnwindEdgeTo(Pred->getTerminator(), NewCleanupPadBB);

    // Setup the switch in the dispatcher.
    auto *SwitchConstant = ConstantInt::get(SwitchType, SwitchIndex);
    SetDispatchValuePN->addIncoming(SwitchConstant, Pred);
    SwitchOnDispatch->addCase(SwitchConstant, CaseBB);
    SwitchIndex++;
  }
}

static void cleanupSinglePredPHIs(Function &F) {
  SmallVector<PHINode *, 32> Worklist;
  for (auto &BB : F) {
    for (auto &Phi : BB.phis()) {
      if (Phi.getNumIncomingValues() == 1) {
        Worklist.push_back(&Phi);
      } else
        break;
    }
  }
  while (!Worklist.empty()) {
    auto *Phi = Worklist.pop_back_val();
    auto *OriginalValue = Phi->getIncomingValue(0);
    Phi->replaceAllUsesWith(OriginalValue);
  }
}

static void rewritePHIs(BasicBlock &BB) {
  // For every incoming edge we will create a block holding all
  // incoming values in a single PHI nodes.
  //
  // loop:
  //    %n.val = phi i32[%n, %entry], [%inc, %loop]
  //
  // It will create:
  //
  // loop.from.entry:
  //    %n.loop.pre = phi i32 [%n, %entry]
  //    br %label loop
  // loop.from.loop:
  //    %inc.loop.pre = phi i32 [%inc, %loop]
  //    br %label loop
  //
  // After this rewrite, further analysis will ignore any phi nodes with more
  // than one incoming edge.

  // TODO: Simplify PHINodes in the basic block to remove duplicate
  // predecessors.

  // Special case for CleanupPad: all EH blocks must have the same unwind edge
  // so we need to create an additional "dispatcher" block.
  if (!BB.empty()) {
    if (auto *CleanupPad =
            dyn_cast_or_null<CleanupPadInst>(BB.getFirstNonPHIIt())) {
      SmallVector<BasicBlock *, 8> Preds(predecessors(&BB));
      for (BasicBlock *Pred : Preds) {
        if (CatchSwitchInst *CS =
                dyn_cast<CatchSwitchInst>(Pred->getTerminator())) {
          // CleanupPad with a CatchSwitch predecessor: therefore this is an
          // unwind destination that needs to be handle specially.
          assert(CS->getUnwindDest() == &BB);
          (void)CS;
          rewritePHIsForCleanupPad(&BB, CleanupPad);
          return;
        }
      }
    }
  }

  LandingPadInst *LandingPad = nullptr;
  PHINode *ReplPHI = nullptr;
  if (!BB.empty()) {
    if ((LandingPad =
             dyn_cast_or_null<LandingPadInst>(BB.getFirstNonPHIIt()))) {
      // ehAwareSplitEdge will clone the LandingPad in all the edge blocks.
      // We replace the original landing pad with a PHINode that will collect the
      // results from all of them.
      ReplPHI = PHINode::Create(LandingPad->getType(), 1, "");
      ReplPHI->insertBefore(LandingPad->getIterator());
      ReplPHI->takeName(LandingPad);
      LandingPad->replaceAllUsesWith(ReplPHI);
      // We will erase the original landing pad at the end of this function after
      // ehAwareSplitEdge cloned it in the transition blocks.
    }
  }

  SmallVector<BasicBlock *, 8> Preds(predecessors(&BB));
  for (BasicBlock *Pred : Preds) {
    auto *IncomingBB = ehAwareSplitEdge(Pred, &BB, LandingPad, ReplPHI);
    IncomingBB->setName(BB.getName() + Twine(".from.") + Pred->getName());

    // Stop the moving of values at ReplPHI, as this is either null or the PHI
    // that replaced the landing pad.
    movePHIValuesToInsertedBlock(&BB, IncomingBB, Pred, ReplPHI);
  }

  if (LandingPad) {
    // Calls to ehAwareSplitEdge function cloned the original lading pad.
    // No longer need it.
    LandingPad->eraseFromParent();
  }
}

static void rewritePHIs(Function &F) {
  SmallVector<BasicBlock *, 8> WorkList;

  for (BasicBlock &BB : F)
    if (auto *PN = dyn_cast<PHINode>(&BB.front()))
      if (PN->getNumIncomingValues() > 1)
        WorkList.push_back(&BB);

  for (BasicBlock *BB : WorkList)
    rewritePHIs(*BB);
}

// Splits the block at a particular instruction unless it is the first
// instruction in the block with a single predecessor.
static BasicBlock *splitBlockIfNotFirst(Instruction *I, const Twine &Name) {
  auto *BB = I->getParent();
  if (&BB->front() == I) {
    if (BB->getSinglePredecessor()) {
      BB->setName(Name);
      return BB;
    }
  }
  return BB->splitBasicBlock(I, Name);
}

// Split above and below a particular instruction so that it
// will be all alone by itself in a block.
static void splitAround(Instruction *I, const Twine &Name) {
  splitBlockIfNotFirst(I, Name);
  splitBlockIfNotFirst(I->getNextNode(), "After" + Name);
}

/// After we split the coroutine, will the given basic block be along
/// an obvious exit path for the resumption function?
static bool willLeaveFunctionImmediatelyAfter(BasicBlock *BB,
                                              unsigned depth = 3) {
  // If we've bottomed out our depth count, stop searching and assume
  // that the path might loop back.
  if (depth == 0) return false;

  // If this is a suspend block, we're about to exit the resumption function.
  if (coro::isSuspendBlock(BB))
    return true;

  // Recurse into the successors.
  for (auto *Succ : successors(BB)) {
    if (!willLeaveFunctionImmediatelyAfter(Succ, depth - 1))
      return false;
  }

  // If none of the successors leads back in a loop, we're on an exit/abort.
  return true;
}

static bool localAllocaNeedsStackSave(CoroAllocaAllocInst *AI) {
  // Look for a free that isn't sufficiently obviously followed by
  // either a suspend or a termination, i.e. something that will leave
  // the coro resumption frame.
  for (auto *U : AI->users()) {
    auto FI = dyn_cast<CoroAllocaFreeInst>(U);
    if (!FI) continue;

    if (!willLeaveFunctionImmediatelyAfter(FI->getParent()))
      return true;
  }

  // If we never found one, we don't need a stack save.
  return false;
}

/// Turn each of the given local allocas into a normal (dynamic) alloca
/// instruction.
static void lowerLocalAllocas(ArrayRef<CoroAllocaAllocInst*> LocalAllocas,
                              SmallVectorImpl<Instruction*> &DeadInsts) {
  for (auto *AI : LocalAllocas) {
    IRBuilder<> Builder(AI);

    // Save the stack depth.  Try to avoid doing this if the stackrestore
    // is going to immediately precede a return or something.
    Value *StackSave = nullptr;
    if (localAllocaNeedsStackSave(AI))
      StackSave = Builder.CreateStackSave();

    // Allocate memory.
    auto Alloca = Builder.CreateAlloca(Builder.getInt8Ty(), AI->getSize());
    Alloca->setAlignment(AI->getAlignment());

    for (auto *U : AI->users()) {
      // Replace gets with the allocation.
      if (isa<CoroAllocaGetInst>(U)) {
        U->replaceAllUsesWith(Alloca);

      // Replace frees with stackrestores.  This is safe because
      // alloca.alloc is required to obey a stack discipline, although we
      // don't enforce that structurally.
      } else {
        auto FI = cast<CoroAllocaFreeInst>(U);
        if (StackSave) {
          Builder.SetInsertPoint(FI);
          Builder.CreateStackRestore(StackSave);
        }
      }
      DeadInsts.push_back(cast<Instruction>(U));
    }

    DeadInsts.push_back(AI);
  }
}

/// Get the current swifterror value.
static Value *emitGetSwiftErrorValue(IRBuilder<> &Builder, Type *ValueTy,
                                     coro::Shape &Shape) {
  // Make a fake function pointer as a sort of intrinsic.
  auto FnTy = FunctionType::get(ValueTy, {}, false);
  auto Fn = ConstantPointerNull::get(Builder.getPtrTy());

  auto Call = Builder.CreateCall(FnTy, Fn, {});
  Shape.SwiftErrorOps.push_back(Call);

  return Call;
}

/// Set the given value as the current swifterror value.
///
/// Returns a slot that can be used as a swifterror slot.
static Value *emitSetSwiftErrorValue(IRBuilder<> &Builder, Value *V,
                                     coro::Shape &Shape) {
  // Make a fake function pointer as a sort of intrinsic.
  auto FnTy = FunctionType::get(Builder.getPtrTy(),
                                {V->getType()}, false);
  auto Fn = ConstantPointerNull::get(Builder.getPtrTy());

  auto Call = Builder.CreateCall(FnTy, Fn, { V });
  Shape.SwiftErrorOps.push_back(Call);

  return Call;
}

/// Set the swifterror value from the given alloca before a call,
/// then put in back in the alloca afterwards.
///
/// Returns an address that will stand in for the swifterror slot
/// until splitting.
static Value *emitSetAndGetSwiftErrorValueAround(Instruction *Call,
                                                 AllocaInst *Alloca,
                                                 coro::Shape &Shape) {
  auto ValueTy = Alloca->getAllocatedType();
  IRBuilder<> Builder(Call);

  // Load the current value from the alloca and set it as the
  // swifterror value.
  auto ValueBeforeCall = Builder.CreateLoad(ValueTy, Alloca);
  auto Addr = emitSetSwiftErrorValue(Builder, ValueBeforeCall, Shape);

  // Move to after the call.  Since swifterror only has a guaranteed
  // value on normal exits, we can ignore implicit and explicit unwind
  // edges.
  if (isa<CallInst>(Call)) {
    Builder.SetInsertPoint(Call->getNextNode());
  } else {
    auto Invoke = cast<InvokeInst>(Call);
    Builder.SetInsertPoint(Invoke->getNormalDest()->getFirstNonPHIOrDbg());
  }

  // Get the current swifterror value and store it to the alloca.
  auto ValueAfterCall = emitGetSwiftErrorValue(Builder, ValueTy, Shape);
  Builder.CreateStore(ValueAfterCall, Alloca);

  return Addr;
}

/// Eliminate a formerly-swifterror alloca by inserting the get/set
/// intrinsics and attempting to MemToReg the alloca away.
static void eliminateSwiftErrorAlloca(Function &F, AllocaInst *Alloca,
                                      coro::Shape &Shape) {
  for (Use &Use : llvm::make_early_inc_range(Alloca->uses())) {
    // swifterror values can only be used in very specific ways.
    // We take advantage of that here.
    auto User = Use.getUser();
    if (isa<LoadInst>(User) || isa<StoreInst>(User))
      continue;

    assert(isa<CallInst>(User) || isa<InvokeInst>(User));
    auto Call = cast<Instruction>(User);

    auto Addr = emitSetAndGetSwiftErrorValueAround(Call, Alloca, Shape);

    // Use the returned slot address as the call argument.
    Use.set(Addr);
  }

  // All the uses should be loads and stores now.
  assert(isAllocaPromotable(Alloca));
}

/// "Eliminate" a swifterror argument by reducing it to the alloca case
/// and then loading and storing in the prologue and epilog.
///
/// The argument keeps the swifterror flag.
static void eliminateSwiftErrorArgument(Function &F, Argument &Arg,
                                        coro::Shape &Shape,
                             SmallVectorImpl<AllocaInst*> &AllocasToPromote) {
  IRBuilder<> Builder(&F.getEntryBlock(),
                      F.getEntryBlock().getFirstNonPHIOrDbg());

  auto ArgTy = cast<PointerType>(Arg.getType());
  auto ValueTy = PointerType::getUnqual(F.getContext());

  // Reduce to the alloca case:

  // Create an alloca and replace all uses of the arg with it.
  auto Alloca = Builder.CreateAlloca(ValueTy, ArgTy->getAddressSpace());
  Arg.replaceAllUsesWith(Alloca);

  // Set an initial value in the alloca.  swifterror is always null on entry.
  auto InitialValue = Constant::getNullValue(ValueTy);
  Builder.CreateStore(InitialValue, Alloca);

  // Find all the suspends in the function and save and restore around them.
  for (auto *Suspend : Shape.CoroSuspends) {
    (void) emitSetAndGetSwiftErrorValueAround(Suspend, Alloca, Shape);
  }

  // Find all the coro.ends in the function and restore the error value.
  for (auto *End : Shape.CoroEnds) {
    Builder.SetInsertPoint(End);
    auto FinalValue = Builder.CreateLoad(ValueTy, Alloca);
    (void) emitSetSwiftErrorValue(Builder, FinalValue, Shape);
  }

  // Now we can use the alloca logic.
  AllocasToPromote.push_back(Alloca);
  eliminateSwiftErrorAlloca(F, Alloca, Shape);
}

/// Eliminate all problematic uses of swifterror arguments and allocas
/// from the function.  We'll fix them up later when splitting the function.
static void eliminateSwiftError(Function &F, coro::Shape &Shape) {
  SmallVector<AllocaInst*, 4> AllocasToPromote;

  // Look for a swifterror argument.
  for (auto &Arg : F.args()) {
    if (!Arg.hasSwiftErrorAttr()) continue;

    eliminateSwiftErrorArgument(F, Arg, Shape, AllocasToPromote);
    break;
  }

  // Look for swifterror allocas.
  for (auto &Inst : F.getEntryBlock()) {
    auto Alloca = dyn_cast<AllocaInst>(&Inst);
    if (!Alloca || !Alloca->isSwiftError()) continue;

    // Clear the swifterror flag.
    Alloca->setSwiftError(false);

    AllocasToPromote.push_back(Alloca);
    eliminateSwiftErrorAlloca(F, Alloca, Shape);
  }

  // If we have any allocas to promote, compute a dominator tree and
  // promote them en masse.
  if (!AllocasToPromote.empty()) {
    DominatorTree DT(F);
    PromoteMemToReg(AllocasToPromote, DT);
  }
}

/// For each local variable that all of its user are only used inside one of
/// suspended region, we sink their lifetime.start markers to the place where
/// after the suspend block. Doing so minimizes the lifetime of each variable,
/// hence minimizing the amount of data we end up putting on the frame.
static void sinkLifetimeStartMarkers(Function &F, coro::Shape &Shape,
                                     SuspendCrossingInfo &Checker,
                                     const DominatorTree &DT) {
  if (F.hasOptNone())
    return;

  // Collect all possible basic blocks which may dominate all uses of allocas.
  SmallPtrSet<BasicBlock *, 4> DomSet;
  DomSet.insert(&F.getEntryBlock());
  for (auto *CSI : Shape.CoroSuspends) {
    BasicBlock *SuspendBlock = CSI->getParent();
    assert(coro::isSuspendBlock(SuspendBlock) &&
           SuspendBlock->getSingleSuccessor() &&
           "should have split coro.suspend into its own block");
    DomSet.insert(SuspendBlock->getSingleSuccessor());
  }

  for (Instruction &I : instructions(F)) {
    AllocaInst* AI = dyn_cast<AllocaInst>(&I);
    if (!AI)
      continue;

    for (BasicBlock *DomBB : DomSet) {
      bool Valid = true;
      SmallVector<Instruction *, 1> Lifetimes;

      auto isLifetimeStart = [](Instruction* I) {
        if (auto* II = dyn_cast<IntrinsicInst>(I))
          return II->getIntrinsicID() == Intrinsic::lifetime_start;
        return false;
      };

      auto collectLifetimeStart = [&](Instruction *U, AllocaInst *AI) {
        if (isLifetimeStart(U)) {
          Lifetimes.push_back(U);
          return true;
        }
        if (!U->hasOneUse() || U->stripPointerCasts() != AI)
          return false;
        if (isLifetimeStart(U->user_back())) {
          Lifetimes.push_back(U->user_back());
          return true;
        }
        return false;
      };

      for (User *U : AI->users()) {
        Instruction *UI = cast<Instruction>(U);
        // For all users except lifetime.start markers, if they are all
        // dominated by one of the basic blocks and do not cross
        // suspend points as well, then there is no need to spill the
        // instruction.
        if (!DT.dominates(DomBB, UI->getParent()) ||
            Checker.isDefinitionAcrossSuspend(DomBB, UI)) {
          // Skip lifetime.start, GEP and bitcast used by lifetime.start
          // markers.
          if (collectLifetimeStart(UI, AI))
            continue;
          Valid = false;
          break;
        }
      }
      // Sink lifetime.start markers to dominate block when they are
      // only used outside the region.
      if (Valid && Lifetimes.size() != 0) {
        auto *NewLifetime = Lifetimes[0]->clone();
        NewLifetime->replaceUsesOfWith(NewLifetime->getOperand(0), AI);
        NewLifetime->insertBefore(DomBB->getTerminator()->getIterator());

        // All the outsided lifetime.start markers are no longer necessary.
        for (Instruction *S : Lifetimes)
          S->eraseFromParent();

        break;
      }
    }
  }
}

static std::optional<std::pair<Value &, DIExpression &>>
salvageDebugInfoImpl(SmallDenseMap<Argument *, AllocaInst *, 4> &ArgToAllocaMap,
                     bool UseEntryValue, Function *F, Value *Storage,
                     DIExpression *Expr, bool SkipOutermostLoad) {
  IRBuilder<> Builder(F->getContext());
  auto InsertPt = F->getEntryBlock().getFirstInsertionPt();
  while (isa<IntrinsicInst>(InsertPt))
    ++InsertPt;
  Builder.SetInsertPoint(&F->getEntryBlock(), InsertPt);

  while (auto *Inst = dyn_cast_or_null<Instruction>(Storage)) {
    if (auto *LdInst = dyn_cast<LoadInst>(Inst)) {
      Storage = LdInst->getPointerOperand();
      // FIXME: This is a heuristic that works around the fact that
      // LLVM IR debug intrinsics cannot yet distinguish between
      // memory and value locations: Because a dbg.declare(alloca) is
      // implicitly a memory location no DW_OP_deref operation for the
      // last direct load from an alloca is necessary.  This condition
      // effectively drops the *last* DW_OP_deref in the expression.
      if (!SkipOutermostLoad)
        Expr = DIExpression::prepend(Expr, DIExpression::DerefBefore);
    } else if (auto *StInst = dyn_cast<StoreInst>(Inst)) {
      Storage = StInst->getValueOperand();
    } else {
      SmallVector<uint64_t, 16> Ops;
      SmallVector<Value *, 0> AdditionalValues;
      Value *Op = llvm::salvageDebugInfoImpl(
          *Inst, Expr ? Expr->getNumLocationOperands() : 0, Ops,
          AdditionalValues);
      if (!Op || !AdditionalValues.empty()) {
        // If salvaging failed or salvaging produced more than one location
        // operand, give up.
        break;
      }
      Storage = Op;
      Expr = DIExpression::appendOpsToArg(Expr, Ops, 0, /*StackValue*/ false);
    }
    SkipOutermostLoad = false;
  }
  if (!Storage)
    return std::nullopt;

  auto *StorageAsArg = dyn_cast<Argument>(Storage);
  const bool IsSwiftAsyncArg =
      StorageAsArg && StorageAsArg->hasAttribute(Attribute::SwiftAsync);

  // Swift async arguments are described by an entry value of the ABI-defined
  // register containing the coroutine context.
  // Entry values in variadic expressions are not supported.
  if (IsSwiftAsyncArg && UseEntryValue && !Expr->isEntryValue() &&
      Expr->isSingleLocationExpression())
    Expr = DIExpression::prepend(Expr, DIExpression::EntryValue);

  // If the coroutine frame is an Argument, store it in an alloca to improve
  // its availability (e.g. registers may be clobbered).
  // Avoid this if the value is guaranteed to be available through other means
  // (e.g. swift ABI guarantees).
  if (StorageAsArg && !IsSwiftAsyncArg) {
    auto &Cached = ArgToAllocaMap[StorageAsArg];
    if (!Cached) {
      Cached = Builder.CreateAlloca(Storage->getType(), 0, nullptr,
                                    Storage->getName() + ".debug");
      Builder.CreateStore(Storage, Cached);
    }
    Storage = Cached;
    // FIXME: LLVM lacks nuanced semantics to differentiate between
    // memory and direct locations at the IR level. The backend will
    // turn a dbg.declare(alloca, ..., DIExpression()) into a memory
    // location. Thus, if there are deref and offset operations in the
    // expression, we need to add a DW_OP_deref at the *start* of the
    // expression to first load the contents of the alloca before
    // adjusting it with the expression.
    Expr = DIExpression::prepend(Expr, DIExpression::DerefBefore);
  }

  Expr = Expr->foldConstantMath();
  return {{*Storage, *Expr}};
}

void coro::salvageDebugInfo(
    SmallDenseMap<Argument *, AllocaInst *, 4> &ArgToAllocaMap,
    DbgVariableRecord &DVR, bool UseEntryValue) {

  Function *F = DVR.getFunction();
  // Follow the pointer arithmetic all the way to the incoming
  // function argument and convert into a DIExpression.
  bool SkipOutermostLoad = DVR.isDbgDeclare();
  Value *OriginalStorage = DVR.getVariableLocationOp(0);

  auto SalvagedInfo =
      ::salvageDebugInfoImpl(ArgToAllocaMap, UseEntryValue, F, OriginalStorage,
                             DVR.getExpression(), SkipOutermostLoad);
  if (!SalvagedInfo)
    return;

  Value *Storage = &SalvagedInfo->first;
  DIExpression *Expr = &SalvagedInfo->second;

  DVR.replaceVariableLocationOp(OriginalStorage, Storage);
  DVR.setExpression(Expr);
  // We only hoist dbg.declare today since it doesn't make sense to hoist
  // dbg.value since it does not have the same function wide guarantees that
  // dbg.declare does.
  if (DVR.getType() == DbgVariableRecord::LocationType::Declare) {
    std::optional<BasicBlock::iterator> InsertPt;
    if (auto *I = dyn_cast<Instruction>(Storage)) {
      InsertPt = I->getInsertionPointAfterDef();
      // Update DILocation only if variable was not inlined.
      DebugLoc ILoc = I->getDebugLoc();
      DebugLoc DVRLoc = DVR.getDebugLoc();
      if (ILoc && DVRLoc &&
          DVRLoc->getScope()->getSubprogram() ==
              ILoc->getScope()->getSubprogram())
        DVR.setDebugLoc(ILoc);
    } else if (isa<Argument>(Storage))
      InsertPt = F->getEntryBlock().begin();
    if (InsertPt) {
      DVR.removeFromParent();
      (*InsertPt)->getParent()->insertDbgRecordBefore(&DVR, *InsertPt);
    }
  }
}

void coro::normalizeCoroutine(Function &F, coro::Shape &Shape,
                              TargetTransformInfo &TTI) {
  // Don't eliminate swifterror in async functions that won't be split.
  if (Shape.ABI != coro::ABI::Async || !Shape.CoroSuspends.empty())
    eliminateSwiftError(F, Shape);

  if (Shape.ABI == coro::ABI::Switch &&
      Shape.SwitchLowering.PromiseAlloca) {
    Shape.getSwitchCoroId()->clearPromise();
  }

  // Make sure that all coro.save, coro.suspend and the fallthrough coro.end
  // intrinsics are in their own blocks to simplify the logic of building up
  // SuspendCrossing data.
  for (auto *CSI : Shape.CoroSuspends) {
    if (auto *Save = CSI->getCoroSave())
      splitAround(Save, "CoroSave");
    splitAround(CSI, "CoroSuspend");
  }

  // Put CoroEnds into their own blocks.
  for (AnyCoroEndInst *CE : Shape.CoroEnds) {
    splitAround(CE, "CoroEnd");

    // Emit the musttail call function in a new block before the CoroEnd.
    // We do this here so that the right suspend crossing info is computed for
    // the uses of the musttail call function call. (Arguments to the coro.end
    // instructions would be ignored)
    if (auto *AsyncEnd = dyn_cast<CoroAsyncEndInst>(CE)) {
      auto *MustTailCallFn = AsyncEnd->getMustTailCallFunction();
      if (!MustTailCallFn)
        continue;
      IRBuilder<> Builder(AsyncEnd);
      SmallVector<Value *, 8> Args(AsyncEnd->args());
      auto Arguments = ArrayRef<Value *>(Args).drop_front(3);
      auto *Call = coro::createMustTailCall(
          AsyncEnd->getDebugLoc(), MustTailCallFn, TTI, Arguments, Builder);
      splitAround(Call, "MustTailCall.Before.CoroEnd");
    }
  }

  // Later code makes structural assumptions about single predecessors phis e.g
  // that they are not live across a suspend point.
  cleanupSinglePredPHIs(F);

  // Transforms multi-edge PHI Nodes, so that any value feeding into a PHI will
  // never have its definition separated from the PHI by the suspend point.
  rewritePHIs(F);
}

void coro::BaseABI::buildCoroutineFrame(bool OptimizeFrame) {
  SuspendCrossingInfo Checker(F, Shape.CoroSuspends, Shape.CoroEnds);
  doRematerializations(F, Checker, IsMaterializable);

  const DominatorTree DT(F);
  if (Shape.ABI != coro::ABI::Async && Shape.ABI != coro::ABI::Retcon &&
      Shape.ABI != coro::ABI::RetconOnce)
    sinkLifetimeStartMarkers(F, Shape, Checker, DT);

  // All values (that are not allocas) that needs to be spilled to the frame.
  coro::SpillInfo Spills;
  // All values defined as allocas that need to live in the frame.
  SmallVector<coro::AllocaInfo, 8> Allocas;

  // Collect the spills for arguments and other not-materializable values.
  coro::collectSpillsFromArgs(Spills, F, Checker);
  SmallVector<Instruction *, 4> DeadInstructions;
  SmallVector<CoroAllocaAllocInst *, 4> LocalAllocas;
  coro::collectSpillsAndAllocasFromInsts(Spills, Allocas, DeadInstructions,
                                         LocalAllocas, F, Checker, DT, Shape);
  coro::collectSpillsFromDbgInfo(Spills, F, Checker);

  LLVM_DEBUG(dumpAllocas(Allocas));
  LLVM_DEBUG(dumpSpills("Spills", Spills));

  if (Shape.ABI == coro::ABI::Retcon || Shape.ABI == coro::ABI::RetconOnce ||
      Shape.ABI == coro::ABI::Async)
    sinkSpillUsesAfterCoroBegin(DT, Shape.CoroBegin, Spills, Allocas);

  // Build frame
  FrameDataInfo FrameData(Spills, Allocas);
  Shape.FrameTy = buildFrameType(F, Shape, FrameData, OptimizeFrame);
  Shape.FramePtr = Shape.CoroBegin;
  // For now, this works for C++ programs only.
  buildFrameDebugInfo(F, Shape, FrameData);
  // Insert spills and reloads
  insertSpills(FrameData, Shape);
  lowerLocalAllocas(LocalAllocas, DeadInstructions);

  for (auto *I : DeadInstructions)
    I->eraseFromParent();
}
