//===-- LowerTypeTests.cpp - type metadata lowering pass ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass lowers type metadata and calls to the llvm.type.test intrinsic.
// See http://llvm.org/docs/TypeMetadata.html for more information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/LowerTypeTests.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSummaryIndexYAML.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TrailingObjects.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;
using namespace lowertypetests;

#define DEBUG_TYPE "lowertypetests"

STATISTIC(ByteArraySizeBits, "Byte array size in bits");
STATISTIC(ByteArraySizeBytes, "Byte array size in bytes");
STATISTIC(NumByteArraysCreated, "Number of byte arrays created");
STATISTIC(NumTypeTestCallsLowered, "Number of type test calls lowered");
STATISTIC(NumTypeIdDisjointSets, "Number of disjoint sets of type identifiers");

static cl::opt<bool> AvoidReuse(
    "lowertypetests-avoid-reuse",
    cl::desc("Try to avoid reuse of byte array addresses using aliases"),
    cl::Hidden, cl::init(true));

static cl::opt<PassSummaryAction> ClSummaryAction(
    "lowertypetests-summary-action",
    cl::desc("What to do with the summary when running this pass"),
    cl::values(clEnumValN(PassSummaryAction::None, "none", "Do nothing"),
               clEnumValN(PassSummaryAction::Import, "import",
                          "Import typeid resolutions from summary and globals"),
               clEnumValN(PassSummaryAction::Export, "export",
                          "Export typeid resolutions to summary and globals")),
    cl::Hidden);

static cl::opt<std::string> ClReadSummary(
    "lowertypetests-read-summary",
    cl::desc("Read summary from given YAML file before running pass"),
    cl::Hidden);

static cl::opt<std::string> ClWriteSummary(
    "lowertypetests-write-summary",
    cl::desc("Write summary to given YAML file after running pass"),
    cl::Hidden);

bool BitSetInfo::containsGlobalOffset(uint64_t Offset) const {
  if (Offset < ByteOffset)
    return false;

  if ((Offset - ByteOffset) % (uint64_t(1) << AlignLog2) != 0)
    return false;

  uint64_t BitOffset = (Offset - ByteOffset) >> AlignLog2;
  if (BitOffset >= BitSize)
    return false;

  return Bits.count(BitOffset);
}

void BitSetInfo::print(raw_ostream &OS) const {
  OS << "offset " << ByteOffset << " size " << BitSize << " align "
     << (1 << AlignLog2);

  if (isAllOnes()) {
    OS << " all-ones\n";
    return;
  }

  OS << " { ";
  for (uint64_t B : Bits)
    OS << B << ' ';
  OS << "}\n";
}

BitSetInfo BitSetBuilder::build() {
  if (Min > Max)
    Min = 0;

  // Normalize each offset against the minimum observed offset, and compute
  // the bitwise OR of each of the offsets. The number of trailing zeros
  // in the mask gives us the log2 of the alignment of all offsets, which
  // allows us to compress the bitset by only storing one bit per aligned
  // address.
  uint64_t Mask = 0;
  for (uint64_t &Offset : Offsets) {
    Offset -= Min;
    Mask |= Offset;
  }

  BitSetInfo BSI;
  BSI.ByteOffset = Min;

  BSI.AlignLog2 = 0;
  if (Mask != 0)
    BSI.AlignLog2 = countTrailingZeros(Mask, ZB_Undefined);

  // Build the compressed bitset while normalizing the offsets against the
  // computed alignment.
  BSI.BitSize = ((Max - Min) >> BSI.AlignLog2) + 1;
  for (uint64_t Offset : Offsets) {
    Offset >>= BSI.AlignLog2;
    BSI.Bits.insert(Offset);
  }

  return BSI;
}

void GlobalLayoutBuilder::addFragment(const std::set<uint64_t> &F) {
  // Create a new fragment to hold the layout for F.
  Fragments.emplace_back();
  std::vector<uint64_t> &Fragment = Fragments.back();
  uint64_t FragmentIndex = Fragments.size() - 1;

  for (auto ObjIndex : F) {
    uint64_t OldFragmentIndex = FragmentMap[ObjIndex];
    if (OldFragmentIndex == 0) {
      // We haven't seen this object index before, so just add it to the current
      // fragment.
      Fragment.push_back(ObjIndex);
    } else {
      // This index belongs to an existing fragment. Copy the elements of the
      // old fragment into this one and clear the old fragment. We don't update
      // the fragment map just yet, this ensures that any further references to
      // indices from the old fragment in this fragment do not insert any more
      // indices.
      std::vector<uint64_t> &OldFragment = Fragments[OldFragmentIndex];
      Fragment.insert(Fragment.end(), OldFragment.begin(), OldFragment.end());
      OldFragment.clear();
    }
  }

  // Update the fragment map to point our object indices to this fragment.
  for (uint64_t ObjIndex : Fragment)
    FragmentMap[ObjIndex] = FragmentIndex;
}

void ByteArrayBuilder::allocate(const std::set<uint64_t> &Bits,
                                uint64_t BitSize, uint64_t &AllocByteOffset,
                                uint8_t &AllocMask) {
  // Find the smallest current allocation.
  unsigned Bit = 0;
  for (unsigned I = 1; I != BitsPerByte; ++I)
    if (BitAllocs[I] < BitAllocs[Bit])
      Bit = I;

  AllocByteOffset = BitAllocs[Bit];

  // Add our size to it.
  unsigned ReqSize = AllocByteOffset + BitSize;
  BitAllocs[Bit] = ReqSize;
  if (Bytes.size() < ReqSize)
    Bytes.resize(ReqSize);

  // Set our bits.
  AllocMask = 1 << Bit;
  for (uint64_t B : Bits)
    Bytes[AllocByteOffset + B] |= AllocMask;
}

namespace {

struct ByteArrayInfo {
  std::set<uint64_t> Bits;
  uint64_t BitSize;
  GlobalVariable *ByteArray;
  GlobalVariable *MaskGlobal;
};

/// A POD-like structure that we use to store a global reference together with
/// its metadata types. In this pass we frequently need to query the set of
/// metadata types referenced by a global, which at the IR level is an expensive
/// operation involving a map lookup; this data structure helps to reduce the
/// number of times we need to do this lookup.
class GlobalTypeMember final : TrailingObjects<GlobalTypeMember, MDNode *> {
  GlobalObject *GO;
  size_t NTypes;

  friend TrailingObjects;
  size_t numTrailingObjects(OverloadToken<MDNode *>) const { return NTypes; }

public:
  static GlobalTypeMember *create(BumpPtrAllocator &Alloc, GlobalObject *GO,
                                  ArrayRef<MDNode *> Types) {
    auto *GTM = static_cast<GlobalTypeMember *>(Alloc.Allocate(
        totalSizeToAlloc<MDNode *>(Types.size()), alignof(GlobalTypeMember)));
    GTM->GO = GO;
    GTM->NTypes = Types.size();
    std::uninitialized_copy(Types.begin(), Types.end(),
                            GTM->getTrailingObjects<MDNode *>());
    return GTM;
  }
  GlobalObject *getGlobal() const {
    return GO;
  }
  ArrayRef<MDNode *> types() const {
    return makeArrayRef(getTrailingObjects<MDNode *>(), NTypes);
  }
};

class LowerTypeTestsModule {
  Module &M;

  PassSummaryAction Action;
  ModuleSummaryIndex *Summary;

  bool LinkerSubsectionsViaSymbols;
  Triple::ArchType Arch;
  Triple::OSType OS;
  Triple::ObjectFormatType ObjectFormat;

  IntegerType *Int1Ty = Type::getInt1Ty(M.getContext());
  IntegerType *Int8Ty = Type::getInt8Ty(M.getContext());
  PointerType *Int8PtrTy = Type::getInt8PtrTy(M.getContext());
  IntegerType *Int32Ty = Type::getInt32Ty(M.getContext());
  PointerType *Int32PtrTy = PointerType::getUnqual(Int32Ty);
  IntegerType *Int64Ty = Type::getInt64Ty(M.getContext());
  IntegerType *IntPtrTy = M.getDataLayout().getIntPtrType(M.getContext(), 0);

  // Indirect function call index assignment counter for WebAssembly
  uint64_t IndirectIndex = 1;

  // Mapping from type identifiers to the call sites that test them, as well as
  // whether the type identifier needs to be exported to ThinLTO backends as
  // part of the regular LTO phase of the ThinLTO pipeline (see exportTypeId).
  struct TypeIdUserInfo {
    std::vector<CallInst *> CallSites;
    bool IsExported = false;
  };
  DenseMap<Metadata *, TypeIdUserInfo> TypeIdUsers;

  /// This structure describes how to lower type tests for a particular type
  /// identifier. It is either built directly from the global analysis (during
  /// regular LTO or the regular LTO phase of ThinLTO), or indirectly using type
  /// identifier summaries and external symbol references (in ThinLTO backends).
  struct TypeIdLowering {
    TypeTestResolution::Kind TheKind;

    /// All except Unsat: the start address within the combined global.
    Constant *OffsetedGlobal;

    /// ByteArray, Inline, AllOnes: log2 of the required global alignment
    /// relative to the start address.
    Constant *AlignLog2;

    /// ByteArray, Inline, AllOnes: one less than the size of the memory region
    /// covering members of this type identifier as a multiple of 2^AlignLog2.
    Constant *SizeM1;

    /// ByteArray: the byte array to test the address against.
    Constant *TheByteArray;

    /// ByteArray: the bit mask to apply to bytes loaded from the byte array.
    Constant *BitMask;

    /// Inline: the bit mask to test the address against.
    Constant *InlineBits;
  };

  std::vector<ByteArrayInfo> ByteArrayInfos;

  Function *WeakInitializerFn = nullptr;

  void exportTypeId(StringRef TypeId, const TypeIdLowering &TIL);
  TypeIdLowering importTypeId(StringRef TypeId);
  void importTypeTest(CallInst *CI);

  BitSetInfo
  buildBitSet(Metadata *TypeId,
              const DenseMap<GlobalTypeMember *, uint64_t> &GlobalLayout);
  ByteArrayInfo *createByteArray(BitSetInfo &BSI);
  void allocateByteArrays();
  Value *createBitSetTest(IRBuilder<> &B, const TypeIdLowering &TIL,
                          Value *BitOffset);
  void lowerTypeTestCalls(
      ArrayRef<Metadata *> TypeIds, Constant *CombinedGlobalAddr,
      const DenseMap<GlobalTypeMember *, uint64_t> &GlobalLayout);
  Value *lowerTypeTestCall(Metadata *TypeId, CallInst *CI,
                           const TypeIdLowering &TIL);
  void buildBitSetsFromGlobalVariables(ArrayRef<Metadata *> TypeIds,
                                       ArrayRef<GlobalTypeMember *> Globals);
  unsigned getJumpTableEntrySize();
  Type *getJumpTableEntryType();
  void createJumpTableEntry(raw_ostream &AsmOS, raw_ostream &ConstraintOS,
                            SmallVectorImpl<Value *> &AsmArgs, Function *Dest);
  void verifyTypeMDNode(GlobalObject *GO, MDNode *Type);
  void buildBitSetsFromFunctions(ArrayRef<Metadata *> TypeIds,
                                 ArrayRef<GlobalTypeMember *> Functions);
  void buildBitSetsFromFunctionsNative(ArrayRef<Metadata *> TypeIds,
                                    ArrayRef<GlobalTypeMember *> Functions);
  void buildBitSetsFromFunctionsWASM(ArrayRef<Metadata *> TypeIds,
                                     ArrayRef<GlobalTypeMember *> Functions);
  void buildBitSetsFromDisjointSet(ArrayRef<Metadata *> TypeIds,
                                   ArrayRef<GlobalTypeMember *> Globals);

  void replaceWeakDeclarationWithJumpTablePtr(Function *F, Constant *JT);
  void moveInitializerToModuleConstructor(GlobalVariable *GV);
  void findGlobalVariableUsersOf(Constant *C,
                                 SmallSetVector<GlobalVariable *, 8> &Out);

  void createJumpTable(Function *F, ArrayRef<GlobalTypeMember *> Functions);

public:
  LowerTypeTestsModule(Module &M, PassSummaryAction Action,
                       ModuleSummaryIndex *Summary);
  bool lower();

  // Lower the module using the action and summary passed as command line
  // arguments. For testing purposes only.
  static bool runForTesting(Module &M);
};

struct LowerTypeTests : public ModulePass {
  static char ID;

  bool UseCommandLine = false;

  PassSummaryAction Action;
  ModuleSummaryIndex *Summary;

  LowerTypeTests() : ModulePass(ID), UseCommandLine(true) {
    initializeLowerTypeTestsPass(*PassRegistry::getPassRegistry());
  }

  LowerTypeTests(PassSummaryAction Action, ModuleSummaryIndex *Summary)
      : ModulePass(ID), Action(Action), Summary(Summary) {
    initializeLowerTypeTestsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;
    if (UseCommandLine)
      return LowerTypeTestsModule::runForTesting(M);
    return LowerTypeTestsModule(M, Action, Summary).lower();
  }
};

} // anonymous namespace

INITIALIZE_PASS(LowerTypeTests, "lowertypetests", "Lower type metadata", false,
                false)
char LowerTypeTests::ID = 0;

ModulePass *llvm::createLowerTypeTestsPass(PassSummaryAction Action,
                                           ModuleSummaryIndex *Summary) {
  return new LowerTypeTests(Action, Summary);
}

/// Build a bit set for TypeId using the object layouts in
/// GlobalLayout.
BitSetInfo LowerTypeTestsModule::buildBitSet(
    Metadata *TypeId,
    const DenseMap<GlobalTypeMember *, uint64_t> &GlobalLayout) {
  BitSetBuilder BSB;

  // Compute the byte offset of each address associated with this type
  // identifier.
  for (auto &GlobalAndOffset : GlobalLayout) {
    for (MDNode *Type : GlobalAndOffset.first->types()) {
      if (Type->getOperand(1) != TypeId)
        continue;
      uint64_t Offset =
          cast<ConstantInt>(
              cast<ConstantAsMetadata>(Type->getOperand(0))->getValue())
              ->getZExtValue();
      BSB.addOffset(GlobalAndOffset.second + Offset);
    }
  }

  return BSB.build();
}

/// Build a test that bit BitOffset mod sizeof(Bits)*8 is set in
/// Bits. This pattern matches to the bt instruction on x86.
static Value *createMaskedBitTest(IRBuilder<> &B, Value *Bits,
                                  Value *BitOffset) {
  auto BitsType = cast<IntegerType>(Bits->getType());
  unsigned BitWidth = BitsType->getBitWidth();

  BitOffset = B.CreateZExtOrTrunc(BitOffset, BitsType);
  Value *BitIndex =
      B.CreateAnd(BitOffset, ConstantInt::get(BitsType, BitWidth - 1));
  Value *BitMask = B.CreateShl(ConstantInt::get(BitsType, 1), BitIndex);
  Value *MaskedBits = B.CreateAnd(Bits, BitMask);
  return B.CreateICmpNE(MaskedBits, ConstantInt::get(BitsType, 0));
}

ByteArrayInfo *LowerTypeTestsModule::createByteArray(BitSetInfo &BSI) {
  // Create globals to stand in for byte arrays and masks. These never actually
  // get initialized, we RAUW and erase them later in allocateByteArrays() once
  // we know the offset and mask to use.
  auto ByteArrayGlobal = new GlobalVariable(
      M, Int8Ty, /*isConstant=*/true, GlobalValue::PrivateLinkage, nullptr);
  auto MaskGlobal = new GlobalVariable(M, Int8Ty, /*isConstant=*/true,
                                       GlobalValue::PrivateLinkage, nullptr);

  ByteArrayInfos.emplace_back();
  ByteArrayInfo *BAI = &ByteArrayInfos.back();

  BAI->Bits = BSI.Bits;
  BAI->BitSize = BSI.BitSize;
  BAI->ByteArray = ByteArrayGlobal;
  BAI->MaskGlobal = MaskGlobal;
  return BAI;
}

void LowerTypeTestsModule::allocateByteArrays() {
  std::stable_sort(ByteArrayInfos.begin(), ByteArrayInfos.end(),
                   [](const ByteArrayInfo &BAI1, const ByteArrayInfo &BAI2) {
                     return BAI1.BitSize > BAI2.BitSize;
                   });

  std::vector<uint64_t> ByteArrayOffsets(ByteArrayInfos.size());

  ByteArrayBuilder BAB;
  for (unsigned I = 0; I != ByteArrayInfos.size(); ++I) {
    ByteArrayInfo *BAI = &ByteArrayInfos[I];

    uint8_t Mask;
    BAB.allocate(BAI->Bits, BAI->BitSize, ByteArrayOffsets[I], Mask);

    BAI->MaskGlobal->replaceAllUsesWith(
        ConstantExpr::getIntToPtr(ConstantInt::get(Int8Ty, Mask), Int8PtrTy));
    BAI->MaskGlobal->eraseFromParent();
  }

  Constant *ByteArrayConst = ConstantDataArray::get(M.getContext(), BAB.Bytes);
  auto ByteArray =
      new GlobalVariable(M, ByteArrayConst->getType(), /*isConstant=*/true,
                         GlobalValue::PrivateLinkage, ByteArrayConst);

  for (unsigned I = 0; I != ByteArrayInfos.size(); ++I) {
    ByteArrayInfo *BAI = &ByteArrayInfos[I];

    Constant *Idxs[] = {ConstantInt::get(IntPtrTy, 0),
                        ConstantInt::get(IntPtrTy, ByteArrayOffsets[I])};
    Constant *GEP = ConstantExpr::getInBoundsGetElementPtr(
        ByteArrayConst->getType(), ByteArray, Idxs);

    // Create an alias instead of RAUW'ing the gep directly. On x86 this ensures
    // that the pc-relative displacement is folded into the lea instead of the
    // test instruction getting another displacement.
    if (LinkerSubsectionsViaSymbols) {
      BAI->ByteArray->replaceAllUsesWith(GEP);
    } else {
      GlobalAlias *Alias = GlobalAlias::create(
          Int8Ty, 0, GlobalValue::PrivateLinkage, "bits", GEP, &M);
      BAI->ByteArray->replaceAllUsesWith(Alias);
    }
    BAI->ByteArray->eraseFromParent();
  }

  ByteArraySizeBits = BAB.BitAllocs[0] + BAB.BitAllocs[1] + BAB.BitAllocs[2] +
                      BAB.BitAllocs[3] + BAB.BitAllocs[4] + BAB.BitAllocs[5] +
                      BAB.BitAllocs[6] + BAB.BitAllocs[7];
  ByteArraySizeBytes = BAB.Bytes.size();
}

/// Build a test that bit BitOffset is set in the type identifier that was
/// lowered to TIL, which must be either an Inline or a ByteArray.
Value *LowerTypeTestsModule::createBitSetTest(IRBuilder<> &B,
                                              const TypeIdLowering &TIL,
                                              Value *BitOffset) {
  if (TIL.TheKind == TypeTestResolution::Inline) {
    // If the bit set is sufficiently small, we can avoid a load by bit testing
    // a constant.
    return createMaskedBitTest(B, TIL.InlineBits, BitOffset);
  } else {
    Constant *ByteArray = TIL.TheByteArray;
    if (!LinkerSubsectionsViaSymbols && AvoidReuse &&
        Action != PassSummaryAction::Import) {
      // Each use of the byte array uses a different alias. This makes the
      // backend less likely to reuse previously computed byte array addresses,
      // improving the security of the CFI mechanism based on this pass.
      // This won't work when importing because TheByteArray is external.
      ByteArray = GlobalAlias::create(Int8Ty, 0, GlobalValue::PrivateLinkage,
                                      "bits_use", ByteArray, &M);
    }

    Value *ByteAddr = B.CreateGEP(Int8Ty, ByteArray, BitOffset);
    Value *Byte = B.CreateLoad(ByteAddr);

    Value *ByteAndMask =
        B.CreateAnd(Byte, ConstantExpr::getPtrToInt(TIL.BitMask, Int8Ty));
    return B.CreateICmpNE(ByteAndMask, ConstantInt::get(Int8Ty, 0));
  }
}

static bool isKnownTypeIdMember(Metadata *TypeId, const DataLayout &DL,
                                Value *V, uint64_t COffset) {
  if (auto GV = dyn_cast<GlobalObject>(V)) {
    SmallVector<MDNode *, 2> Types;
    GV->getMetadata(LLVMContext::MD_type, Types);
    for (MDNode *Type : Types) {
      if (Type->getOperand(1) != TypeId)
        continue;
      uint64_t Offset =
          cast<ConstantInt>(
              cast<ConstantAsMetadata>(Type->getOperand(0))->getValue())
              ->getZExtValue();
      if (COffset == Offset)
        return true;
    }
    return false;
  }

  if (auto GEP = dyn_cast<GEPOperator>(V)) {
    APInt APOffset(DL.getPointerSizeInBits(0), 0);
    bool Result = GEP->accumulateConstantOffset(DL, APOffset);
    if (!Result)
      return false;
    COffset += APOffset.getZExtValue();
    return isKnownTypeIdMember(TypeId, DL, GEP->getPointerOperand(), COffset);
  }

  if (auto Op = dyn_cast<Operator>(V)) {
    if (Op->getOpcode() == Instruction::BitCast)
      return isKnownTypeIdMember(TypeId, DL, Op->getOperand(0), COffset);

    if (Op->getOpcode() == Instruction::Select)
      return isKnownTypeIdMember(TypeId, DL, Op->getOperand(1), COffset) &&
             isKnownTypeIdMember(TypeId, DL, Op->getOperand(2), COffset);
  }

  return false;
}

/// Lower a llvm.type.test call to its implementation. Returns the value to
/// replace the call with.
Value *LowerTypeTestsModule::lowerTypeTestCall(Metadata *TypeId, CallInst *CI,
                                               const TypeIdLowering &TIL) {
  if (TIL.TheKind == TypeTestResolution::Unsat)
    return ConstantInt::getFalse(M.getContext());

  Value *Ptr = CI->getArgOperand(0);
  const DataLayout &DL = M.getDataLayout();
  if (isKnownTypeIdMember(TypeId, DL, Ptr, 0))
    return ConstantInt::getTrue(M.getContext());

  BasicBlock *InitialBB = CI->getParent();

  IRBuilder<> B(CI);

  Value *PtrAsInt = B.CreatePtrToInt(Ptr, IntPtrTy);

  Constant *OffsetedGlobalAsInt =
      ConstantExpr::getPtrToInt(TIL.OffsetedGlobal, IntPtrTy);
  if (TIL.TheKind == TypeTestResolution::Single)
    return B.CreateICmpEQ(PtrAsInt, OffsetedGlobalAsInt);

  Value *PtrOffset = B.CreateSub(PtrAsInt, OffsetedGlobalAsInt);

  // We need to check that the offset both falls within our range and is
  // suitably aligned. We can check both properties at the same time by
  // performing a right rotate by log2(alignment) followed by an integer
  // comparison against the bitset size. The rotate will move the lower
  // order bits that need to be zero into the higher order bits of the
  // result, causing the comparison to fail if they are nonzero. The rotate
  // also conveniently gives us a bit offset to use during the load from
  // the bitset.
  Value *OffsetSHR =
      B.CreateLShr(PtrOffset, ConstantExpr::getZExt(TIL.AlignLog2, IntPtrTy));
  Value *OffsetSHL = B.CreateShl(
      PtrOffset, ConstantExpr::getZExt(
                     ConstantExpr::getSub(
                         ConstantInt::get(Int8Ty, DL.getPointerSizeInBits(0)),
                         TIL.AlignLog2),
                     IntPtrTy));
  Value *BitOffset = B.CreateOr(OffsetSHR, OffsetSHL);

  Value *OffsetInRange = B.CreateICmpULE(BitOffset, TIL.SizeM1);

  // If the bit set is all ones, testing against it is unnecessary.
  if (TIL.TheKind == TypeTestResolution::AllOnes)
    return OffsetInRange;

  TerminatorInst *Term = SplitBlockAndInsertIfThen(OffsetInRange, CI, false);
  IRBuilder<> ThenB(Term);

  // Now that we know that the offset is in range and aligned, load the
  // appropriate bit from the bitset.
  Value *Bit = createBitSetTest(ThenB, TIL, BitOffset);

  // The value we want is 0 if we came directly from the initial block
  // (having failed the range or alignment checks), or the loaded bit if
  // we came from the block in which we loaded it.
  B.SetInsertPoint(CI);
  PHINode *P = B.CreatePHI(Int1Ty, 2);
  P->addIncoming(ConstantInt::get(Int1Ty, 0), InitialBB);
  P->addIncoming(Bit, ThenB.GetInsertBlock());
  return P;
}

/// Given a disjoint set of type identifiers and globals, lay out the globals,
/// build the bit sets and lower the llvm.type.test calls.
void LowerTypeTestsModule::buildBitSetsFromGlobalVariables(
    ArrayRef<Metadata *> TypeIds, ArrayRef<GlobalTypeMember *> Globals) {
  // Build a new global with the combined contents of the referenced globals.
  // This global is a struct whose even-indexed elements contain the original
  // contents of the referenced globals and whose odd-indexed elements contain
  // any padding required to align the next element to the next power of 2.
  std::vector<Constant *> GlobalInits;
  const DataLayout &DL = M.getDataLayout();
  for (GlobalTypeMember *G : Globals) {
    GlobalVariable *GV = cast<GlobalVariable>(G->getGlobal());
    GlobalInits.push_back(GV->getInitializer());
    uint64_t InitSize = DL.getTypeAllocSize(GV->getValueType());

    // Compute the amount of padding required.
    uint64_t Padding = NextPowerOf2(InitSize - 1) - InitSize;

    // Cap at 128 was found experimentally to have a good data/instruction
    // overhead tradeoff.
    if (Padding > 128)
      Padding = alignTo(InitSize, 128) - InitSize;

    GlobalInits.push_back(
        ConstantAggregateZero::get(ArrayType::get(Int8Ty, Padding)));
  }
  if (!GlobalInits.empty())
    GlobalInits.pop_back();
  Constant *NewInit = ConstantStruct::getAnon(M.getContext(), GlobalInits);
  auto *CombinedGlobal =
      new GlobalVariable(M, NewInit->getType(), /*isConstant=*/true,
                         GlobalValue::PrivateLinkage, NewInit);

  StructType *NewTy = cast<StructType>(NewInit->getType());
  const StructLayout *CombinedGlobalLayout = DL.getStructLayout(NewTy);

  // Compute the offsets of the original globals within the new global.
  DenseMap<GlobalTypeMember *, uint64_t> GlobalLayout;
  for (unsigned I = 0; I != Globals.size(); ++I)
    // Multiply by 2 to account for padding elements.
    GlobalLayout[Globals[I]] = CombinedGlobalLayout->getElementOffset(I * 2);

  lowerTypeTestCalls(TypeIds, CombinedGlobal, GlobalLayout);

  // Build aliases pointing to offsets into the combined global for each
  // global from which we built the combined global, and replace references
  // to the original globals with references to the aliases.
  for (unsigned I = 0; I != Globals.size(); ++I) {
    GlobalVariable *GV = cast<GlobalVariable>(Globals[I]->getGlobal());

    // Multiply by 2 to account for padding elements.
    Constant *CombinedGlobalIdxs[] = {ConstantInt::get(Int32Ty, 0),
                                      ConstantInt::get(Int32Ty, I * 2)};
    Constant *CombinedGlobalElemPtr = ConstantExpr::getGetElementPtr(
        NewInit->getType(), CombinedGlobal, CombinedGlobalIdxs);
    if (LinkerSubsectionsViaSymbols) {
      GV->replaceAllUsesWith(CombinedGlobalElemPtr);
    } else {
      assert(GV->getType()->getAddressSpace() == 0);
      GlobalAlias *GAlias = GlobalAlias::create(NewTy->getElementType(I * 2), 0,
                                                GV->getLinkage(), "",
                                                CombinedGlobalElemPtr, &M);
      GAlias->setVisibility(GV->getVisibility());
      GAlias->takeName(GV);
      GV->replaceAllUsesWith(GAlias);
    }
    GV->eraseFromParent();
  }
}

/// Export the given type identifier so that ThinLTO backends may import it.
/// Type identifiers are exported by adding coarse-grained information about how
/// to test the type identifier to the summary, and creating symbols in the
/// object file (aliases and absolute symbols) containing fine-grained
/// information about the type identifier.
void LowerTypeTestsModule::exportTypeId(StringRef TypeId,
                                        const TypeIdLowering &TIL) {
  TypeTestResolution &TTRes = Summary->getTypeIdSummary(TypeId).TTRes;
  TTRes.TheKind = TIL.TheKind;

  auto ExportGlobal = [&](StringRef Name, Constant *C) {
    GlobalAlias *GA =
        GlobalAlias::create(Int8Ty, 0, GlobalValue::ExternalLinkage,
                            "__typeid_" + TypeId + "_" + Name, C, &M);
    GA->setVisibility(GlobalValue::HiddenVisibility);
  };

  if (TIL.TheKind != TypeTestResolution::Unsat)
    ExportGlobal("global_addr", TIL.OffsetedGlobal);

  if (TIL.TheKind == TypeTestResolution::ByteArray ||
      TIL.TheKind == TypeTestResolution::Inline ||
      TIL.TheKind == TypeTestResolution::AllOnes) {
    ExportGlobal("align", ConstantExpr::getIntToPtr(TIL.AlignLog2, Int8PtrTy));
    ExportGlobal("size_m1", ConstantExpr::getIntToPtr(TIL.SizeM1, Int8PtrTy));

    uint64_t BitSize = cast<ConstantInt>(TIL.SizeM1)->getZExtValue() + 1;
    if (TIL.TheKind == TypeTestResolution::Inline)
      TTRes.SizeM1BitWidth = (BitSize <= 32) ? 5 : 6;
    else
      TTRes.SizeM1BitWidth = (BitSize <= 128) ? 7 : 32;
  }

  if (TIL.TheKind == TypeTestResolution::ByteArray) {
    ExportGlobal("byte_array", TIL.TheByteArray);
    ExportGlobal("bit_mask", TIL.BitMask);
  }

  if (TIL.TheKind == TypeTestResolution::Inline)
    ExportGlobal("inline_bits",
                 ConstantExpr::getIntToPtr(TIL.InlineBits, Int8PtrTy));
}

LowerTypeTestsModule::TypeIdLowering
LowerTypeTestsModule::importTypeId(StringRef TypeId) {
  TypeTestResolution &TTRes = Summary->getTypeIdSummary(TypeId).TTRes;

  TypeIdLowering TIL;
  TIL.TheKind = TTRes.TheKind;

  auto ImportGlobal = [&](StringRef Name, unsigned AbsWidth) {
    Constant *C =
        M.getOrInsertGlobal(("__typeid_" + TypeId + "_" + Name).str(), Int8Ty);
    auto *GV = dyn_cast<GlobalVariable>(C);
    // We only need to set metadata if the global is newly created, in which
    // case it would not have hidden visibility.
    if (!GV || GV->getVisibility() == GlobalValue::HiddenVisibility)
      return C;

    GV->setVisibility(GlobalValue::HiddenVisibility);
    auto SetAbsRange = [&](uint64_t Min, uint64_t Max) {
      auto *MinC = ConstantAsMetadata::get(ConstantInt::get(IntPtrTy, Min));
      auto *MaxC = ConstantAsMetadata::get(ConstantInt::get(IntPtrTy, Max));
      GV->setMetadata(LLVMContext::MD_absolute_symbol,
                      MDNode::get(M.getContext(), {MinC, MaxC}));
    };
    if (AbsWidth == IntPtrTy->getBitWidth())
      SetAbsRange(~0ull, ~0ull); // Full set.
    else if (AbsWidth)
      SetAbsRange(0, 1ull << AbsWidth);
    return C;
  };

  if (TIL.TheKind != TypeTestResolution::Unsat)
    TIL.OffsetedGlobal = ImportGlobal("global_addr", 0);

  if (TIL.TheKind == TypeTestResolution::ByteArray ||
      TIL.TheKind == TypeTestResolution::Inline ||
      TIL.TheKind == TypeTestResolution::AllOnes) {
    TIL.AlignLog2 = ConstantExpr::getPtrToInt(ImportGlobal("align", 8), Int8Ty);
    TIL.SizeM1 = ConstantExpr::getPtrToInt(
        ImportGlobal("size_m1", TTRes.SizeM1BitWidth), IntPtrTy);
  }

  if (TIL.TheKind == TypeTestResolution::ByteArray) {
    TIL.TheByteArray = ImportGlobal("byte_array", 0);
    TIL.BitMask = ImportGlobal("bit_mask", 8);
  }

  if (TIL.TheKind == TypeTestResolution::Inline)
    TIL.InlineBits = ConstantExpr::getPtrToInt(
        ImportGlobal("inline_bits", 1 << TTRes.SizeM1BitWidth),
        TTRes.SizeM1BitWidth <= 5 ? Int32Ty : Int64Ty);

  return TIL;
}

void LowerTypeTestsModule::importTypeTest(CallInst *CI) {
  auto TypeIdMDVal = dyn_cast<MetadataAsValue>(CI->getArgOperand(1));
  if (!TypeIdMDVal)
    report_fatal_error("Second argument of llvm.type.test must be metadata");

  auto TypeIdStr = dyn_cast<MDString>(TypeIdMDVal->getMetadata());
  if (!TypeIdStr)
    report_fatal_error(
        "Second argument of llvm.type.test must be a metadata string");

  TypeIdLowering TIL = importTypeId(TypeIdStr->getString());
  Value *Lowered = lowerTypeTestCall(TypeIdStr, CI, TIL);
  CI->replaceAllUsesWith(Lowered);
  CI->eraseFromParent();
}

void LowerTypeTestsModule::lowerTypeTestCalls(
    ArrayRef<Metadata *> TypeIds, Constant *CombinedGlobalAddr,
    const DenseMap<GlobalTypeMember *, uint64_t> &GlobalLayout) {
  CombinedGlobalAddr = ConstantExpr::getBitCast(CombinedGlobalAddr, Int8PtrTy);

  // For each type identifier in this disjoint set...
  for (Metadata *TypeId : TypeIds) {
    // Build the bitset.
    BitSetInfo BSI = buildBitSet(TypeId, GlobalLayout);
    DEBUG({
      if (auto MDS = dyn_cast<MDString>(TypeId))
        dbgs() << MDS->getString() << ": ";
      else
        dbgs() << "<unnamed>: ";
      BSI.print(dbgs());
    });

    TypeIdLowering TIL;
    TIL.OffsetedGlobal = ConstantExpr::getGetElementPtr(
        Int8Ty, CombinedGlobalAddr, ConstantInt::get(IntPtrTy, BSI.ByteOffset)),
    TIL.AlignLog2 = ConstantInt::get(Int8Ty, BSI.AlignLog2);
    TIL.SizeM1 = ConstantInt::get(IntPtrTy, BSI.BitSize - 1);
    if (BSI.isAllOnes()) {
      TIL.TheKind = (BSI.BitSize == 1) ? TypeTestResolution::Single
                                       : TypeTestResolution::AllOnes;
    } else if (BSI.BitSize <= 64) {
      TIL.TheKind = TypeTestResolution::Inline;
      uint64_t InlineBits = 0;
      for (auto Bit : BSI.Bits)
        InlineBits |= uint64_t(1) << Bit;
      if (InlineBits == 0)
        TIL.TheKind = TypeTestResolution::Unsat;
      else
        TIL.InlineBits = ConstantInt::get(
            (BSI.BitSize <= 32) ? Int32Ty : Int64Ty, InlineBits);
    } else {
      TIL.TheKind = TypeTestResolution::ByteArray;
      ++NumByteArraysCreated;
      ByteArrayInfo *BAI = createByteArray(BSI);
      TIL.TheByteArray = BAI->ByteArray;
      TIL.BitMask = BAI->MaskGlobal;
    }

    TypeIdUserInfo &TIUI = TypeIdUsers[TypeId];

    if (TIUI.IsExported)
      exportTypeId(cast<MDString>(TypeId)->getString(), TIL);

    // Lower each call to llvm.type.test for this type identifier.
    for (CallInst *CI : TIUI.CallSites) {
      ++NumTypeTestCallsLowered;
      Value *Lowered = lowerTypeTestCall(TypeId, CI, TIL);
      CI->replaceAllUsesWith(Lowered);
      CI->eraseFromParent();
    }
  }
}

void LowerTypeTestsModule::verifyTypeMDNode(GlobalObject *GO, MDNode *Type) {
  if (Type->getNumOperands() != 2)
    report_fatal_error("All operands of type metadata must have 2 elements");

  if (GO->isThreadLocal())
    report_fatal_error("Bit set element may not be thread-local");
  if (isa<GlobalVariable>(GO) && GO->hasSection())
    report_fatal_error(
        "A member of a type identifier may not have an explicit section");

  // FIXME: We previously checked that global var member of a type identifier
  // must be a definition, but the IR linker may leave type metadata on
  // declarations. We should restore this check after fixing PR31759.

  auto OffsetConstMD = dyn_cast<ConstantAsMetadata>(Type->getOperand(0));
  if (!OffsetConstMD)
    report_fatal_error("Type offset must be a constant");
  auto OffsetInt = dyn_cast<ConstantInt>(OffsetConstMD->getValue());
  if (!OffsetInt)
    report_fatal_error("Type offset must be an integer constant");
}

static const unsigned kX86JumpTableEntrySize = 8;
static const unsigned kARMJumpTableEntrySize = 4;

unsigned LowerTypeTestsModule::getJumpTableEntrySize() {
  switch (Arch) {
    case Triple::x86:
    case Triple::x86_64:
      return kX86JumpTableEntrySize;
    case Triple::arm:
    case Triple::thumb:
    case Triple::aarch64:
      return kARMJumpTableEntrySize;
    default:
      report_fatal_error("Unsupported architecture for jump tables");
  }
}

// Create a jump table entry for the target. This consists of an instruction
// sequence containing a relative branch to Dest. Appends inline asm text,
// constraints and arguments to AsmOS, ConstraintOS and AsmArgs.
void LowerTypeTestsModule::createJumpTableEntry(
    raw_ostream &AsmOS, raw_ostream &ConstraintOS,
    SmallVectorImpl<Value *> &AsmArgs, Function *Dest) {
  unsigned ArgIndex = AsmArgs.size();

  if (Arch == Triple::x86 || Arch == Triple::x86_64) {
    AsmOS << "jmp ${" << ArgIndex << ":c}@plt\n";
    AsmOS << "int3\nint3\nint3\n";
  } else if (Arch == Triple::arm || Arch == Triple::aarch64) {
    AsmOS << "b $" << ArgIndex << "\n";
  } else if (Arch == Triple::thumb) {
    AsmOS << "b.w $" << ArgIndex << "\n";
  } else {
    report_fatal_error("Unsupported architecture for jump tables");
  }

  ConstraintOS << (ArgIndex > 0 ? ",s" : "s");
  AsmArgs.push_back(Dest);
}

Type *LowerTypeTestsModule::getJumpTableEntryType() {
  return ArrayType::get(Int8Ty, getJumpTableEntrySize());
}

/// Given a disjoint set of type identifiers and functions, build the bit sets
/// and lower the llvm.type.test calls, architecture dependently.
void LowerTypeTestsModule::buildBitSetsFromFunctions(
    ArrayRef<Metadata *> TypeIds, ArrayRef<GlobalTypeMember *> Functions) {
  if (Arch == Triple::x86 || Arch == Triple::x86_64 || Arch == Triple::arm ||
      Arch == Triple::thumb || Arch == Triple::aarch64)
    buildBitSetsFromFunctionsNative(TypeIds, Functions);
  else if (Arch == Triple::wasm32 || Arch == Triple::wasm64)
    buildBitSetsFromFunctionsWASM(TypeIds, Functions);
  else
    report_fatal_error("Unsupported architecture for jump tables");
}

void LowerTypeTestsModule::moveInitializerToModuleConstructor(
    GlobalVariable *GV) {
  if (WeakInitializerFn == nullptr) {
    WeakInitializerFn = Function::Create(
        FunctionType::get(Type::getVoidTy(M.getContext()),
                          /* IsVarArg */ false),
        GlobalValue::InternalLinkage, "__cfi_global_var_init", &M);
    BasicBlock *BB =
        BasicBlock::Create(M.getContext(), "entry", WeakInitializerFn);
    ReturnInst::Create(M.getContext(), BB);
    WeakInitializerFn->setSection(
        ObjectFormat == Triple::MachO
            ? "__TEXT,__StaticInit,regular,pure_instructions"
            : ".text.startup");
    // This code is equivalent to relocation application, and should run at the
    // earliest possible time (i.e. with the highest priority).
    appendToGlobalCtors(M, WeakInitializerFn, /* Priority */ 0);
  }

  IRBuilder<> IRB(WeakInitializerFn->getEntryBlock().getTerminator());
  GV->setConstant(false);
  IRB.CreateAlignedStore(GV->getInitializer(), GV, GV->getAlignment());
  GV->setInitializer(Constant::getNullValue(GV->getValueType()));
}

void LowerTypeTestsModule::findGlobalVariableUsersOf(
    Constant *C, SmallSetVector<GlobalVariable *, 8> &Out) {
  for (auto *U : C->users()){
    if (auto *GV = dyn_cast<GlobalVariable>(U))
      Out.insert(GV);
    else if (auto *C2 = dyn_cast<Constant>(U))
      findGlobalVariableUsersOf(C2, Out);
  }
}

// Replace all uses of F with (F ? JT : 0).
void LowerTypeTestsModule::replaceWeakDeclarationWithJumpTablePtr(
    Function *F, Constant *JT) {
  // The target expression can not appear in a constant initializer on most
  // (all?) targets. Switch to a runtime initializer.
  SmallSetVector<GlobalVariable *, 8> GlobalVarUsers;
  findGlobalVariableUsersOf(F, GlobalVarUsers);
  for (auto GV : GlobalVarUsers)
    moveInitializerToModuleConstructor(GV);

  // Can not RAUW F with an expression that uses F. Replace with a temporary
  // placeholder first.
  Function *PlaceholderFn =
      Function::Create(cast<FunctionType>(F->getValueType()),
                       GlobalValue::ExternalWeakLinkage, "", &M);
  F->replaceAllUsesWith(PlaceholderFn);

  Constant *Target = ConstantExpr::getSelect(
      ConstantExpr::getICmp(CmpInst::ICMP_NE, F,
                            Constant::getNullValue(F->getType())),
      JT, Constant::getNullValue(F->getType()));
  PlaceholderFn->replaceAllUsesWith(Target);
  PlaceholderFn->eraseFromParent();
}

void LowerTypeTestsModule::createJumpTable(
    Function *F, ArrayRef<GlobalTypeMember *> Functions) {
  std::string AsmStr, ConstraintStr;
  raw_string_ostream AsmOS(AsmStr), ConstraintOS(ConstraintStr);
  SmallVector<Value *, 16> AsmArgs;
  AsmArgs.reserve(Functions.size() * 2);

  for (unsigned I = 0; I != Functions.size(); ++I)
    createJumpTableEntry(AsmOS, ConstraintOS, AsmArgs,
                         cast<Function>(Functions[I]->getGlobal()));

  // Try to emit the jump table at the end of the text segment.
  // Jump table must come after __cfi_check in the cross-dso mode.
  // FIXME: this magic section name seems to do the trick.
  F->setSection(ObjectFormat == Triple::MachO
                    ? "__TEXT,__text,regular,pure_instructions"
                    : ".text.cfi");
  // Align the whole table by entry size.
  F->setAlignment(getJumpTableEntrySize());
  // Skip prologue.
  // Disabled on win32 due to https://llvm.org/bugs/show_bug.cgi?id=28641#c3.
  // Luckily, this function does not get any prologue even without the
  // attribute.
  if (OS != Triple::Win32)
    F->addFnAttr(llvm::Attribute::Naked);
  // Thumb jump table assembly needs Thumb2. The following attribute is added by
  // Clang for -march=armv7.
  if (Arch == Triple::thumb)
    F->addFnAttr("target-cpu", "cortex-a8");

  BasicBlock *BB = BasicBlock::Create(M.getContext(), "entry", F);
  IRBuilder<> IRB(BB);

  SmallVector<Type *, 16> ArgTypes;
  ArgTypes.reserve(AsmArgs.size());
  for (const auto &Arg : AsmArgs)
    ArgTypes.push_back(Arg->getType());
  InlineAsm *JumpTableAsm =
      InlineAsm::get(FunctionType::get(IRB.getVoidTy(), ArgTypes, false),
                     AsmOS.str(), ConstraintOS.str(),
                     /*hasSideEffects=*/true);

  IRB.CreateCall(JumpTableAsm, AsmArgs);
  IRB.CreateUnreachable();
}

/// Given a disjoint set of type identifiers and functions, build a jump table
/// for the functions, build the bit sets and lower the llvm.type.test calls.
void LowerTypeTestsModule::buildBitSetsFromFunctionsNative(
    ArrayRef<Metadata *> TypeIds, ArrayRef<GlobalTypeMember *> Functions) {
  // Unlike the global bitset builder, the function bitset builder cannot
  // re-arrange functions in a particular order and base its calculations on the
  // layout of the functions' entry points, as we have no idea how large a
  // particular function will end up being (the size could even depend on what
  // this pass does!) Instead, we build a jump table, which is a block of code
  // consisting of one branch instruction for each of the functions in the bit
  // set that branches to the target function, and redirect any taken function
  // addresses to the corresponding jump table entry. In the object file's
  // symbol table, the symbols for the target functions also refer to the jump
  // table entries, so that addresses taken outside the module will pass any
  // verification done inside the module.
  //
  // In more concrete terms, suppose we have three functions f, g, h which are
  // of the same type, and a function foo that returns their addresses:
  //
  // f:
  // mov 0, %eax
  // ret
  //
  // g:
  // mov 1, %eax
  // ret
  //
  // h:
  // mov 2, %eax
  // ret
  //
  // foo:
  // mov f, %eax
  // mov g, %edx
  // mov h, %ecx
  // ret
  //
  // We output the jump table as module-level inline asm string. The end result
  // will (conceptually) look like this:
  //
  // f = .cfi.jumptable
  // g = .cfi.jumptable + 4
  // h = .cfi.jumptable + 8
  // .cfi.jumptable:
  // jmp f.cfi  ; 5 bytes
  // int3       ; 1 byte
  // int3       ; 1 byte
  // int3       ; 1 byte
  // jmp g.cfi  ; 5 bytes
  // int3       ; 1 byte
  // int3       ; 1 byte
  // int3       ; 1 byte
  // jmp h.cfi  ; 5 bytes
  // int3       ; 1 byte
  // int3       ; 1 byte
  // int3       ; 1 byte
  //
  // f.cfi:
  // mov 0, %eax
  // ret
  //
  // g.cfi:
  // mov 1, %eax
  // ret
  //
  // h.cfi:
  // mov 2, %eax
  // ret
  //
  // foo:
  // mov f, %eax
  // mov g, %edx
  // mov h, %ecx
  // ret
  //
  // Because the addresses of f, g, h are evenly spaced at a power of 2, in the
  // normal case the check can be carried out using the same kind of simple
  // arithmetic that we normally use for globals.

  // FIXME: find a better way to represent the jumptable in the IR.

  assert(!Functions.empty());

  // Build a simple layout based on the regular layout of jump tables.
  DenseMap<GlobalTypeMember *, uint64_t> GlobalLayout;
  unsigned EntrySize = getJumpTableEntrySize();
  for (unsigned I = 0; I != Functions.size(); ++I)
    GlobalLayout[Functions[I]] = I * EntrySize;

  Function *JumpTableFn =
      Function::Create(FunctionType::get(Type::getVoidTy(M.getContext()),
                                         /* IsVarArg */ false),
                       GlobalValue::PrivateLinkage, ".cfi.jumptable", &M);
  ArrayType *JumpTableType =
      ArrayType::get(getJumpTableEntryType(), Functions.size());
  auto JumpTable =
      ConstantExpr::getPointerCast(JumpTableFn, JumpTableType->getPointerTo(0));

  lowerTypeTestCalls(TypeIds, JumpTable, GlobalLayout);

  // Build aliases pointing to offsets into the jump table, and replace
  // references to the original functions with references to the aliases.
  for (unsigned I = 0; I != Functions.size(); ++I) {
    Function *F = cast<Function>(Functions[I]->getGlobal());

    Constant *CombinedGlobalElemPtr = ConstantExpr::getBitCast(
        ConstantExpr::getInBoundsGetElementPtr(
            JumpTableType, JumpTable,
            ArrayRef<Constant *>{ConstantInt::get(IntPtrTy, 0),
                                 ConstantInt::get(IntPtrTy, I)}),
        F->getType());
    if (LinkerSubsectionsViaSymbols || F->isDeclarationForLinker()) {

      if (F->isWeakForLinker())
        replaceWeakDeclarationWithJumpTablePtr(F, CombinedGlobalElemPtr);
      else
        F->replaceAllUsesWith(CombinedGlobalElemPtr);
    } else {
      assert(F->getType()->getAddressSpace() == 0);

      GlobalAlias *FAlias = GlobalAlias::create(F->getValueType(), 0,
                                                F->getLinkage(), "",
                                                CombinedGlobalElemPtr, &M);
      FAlias->setVisibility(F->getVisibility());
      FAlias->takeName(F);
      if (FAlias->hasName())
        F->setName(FAlias->getName() + ".cfi");
      F->replaceAllUsesWith(FAlias);
    }
    if (!F->isDeclarationForLinker())
      F->setLinkage(GlobalValue::InternalLinkage);
  }

  createJumpTable(JumpTableFn, Functions);
}

/// Assign a dummy layout using an incrementing counter, tag each function
/// with its index represented as metadata, and lower each type test to an
/// integer range comparison. During generation of the indirect function call
/// table in the backend, it will assign the given indexes.
/// Note: Dynamic linking is not supported, as the WebAssembly ABI has not yet
/// been finalized.
void LowerTypeTestsModule::buildBitSetsFromFunctionsWASM(
    ArrayRef<Metadata *> TypeIds, ArrayRef<GlobalTypeMember *> Functions) {
  assert(!Functions.empty());

  // Build consecutive monotonic integer ranges for each call target set
  DenseMap<GlobalTypeMember *, uint64_t> GlobalLayout;

  for (GlobalTypeMember *GTM : Functions) {
    Function *F = cast<Function>(GTM->getGlobal());

    // Skip functions that are not address taken, to avoid bloating the table
    if (!F->hasAddressTaken())
      continue;

    // Store metadata with the index for each function
    MDNode *MD = MDNode::get(F->getContext(),
                             ArrayRef<Metadata *>(ConstantAsMetadata::get(
                                 ConstantInt::get(Int64Ty, IndirectIndex))));
    F->setMetadata("wasm.index", MD);

    // Assign the counter value
    GlobalLayout[GTM] = IndirectIndex++;
  }

  // The indirect function table index space starts at zero, so pass a NULL
  // pointer as the subtracted "jump table" offset.
  lowerTypeTestCalls(TypeIds, ConstantPointerNull::get(Int32PtrTy),
                     GlobalLayout);
}

void LowerTypeTestsModule::buildBitSetsFromDisjointSet(
    ArrayRef<Metadata *> TypeIds, ArrayRef<GlobalTypeMember *> Globals) {
  llvm::DenseMap<Metadata *, uint64_t> TypeIdIndices;
  for (unsigned I = 0; I != TypeIds.size(); ++I)
    TypeIdIndices[TypeIds[I]] = I;

  // For each type identifier, build a set of indices that refer to members of
  // the type identifier.
  std::vector<std::set<uint64_t>> TypeMembers(TypeIds.size());
  unsigned GlobalIndex = 0;
  for (GlobalTypeMember *GTM : Globals) {
    for (MDNode *Type : GTM->types()) {
      // Type = { offset, type identifier }
      unsigned TypeIdIndex = TypeIdIndices[Type->getOperand(1)];
      TypeMembers[TypeIdIndex].insert(GlobalIndex);
    }
    GlobalIndex++;
  }

  // Order the sets of indices by size. The GlobalLayoutBuilder works best
  // when given small index sets first.
  std::stable_sort(
      TypeMembers.begin(), TypeMembers.end(),
      [](const std::set<uint64_t> &O1, const std::set<uint64_t> &O2) {
        return O1.size() < O2.size();
      });

  // Create a GlobalLayoutBuilder and provide it with index sets as layout
  // fragments. The GlobalLayoutBuilder tries to lay out members of fragments as
  // close together as possible.
  GlobalLayoutBuilder GLB(Globals.size());
  for (auto &&MemSet : TypeMembers)
    GLB.addFragment(MemSet);

  // Build the bitsets from this disjoint set.
  if (Globals.empty() || isa<GlobalVariable>(Globals[0]->getGlobal())) {
    // Build a vector of global variables with the computed layout.
    std::vector<GlobalTypeMember *> OrderedGVs(Globals.size());
    auto OGI = OrderedGVs.begin();
    for (auto &&F : GLB.Fragments) {
      for (auto &&Offset : F) {
        auto GV = dyn_cast<GlobalVariable>(Globals[Offset]->getGlobal());
        if (!GV)
          report_fatal_error("Type identifier may not contain both global "
                             "variables and functions");
        *OGI++ = Globals[Offset];
      }
    }

    buildBitSetsFromGlobalVariables(TypeIds, OrderedGVs);
  } else {
    // Build a vector of functions with the computed layout.
    std::vector<GlobalTypeMember *> OrderedFns(Globals.size());
    auto OFI = OrderedFns.begin();
    for (auto &&F : GLB.Fragments) {
      for (auto &&Offset : F) {
        auto Fn = dyn_cast<Function>(Globals[Offset]->getGlobal());
        if (!Fn)
          report_fatal_error("Type identifier may not contain both global "
                             "variables and functions");
        *OFI++ = Globals[Offset];
      }
    }

    buildBitSetsFromFunctions(TypeIds, OrderedFns);
  }
}

/// Lower all type tests in this module.
LowerTypeTestsModule::LowerTypeTestsModule(Module &M, PassSummaryAction Action,
                                           ModuleSummaryIndex *Summary)
    : M(M), Action(Action), Summary(Summary) {
  Triple TargetTriple(M.getTargetTriple());
  LinkerSubsectionsViaSymbols = TargetTriple.isMacOSX();
  Arch = TargetTriple.getArch();
  OS = TargetTriple.getOS();
  ObjectFormat = TargetTriple.getObjectFormat();
}

bool LowerTypeTestsModule::runForTesting(Module &M) {
  ModuleSummaryIndex Summary;

  // Handle the command-line summary arguments. This code is for testing
  // purposes only, so we handle errors directly.
  if (!ClReadSummary.empty()) {
    ExitOnError ExitOnErr("-lowertypetests-read-summary: " + ClReadSummary +
                          ": ");
    auto ReadSummaryFile =
        ExitOnErr(errorOrToExpected(MemoryBuffer::getFile(ClReadSummary)));

    yaml::Input In(ReadSummaryFile->getBuffer());
    In >> Summary;
    ExitOnErr(errorCodeToError(In.error()));
  }

  bool Changed = LowerTypeTestsModule(M, ClSummaryAction, &Summary).lower();

  if (!ClWriteSummary.empty()) {
    ExitOnError ExitOnErr("-lowertypetests-write-summary: " + ClWriteSummary +
                          ": ");
    std::error_code EC;
    raw_fd_ostream OS(ClWriteSummary, EC, sys::fs::F_Text);
    ExitOnErr(errorCodeToError(EC));

    yaml::Output Out(OS);
    Out << Summary;
  }

  return Changed;
}

bool LowerTypeTestsModule::lower() {
  Function *TypeTestFunc =
      M.getFunction(Intrinsic::getName(Intrinsic::type_test));
  if ((!TypeTestFunc || TypeTestFunc->use_empty()) &&
      Action != PassSummaryAction::Export)
    return false;

  if (Action == PassSummaryAction::Import) {
    for (auto UI = TypeTestFunc->use_begin(), UE = TypeTestFunc->use_end();
         UI != UE;) {
      auto *CI = cast<CallInst>((*UI++).getUser());
      importTypeTest(CI);
    }
    return true;
  }

  // Equivalence class set containing type identifiers and the globals that
  // reference them. This is used to partition the set of type identifiers in
  // the module into disjoint sets.
  typedef EquivalenceClasses<PointerUnion<GlobalTypeMember *, Metadata *>>
      GlobalClassesTy;
  GlobalClassesTy GlobalClasses;

  // Verify the type metadata and build a few data structures to let us
  // efficiently enumerate the type identifiers associated with a global:
  // a list of GlobalTypeMembers (a GlobalObject stored alongside a vector
  // of associated type metadata) and a mapping from type identifiers to their
  // list of GlobalTypeMembers and last observed index in the list of globals.
  // The indices will be used later to deterministically order the list of type
  // identifiers.
  BumpPtrAllocator Alloc;
  struct TIInfo {
    unsigned Index;
    std::vector<GlobalTypeMember *> RefGlobals;
  };
  llvm::DenseMap<Metadata *, TIInfo> TypeIdInfo;
  unsigned I = 0;
  SmallVector<MDNode *, 2> Types;
  for (GlobalObject &GO : M.global_objects()) {
    if (isa<GlobalVariable>(GO) && GO.isDeclarationForLinker())
      continue;

    Types.clear();
    GO.getMetadata(LLVMContext::MD_type, Types);
    if (Types.empty())
      continue;

    auto *GTM = GlobalTypeMember::create(Alloc, &GO, Types);
    for (MDNode *Type : Types) {
      verifyTypeMDNode(&GO, Type);
      auto &Info = TypeIdInfo[cast<MDNode>(Type)->getOperand(1)];
      Info.Index = ++I;
      Info.RefGlobals.push_back(GTM);
    }
  }

  auto AddTypeIdUse = [&](Metadata *TypeId) -> TypeIdUserInfo & {
    // Add the call site to the list of call sites for this type identifier. We
    // also use TypeIdUsers to keep track of whether we have seen this type
    // identifier before. If we have, we don't need to re-add the referenced
    // globals to the equivalence class.
    auto Ins = TypeIdUsers.insert({TypeId, {}});
    if (Ins.second) {
      // Add the type identifier to the equivalence class.
      GlobalClassesTy::iterator GCI = GlobalClasses.insert(TypeId);
      GlobalClassesTy::member_iterator CurSet = GlobalClasses.findLeader(GCI);

      // Add the referenced globals to the type identifier's equivalence class.
      for (GlobalTypeMember *GTM : TypeIdInfo[TypeId].RefGlobals)
        CurSet = GlobalClasses.unionSets(
            CurSet, GlobalClasses.findLeader(GlobalClasses.insert(GTM)));
    }

    return Ins.first->second;
  };

  if (TypeTestFunc) {
    for (const Use &U : TypeTestFunc->uses()) {
      auto CI = cast<CallInst>(U.getUser());

      auto TypeIdMDVal = dyn_cast<MetadataAsValue>(CI->getArgOperand(1));
      if (!TypeIdMDVal)
        report_fatal_error("Second argument of llvm.type.test must be metadata");
      auto TypeId = TypeIdMDVal->getMetadata();
      AddTypeIdUse(TypeId).CallSites.push_back(CI);
    }
  }

  if (Action == PassSummaryAction::Export) {
    DenseMap<GlobalValue::GUID, TinyPtrVector<Metadata *>> MetadataByGUID;
    for (auto &P : TypeIdInfo) {
      if (auto *TypeId = dyn_cast<MDString>(P.first))
        MetadataByGUID[GlobalValue::getGUID(TypeId->getString())].push_back(
            TypeId);
    }

    for (auto &P : *Summary) {
      for (auto &S : P.second) {
        auto *FS = dyn_cast<FunctionSummary>(S.get());
        if (!FS)
          continue;
        // FIXME: Only add live functions.
        for (GlobalValue::GUID G : FS->type_tests())
          for (Metadata *MD : MetadataByGUID[G])
            AddTypeIdUse(MD).IsExported = true;
      }
    }
  }

  if (GlobalClasses.empty())
    return false;

  // Build a list of disjoint sets ordered by their maximum global index for
  // determinism.
  std::vector<std::pair<GlobalClassesTy::iterator, unsigned>> Sets;
  for (GlobalClassesTy::iterator I = GlobalClasses.begin(),
                                 E = GlobalClasses.end();
       I != E; ++I) {
    if (!I->isLeader())
      continue;
    ++NumTypeIdDisjointSets;

    unsigned MaxIndex = 0;
    for (GlobalClassesTy::member_iterator MI = GlobalClasses.member_begin(I);
         MI != GlobalClasses.member_end(); ++MI) {
      if ((*MI).is<Metadata *>())
        MaxIndex = std::max(MaxIndex, TypeIdInfo[MI->get<Metadata *>()].Index);
    }
    Sets.emplace_back(I, MaxIndex);
  }
  std::sort(Sets.begin(), Sets.end(),
            [](const std::pair<GlobalClassesTy::iterator, unsigned> &S1,
               const std::pair<GlobalClassesTy::iterator, unsigned> &S2) {
              return S1.second < S2.second;
            });

  // For each disjoint set we found...
  for (const auto &S : Sets) {
    // Build the list of type identifiers in this disjoint set.
    std::vector<Metadata *> TypeIds;
    std::vector<GlobalTypeMember *> Globals;
    for (GlobalClassesTy::member_iterator MI =
             GlobalClasses.member_begin(S.first);
         MI != GlobalClasses.member_end(); ++MI) {
      if ((*MI).is<Metadata *>())
        TypeIds.push_back(MI->get<Metadata *>());
      else
        Globals.push_back(MI->get<GlobalTypeMember *>());
    }

    // Order type identifiers by global index for determinism. This ordering is
    // stable as there is a one-to-one mapping between metadata and indices.
    std::sort(TypeIds.begin(), TypeIds.end(), [&](Metadata *M1, Metadata *M2) {
      return TypeIdInfo[M1].Index < TypeIdInfo[M2].Index;
    });

    // Build bitsets for this disjoint set.
    buildBitSetsFromDisjointSet(TypeIds, Globals);
  }

  allocateByteArrays();

  return true;
}

PreservedAnalyses LowerTypeTestsPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  bool Changed =
      LowerTypeTestsModule(M, PassSummaryAction::None, /*Summary=*/nullptr)
          .lower();
  if (!Changed)
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}
