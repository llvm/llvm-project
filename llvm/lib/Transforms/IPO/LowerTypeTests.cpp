//===- LowerTypeTests.cpp - type metadata lowering pass -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers type metadata and calls to the llvm.type.test intrinsic.
// It also ensures that globals are properly laid out for the
// llvm.icall.branch.funnel intrinsic.
// See http://llvm.org/docs/TypeMetadata.html for more information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/LowerTypeTests.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TypeMetadataUtils.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/ModuleSummaryIndexYAML.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TrailingObjects.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

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

static cl::opt<DropTestKind>
    ClDropTypeTests("lowertypetests-drop-type-tests",
                    cl::desc("Simply drop type test sequences"),
                    cl::values(clEnumValN(DropTestKind::None, "none",
                                          "Do not drop any type tests"),
                               clEnumValN(DropTestKind::Assume, "assume",
                                          "Drop type test assume sequences"),
                               clEnumValN(DropTestKind::All, "all",
                                          "Drop all type test sequences")),
                    cl::Hidden, cl::init(DropTestKind::None));

bool BitSetInfo::containsGlobalOffset(uint64_t Offset) const {
  if (Offset < ByteOffset)
    return false;

  if ((Offset - ByteOffset) % (uint64_t(1) << AlignLog2) != 0)
    return false;

  uint64_t BitOffset = (Offset - ByteOffset) >> AlignLog2;
  if (BitOffset >= BitSize)
    return false;

  return Bits.count(BitSize - 1 - BitOffset);
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
    BSI.AlignLog2 = llvm::countr_zero(Mask);

  // Build the compressed bitset while normalizing the offsets against the
  // computed alignment.
  BSI.BitSize = ((Max - Min) >> BSI.AlignLog2) + 1;
  for (uint64_t Offset : Offsets) {
    Offset >>= BSI.AlignLog2;
    // We invert the order of bits when adding them to the bitset. This is
    // because the offset that we test against is computed by subtracting the
    // address that we are testing from the global's address, which means that
    // the offset increases as the tested address decreases.
    BSI.Bits.insert(BSI.BitSize - 1 - Offset);
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
      llvm::append_range(Fragment, OldFragment);
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

bool lowertypetests::isJumpTableCanonical(Function *F) {
  if (F->isDeclarationForLinker())
    return false;
  auto *CI = mdconst::extract_or_null<ConstantInt>(
      F->getParent()->getModuleFlag("CFI Canonical Jump Tables"));
  if (!CI || !CI->isZero())
    return true;
  return F->hasFnAttribute("cfi-canonical-jump-table");
}

namespace {

struct ByteArrayInfo {
  std::set<uint64_t> Bits;
  uint64_t BitSize;
  GlobalVariable *ByteArray;
  GlobalVariable *MaskGlobal;
  uint8_t *MaskPtr = nullptr;
};

/// A POD-like structure that we use to store a global reference together with
/// its metadata types. In this pass we frequently need to query the set of
/// metadata types referenced by a global, which at the IR level is an expensive
/// operation involving a map lookup; this data structure helps to reduce the
/// number of times we need to do this lookup.
class GlobalTypeMember final : TrailingObjects<GlobalTypeMember, MDNode *> {
  friend TrailingObjects;

  GlobalObject *GO;
  size_t NTypes;

  // For functions: true if the jump table is canonical. This essentially means
  // whether the canonical address (i.e. the symbol table entry) of the function
  // is provided by the local jump table. This is normally the same as whether
  // the function is defined locally, but if canonical jump tables are disabled
  // by the user then the jump table never provides a canonical definition.
  bool IsJumpTableCanonical;

  // For functions: true if this function is either defined or used in a thinlto
  // module and its jumptable entry needs to be exported to thinlto backends.
  bool IsExported;

public:
  static GlobalTypeMember *create(BumpPtrAllocator &Alloc, GlobalObject *GO,
                                  bool IsJumpTableCanonical, bool IsExported,
                                  ArrayRef<MDNode *> Types) {
    auto *GTM = static_cast<GlobalTypeMember *>(Alloc.Allocate(
        totalSizeToAlloc<MDNode *>(Types.size()), alignof(GlobalTypeMember)));
    GTM->GO = GO;
    GTM->NTypes = Types.size();
    GTM->IsJumpTableCanonical = IsJumpTableCanonical;
    GTM->IsExported = IsExported;
    llvm::copy(Types, GTM->getTrailingObjects());
    return GTM;
  }

  GlobalObject *getGlobal() const {
    return GO;
  }

  bool isJumpTableCanonical() const {
    return IsJumpTableCanonical;
  }

  bool isExported() const {
    return IsExported;
  }

  ArrayRef<MDNode *> types() const { return getTrailingObjects(NTypes); }
};

struct ICallBranchFunnel final
    : TrailingObjects<ICallBranchFunnel, GlobalTypeMember *> {
  static ICallBranchFunnel *create(BumpPtrAllocator &Alloc, CallInst *CI,
                                   ArrayRef<GlobalTypeMember *> Targets,
                                   unsigned UniqueId) {
    auto *Call = static_cast<ICallBranchFunnel *>(
        Alloc.Allocate(totalSizeToAlloc<GlobalTypeMember *>(Targets.size()),
                       alignof(ICallBranchFunnel)));
    Call->CI = CI;
    Call->UniqueId = UniqueId;
    Call->NTargets = Targets.size();
    llvm::copy(Targets, Call->getTrailingObjects());
    return Call;
  }

  CallInst *CI;
  ArrayRef<GlobalTypeMember *> targets() const {
    return getTrailingObjects(NTargets);
  }

  unsigned UniqueId;

private:
  size_t NTargets;
};

struct ScopedSaveAliaseesAndUsed {
  Module &M;
  SmallVector<GlobalValue *, 4> Used, CompilerUsed;
  std::vector<std::pair<GlobalAlias *, Function *>> FunctionAliases;
  std::vector<std::pair<GlobalIFunc *, Function *>> ResolverIFuncs;

  // This function only removes functions from llvm.used and llvm.compiler.used.
  // We cannot remove global variables because they need to follow RAUW, as
  // they may be deleted by buildBitSetsFromGlobalVariables.
  void collectAndEraseUsedFunctions(Module &M,
                                    SmallVectorImpl<GlobalValue *> &Vec,
                                    bool CompilerUsed) {
    auto *GV = collectUsedGlobalVariables(M, Vec, CompilerUsed);
    if (!GV)
      return;
    // There's no API to only remove certain array elements from
    // llvm.used/llvm.compiler.used, so we remove all of them and add back only
    // the non-functions.
    GV->eraseFromParent();
    auto NonFuncBegin =
        std::stable_partition(Vec.begin(), Vec.end(), [](GlobalValue *GV) {
          return isa<Function>(GV);
        });
    if (CompilerUsed)
      appendToCompilerUsed(M, {NonFuncBegin, Vec.end()});
    else
      appendToUsed(M, {NonFuncBegin, Vec.end()});
    Vec.resize(NonFuncBegin - Vec.begin());
  }

  ScopedSaveAliaseesAndUsed(Module &M) : M(M) {
    // The users of this class want to replace all function references except
    // for aliases and llvm.used/llvm.compiler.used with references to a jump
    // table. We avoid replacing aliases in order to avoid introducing a double
    // indirection (or an alias pointing to a declaration in ThinLTO mode), and
    // we avoid replacing llvm.used/llvm.compiler.used because these global
    // variables describe properties of the global, not the jump table (besides,
    // offseted references to the jump table in llvm.used are invalid).
    // Unfortunately, LLVM doesn't have a "RAUW except for these (possibly
    // indirect) users", so what we do is save the list of globals referenced by
    // llvm.used/llvm.compiler.used and aliases, erase the used lists, let RAUW
    // replace the aliasees and then set them back to their original values at
    // the end.
    collectAndEraseUsedFunctions(M, Used, false);
    collectAndEraseUsedFunctions(M, CompilerUsed, true);

    for (auto &GA : M.aliases()) {
      // FIXME: This should look past all aliases not just interposable ones,
      // see discussion on D65118.
      if (auto *F = dyn_cast<Function>(GA.getAliasee()->stripPointerCasts()))
        FunctionAliases.push_back({&GA, F});
    }

    for (auto &GI : M.ifuncs())
      if (auto *F = dyn_cast<Function>(GI.getResolver()->stripPointerCasts()))
        ResolverIFuncs.push_back({&GI, F});
  }

  ~ScopedSaveAliaseesAndUsed() {
    appendToUsed(M, Used);
    appendToCompilerUsed(M, CompilerUsed);

    for (auto P : FunctionAliases)
      P.first->setAliasee(P.second);

    for (auto P : ResolverIFuncs) {
      // This does not preserve pointer casts that may have been stripped by the
      // constructor, but the resolver's type is different from that of the
      // ifunc anyway.
      P.first->setResolver(P.second);
    }
  }
};

class LowerTypeTestsModule {
  Module &M;

  ModuleSummaryIndex *ExportSummary;
  const ModuleSummaryIndex *ImportSummary;
  // Set when the client has invoked this to simply drop all type test assume
  // sequences.
  DropTestKind DropTypeTests;

  Triple::ArchType Arch;
  Triple::OSType OS;
  Triple::ObjectFormatType ObjectFormat;

  // Determines which kind of Thumb jump table we generate. If arch is
  // either 'arm' or 'thumb' we need to find this out, because
  // selectJumpTableArmEncoding may decide to use Thumb in either case.
  bool CanUseArmJumpTable = false, CanUseThumbBWJumpTable = false;

  // Cache variable used by hasBranchTargetEnforcement().
  int HasBranchTargetEnforcement = -1;

  IntegerType *Int1Ty = Type::getInt1Ty(M.getContext());
  IntegerType *Int8Ty = Type::getInt8Ty(M.getContext());
  PointerType *PtrTy = PointerType::getUnqual(M.getContext());
  ArrayType *Int8Arr0Ty = ArrayType::get(Type::getInt8Ty(M.getContext()), 0);
  IntegerType *Int32Ty = Type::getInt32Ty(M.getContext());
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
    TypeTestResolution::Kind TheKind = TypeTestResolution::Unsat;

    /// All except Unsat: the address of the last element within the combined
    /// global.
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

  GlobalVariable *GlobalAnnotation;
  DenseSet<Value *> FunctionAnnotations;

  bool shouldExportConstantsAsAbsoluteSymbols();
  uint8_t *exportTypeId(StringRef TypeId, const TypeIdLowering &TIL);
  TypeIdLowering importTypeId(StringRef TypeId);
  void importTypeTest(CallInst *CI);
  void importFunction(Function *F, bool isJumpTableCanonical);

  ByteArrayInfo *createByteArray(const BitSetInfo &BSI);
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
  Triple::ArchType
  selectJumpTableArmEncoding(ArrayRef<GlobalTypeMember *> Functions);
  bool hasBranchTargetEnforcement();
  unsigned getJumpTableEntrySize(Triple::ArchType JumpTableArch);
  InlineAsm *createJumpTableEntryAsm(Triple::ArchType JumpTableArch);
  void verifyTypeMDNode(GlobalObject *GO, MDNode *Type);
  void buildBitSetsFromFunctions(ArrayRef<Metadata *> TypeIds,
                                 ArrayRef<GlobalTypeMember *> Functions);
  void buildBitSetsFromFunctionsNative(ArrayRef<Metadata *> TypeIds,
                                       ArrayRef<GlobalTypeMember *> Functions);
  void buildBitSetsFromFunctionsWASM(ArrayRef<Metadata *> TypeIds,
                                     ArrayRef<GlobalTypeMember *> Functions);
  void
  buildBitSetsFromDisjointSet(ArrayRef<Metadata *> TypeIds,
                              ArrayRef<GlobalTypeMember *> Globals,
                              ArrayRef<ICallBranchFunnel *> ICallBranchFunnels);

  void replaceWeakDeclarationWithJumpTablePtr(Function *F, Constant *JT,
                                              bool IsJumpTableCanonical);
  void moveInitializerToModuleConstructor(GlobalVariable *GV);
  void findGlobalVariableUsersOf(Constant *C,
                                 SmallSetVector<GlobalVariable *, 8> &Out);

  void createJumpTable(Function *F, ArrayRef<GlobalTypeMember *> Functions,
                       Triple::ArchType JumpTableArch);

  /// replaceCfiUses - Go through the uses list for this definition
  /// and make each use point to "V" instead of "this" when the use is outside
  /// the block. 'This's use list is expected to have at least one element.
  /// Unlike replaceAllUsesWith this function skips blockaddr and direct call
  /// uses.
  void replaceCfiUses(Function *Old, Value *New, bool IsJumpTableCanonical);

  /// replaceDirectCalls - Go through the uses list for this definition and
  /// replace each use, which is a direct function call.
  void replaceDirectCalls(Value *Old, Value *New);

  bool isFunctionAnnotation(Value *V) const {
    return FunctionAnnotations.contains(V);
  }

  void maybeReplaceComdat(Function *F, StringRef OriginalName);

public:
  LowerTypeTestsModule(Module &M, ModuleAnalysisManager &AM,
                       ModuleSummaryIndex *ExportSummary,
                       const ModuleSummaryIndex *ImportSummary,
                       DropTestKind DropTypeTests);

  bool lower();

  // Lower the module using the action and summary passed as command line
  // arguments. For testing purposes only.
  static bool runForTesting(Module &M, ModuleAnalysisManager &AM);
};
} // end anonymous namespace

/// Build a bit set for list of offsets.
static BitSetInfo buildBitSet(ArrayRef<uint64_t> Offsets) {
  // Compute the byte offset of each address associated with this type
  // identifier.
  return BitSetBuilder(Offsets).build();
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

ByteArrayInfo *LowerTypeTestsModule::createByteArray(const BitSetInfo &BSI) {
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
  llvm::stable_sort(ByteArrayInfos,
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
        ConstantExpr::getIntToPtr(ConstantInt::get(Int8Ty, Mask), PtrTy));
    BAI->MaskGlobal->eraseFromParent();
    if (BAI->MaskPtr)
      *BAI->MaskPtr = Mask;
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
    GlobalAlias *Alias = GlobalAlias::create(
        Int8Ty, 0, GlobalValue::PrivateLinkage, "bits", GEP, &M);
    BAI->ByteArray->replaceAllUsesWith(Alias);
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
    if (AvoidReuse && !ImportSummary) {
      // Each use of the byte array uses a different alias. This makes the
      // backend less likely to reuse previously computed byte array addresses,
      // improving the security of the CFI mechanism based on this pass.
      // This won't work when importing because TheByteArray is external.
      ByteArray = GlobalAlias::create(Int8Ty, 0, GlobalValue::PrivateLinkage,
                                      "bits_use", ByteArray, &M);
    }

    Value *ByteAddr = B.CreateGEP(Int8Ty, ByteArray, BitOffset);
    Value *Byte = B.CreateLoad(Int8Ty, ByteAddr);

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
    APInt APOffset(DL.getIndexSizeInBits(0), 0);
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
  // Delay lowering if the resolution is currently unknown.
  if (TIL.TheKind == TypeTestResolution::Unknown)
    return nullptr;
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

  // Here we compute `last element - address`. The reason why we do this instead
  // of computing `address - first element` is that it leads to a slightly
  // shorter instruction sequence on x86. Because it doesn't matter how we do
  // the subtraction on other architectures, we do so unconditionally.
  Value *PtrOffset = B.CreateSub(OffsetedGlobalAsInt, PtrAsInt);

  // We need to check that the offset both falls within our range and is
  // suitably aligned. We can check both properties at the same time by
  // performing a right rotate by log2(alignment) followed by an integer
  // comparison against the bitset size. The rotate will move the lower
  // order bits that need to be zero into the higher order bits of the
  // result, causing the comparison to fail if they are nonzero. The rotate
  // also conveniently gives us a bit offset to use during the load from
  // the bitset.
  Value *BitOffset = B.CreateIntrinsic(IntPtrTy, Intrinsic::fshr,
                                       {PtrOffset, PtrOffset, TIL.AlignLog2});

  Value *OffsetInRange = B.CreateICmpULE(BitOffset, TIL.SizeM1);

  // If the bit set is all ones, testing against it is unnecessary.
  if (TIL.TheKind == TypeTestResolution::AllOnes)
    return OffsetInRange;

  // See if the intrinsic is used in the following common pattern:
  //   br(llvm.type.test(...), thenbb, elsebb)
  // where nothing happens between the type test and the br.
  // If so, create slightly simpler IR.
  if (CI->hasOneUse())
    if (auto *Br = dyn_cast<BranchInst>(*CI->user_begin()))
      if (CI->getNextNode() == Br) {
        BasicBlock *Then = InitialBB->splitBasicBlock(CI->getIterator());
        BasicBlock *Else = Br->getSuccessor(1);
        BranchInst *NewBr = BranchInst::Create(Then, Else, OffsetInRange);
        NewBr->setMetadata(LLVMContext::MD_prof,
                           Br->getMetadata(LLVMContext::MD_prof));
        ReplaceInstWithInst(InitialBB->getTerminator(), NewBr);

        // Update phis in Else resulting from InitialBB being split
        for (auto &Phi : Else->phis())
          Phi.addIncoming(Phi.getIncomingValueForBlock(Then), InitialBB);

        IRBuilder<> ThenB(CI);
        return createBitSetTest(ThenB, TIL, BitOffset);
      }

  IRBuilder<> ThenB(SplitBlockAndInsertIfThen(OffsetInRange, CI, false));

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
  // any padding required to align the next element to the next power of 2 plus
  // any additional padding required to meet its alignment requirements.
  std::vector<Constant *> GlobalInits;
  const DataLayout &DL = M.getDataLayout();
  DenseMap<GlobalTypeMember *, uint64_t> GlobalLayout;
  Align MaxAlign;
  uint64_t CurOffset = 0;
  uint64_t DesiredPadding = 0;
  for (GlobalTypeMember *G : Globals) {
    auto *GV = cast<GlobalVariable>(G->getGlobal());
    Align Alignment =
        DL.getValueOrABITypeAlignment(GV->getAlign(), GV->getValueType());
    MaxAlign = std::max(MaxAlign, Alignment);
    uint64_t GVOffset = alignTo(CurOffset + DesiredPadding, Alignment);
    GlobalLayout[G] = GVOffset;
    if (GVOffset != 0) {
      uint64_t Padding = GVOffset - CurOffset;
      GlobalInits.push_back(
          ConstantAggregateZero::get(ArrayType::get(Int8Ty, Padding)));
    }

    GlobalInits.push_back(GV->getInitializer());
    uint64_t InitSize = DL.getTypeAllocSize(GV->getValueType());
    CurOffset = GVOffset + InitSize;

    // Compute the amount of padding that we'd like for the next element.
    DesiredPadding = NextPowerOf2(InitSize - 1) - InitSize;

    // Experiments of different caps with Chromium on both x64 and ARM64
    // have shown that the 32-byte cap generates the smallest binary on
    // both platforms while different caps yield similar performance.
    // (see https://lists.llvm.org/pipermail/llvm-dev/2018-July/124694.html)
    if (DesiredPadding > 32)
      DesiredPadding = alignTo(InitSize, 32) - InitSize;
  }

  Constant *NewInit = ConstantStruct::getAnon(M.getContext(), GlobalInits);
  auto *CombinedGlobal =
      new GlobalVariable(M, NewInit->getType(), /*isConstant=*/true,
                         GlobalValue::PrivateLinkage, NewInit);
  CombinedGlobal->setAlignment(MaxAlign);

  StructType *NewTy = cast<StructType>(NewInit->getType());
  lowerTypeTestCalls(TypeIds, CombinedGlobal, GlobalLayout);

  // Build aliases pointing to offsets into the combined global for each
  // global from which we built the combined global, and replace references
  // to the original globals with references to the aliases.
  for (unsigned I = 0; I != Globals.size(); ++I) {
    GlobalVariable *GV = cast<GlobalVariable>(Globals[I]->getGlobal());

    // Multiply by 2 to account for padding elements.
    Constant *CombinedGlobalIdxs[] = {ConstantInt::get(Int32Ty, 0),
                                      ConstantInt::get(Int32Ty, I * 2)};
    Constant *CombinedGlobalElemPtr = ConstantExpr::getInBoundsGetElementPtr(
        NewInit->getType(), CombinedGlobal, CombinedGlobalIdxs);
    assert(GV->getType()->getAddressSpace() == 0);
    GlobalAlias *GAlias =
        GlobalAlias::create(NewTy->getElementType(I * 2), 0, GV->getLinkage(),
                            "", CombinedGlobalElemPtr, &M);
    GAlias->setVisibility(GV->getVisibility());
    GAlias->takeName(GV);
    GV->replaceAllUsesWith(GAlias);
    GV->eraseFromParent();
  }
}

bool LowerTypeTestsModule::shouldExportConstantsAsAbsoluteSymbols() {
  return (Arch == Triple::x86 || Arch == Triple::x86_64) &&
         ObjectFormat == Triple::ELF;
}

/// Export the given type identifier so that ThinLTO backends may import it.
/// Type identifiers are exported by adding coarse-grained information about how
/// to test the type identifier to the summary, and creating symbols in the
/// object file (aliases and absolute symbols) containing fine-grained
/// information about the type identifier.
///
/// Returns a pointer to the location in which to store the bitmask, if
/// applicable.
uint8_t *LowerTypeTestsModule::exportTypeId(StringRef TypeId,
                                            const TypeIdLowering &TIL) {
  TypeTestResolution &TTRes =
      ExportSummary->getOrInsertTypeIdSummary(TypeId).TTRes;
  TTRes.TheKind = TIL.TheKind;

  auto ExportGlobal = [&](StringRef Name, Constant *C) {
    GlobalAlias *GA =
        GlobalAlias::create(Int8Ty, 0, GlobalValue::ExternalLinkage,
                            "__typeid_" + TypeId + "_" + Name, C, &M);
    GA->setVisibility(GlobalValue::HiddenVisibility);
  };

  auto ExportConstant = [&](StringRef Name, uint64_t &Storage, Constant *C) {
    if (shouldExportConstantsAsAbsoluteSymbols())
      ExportGlobal(Name, ConstantExpr::getIntToPtr(C, PtrTy));
    else
      Storage = cast<ConstantInt>(C)->getZExtValue();
  };

  if (TIL.TheKind != TypeTestResolution::Unsat)
    ExportGlobal("global_addr", TIL.OffsetedGlobal);

  if (TIL.TheKind == TypeTestResolution::ByteArray ||
      TIL.TheKind == TypeTestResolution::Inline ||
      TIL.TheKind == TypeTestResolution::AllOnes) {
    ExportConstant("align", TTRes.AlignLog2, TIL.AlignLog2);
    ExportConstant("size_m1", TTRes.SizeM1, TIL.SizeM1);

    uint64_t BitSize = cast<ConstantInt>(TIL.SizeM1)->getZExtValue() + 1;
    if (TIL.TheKind == TypeTestResolution::Inline)
      TTRes.SizeM1BitWidth = (BitSize <= 32) ? 5 : 6;
    else
      TTRes.SizeM1BitWidth = (BitSize <= 128) ? 7 : 32;
  }

  if (TIL.TheKind == TypeTestResolution::ByteArray) {
    ExportGlobal("byte_array", TIL.TheByteArray);
    if (shouldExportConstantsAsAbsoluteSymbols())
      ExportGlobal("bit_mask", TIL.BitMask);
    else
      return &TTRes.BitMask;
  }

  if (TIL.TheKind == TypeTestResolution::Inline)
    ExportConstant("inline_bits", TTRes.InlineBits, TIL.InlineBits);

  return nullptr;
}

LowerTypeTestsModule::TypeIdLowering
LowerTypeTestsModule::importTypeId(StringRef TypeId) {
  const TypeIdSummary *TidSummary = ImportSummary->getTypeIdSummary(TypeId);
  if (!TidSummary)
    return {}; // Unsat: no globals match this type id.
  const TypeTestResolution &TTRes = TidSummary->TTRes;

  TypeIdLowering TIL;
  TIL.TheKind = TTRes.TheKind;

  auto ImportGlobal = [&](StringRef Name) {
    // Give the global a type of length 0 so that it is not assumed not to alias
    // with any other global.
    GlobalVariable *GV = M.getOrInsertGlobal(
        ("__typeid_" + TypeId + "_" + Name).str(), Int8Arr0Ty);
    GV->setVisibility(GlobalValue::HiddenVisibility);
    return GV;
  };

  auto ImportConstant = [&](StringRef Name, uint64_t Const, unsigned AbsWidth,
                            Type *Ty) {
    if (!shouldExportConstantsAsAbsoluteSymbols()) {
      Constant *C =
          ConstantInt::get(isa<IntegerType>(Ty) ? Ty : Int64Ty, Const);
      if (!isa<IntegerType>(Ty))
        C = ConstantExpr::getIntToPtr(C, Ty);
      return C;
    }

    Constant *C = ImportGlobal(Name);
    auto *GV = cast<GlobalVariable>(C->stripPointerCasts());
    if (isa<IntegerType>(Ty))
      C = ConstantExpr::getPtrToInt(C, Ty);
    if (GV->getMetadata(LLVMContext::MD_absolute_symbol))
      return C;

    auto SetAbsRange = [&](uint64_t Min, uint64_t Max) {
      auto *MinC = ConstantAsMetadata::get(ConstantInt::get(IntPtrTy, Min));
      auto *MaxC = ConstantAsMetadata::get(ConstantInt::get(IntPtrTy, Max));
      GV->setMetadata(LLVMContext::MD_absolute_symbol,
                      MDNode::get(M.getContext(), {MinC, MaxC}));
    };
    if (AbsWidth == IntPtrTy->getBitWidth())
      SetAbsRange(~0ull, ~0ull); // Full set.
    else
      SetAbsRange(0, 1ull << AbsWidth);
    return C;
  };

  if (TIL.TheKind != TypeTestResolution::Unsat) {
    auto *GV = ImportGlobal("global_addr");
    // This is either a vtable (in .data.rel.ro) or a jump table (in .text).
    // Either way it's expected to be in the low 2 GiB, so set the small code
    // model.
    //
    // For .data.rel.ro, we currently place all such sections in the low 2 GiB
    // [1], and for .text the sections are expected to be in the low 2 GiB under
    // the small and medium code models [2] and this pass only supports those
    // code models (e.g. jump tables use jmp instead of movabs/jmp).
    //
    // [1]https://github.com/llvm/llvm-project/pull/137742
    // [2]https://maskray.me/blog/2023-05-14-relocation-overflow-and-code-models
    GV->setCodeModel(CodeModel::Small);
    TIL.OffsetedGlobal = GV;
  }

  if (TIL.TheKind == TypeTestResolution::ByteArray ||
      TIL.TheKind == TypeTestResolution::Inline ||
      TIL.TheKind == TypeTestResolution::AllOnes) {
    TIL.AlignLog2 = ImportConstant("align", TTRes.AlignLog2, 8, IntPtrTy);
    TIL.SizeM1 =
        ImportConstant("size_m1", TTRes.SizeM1, TTRes.SizeM1BitWidth, IntPtrTy);
  }

  if (TIL.TheKind == TypeTestResolution::ByteArray) {
    TIL.TheByteArray = ImportGlobal("byte_array");
    TIL.BitMask = ImportConstant("bit_mask", TTRes.BitMask, 8, PtrTy);
  }

  if (TIL.TheKind == TypeTestResolution::Inline)
    TIL.InlineBits = ImportConstant(
        "inline_bits", TTRes.InlineBits, 1 << TTRes.SizeM1BitWidth,
        TTRes.SizeM1BitWidth <= 5 ? Int32Ty : Int64Ty);

  return TIL;
}

void LowerTypeTestsModule::importTypeTest(CallInst *CI) {
  auto TypeIdMDVal = dyn_cast<MetadataAsValue>(CI->getArgOperand(1));
  if (!TypeIdMDVal)
    report_fatal_error("Second argument of llvm.type.test must be metadata");

  auto TypeIdStr = dyn_cast<MDString>(TypeIdMDVal->getMetadata());
  // If this is a local unpromoted type, which doesn't have a metadata string,
  // treat as Unknown and delay lowering, so that we can still utilize it for
  // later optimizations.
  if (!TypeIdStr)
    return;

  TypeIdLowering TIL = importTypeId(TypeIdStr->getString());
  Value *Lowered = lowerTypeTestCall(TypeIdStr, CI, TIL);
  if (Lowered) {
    CI->replaceAllUsesWith(Lowered);
    CI->eraseFromParent();
  }
}

void LowerTypeTestsModule::maybeReplaceComdat(Function *F,
                                              StringRef OriginalName) {
  // For COFF we should also rename the comdat if this function also
  // happens to be the key function. Even if the comdat name changes, this
  // should still be fine since comdat and symbol resolution happens
  // before LTO, so all symbols which would prevail have been selected.
  if (F->hasComdat() && ObjectFormat == Triple::COFF &&
      F->getComdat()->getName() == OriginalName) {
    Comdat *OldComdat = F->getComdat();
    Comdat *NewComdat = M.getOrInsertComdat(F->getName());
    for (GlobalObject &GO : M.global_objects()) {
      if (GO.getComdat() == OldComdat)
        GO.setComdat(NewComdat);
    }
  }
}

// ThinLTO backend: the function F has a jump table entry; update this module
// accordingly. isJumpTableCanonical describes the type of the jump table entry.
void LowerTypeTestsModule::importFunction(Function *F,
                                          bool isJumpTableCanonical) {
  assert(F->getType()->getAddressSpace() == 0);

  GlobalValue::VisibilityTypes Visibility = F->getVisibility();
  std::string Name = std::string(F->getName());

  if (F->isDeclarationForLinker() && isJumpTableCanonical) {
    // Non-dso_local functions may be overriden at run time,
    // don't short curcuit them
    if (F->isDSOLocal()) {
      Function *RealF = Function::Create(F->getFunctionType(),
                                         GlobalValue::ExternalLinkage,
                                         F->getAddressSpace(),
                                         Name + ".cfi", &M);
      RealF->setVisibility(GlobalVariable::HiddenVisibility);
      replaceDirectCalls(F, RealF);
    }
    return;
  }

  Function *FDecl;
  if (!isJumpTableCanonical) {
    // Either a declaration of an external function or a reference to a locally
    // defined jump table.
    FDecl = Function::Create(F->getFunctionType(), GlobalValue::ExternalLinkage,
                             F->getAddressSpace(), Name + ".cfi_jt", &M);
    FDecl->setVisibility(GlobalValue::HiddenVisibility);
  } else {
    F->setName(Name + ".cfi");
    maybeReplaceComdat(F, Name);
    FDecl = Function::Create(F->getFunctionType(), GlobalValue::ExternalLinkage,
                             F->getAddressSpace(), Name, &M);
    FDecl->setVisibility(Visibility);
    Visibility = GlobalValue::HiddenVisibility;

    // Update aliases pointing to this function to also include the ".cfi" suffix,
    // We expect the jump table entry to either point to the real function or an
    // alias. Redirect all other users to the jump table entry.
    for (auto &U : F->uses()) {
      if (auto *A = dyn_cast<GlobalAlias>(U.getUser())) {
        std::string AliasName = A->getName().str() + ".cfi";
        Function *AliasDecl = Function::Create(
            F->getFunctionType(), GlobalValue::ExternalLinkage,
            F->getAddressSpace(), "", &M);
        AliasDecl->takeName(A);
        A->replaceAllUsesWith(AliasDecl);
        A->setName(AliasName);
      }
    }
  }

  if (F->hasExternalWeakLinkage())
    replaceWeakDeclarationWithJumpTablePtr(F, FDecl, isJumpTableCanonical);
  else
    replaceCfiUses(F, FDecl, isJumpTableCanonical);

  // Set visibility late because it's used in replaceCfiUses() to determine
  // whether uses need to be replaced.
  F->setVisibility(Visibility);
}

static auto
buildBitSets(ArrayRef<Metadata *> TypeIds,
             const DenseMap<GlobalTypeMember *, uint64_t> &GlobalLayout) {
  DenseMap<Metadata *, SmallVector<uint64_t, 16>> OffsetsByTypeID;
  // Pre-populate the map with interesting type identifiers.
  for (Metadata *TypeId : TypeIds)
    OffsetsByTypeID[TypeId];
  for (const auto &[Mem, MemOff] : GlobalLayout) {
    for (MDNode *Type : Mem->types()) {
      auto It = OffsetsByTypeID.find(Type->getOperand(1));
      if (It == OffsetsByTypeID.end())
        continue;
      uint64_t Offset =
          cast<ConstantInt>(
              cast<ConstantAsMetadata>(Type->getOperand(0))->getValue())
              ->getZExtValue();
      It->second.push_back(MemOff + Offset);
    }
  }

  SmallVector<std::pair<Metadata *, BitSetInfo>> BitSets;
  BitSets.reserve(TypeIds.size());
  for (Metadata *TypeId : TypeIds) {
    BitSets.emplace_back(TypeId, buildBitSet(OffsetsByTypeID[TypeId]));
    LLVM_DEBUG({
      if (auto MDS = dyn_cast<MDString>(TypeId))
        dbgs() << MDS->getString() << ": ";
      else
        dbgs() << "<unnamed>: ";
      BitSets.back().second.print(dbgs());
    });
  }

  return BitSets;
}

void LowerTypeTestsModule::lowerTypeTestCalls(
    ArrayRef<Metadata *> TypeIds, Constant *CombinedGlobalAddr,
    const DenseMap<GlobalTypeMember *, uint64_t> &GlobalLayout) {
  // For each type identifier in this disjoint set...
  for (const auto &[TypeId, BSI] : buildBitSets(TypeIds, GlobalLayout)) {
    ByteArrayInfo *BAI = nullptr;
    TypeIdLowering TIL;

    uint64_t GlobalOffset =
        BSI.ByteOffset + ((BSI.BitSize - 1) << BSI.AlignLog2);
    TIL.OffsetedGlobal = ConstantExpr::getGetElementPtr(
        Int8Ty, CombinedGlobalAddr, ConstantInt::get(IntPtrTy, GlobalOffset)),
    TIL.AlignLog2 = ConstantInt::get(IntPtrTy, BSI.AlignLog2);
    TIL.SizeM1 = ConstantInt::get(IntPtrTy, BSI.BitSize - 1);
    if (BSI.isAllOnes()) {
      TIL.TheKind = (BSI.BitSize == 1) ? TypeTestResolution::Single
                                       : TypeTestResolution::AllOnes;
    } else if (BSI.BitSize <= IntPtrTy->getBitWidth()) {
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
      BAI = createByteArray(BSI);
      TIL.TheByteArray = BAI->ByteArray;
      TIL.BitMask = BAI->MaskGlobal;
    }

    TypeIdUserInfo &TIUI = TypeIdUsers[TypeId];

    if (TIUI.IsExported) {
      uint8_t *MaskPtr = exportTypeId(cast<MDString>(TypeId)->getString(), TIL);
      if (BAI)
        BAI->MaskPtr = MaskPtr;
    }

    // Lower each call to llvm.type.test for this type identifier.
    for (CallInst *CI : TIUI.CallSites) {
      ++NumTypeTestCallsLowered;
      Value *Lowered = lowerTypeTestCall(TypeId, CI, TIL);
      if (Lowered) {
        CI->replaceAllUsesWith(Lowered);
        CI->eraseFromParent();
      }
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
static const unsigned kX86IBTJumpTableEntrySize = 16;
static const unsigned kARMJumpTableEntrySize = 4;
static const unsigned kARMBTIJumpTableEntrySize = 8;
static const unsigned kARMv6MJumpTableEntrySize = 16;
static const unsigned kRISCVJumpTableEntrySize = 8;
static const unsigned kLOONGARCH64JumpTableEntrySize = 8;

bool LowerTypeTestsModule::hasBranchTargetEnforcement() {
  if (HasBranchTargetEnforcement == -1) {
    // First time this query has been called. Find out the answer by checking
    // the module flags.
    if (const auto *BTE = mdconst::extract_or_null<ConstantInt>(
          M.getModuleFlag("branch-target-enforcement")))
      HasBranchTargetEnforcement = (BTE->getZExtValue() != 0);
    else
      HasBranchTargetEnforcement = 0;
  }
  return HasBranchTargetEnforcement;
}

unsigned
LowerTypeTestsModule::getJumpTableEntrySize(Triple::ArchType JumpTableArch) {
  switch (JumpTableArch) {
  case Triple::x86:
  case Triple::x86_64:
    if (const auto *MD = mdconst::extract_or_null<ConstantInt>(
            M.getModuleFlag("cf-protection-branch")))
      if (MD->getZExtValue())
        return kX86IBTJumpTableEntrySize;
    return kX86JumpTableEntrySize;
  case Triple::arm:
    return kARMJumpTableEntrySize;
  case Triple::thumb:
    if (CanUseThumbBWJumpTable) {
      if (hasBranchTargetEnforcement())
        return kARMBTIJumpTableEntrySize;
      return kARMJumpTableEntrySize;
    } else {
      return kARMv6MJumpTableEntrySize;
    }
  case Triple::aarch64:
    if (hasBranchTargetEnforcement())
      return kARMBTIJumpTableEntrySize;
    return kARMJumpTableEntrySize;
  case Triple::riscv32:
  case Triple::riscv64:
    return kRISCVJumpTableEntrySize;
  case Triple::loongarch64:
    return kLOONGARCH64JumpTableEntrySize;
  default:
    report_fatal_error("Unsupported architecture for jump tables");
  }
}

// Create an inline asm constant representing a jump table entry for the target.
// This consists of an instruction sequence containing a relative branch to
// Dest.
InlineAsm *
LowerTypeTestsModule::createJumpTableEntryAsm(Triple::ArchType JumpTableArch) {
  std::string Asm;
  raw_string_ostream AsmOS(Asm);

  if (JumpTableArch == Triple::x86 || JumpTableArch == Triple::x86_64) {
    bool Endbr = false;
    if (const auto *MD = mdconst::extract_or_null<ConstantInt>(
            M.getModuleFlag("cf-protection-branch")))
      Endbr = !MD->isZero();
    if (Endbr)
      AsmOS << (JumpTableArch == Triple::x86 ? "endbr32\n" : "endbr64\n");
    AsmOS << "jmp ${0:c}@plt\n";
    if (Endbr)
      AsmOS << ".balign 16, 0xcc\n";
    else
      AsmOS << "int3\nint3\nint3\n";
  } else if (JumpTableArch == Triple::arm) {
    AsmOS << "b $0\n";
  } else if (JumpTableArch == Triple::aarch64) {
    if (hasBranchTargetEnforcement())
      AsmOS << "bti c\n";
    AsmOS << "b $0\n";
  } else if (JumpTableArch == Triple::thumb) {
    if (!CanUseThumbBWJumpTable) {
      // In Armv6-M, this sequence will generate a branch without corrupting
      // any registers. We use two stack words; in the second, we construct the
      // address we'll pop into pc, and the first is used to save and restore
      // r0 which we use as a temporary register.
      //
      // To support position-independent use cases, the offset of the target
      // function is stored as a relative offset (which will expand into an
      // R_ARM_REL32 relocation in ELF, and presumably the equivalent in other
      // object file types), and added to pc after we load it. (The alternative
      // B.W is automatically pc-relative.)
      //
      // There are five 16-bit Thumb instructions here, so the .balign 4 adds a
      // sixth halfword of padding, and then the offset consumes a further 4
      // bytes, for a total of 16, which is very convenient since entries in
      // this jump table need to have power-of-two size.
      AsmOS << "push {r0,r1}\n"
            << "ldr r0, 1f\n"
            << "0: add r0, r0, pc\n"
            << "str r0, [sp, #4]\n"
            << "pop {r0,pc}\n"
            << ".balign 4\n"
            << "1: .word $0 - (0b + 4)\n";
    } else {
      if (hasBranchTargetEnforcement())
        AsmOS << "bti\n";
      AsmOS << "b.w $0\n";
    }
  } else if (JumpTableArch == Triple::riscv32 ||
             JumpTableArch == Triple::riscv64) {
    AsmOS << "tail $0@plt\n";
  } else if (JumpTableArch == Triple::loongarch64) {
    AsmOS << "pcalau12i $$t0, %pc_hi20($0)\n"
          << "jirl $$r0, $$t0, %pc_lo12($0)\n";
  } else {
    report_fatal_error("Unsupported architecture for jump tables");
  }

  return InlineAsm::get(
      FunctionType::get(Type::getVoidTy(M.getContext()), PtrTy, false),
      AsmOS.str(), "s",
      /*hasSideEffects=*/true);
}

/// Given a disjoint set of type identifiers and functions, build the bit sets
/// and lower the llvm.type.test calls, architecture dependently.
void LowerTypeTestsModule::buildBitSetsFromFunctions(
    ArrayRef<Metadata *> TypeIds, ArrayRef<GlobalTypeMember *> Functions) {
  if (Arch == Triple::x86 || Arch == Triple::x86_64 || Arch == Triple::arm ||
      Arch == Triple::thumb || Arch == Triple::aarch64 ||
      Arch == Triple::riscv32 || Arch == Triple::riscv64 ||
      Arch == Triple::loongarch64)
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
        GlobalValue::InternalLinkage,
        M.getDataLayout().getProgramAddressSpace(),
        "__cfi_global_var_init", &M);
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
  IRB.CreateAlignedStore(GV->getInitializer(), GV, GV->getAlign());
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
    Function *F, Constant *JT, bool IsJumpTableCanonical) {
  // The target expression can not appear in a constant initializer on most
  // (all?) targets. Switch to a runtime initializer.
  SmallSetVector<GlobalVariable *, 8> GlobalVarUsers;
  findGlobalVariableUsersOf(F, GlobalVarUsers);
  for (auto *GV : GlobalVarUsers) {
    if (GV == GlobalAnnotation)
      continue;
    moveInitializerToModuleConstructor(GV);
  }

  // Can not RAUW F with an expression that uses F. Replace with a temporary
  // placeholder first.
  Function *PlaceholderFn =
      Function::Create(cast<FunctionType>(F->getValueType()),
                       GlobalValue::ExternalWeakLinkage,
                       F->getAddressSpace(), "", &M);
  replaceCfiUses(F, PlaceholderFn, IsJumpTableCanonical);

  convertUsersOfConstantsToInstructions(PlaceholderFn);
  // Don't use range based loop, because use list will be modified.
  while (!PlaceholderFn->use_empty()) {
    Use &U = *PlaceholderFn->use_begin();
    auto *InsertPt = dyn_cast<Instruction>(U.getUser());
    assert(InsertPt && "Non-instruction users should have been eliminated");
    auto *PN = dyn_cast<PHINode>(InsertPt);
    if (PN)
      InsertPt = PN->getIncomingBlock(U)->getTerminator();
    IRBuilder Builder(InsertPt);
    Value *ICmp = Builder.CreateICmp(CmpInst::ICMP_NE, F,
                                     Constant::getNullValue(F->getType()));
    Value *Select = Builder.CreateSelect(ICmp, JT,
                                         Constant::getNullValue(F->getType()));
    // For phi nodes, we need to update the incoming value for all operands
    // with the same predecessor.
    if (PN)
      PN->setIncomingValueForBlock(InsertPt->getParent(), Select);
    else
      U.set(Select);
  }
  PlaceholderFn->eraseFromParent();
}

static bool isThumbFunction(Function *F, Triple::ArchType ModuleArch) {
  Attribute TFAttr = F->getFnAttribute("target-features");
  if (TFAttr.isValid()) {
    SmallVector<StringRef, 6> Features;
    TFAttr.getValueAsString().split(Features, ',');
    for (StringRef Feature : Features) {
      if (Feature == "-thumb-mode")
        return false;
      else if (Feature == "+thumb-mode")
        return true;
    }
  }

  return ModuleArch == Triple::thumb;
}

// Each jump table must be either ARM or Thumb as a whole for the bit-test math
// to work. Pick one that matches the majority of members to minimize interop
// veneers inserted by the linker.
Triple::ArchType LowerTypeTestsModule::selectJumpTableArmEncoding(
    ArrayRef<GlobalTypeMember *> Functions) {
  if (Arch != Triple::arm && Arch != Triple::thumb)
    return Arch;

  if (!CanUseThumbBWJumpTable && CanUseArmJumpTable) {
    // In architectures that provide Arm and Thumb-1 but not Thumb-2,
    // we should always prefer the Arm jump table format, because the
    // Thumb-1 one is larger and slower.
    return Triple::arm;
  }

  // Otherwise, go with majority vote.
  unsigned ArmCount = 0, ThumbCount = 0;
  for (const auto GTM : Functions) {
    if (!GTM->isJumpTableCanonical()) {
      // PLT stubs are always ARM.
      // FIXME: This is the wrong heuristic for non-canonical jump tables.
      ++ArmCount;
      continue;
    }

    Function *F = cast<Function>(GTM->getGlobal());
    ++(isThumbFunction(F, Arch) ? ThumbCount : ArmCount);
  }

  return ArmCount > ThumbCount ? Triple::arm : Triple::thumb;
}

void LowerTypeTestsModule::createJumpTable(
    Function *F, ArrayRef<GlobalTypeMember *> Functions,
    Triple::ArchType JumpTableArch) {
  BasicBlock *BB = BasicBlock::Create(M.getContext(), "entry", F);
  IRBuilder<> IRB(BB);

  InlineAsm *JumpTableAsm = createJumpTableEntryAsm(JumpTableArch);

  // Check if all entries have the NoUnwind attribute.
  // If all entries have it, we can safely mark the
  // cfi.jumptable as NoUnwind, otherwise, direct calls
  // to the jump table will not handle exceptions properly
  bool areAllEntriesNounwind = true;
  for (GlobalTypeMember *GTM : Functions) {
    if (!llvm::cast<llvm::Function>(GTM->getGlobal())
             ->hasFnAttribute(llvm::Attribute::NoUnwind)) {
      areAllEntriesNounwind = false;
    }
    IRB.CreateCall(JumpTableAsm, GTM->getGlobal());
  }
  IRB.CreateUnreachable();

  // Align the whole table by entry size.
  F->setAlignment(Align(getJumpTableEntrySize(JumpTableArch)));
  // Skip prologue.
  // Disabled on win32 due to https://llvm.org/bugs/show_bug.cgi?id=28641#c3.
  // Luckily, this function does not get any prologue even without the
  // attribute.
  if (OS != Triple::Win32)
    F->addFnAttr(Attribute::Naked);
  if (JumpTableArch == Triple::arm)
    F->addFnAttr("target-features", "-thumb-mode");
  if (JumpTableArch == Triple::thumb) {
    if (hasBranchTargetEnforcement()) {
      // If we're generating a Thumb jump table with BTI, add a target-features
      // setting to ensure BTI can be assembled.
      F->addFnAttr("target-features", "+thumb-mode,+pacbti");
    } else {
      F->addFnAttr("target-features", "+thumb-mode");
      if (CanUseThumbBWJumpTable) {
        // Thumb jump table assembly needs Thumb2. The following attribute is
        // added by Clang for -march=armv7.
        F->addFnAttr("target-cpu", "cortex-a8");
      }
    }
  }
  // When -mbranch-protection= is used, the inline asm adds a BTI. Suppress BTI
  // for the function to avoid double BTI. This is a no-op without
  // -mbranch-protection=.
  if (JumpTableArch == Triple::aarch64 || JumpTableArch == Triple::thumb) {
    if (F->hasFnAttribute("branch-target-enforcement"))
      F->removeFnAttr("branch-target-enforcement");
    if (F->hasFnAttribute("sign-return-address"))
      F->removeFnAttr("sign-return-address");
  }
  if (JumpTableArch == Triple::riscv32 || JumpTableArch == Triple::riscv64) {
    // Make sure the jump table assembly is not modified by the assembler or
    // the linker.
    F->addFnAttr("target-features", "-c,-relax");
  }
  // When -fcf-protection= is used, the inline asm adds an ENDBR. Suppress ENDBR
  // for the function to avoid double ENDBR. This is a no-op without
  // -fcf-protection=.
  if (JumpTableArch == Triple::x86 || JumpTableArch == Triple::x86_64)
    F->addFnAttr(Attribute::NoCfCheck);

  // Make sure we don't emit .eh_frame for this function if it isn't needed.
  if (areAllEntriesNounwind)
    F->addFnAttr(Attribute::NoUnwind);

  // Make sure we do not inline any calls to the cfi.jumptable.
  F->addFnAttr(Attribute::NoInline);
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

  // Decide on the jump table encoding, so that we know how big the
  // entries will be.
  Triple::ArchType JumpTableArch = selectJumpTableArmEncoding(Functions);

  // Build a simple layout based on the regular layout of jump tables.
  DenseMap<GlobalTypeMember *, uint64_t> GlobalLayout;
  unsigned EntrySize = getJumpTableEntrySize(JumpTableArch);
  for (unsigned I = 0; I != Functions.size(); ++I)
    GlobalLayout[Functions[I]] = I * EntrySize;

  Function *JumpTableFn =
      Function::Create(FunctionType::get(Type::getVoidTy(M.getContext()),
                                         /* IsVarArg */ false),
                       GlobalValue::PrivateLinkage,
                       M.getDataLayout().getProgramAddressSpace(),
                       ".cfi.jumptable", &M);
  ArrayType *JumpTableEntryType = ArrayType::get(Int8Ty, EntrySize);
  ArrayType *JumpTableType =
      ArrayType::get(JumpTableEntryType, Functions.size());
  auto JumpTable = ConstantExpr::getPointerCast(
      JumpTableFn, PointerType::getUnqual(M.getContext()));

  lowerTypeTestCalls(TypeIds, JumpTable, GlobalLayout);

  // Build aliases pointing to offsets into the jump table, and replace
  // references to the original functions with references to the aliases.
  for (unsigned I = 0; I != Functions.size(); ++I) {
    Function *F = cast<Function>(Functions[I]->getGlobal());
    bool IsJumpTableCanonical = Functions[I]->isJumpTableCanonical();

    Constant *CombinedGlobalElemPtr = ConstantExpr::getInBoundsGetElementPtr(
        JumpTableType, JumpTable,
        ArrayRef<Constant *>{ConstantInt::get(IntPtrTy, 0),
                             ConstantInt::get(IntPtrTy, I)});

    const bool IsExported = Functions[I]->isExported();
    if (!IsJumpTableCanonical) {
      GlobalValue::LinkageTypes LT = IsExported ? GlobalValue::ExternalLinkage
                                                : GlobalValue::InternalLinkage;
      GlobalAlias *JtAlias = GlobalAlias::create(JumpTableEntryType, 0, LT,
                                                 F->getName() + ".cfi_jt",
                                                 CombinedGlobalElemPtr, &M);
      if (IsExported)
        JtAlias->setVisibility(GlobalValue::HiddenVisibility);
      else
        appendToUsed(M, {JtAlias});
    }

    if (IsExported) {
      if (IsJumpTableCanonical)
        ExportSummary->cfiFunctionDefs().emplace(F->getName());
      else
        ExportSummary->cfiFunctionDecls().emplace(F->getName());
    }

    if (!IsJumpTableCanonical) {
      if (F->hasExternalWeakLinkage())
        replaceWeakDeclarationWithJumpTablePtr(F, CombinedGlobalElemPtr,
                                               IsJumpTableCanonical);
      else
        replaceCfiUses(F, CombinedGlobalElemPtr, IsJumpTableCanonical);
    } else {
      assert(F->getType()->getAddressSpace() == 0);

      GlobalAlias *FAlias =
          GlobalAlias::create(JumpTableEntryType, 0, F->getLinkage(), "",
                              CombinedGlobalElemPtr, &M);
      FAlias->setVisibility(F->getVisibility());
      FAlias->takeName(F);
      if (FAlias->hasName()) {
        F->setName(FAlias->getName() + ".cfi");
        maybeReplaceComdat(F, FAlias->getName());
      }
      replaceCfiUses(F, FAlias, IsJumpTableCanonical);
      if (!F->hasLocalLinkage())
        F->setVisibility(GlobalVariable::HiddenVisibility);
    }
  }

  createJumpTable(JumpTableFn, Functions, JumpTableArch);
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
  lowerTypeTestCalls(TypeIds, ConstantPointerNull::get(PtrTy),
                     GlobalLayout);
}

void LowerTypeTestsModule::buildBitSetsFromDisjointSet(
    ArrayRef<Metadata *> TypeIds, ArrayRef<GlobalTypeMember *> Globals,
    ArrayRef<ICallBranchFunnel *> ICallBranchFunnels) {
  DenseMap<Metadata *, uint64_t> TypeIdIndices;
  for (unsigned I = 0; I != TypeIds.size(); ++I)
    TypeIdIndices[TypeIds[I]] = I;

  // For each type identifier, build a set of indices that refer to members of
  // the type identifier.
  std::vector<std::set<uint64_t>> TypeMembers(TypeIds.size());
  unsigned GlobalIndex = 0;
  DenseMap<GlobalTypeMember *, uint64_t> GlobalIndices;
  for (GlobalTypeMember *GTM : Globals) {
    for (MDNode *Type : GTM->types()) {
      // Type = { offset, type identifier }
      auto I = TypeIdIndices.find(Type->getOperand(1));
      if (I != TypeIdIndices.end())
        TypeMembers[I->second].insert(GlobalIndex);
    }
    GlobalIndices[GTM] = GlobalIndex;
    GlobalIndex++;
  }

  for (ICallBranchFunnel *JT : ICallBranchFunnels) {
    TypeMembers.emplace_back();
    std::set<uint64_t> &TMSet = TypeMembers.back();
    for (GlobalTypeMember *T : JT->targets())
      TMSet.insert(GlobalIndices[T]);
  }

  // Order the sets of indices by size. The GlobalLayoutBuilder works best
  // when given small index sets first.
  llvm::stable_sort(TypeMembers, [](const std::set<uint64_t> &O1,
                                    const std::set<uint64_t> &O2) {
    return O1.size() < O2.size();
  });

  // Create a GlobalLayoutBuilder and provide it with index sets as layout
  // fragments. The GlobalLayoutBuilder tries to lay out members of fragments as
  // close together as possible.
  GlobalLayoutBuilder GLB(Globals.size());
  for (auto &&MemSet : TypeMembers)
    GLB.addFragment(MemSet);

  // Build a vector of globals with the computed layout.
  bool IsGlobalSet =
      Globals.empty() || isa<GlobalVariable>(Globals[0]->getGlobal());
  std::vector<GlobalTypeMember *> OrderedGTMs(Globals.size());
  auto OGTMI = OrderedGTMs.begin();
  for (auto &&F : GLB.Fragments) {
    for (auto &&Offset : F) {
      if (IsGlobalSet != isa<GlobalVariable>(Globals[Offset]->getGlobal()))
        report_fatal_error("Type identifier may not contain both global "
                           "variables and functions");
      *OGTMI++ = Globals[Offset];
    }
  }

  // Build the bitsets from this disjoint set.
  if (IsGlobalSet)
    buildBitSetsFromGlobalVariables(TypeIds, OrderedGTMs);
  else
    buildBitSetsFromFunctions(TypeIds, OrderedGTMs);
}

/// Lower all type tests in this module.
LowerTypeTestsModule::LowerTypeTestsModule(
    Module &M, ModuleAnalysisManager &AM, ModuleSummaryIndex *ExportSummary,
    const ModuleSummaryIndex *ImportSummary, DropTestKind DropTypeTests)
    : M(M), ExportSummary(ExportSummary), ImportSummary(ImportSummary),
      DropTypeTests(ClDropTypeTests > DropTypeTests ? ClDropTypeTests
                                                    : DropTypeTests) {
  assert(!(ExportSummary && ImportSummary));
  Triple TargetTriple(M.getTargetTriple());
  Arch = TargetTriple.getArch();
  if (Arch == Triple::arm)
    CanUseArmJumpTable = true;
  if (Arch == Triple::arm || Arch == Triple::thumb) {
    auto &FAM =
        AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
    for (Function &F : M) {
      // Skip declarations since we should not query the TTI for them.
      if (F.isDeclaration())
        continue;
      auto &TTI = FAM.getResult<TargetIRAnalysis>(F);
      if (TTI.hasArmWideBranch(false))
        CanUseArmJumpTable = true;
      if (TTI.hasArmWideBranch(true))
        CanUseThumbBWJumpTable = true;
    }
  }
  OS = TargetTriple.getOS();
  ObjectFormat = TargetTriple.getObjectFormat();

  // Function annotation describes or applies to function itself, and
  // shouldn't be associated with jump table thunk generated for CFI.
  GlobalAnnotation = M.getGlobalVariable("llvm.global.annotations");
  if (GlobalAnnotation && GlobalAnnotation->hasInitializer()) {
    const ConstantArray *CA =
        cast<ConstantArray>(GlobalAnnotation->getInitializer());
    FunctionAnnotations.insert_range(CA->operands());
  }
}

bool LowerTypeTestsModule::runForTesting(Module &M, ModuleAnalysisManager &AM) {
  ModuleSummaryIndex Summary(/*HaveGVs=*/false);

  // Handle the command-line summary arguments. This code is for testing
  // purposes only, so we handle errors directly.
  if (!ClReadSummary.empty()) {
    ExitOnError ExitOnErr("-lowertypetests-read-summary: " + ClReadSummary +
                          ": ");
    auto ReadSummaryFile = ExitOnErr(errorOrToExpected(
        MemoryBuffer::getFile(ClReadSummary, /*IsText=*/true)));

    yaml::Input In(ReadSummaryFile->getBuffer());
    In >> Summary;
    ExitOnErr(errorCodeToError(In.error()));
  }

  bool Changed =
      LowerTypeTestsModule(
          M, AM,
          ClSummaryAction == PassSummaryAction::Export ? &Summary : nullptr,
          ClSummaryAction == PassSummaryAction::Import ? &Summary : nullptr,
          /*DropTypeTests=*/DropTestKind::None)
          .lower();

  if (!ClWriteSummary.empty()) {
    ExitOnError ExitOnErr("-lowertypetests-write-summary: " + ClWriteSummary +
                          ": ");
    std::error_code EC;
    raw_fd_ostream OS(ClWriteSummary, EC, sys::fs::OF_TextWithCRLF);
    ExitOnErr(errorCodeToError(EC));

    yaml::Output Out(OS);
    Out << Summary;
  }

  return Changed;
}

static bool isDirectCall(Use& U) {
  auto *Usr = dyn_cast<CallInst>(U.getUser());
  if (Usr) {
    auto *CB = dyn_cast<CallBase>(Usr);
    if (CB && CB->isCallee(&U))
      return true;
  }
  return false;
}

void LowerTypeTestsModule::replaceCfiUses(Function *Old, Value *New,
                                          bool IsJumpTableCanonical) {
  SmallSetVector<Constant *, 4> Constants;
  for (Use &U : llvm::make_early_inc_range(Old->uses())) {
    // Skip no_cfi values, which refer to the function body instead of the jump
    // table.
    if (isa<NoCFIValue>(U.getUser()))
      continue;

    // Skip direct calls to externally defined or non-dso_local functions.
    if (isDirectCall(U) && (Old->isDSOLocal() || !IsJumpTableCanonical))
      continue;

    // Skip function annotation.
    if (isFunctionAnnotation(U.getUser()))
      continue;

    // Must handle Constants specially, we cannot call replaceUsesOfWith on a
    // constant because they are uniqued.
    if (auto *C = dyn_cast<Constant>(U.getUser())) {
      if (!isa<GlobalValue>(C)) {
        // Save unique users to avoid processing operand replacement
        // more than once.
        Constants.insert(C);
        continue;
      }
    }

    U.set(New);
  }

  // Process operand replacement of saved constants.
  for (auto *C : Constants)
    C->handleOperandChange(Old, New);
}

void LowerTypeTestsModule::replaceDirectCalls(Value *Old, Value *New) {
  Old->replaceUsesWithIf(New, isDirectCall);
}

static void dropTypeTests(Module &M, Function &TypeTestFunc,
                          bool ShouldDropAll) {
  for (Use &U : llvm::make_early_inc_range(TypeTestFunc.uses())) {
    auto *CI = cast<CallInst>(U.getUser());
    // Find and erase llvm.assume intrinsics for this llvm.type.test call.
    for (Use &CIU : llvm::make_early_inc_range(CI->uses()))
      if (auto *Assume = dyn_cast<AssumeInst>(CIU.getUser()))
        Assume->eraseFromParent();
    // If the assume was merged with another assume, we might have a use on a
    // phi (which will feed the assume). Simply replace the use on the phi
    // with "true" and leave the merged assume.
    //
    // If ShouldDropAll is set, then we  we need to update any remaining uses,
    // regardless of the instruction type.
    if (!CI->use_empty()) {
      assert(ShouldDropAll || all_of(CI->users(), [](User *U) -> bool {
               return isa<PHINode>(U);
             }));
      CI->replaceAllUsesWith(ConstantInt::getTrue(M.getContext()));
    }
    CI->eraseFromParent();
  }
}

bool LowerTypeTestsModule::lower() {
  Function *TypeTestFunc =
      Intrinsic::getDeclarationIfExists(&M, Intrinsic::type_test);

  if (DropTypeTests != DropTestKind::None) {
    bool ShouldDropAll = DropTypeTests == DropTestKind::All;
    if (TypeTestFunc)
      dropTypeTests(M, *TypeTestFunc, ShouldDropAll);
    // Normally we'd have already removed all @llvm.public.type.test calls,
    // except for in the case where we originally were performing ThinLTO but
    // decided not to in the backend.
    Function *PublicTypeTestFunc =
        Intrinsic::getDeclarationIfExists(&M, Intrinsic::public_type_test);
    if (PublicTypeTestFunc)
      dropTypeTests(M, *PublicTypeTestFunc, ShouldDropAll);
    if (TypeTestFunc || PublicTypeTestFunc) {
      // We have deleted the type intrinsics, so we no longer have enough
      // information to reason about the liveness of virtual function pointers
      // in GlobalDCE.
      for (GlobalVariable &GV : M.globals())
        GV.eraseMetadata(LLVMContext::MD_vcall_visibility);
      return true;
    }
    return false;
  }

  // If only some of the modules were split, we cannot correctly perform
  // this transformation. We already checked for the presense of type tests
  // with partially split modules during the thin link, and would have emitted
  // an error if any were found, so here we can simply return.
  if ((ExportSummary && ExportSummary->partiallySplitLTOUnits()) ||
      (ImportSummary && ImportSummary->partiallySplitLTOUnits()))
    return false;

  Function *ICallBranchFunnelFunc =
      Intrinsic::getDeclarationIfExists(&M, Intrinsic::icall_branch_funnel);
  if ((!TypeTestFunc || TypeTestFunc->use_empty()) &&
      (!ICallBranchFunnelFunc || ICallBranchFunnelFunc->use_empty()) &&
      !ExportSummary && !ImportSummary)
    return false;

  if (ImportSummary) {
    if (TypeTestFunc)
      for (Use &U : llvm::make_early_inc_range(TypeTestFunc->uses()))
        importTypeTest(cast<CallInst>(U.getUser()));

    if (ICallBranchFunnelFunc && !ICallBranchFunnelFunc->use_empty())
      report_fatal_error(
          "unexpected call to llvm.icall.branch.funnel during import phase");

    SmallVector<Function *, 8> Defs;
    SmallVector<Function *, 8> Decls;
    for (auto &F : M) {
      // CFI functions are either external, or promoted. A local function may
      // have the same name, but it's not the one we are looking for.
      if (F.hasLocalLinkage())
        continue;
      if (ImportSummary->cfiFunctionDefs().count(F.getName()))
        Defs.push_back(&F);
      else if (ImportSummary->cfiFunctionDecls().count(F.getName()))
        Decls.push_back(&F);
    }

    {
      ScopedSaveAliaseesAndUsed S(M);
      for (auto *F : Defs)
        importFunction(F, /*isJumpTableCanonical*/ true);
      for (auto *F : Decls)
        importFunction(F, /*isJumpTableCanonical*/ false);
    }

    return true;
  }

  // Equivalence class set containing type identifiers and the globals that
  // reference them. This is used to partition the set of type identifiers in
  // the module into disjoint sets.
  using GlobalClassesTy = EquivalenceClasses<
      PointerUnion<GlobalTypeMember *, Metadata *, ICallBranchFunnel *>>;
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
    unsigned UniqueId;
    std::vector<GlobalTypeMember *> RefGlobals;
  };
  DenseMap<Metadata *, TIInfo> TypeIdInfo;
  unsigned CurUniqueId = 0;
  SmallVector<MDNode *, 2> Types;

  // Cross-DSO CFI emits jumptable entries for exported functions as well as
  // address taken functions in case they are address taken in other modules.
  const bool CrossDsoCfi = M.getModuleFlag("Cross-DSO CFI") != nullptr;

  struct ExportedFunctionInfo {
    CfiFunctionLinkage Linkage;
    MDNode *FuncMD; // {name, linkage, type[, type...]}
  };
  MapVector<StringRef, ExportedFunctionInfo> ExportedFunctions;
  if (ExportSummary) {
    NamedMDNode *CfiFunctionsMD = M.getNamedMetadata("cfi.functions");
    if (CfiFunctionsMD) {
      // A set of all functions that are address taken by a live global object.
      DenseSet<GlobalValue::GUID> AddressTaken;
      for (auto &I : *ExportSummary)
        for (auto &GVS : I.second.SummaryList)
          if (GVS->isLive())
            for (const auto &Ref : GVS->refs()) {
              AddressTaken.insert(Ref.getGUID());
              for (auto &RefGVS : Ref.getSummaryList())
                if (auto Alias = dyn_cast<AliasSummary>(RefGVS.get()))
                  AddressTaken.insert(Alias->getAliaseeGUID());
            }
      auto IsAddressTaken = [&](GlobalValue::GUID GUID) {
        if (AddressTaken.count(GUID))
          return true;
        auto VI = ExportSummary->getValueInfo(GUID);
        if (!VI)
          return false;
        for (auto &I : VI.getSummaryList())
          if (auto Alias = dyn_cast<AliasSummary>(I.get()))
            if (AddressTaken.count(Alias->getAliaseeGUID()))
              return true;
        return false;
      };
      for (auto *FuncMD : CfiFunctionsMD->operands()) {
        assert(FuncMD->getNumOperands() >= 2);
        StringRef FunctionName =
            cast<MDString>(FuncMD->getOperand(0))->getString();
        CfiFunctionLinkage Linkage = static_cast<CfiFunctionLinkage>(
            cast<ConstantAsMetadata>(FuncMD->getOperand(1))
                ->getValue()
                ->getUniqueInteger()
                .getZExtValue());
        const GlobalValue::GUID GUID =
            GlobalValue::getGUIDAssumingExternalLinkage(
                GlobalValue::dropLLVMManglingEscape(FunctionName));
        // Do not emit jumptable entries for functions that are not-live and
        // have no live references (and are not exported with cross-DSO CFI.)
        if (!ExportSummary->isGUIDLive(GUID))
          continue;
        if (!IsAddressTaken(GUID)) {
          if (!CrossDsoCfi || Linkage != CFL_Definition)
            continue;

          bool Exported = false;
          if (auto VI = ExportSummary->getValueInfo(GUID))
            for (const auto &GVS : VI.getSummaryList())
              if (GVS->isLive() && !GlobalValue::isLocalLinkage(GVS->linkage()))
                Exported = true;

          if (!Exported)
            continue;
        }
        auto P = ExportedFunctions.insert({FunctionName, {Linkage, FuncMD}});
        if (!P.second && P.first->second.Linkage != CFL_Definition)
          P.first->second = {Linkage, FuncMD};
      }

      for (const auto &P : ExportedFunctions) {
        StringRef FunctionName = P.first;
        CfiFunctionLinkage Linkage = P.second.Linkage;
        MDNode *FuncMD = P.second.FuncMD;
        Function *F = M.getFunction(FunctionName);
        if (F && F->hasLocalLinkage()) {
          // Locally defined function that happens to have the same name as a
          // function defined in a ThinLTO module. Rename it to move it out of
          // the way of the external reference that we're about to create.
          // Note that setName will find a unique name for the function, so even
          // if there is an existing function with the suffix there won't be a
          // name collision.
          F->setName(F->getName() + ".1");
          F = nullptr;
        }

        if (!F)
          F = Function::Create(
              FunctionType::get(Type::getVoidTy(M.getContext()), false),
              GlobalVariable::ExternalLinkage,
              M.getDataLayout().getProgramAddressSpace(), FunctionName, &M);

        // If the function is available_externally, remove its definition so
        // that it is handled the same way as a declaration. Later we will try
        // to create an alias using this function's linkage, which will fail if
        // the linkage is available_externally. This will also result in us
        // following the code path below to replace the type metadata.
        if (F->hasAvailableExternallyLinkage()) {
          F->setLinkage(GlobalValue::ExternalLinkage);
          F->deleteBody();
          F->setComdat(nullptr);
          F->clearMetadata();
        }

        // Update the linkage for extern_weak declarations when a definition
        // exists.
        if (Linkage == CFL_Definition && F->hasExternalWeakLinkage())
          F->setLinkage(GlobalValue::ExternalLinkage);

        // If the function in the full LTO module is a declaration, replace its
        // type metadata with the type metadata we found in cfi.functions. That
        // metadata is presumed to be more accurate than the metadata attached
        // to the declaration.
        if (F->isDeclaration()) {
          if (Linkage == CFL_WeakDeclaration)
            F->setLinkage(GlobalValue::ExternalWeakLinkage);

          F->eraseMetadata(LLVMContext::MD_type);
          for (unsigned I = 2; I < FuncMD->getNumOperands(); ++I)
            F->addMetadata(LLVMContext::MD_type,
                           *cast<MDNode>(FuncMD->getOperand(I).get()));
        }
      }
    }
  }

  struct AliasToCreate {
    Function *Alias;
    std::string TargetName;
  };
  std::vector<AliasToCreate> AliasesToCreate;

  // Parse alias data to replace stand-in function declarations for aliases
  // with an alias to the intended target.
  if (ExportSummary) {
    if (NamedMDNode *AliasesMD = M.getNamedMetadata("aliases")) {
      for (auto *AliasMD : AliasesMD->operands()) {
        SmallVector<Function *> Aliases;
        for (Metadata *MD : AliasMD->operands()) {
          auto *MDS = dyn_cast<MDString>(MD);
          if (!MDS)
            continue;
          StringRef AliasName = MDS->getString();
          if (!ExportedFunctions.count(AliasName))
            continue;
          auto *AliasF = M.getFunction(AliasName);
          if (AliasF)
            Aliases.push_back(AliasF);
        }

        if (Aliases.empty())
          continue;

        for (unsigned I = 1; I != Aliases.size(); ++I) {
          auto *AliasF = Aliases[I];
          ExportedFunctions.erase(AliasF->getName());
          AliasesToCreate.push_back(
              {AliasF, std::string(Aliases[0]->getName())});
        }
      }
    }
  }

  DenseMap<GlobalObject *, GlobalTypeMember *> GlobalTypeMembers;
  for (GlobalObject &GO : M.global_objects()) {
    if (isa<GlobalVariable>(GO) && GO.isDeclarationForLinker())
      continue;

    Types.clear();
    GO.getMetadata(LLVMContext::MD_type, Types);

    bool IsJumpTableCanonical = false;
    bool IsExported = false;
    if (Function *F = dyn_cast<Function>(&GO)) {
      IsJumpTableCanonical = isJumpTableCanonical(F);
      if (auto It = ExportedFunctions.find(F->getName());
          It != ExportedFunctions.end()) {
        IsJumpTableCanonical |= It->second.Linkage == CFL_Definition;
        IsExported = true;
      // TODO: The logic here checks only that the function is address taken,
      // not that the address takers are live. This can be updated to check
      // their liveness and emit fewer jumptable entries once monolithic LTO
      // builds also emit summaries.
      } else if (!F->hasAddressTaken()) {
        if (!CrossDsoCfi || !IsJumpTableCanonical || F->hasLocalLinkage())
          continue;
      }
    }

    auto *GTM = GlobalTypeMember::create(Alloc, &GO, IsJumpTableCanonical,
                                         IsExported, Types);
    GlobalTypeMembers[&GO] = GTM;
    for (MDNode *Type : Types) {
      verifyTypeMDNode(&GO, Type);
      auto &Info = TypeIdInfo[Type->getOperand(1)];
      Info.UniqueId = ++CurUniqueId;
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
      auto &GCI = GlobalClasses.insert(TypeId);
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
      // If this type test is only used by llvm.assume instructions, it
      // was used for whole program devirtualization, and is being kept
      // for use by other optimization passes. We do not need or want to
      // lower it here. We also don't want to rewrite any associated globals
      // unnecessarily. These will be removed by a subsequent LTT invocation
      // with the DropTypeTests flag set.
      bool OnlyAssumeUses = !CI->use_empty();
      for (const Use &CIU : CI->uses()) {
        if (isa<AssumeInst>(CIU.getUser()))
          continue;
        OnlyAssumeUses = false;
        break;
      }
      if (OnlyAssumeUses)
        continue;

      auto TypeIdMDVal = dyn_cast<MetadataAsValue>(CI->getArgOperand(1));
      if (!TypeIdMDVal)
        report_fatal_error("Second argument of llvm.type.test must be metadata");
      auto TypeId = TypeIdMDVal->getMetadata();
      AddTypeIdUse(TypeId).CallSites.push_back(CI);
    }
  }

  if (ICallBranchFunnelFunc) {
    for (const Use &U : ICallBranchFunnelFunc->uses()) {
      if (Arch != Triple::x86_64)
        report_fatal_error(
            "llvm.icall.branch.funnel not supported on this target");

      auto CI = cast<CallInst>(U.getUser());

      std::vector<GlobalTypeMember *> Targets;
      if (CI->arg_size() % 2 != 1)
        report_fatal_error("number of arguments should be odd");

      GlobalClassesTy::member_iterator CurSet;
      for (unsigned I = 1; I != CI->arg_size(); I += 2) {
        int64_t Offset;
        auto *Base = dyn_cast<GlobalObject>(GetPointerBaseWithConstantOffset(
            CI->getOperand(I), Offset, M.getDataLayout()));
        if (!Base)
          report_fatal_error(
              "Expected branch funnel operand to be global value");

        GlobalTypeMember *GTM = GlobalTypeMembers[Base];
        Targets.push_back(GTM);
        GlobalClassesTy::member_iterator NewSet =
            GlobalClasses.findLeader(GlobalClasses.insert(GTM));
        if (I == 1)
          CurSet = NewSet;
        else
          CurSet = GlobalClasses.unionSets(CurSet, NewSet);
      }

      GlobalClasses.unionSets(
          CurSet, GlobalClasses.findLeader(
                      GlobalClasses.insert(ICallBranchFunnel::create(
                          Alloc, CI, Targets, ++CurUniqueId))));
    }
  }

  if (ExportSummary) {
    DenseMap<GlobalValue::GUID, TinyPtrVector<Metadata *>> MetadataByGUID;
    for (auto &P : TypeIdInfo) {
      if (auto *TypeId = dyn_cast<MDString>(P.first))
        MetadataByGUID[GlobalValue::getGUIDAssumingExternalLinkage(
                           TypeId->getString())]
            .push_back(TypeId);
    }

    for (auto &P : *ExportSummary) {
      for (auto &S : P.second.SummaryList) {
        if (!ExportSummary->isGlobalValueLive(S.get()))
          continue;
        if (auto *FS = dyn_cast<FunctionSummary>(S->getBaseObject()))
          for (GlobalValue::GUID G : FS->type_tests())
            for (Metadata *MD : MetadataByGUID[G])
              AddTypeIdUse(MD).IsExported = true;
      }
    }
  }

  if (GlobalClasses.empty())
    return false;

  {
    ScopedSaveAliaseesAndUsed S(M);
    // For each disjoint set we found...
    for (const auto &C : GlobalClasses) {
      if (!C->isLeader())
        continue;

      ++NumTypeIdDisjointSets;
      // Build the list of type identifiers in this disjoint set.
      std::vector<Metadata *> TypeIds;
      std::vector<GlobalTypeMember *> Globals;
      std::vector<ICallBranchFunnel *> ICallBranchFunnels;
      for (auto M : GlobalClasses.members(*C)) {
        if (isa<Metadata *>(M))
          TypeIds.push_back(cast<Metadata *>(M));
        else if (isa<GlobalTypeMember *>(M))
          Globals.push_back(cast<GlobalTypeMember *>(M));
        else
          ICallBranchFunnels.push_back(cast<ICallBranchFunnel *>(M));
      }

      // Order type identifiers by unique ID for determinism. This ordering is
      // stable as there is a one-to-one mapping between metadata and unique
      // IDs.
      llvm::sort(TypeIds, [&](Metadata *M1, Metadata *M2) {
        return TypeIdInfo[M1].UniqueId < TypeIdInfo[M2].UniqueId;
      });

      // Same for the branch funnels.
      llvm::sort(ICallBranchFunnels,
                 [&](ICallBranchFunnel *F1, ICallBranchFunnel *F2) {
                   return F1->UniqueId < F2->UniqueId;
                 });

      // Build bitsets for this disjoint set.
      buildBitSetsFromDisjointSet(TypeIds, Globals, ICallBranchFunnels);
    }
  }

  allocateByteArrays();

  for (auto A : AliasesToCreate) {
    auto *Target = M.getNamedValue(A.TargetName);
    if (!isa<GlobalAlias>(Target))
      continue;
    auto *AliasGA = GlobalAlias::create("", Target);
    AliasGA->setVisibility(A.Alias->getVisibility());
    AliasGA->setLinkage(A.Alias->getLinkage());
    AliasGA->takeName(A.Alias);
    A.Alias->replaceAllUsesWith(AliasGA);
    A.Alias->eraseFromParent();
  }

  // Emit .symver directives for exported functions, if they exist.
  if (ExportSummary) {
    if (NamedMDNode *SymversMD = M.getNamedMetadata("symvers")) {
      for (auto *Symver : SymversMD->operands()) {
        assert(Symver->getNumOperands() >= 2);
        StringRef SymbolName =
            cast<MDString>(Symver->getOperand(0))->getString();
        StringRef Alias = cast<MDString>(Symver->getOperand(1))->getString();

        if (!ExportedFunctions.count(SymbolName))
          continue;

        M.appendModuleInlineAsm(
            (llvm::Twine(".symver ") + SymbolName + ", " + Alias).str());
      }
    }
  }

  return true;
}

PreservedAnalyses LowerTypeTestsPass::run(Module &M,
                                          ModuleAnalysisManager &AM) {
  bool Changed;
  if (UseCommandLine)
    Changed = LowerTypeTestsModule::runForTesting(M, AM);
  else
    Changed =
        LowerTypeTestsModule(M, AM, ExportSummary, ImportSummary, DropTypeTests)
            .lower();
  if (!Changed)
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

PreservedAnalyses SimplifyTypeTestsPass::run(Module &M,
                                             ModuleAnalysisManager &AM) {
  bool Changed = false;
  // Figure out whether inlining has exposed a constant address to a lowered
  // type test, and remove the test if so and the address is known to pass the
  // test. Unfortunately this pass ends up needing to reverse engineer what
  // LowerTypeTests did; this is currently inherent to the design of ThinLTO
  // importing where LowerTypeTests needs to run at the start.
  //
  // We look for things like:
  //
  // sub (i64 ptrtoint (ptr @_Z2fpv to i64), i64 ptrtoint (ptr
  // @__typeid__ZTSFvvE_global_addr to i64))
  //
  // which gets replaced with 0 if _Z2fpv (more specifically _Z2fpv.cfi, the
  // function referred to by the jump table) is a member of the type _ZTSFvv, as
  // well as things like
  //
  // icmp eq ptr @_Z2fpv, @__typeid__ZTSFvvE_global_addr
  //
  // which gets replaced with true if _Z2fpv is a member.
  for (auto &GV : M.globals()) {
    if (!GV.getName().starts_with("__typeid_") ||
        !GV.getName().ends_with("_global_addr"))
      continue;
    // __typeid_foo_global_addr -> foo
    auto *MD = MDString::get(M.getContext(),
                             GV.getName().substr(9, GV.getName().size() - 21));
    auto MaySimplifyPtr = [&](Value *Ptr) {
      if (auto *GV = dyn_cast<GlobalValue>(Ptr))
        if (auto *CFIGV = M.getNamedValue((GV->getName() + ".cfi").str()))
          Ptr = CFIGV;
      return isKnownTypeIdMember(MD, M.getDataLayout(), Ptr, 0);
    };
    auto MaySimplifyInt = [&](Value *Op) {
      auto *PtrAsInt = dyn_cast<ConstantExpr>(Op);
      if (!PtrAsInt || PtrAsInt->getOpcode() != Instruction::PtrToInt)
        return false;
      return MaySimplifyPtr(PtrAsInt->getOperand(0));
    };
    for (User *U : make_early_inc_range(GV.users())) {
      if (auto *CI = dyn_cast<ICmpInst>(U)) {
        if (CI->getPredicate() == CmpInst::ICMP_EQ &&
            MaySimplifyPtr(CI->getOperand(0))) {
          // This is an equality comparison (TypeTestResolution::Single case in
          // lowerTypeTestCall). In this case we just replace the comparison
          // with true.
          CI->replaceAllUsesWith(ConstantInt::getTrue(M.getContext()));
          CI->eraseFromParent();
          Changed = true;
          continue;
        }
      }
      auto *CE = dyn_cast<ConstantExpr>(U);
      if (!CE || CE->getOpcode() != Instruction::PtrToInt)
        continue;
      for (Use &U : make_early_inc_range(CE->uses())) {
        auto *CE = dyn_cast<ConstantExpr>(U.getUser());
        if (U.getOperandNo() == 0 && CE &&
            CE->getOpcode() == Instruction::Sub &&
            MaySimplifyInt(CE->getOperand(1))) {
          // This is a computation of PtrOffset as generated by
          // LowerTypeTestsModule::lowerTypeTestCall above. If
          // isKnownTypeIdMember passes we just pretend it evaluated to 0. This
          // should cause later passes to remove the range and alignment checks.
          // The bitset checks won't be removed but those are uncommon.
          CE->replaceAllUsesWith(ConstantInt::get(CE->getType(), 0));
          Changed = true;
        }
        auto *CI = dyn_cast<ICmpInst>(U.getUser());
        if (U.getOperandNo() == 1 && CI &&
            CI->getPredicate() == CmpInst::ICMP_EQ &&
            MaySimplifyInt(CI->getOperand(0))) {
          // This is an equality comparison. Unlike in the case above it
          // remained as an integer compare.
          CI->replaceAllUsesWith(ConstantInt::getTrue(M.getContext()));
          CI->eraseFromParent();
          Changed = true;
        }
      }
    }
  }

  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA = PreservedAnalyses::none();
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<PostDominatorTreeAnalysis>();
  PA.preserve<LoopAnalysis>();
  return PA;
}
