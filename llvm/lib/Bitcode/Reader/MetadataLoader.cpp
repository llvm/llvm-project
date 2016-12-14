//===- MetadataLoader.cpp - Internal BitcodeReader implementation ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MetadataLoader.h"
#include "ValueList.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Comdat.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GVMaterializer.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalIFunc.h"
#include "llvm/IR/GlobalIndirectSymbol.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/OperandTraits.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/TrackingMDRef.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

using namespace llvm;

namespace {

static int64_t unrotateSign(uint64_t U) { return U & 1 ? ~(U >> 1) : U >> 1; }

class BitcodeReaderMetadataList {
  unsigned NumFwdRefs;
  bool AnyFwdRefs;
  unsigned MinFwdRef;
  unsigned MaxFwdRef;

  /// Array of metadata references.
  ///
  /// Don't use std::vector here.  Some versions of libc++ copy (instead of
  /// move) on resize, and TrackingMDRef is very expensive to copy.
  SmallVector<TrackingMDRef, 1> MetadataPtrs;

  /// Structures for resolving old type refs.
  struct {
    SmallDenseMap<MDString *, TempMDTuple, 1> Unknown;
    SmallDenseMap<MDString *, DICompositeType *, 1> Final;
    SmallDenseMap<MDString *, DICompositeType *, 1> FwdDecls;
    SmallVector<std::pair<TrackingMDRef, TempMDTuple>, 1> Arrays;
  } OldTypeRefs;

  LLVMContext &Context;

public:
  BitcodeReaderMetadataList(LLVMContext &C)
      : NumFwdRefs(0), AnyFwdRefs(false), Context(C) {}

  // vector compatibility methods
  unsigned size() const { return MetadataPtrs.size(); }
  void resize(unsigned N) { MetadataPtrs.resize(N); }
  void push_back(Metadata *MD) { MetadataPtrs.emplace_back(MD); }
  void clear() { MetadataPtrs.clear(); }
  Metadata *back() const { return MetadataPtrs.back(); }
  void pop_back() { MetadataPtrs.pop_back(); }
  bool empty() const { return MetadataPtrs.empty(); }

  Metadata *operator[](unsigned i) const {
    assert(i < MetadataPtrs.size());
    return MetadataPtrs[i];
  }

  Metadata *lookup(unsigned I) const {
    if (I < MetadataPtrs.size())
      return MetadataPtrs[I];
    return nullptr;
  }

  void shrinkTo(unsigned N) {
    assert(N <= size() && "Invalid shrinkTo request!");
    assert(!AnyFwdRefs && "Unexpected forward refs");
    MetadataPtrs.resize(N);
  }

  /// Return the given metadata, creating a replaceable forward reference if
  /// necessary.
  Metadata *getMetadataFwdRef(unsigned Idx);

  /// Return the the given metadata only if it is fully resolved.
  ///
  /// Gives the same result as \a lookup(), unless \a MDNode::isResolved()
  /// would give \c false.
  Metadata *getMetadataIfResolved(unsigned Idx);

  MDNode *getMDNodeFwdRefOrNull(unsigned Idx);
  void assignValue(Metadata *MD, unsigned Idx);
  void tryToResolveCycles();
  bool hasFwdRefs() const { return AnyFwdRefs; }

  /// Upgrade a type that had an MDString reference.
  void addTypeRef(MDString &UUID, DICompositeType &CT);

  /// Upgrade a type that had an MDString reference.
  Metadata *upgradeTypeRef(Metadata *MaybeUUID);

  /// Upgrade a type ref array that may have MDString references.
  Metadata *upgradeTypeRefArray(Metadata *MaybeTuple);

private:
  Metadata *resolveTypeRefArray(Metadata *MaybeTuple);
};

void BitcodeReaderMetadataList::assignValue(Metadata *MD, unsigned Idx) {
  if (Idx == size()) {
    push_back(MD);
    return;
  }

  if (Idx >= size())
    resize(Idx + 1);

  TrackingMDRef &OldMD = MetadataPtrs[Idx];
  if (!OldMD) {
    OldMD.reset(MD);
    return;
  }

  // If there was a forward reference to this value, replace it.
  TempMDTuple PrevMD(cast<MDTuple>(OldMD.get()));
  PrevMD->replaceAllUsesWith(MD);
  --NumFwdRefs;
}

Metadata *BitcodeReaderMetadataList::getMetadataFwdRef(unsigned Idx) {
  if (Idx >= size())
    resize(Idx + 1);

  if (Metadata *MD = MetadataPtrs[Idx])
    return MD;

  // Track forward refs to be resolved later.
  if (AnyFwdRefs) {
    MinFwdRef = std::min(MinFwdRef, Idx);
    MaxFwdRef = std::max(MaxFwdRef, Idx);
  } else {
    AnyFwdRefs = true;
    MinFwdRef = MaxFwdRef = Idx;
  }
  ++NumFwdRefs;

  // Create and return a placeholder, which will later be RAUW'd.
  Metadata *MD = MDNode::getTemporary(Context, None).release();
  MetadataPtrs[Idx].reset(MD);
  return MD;
}

Metadata *BitcodeReaderMetadataList::getMetadataIfResolved(unsigned Idx) {
  Metadata *MD = lookup(Idx);
  if (auto *N = dyn_cast_or_null<MDNode>(MD))
    if (!N->isResolved())
      return nullptr;
  return MD;
}

MDNode *BitcodeReaderMetadataList::getMDNodeFwdRefOrNull(unsigned Idx) {
  return dyn_cast_or_null<MDNode>(getMetadataFwdRef(Idx));
}

void BitcodeReaderMetadataList::tryToResolveCycles() {
  if (NumFwdRefs)
    // Still forward references... can't resolve cycles.
    return;

  bool DidReplaceTypeRefs = false;

  // Give up on finding a full definition for any forward decls that remain.
  for (const auto &Ref : OldTypeRefs.FwdDecls)
    OldTypeRefs.Final.insert(Ref);
  OldTypeRefs.FwdDecls.clear();

  // Upgrade from old type ref arrays.  In strange cases, this could add to
  // OldTypeRefs.Unknown.
  for (const auto &Array : OldTypeRefs.Arrays) {
    DidReplaceTypeRefs = true;
    Array.second->replaceAllUsesWith(resolveTypeRefArray(Array.first.get()));
  }
  OldTypeRefs.Arrays.clear();

  // Replace old string-based type refs with the resolved node, if possible.
  // If we haven't seen the node, leave it to the verifier to complain about
  // the invalid string reference.
  for (const auto &Ref : OldTypeRefs.Unknown) {
    DidReplaceTypeRefs = true;
    if (DICompositeType *CT = OldTypeRefs.Final.lookup(Ref.first))
      Ref.second->replaceAllUsesWith(CT);
    else
      Ref.second->replaceAllUsesWith(Ref.first);
  }
  OldTypeRefs.Unknown.clear();

  // Make sure all the upgraded types are resolved.
  if (DidReplaceTypeRefs) {
    AnyFwdRefs = true;
    MinFwdRef = 0;
    MaxFwdRef = MetadataPtrs.size() - 1;
  }

  if (!AnyFwdRefs)
    // Nothing to do.
    return;

  // Resolve any cycles.
  for (unsigned I = MinFwdRef, E = MaxFwdRef + 1; I != E; ++I) {
    auto &MD = MetadataPtrs[I];
    auto *N = dyn_cast_or_null<MDNode>(MD);
    if (!N)
      continue;

    assert(!N->isTemporary() && "Unexpected forward reference");
    N->resolveCycles();
  }

  // Make sure we return early again until there's another forward ref.
  AnyFwdRefs = false;
}

void BitcodeReaderMetadataList::addTypeRef(MDString &UUID,
                                           DICompositeType &CT) {
  assert(CT.getRawIdentifier() == &UUID && "Mismatched UUID");
  if (CT.isForwardDecl())
    OldTypeRefs.FwdDecls.insert(std::make_pair(&UUID, &CT));
  else
    OldTypeRefs.Final.insert(std::make_pair(&UUID, &CT));
}

Metadata *BitcodeReaderMetadataList::upgradeTypeRef(Metadata *MaybeUUID) {
  auto *UUID = dyn_cast_or_null<MDString>(MaybeUUID);
  if (LLVM_LIKELY(!UUID))
    return MaybeUUID;

  if (auto *CT = OldTypeRefs.Final.lookup(UUID))
    return CT;

  auto &Ref = OldTypeRefs.Unknown[UUID];
  if (!Ref)
    Ref = MDNode::getTemporary(Context, None);
  return Ref.get();
}

Metadata *BitcodeReaderMetadataList::upgradeTypeRefArray(Metadata *MaybeTuple) {
  auto *Tuple = dyn_cast_or_null<MDTuple>(MaybeTuple);
  if (!Tuple || Tuple->isDistinct())
    return MaybeTuple;

  // Look through the array immediately if possible.
  if (!Tuple->isTemporary())
    return resolveTypeRefArray(Tuple);

  // Create and return a placeholder to use for now.  Eventually
  // resolveTypeRefArrays() will be resolve this forward reference.
  OldTypeRefs.Arrays.emplace_back(
      std::piecewise_construct, std::forward_as_tuple(Tuple),
      std::forward_as_tuple(MDTuple::getTemporary(Context, None)));
  return OldTypeRefs.Arrays.back().second.get();
}

Metadata *BitcodeReaderMetadataList::resolveTypeRefArray(Metadata *MaybeTuple) {
  auto *Tuple = dyn_cast_or_null<MDTuple>(MaybeTuple);
  if (!Tuple || Tuple->isDistinct())
    return MaybeTuple;

  // Look through the DITypeRefArray, upgrading each DITypeRef.
  SmallVector<Metadata *, 32> Ops;
  Ops.reserve(Tuple->getNumOperands());
  for (Metadata *MD : Tuple->operands())
    Ops.push_back(upgradeTypeRef(MD));

  return MDTuple::get(Context, Ops);
}

namespace {

class PlaceholderQueue {
  // Placeholders would thrash around when moved, so store in a std::deque
  // instead of some sort of vector.
  std::deque<DistinctMDOperandPlaceholder> PHs;

public:
  DistinctMDOperandPlaceholder &getPlaceholderOp(unsigned ID);
  void flush(BitcodeReaderMetadataList &MetadataList);
};

} // end anonymous namespace

DistinctMDOperandPlaceholder &PlaceholderQueue::getPlaceholderOp(unsigned ID) {
  PHs.emplace_back(ID);
  return PHs.back();
}

void PlaceholderQueue::flush(BitcodeReaderMetadataList &MetadataList) {
  while (!PHs.empty()) {
    PHs.front().replaceUseWith(
        MetadataList.getMetadataFwdRef(PHs.front().getID()));
    PHs.pop_front();
  }
}

} // anonynous namespace

class MetadataLoader::MetadataLoaderImpl {
  BitcodeReaderMetadataList MetadataList;
  BitcodeReaderValueList &ValueList;
  BitstreamCursor &Stream;
  LLVMContext &Context;
  Module &TheModule;
  std::function<Type *(unsigned)> getTypeByID;

  /// Functions that need to be matched with subprograms when upgrading old
  /// metadata.
  SmallDenseMap<Function *, DISubprogram *, 16> FunctionsWithSPs;

  // Map the bitcode's custom MDKind ID to the Module's MDKind ID.
  DenseMap<unsigned, unsigned> MDKindMap;

  bool HasSeenOldLoopTags = false;

  Error parseMetadataStrings(ArrayRef<uint64_t> Record, StringRef Blob,
                             unsigned &NextMetadataNo);
  Error parseGlobalObjectAttachment(GlobalObject &GO,
                                    ArrayRef<uint64_t> Record);
  Error parseMetadataKindRecord(SmallVectorImpl<uint64_t> &Record);

public:
  MetadataLoaderImpl(BitstreamCursor &Stream, Module &TheModule,
                     BitcodeReaderValueList &ValueList,
                     std::function<Type *(unsigned)> getTypeByID)
      : MetadataList(TheModule.getContext()), ValueList(ValueList),
        Stream(Stream), Context(TheModule.getContext()), TheModule(TheModule),
        getTypeByID(getTypeByID) {}

  Error parseMetadata(bool ModuleLevel);

  bool hasFwdRefs() const { return MetadataList.hasFwdRefs(); }
  Metadata *getMetadataFwdRef(unsigned Idx) {
    return MetadataList.getMetadataFwdRef(Idx);
  }

  MDNode *getMDNodeFwdRefOrNull(unsigned Idx) {
    return MetadataList.getMDNodeFwdRefOrNull(Idx);
  }

  DISubprogram *lookupSubprogramForFunction(Function *F) {
    return FunctionsWithSPs.lookup(F);
  }

  bool hasSeenOldLoopTags() { return HasSeenOldLoopTags; }

  Error parseMetadataAttachment(
      Function &F, const SmallVectorImpl<Instruction *> &InstructionList);

  Error parseMetadataKinds();

  unsigned size() const { return MetadataList.size(); }
  void shrinkTo(unsigned N) { MetadataList.shrinkTo(N); }
};

Error error(const Twine &Message) {
  return make_error<StringError>(
      Message, make_error_code(BitcodeError::CorruptedBitcode));
}

/// Parse a METADATA_BLOCK. If ModuleLevel is true then we are parsing
/// module level metadata.
Error MetadataLoader::MetadataLoaderImpl::parseMetadata(bool ModuleLevel) {
  if (!ModuleLevel && MetadataList.hasFwdRefs())
    return error("Invalid metadata: fwd refs into function blocks");

  if (Stream.EnterSubBlock(bitc::METADATA_BLOCK_ID))
    return error("Invalid record");

  unsigned NextMetadataNo = MetadataList.size();
  std::vector<std::pair<DICompileUnit *, Metadata *>> CUSubprograms;
  SmallVector<uint64_t, 64> Record;

  PlaceholderQueue Placeholders;
  bool IsDistinct;
  auto getMD = [&](unsigned ID) -> Metadata * {
    if (!IsDistinct)
      return MetadataList.getMetadataFwdRef(ID);
    if (auto *MD = MetadataList.getMetadataIfResolved(ID))
      return MD;
    return &Placeholders.getPlaceholderOp(ID);
  };
  auto getMDOrNull = [&](unsigned ID) -> Metadata * {
    if (ID)
      return getMD(ID - 1);
    return nullptr;
  };
  auto getMDOrNullWithoutPlaceholders = [&](unsigned ID) -> Metadata * {
    if (ID)
      return MetadataList.getMetadataFwdRef(ID - 1);
    return nullptr;
  };
  auto getMDString = [&](unsigned ID) -> MDString * {
    // This requires that the ID is not really a forward reference.  In
    // particular, the MDString must already have been resolved.
    return cast_or_null<MDString>(getMDOrNull(ID));
  };

  // Support for old type refs.
  auto getDITypeRefOrNull = [&](unsigned ID) {
    return MetadataList.upgradeTypeRef(getMDOrNull(ID));
  };

#define GET_OR_DISTINCT(CLASS, ARGS)                                           \
  (IsDistinct ? CLASS::getDistinct ARGS : CLASS::get ARGS)

  // Read all the records.
  while (true) {
    BitstreamEntry Entry = Stream.advanceSkippingSubblocks();

    switch (Entry.Kind) {
    case BitstreamEntry::SubBlock: // Handled for us already.
    case BitstreamEntry::Error:
      return error("Malformed block");
    case BitstreamEntry::EndBlock:
      // Upgrade old-style CU <-> SP pointers to point from SP to CU.
      for (auto CU_SP : CUSubprograms)
        if (auto *SPs = dyn_cast_or_null<MDTuple>(CU_SP.second))
          for (auto &Op : SPs->operands())
            if (auto *SP = dyn_cast_or_null<MDNode>(Op))
              SP->replaceOperandWith(7, CU_SP.first);

      MetadataList.tryToResolveCycles();
      Placeholders.flush(MetadataList);
      return Error::success();
    case BitstreamEntry::Record:
      // The interesting case.
      break;
    }

    // Read a record.
    Record.clear();
    StringRef Blob;
    unsigned Code = Stream.readRecord(Entry.ID, Record, &Blob);
    IsDistinct = false;
    switch (Code) {
    default: // Default behavior: ignore.
      break;
    case bitc::METADATA_NAME: {
      // Read name of the named metadata.
      SmallString<8> Name(Record.begin(), Record.end());
      Record.clear();
      Code = Stream.ReadCode();

      unsigned NextBitCode = Stream.readRecord(Code, Record);
      if (NextBitCode != bitc::METADATA_NAMED_NODE)
        return error("METADATA_NAME not followed by METADATA_NAMED_NODE");

      // Read named metadata elements.
      unsigned Size = Record.size();
      NamedMDNode *NMD = TheModule.getOrInsertNamedMetadata(Name);
      for (unsigned i = 0; i != Size; ++i) {
        MDNode *MD = MetadataList.getMDNodeFwdRefOrNull(Record[i]);
        if (!MD)
          return error("Invalid record");
        NMD->addOperand(MD);
      }
      break;
    }
    case bitc::METADATA_OLD_FN_NODE: {
      // FIXME: Remove in 4.0.
      // This is a LocalAsMetadata record, the only type of function-local
      // metadata.
      if (Record.size() % 2 == 1)
        return error("Invalid record");

      // If this isn't a LocalAsMetadata record, we're dropping it.  This used
      // to be legal, but there's no upgrade path.
      auto dropRecord = [&] {
        MetadataList.assignValue(MDNode::get(Context, None), NextMetadataNo++);
      };
      if (Record.size() != 2) {
        dropRecord();
        break;
      }

      Type *Ty = getTypeByID(Record[0]);
      if (Ty->isMetadataTy() || Ty->isVoidTy()) {
        dropRecord();
        break;
      }

      MetadataList.assignValue(
          LocalAsMetadata::get(ValueList.getValueFwdRef(Record[1], Ty)),
          NextMetadataNo++);
      break;
    }
    case bitc::METADATA_OLD_NODE: {
      // FIXME: Remove in 4.0.
      if (Record.size() % 2 == 1)
        return error("Invalid record");

      unsigned Size = Record.size();
      SmallVector<Metadata *, 8> Elts;
      for (unsigned i = 0; i != Size; i += 2) {
        Type *Ty = getTypeByID(Record[i]);
        if (!Ty)
          return error("Invalid record");
        if (Ty->isMetadataTy())
          Elts.push_back(getMD(Record[i + 1]));
        else if (!Ty->isVoidTy()) {
          auto *MD =
              ValueAsMetadata::get(ValueList.getValueFwdRef(Record[i + 1], Ty));
          assert(isa<ConstantAsMetadata>(MD) &&
                 "Expected non-function-local metadata");
          Elts.push_back(MD);
        } else
          Elts.push_back(nullptr);
      }
      MetadataList.assignValue(MDNode::get(Context, Elts), NextMetadataNo++);
      break;
    }
    case bitc::METADATA_VALUE: {
      if (Record.size() != 2)
        return error("Invalid record");

      Type *Ty = getTypeByID(Record[0]);
      if (Ty->isMetadataTy() || Ty->isVoidTy())
        return error("Invalid record");

      MetadataList.assignValue(
          ValueAsMetadata::get(ValueList.getValueFwdRef(Record[1], Ty)),
          NextMetadataNo++);
      break;
    }
    case bitc::METADATA_DISTINCT_NODE:
      IsDistinct = true;
      LLVM_FALLTHROUGH;
    case bitc::METADATA_NODE: {
      SmallVector<Metadata *, 8> Elts;
      Elts.reserve(Record.size());
      for (unsigned ID : Record)
        Elts.push_back(getMDOrNull(ID));
      MetadataList.assignValue(IsDistinct ? MDNode::getDistinct(Context, Elts)
                                          : MDNode::get(Context, Elts),
                               NextMetadataNo++);
      break;
    }
    case bitc::METADATA_LOCATION: {
      if (Record.size() != 5)
        return error("Invalid record");

      IsDistinct = Record[0];
      unsigned Line = Record[1];
      unsigned Column = Record[2];
      Metadata *Scope = getMD(Record[3]);
      Metadata *InlinedAt = getMDOrNull(Record[4]);
      MetadataList.assignValue(
          GET_OR_DISTINCT(DILocation,
                          (Context, Line, Column, Scope, InlinedAt)),
          NextMetadataNo++);
      break;
    }
    case bitc::METADATA_GENERIC_DEBUG: {
      if (Record.size() < 4)
        return error("Invalid record");

      IsDistinct = Record[0];
      unsigned Tag = Record[1];
      unsigned Version = Record[2];

      if (Tag >= 1u << 16 || Version != 0)
        return error("Invalid record");

      auto *Header = getMDString(Record[3]);
      SmallVector<Metadata *, 8> DwarfOps;
      for (unsigned I = 4, E = Record.size(); I != E; ++I)
        DwarfOps.push_back(getMDOrNull(Record[I]));
      MetadataList.assignValue(
          GET_OR_DISTINCT(GenericDINode, (Context, Tag, Header, DwarfOps)),
          NextMetadataNo++);
      break;
    }
    case bitc::METADATA_SUBRANGE: {
      if (Record.size() != 3)
        return error("Invalid record");

      IsDistinct = Record[0];
      MetadataList.assignValue(
          GET_OR_DISTINCT(DISubrange,
                          (Context, Record[1], unrotateSign(Record[2]))),
          NextMetadataNo++);
      break;
    }
    case bitc::METADATA_ENUMERATOR: {
      if (Record.size() != 3)
        return error("Invalid record");

      IsDistinct = Record[0];
      MetadataList.assignValue(
          GET_OR_DISTINCT(DIEnumerator, (Context, unrotateSign(Record[1]),
                                         getMDString(Record[2]))),
          NextMetadataNo++);
      break;
    }
    case bitc::METADATA_BASIC_TYPE: {
      if (Record.size() != 6)
        return error("Invalid record");

      IsDistinct = Record[0];
      MetadataList.assignValue(
          GET_OR_DISTINCT(DIBasicType,
                          (Context, Record[1], getMDString(Record[2]),
                           Record[3], Record[4], Record[5])),
          NextMetadataNo++);
      break;
    }
    case bitc::METADATA_DERIVED_TYPE: {
      if (Record.size() != 12)
        return error("Invalid record");

      IsDistinct = Record[0];
      DINode::DIFlags Flags = static_cast<DINode::DIFlags>(Record[10]);
      MetadataList.assignValue(
          GET_OR_DISTINCT(DIDerivedType,
                          (Context, Record[1], getMDString(Record[2]),
                           getMDOrNull(Record[3]), Record[4],
                           getDITypeRefOrNull(Record[5]),
                           getDITypeRefOrNull(Record[6]), Record[7], Record[8],
                           Record[9], Flags, getDITypeRefOrNull(Record[11]))),
          NextMetadataNo++);
      break;
    }
    case bitc::METADATA_COMPOSITE_TYPE: {
      if (Record.size() != 16)
        return error("Invalid record");

      // If we have a UUID and this is not a forward declaration, lookup the
      // mapping.
      IsDistinct = Record[0] & 0x1;
      bool IsNotUsedInTypeRef = Record[0] >= 2;
      unsigned Tag = Record[1];
      MDString *Name = getMDString(Record[2]);
      Metadata *File = getMDOrNull(Record[3]);
      unsigned Line = Record[4];
      Metadata *Scope = getDITypeRefOrNull(Record[5]);
      Metadata *BaseType = getDITypeRefOrNull(Record[6]);
      uint64_t SizeInBits = Record[7];
      if (Record[8] > (uint64_t)std::numeric_limits<uint32_t>::max())
        return error("Alignment value is too large");
      uint32_t AlignInBits = Record[8];
      uint64_t OffsetInBits = Record[9];
      DINode::DIFlags Flags = static_cast<DINode::DIFlags>(Record[10]);
      Metadata *Elements = getMDOrNull(Record[11]);
      unsigned RuntimeLang = Record[12];
      Metadata *VTableHolder = getDITypeRefOrNull(Record[13]);
      Metadata *TemplateParams = getMDOrNull(Record[14]);
      auto *Identifier = getMDString(Record[15]);
      DICompositeType *CT = nullptr;
      if (Identifier)
        CT = DICompositeType::buildODRType(
            Context, *Identifier, Tag, Name, File, Line, Scope, BaseType,
            SizeInBits, AlignInBits, OffsetInBits, Flags, Elements, RuntimeLang,
            VTableHolder, TemplateParams);

      // Create a node if we didn't get a lazy ODR type.
      if (!CT)
        CT = GET_OR_DISTINCT(DICompositeType,
                             (Context, Tag, Name, File, Line, Scope, BaseType,
                              SizeInBits, AlignInBits, OffsetInBits, Flags,
                              Elements, RuntimeLang, VTableHolder,
                              TemplateParams, Identifier));
      if (!IsNotUsedInTypeRef && Identifier)
        MetadataList.addTypeRef(*Identifier, *cast<DICompositeType>(CT));

      MetadataList.assignValue(CT, NextMetadataNo++);
      break;
    }
    case bitc::METADATA_SUBROUTINE_TYPE: {
      if (Record.size() < 3 || Record.size() > 4)
        return error("Invalid record");
      bool IsOldTypeRefArray = Record[0] < 2;
      unsigned CC = (Record.size() > 3) ? Record[3] : 0;

      IsDistinct = Record[0] & 0x1;
      DINode::DIFlags Flags = static_cast<DINode::DIFlags>(Record[1]);
      Metadata *Types = getMDOrNull(Record[2]);
      if (LLVM_UNLIKELY(IsOldTypeRefArray))
        Types = MetadataList.upgradeTypeRefArray(Types);

      MetadataList.assignValue(
          GET_OR_DISTINCT(DISubroutineType, (Context, Flags, CC, Types)),
          NextMetadataNo++);
      break;
    }

    case bitc::METADATA_MODULE: {
      if (Record.size() != 6)
        return error("Invalid record");

      IsDistinct = Record[0];
      MetadataList.assignValue(
          GET_OR_DISTINCT(DIModule,
                          (Context, getMDOrNull(Record[1]),
                           getMDString(Record[2]), getMDString(Record[3]),
                           getMDString(Record[4]), getMDString(Record[5]))),
          NextMetadataNo++);
      break;
    }

    case bitc::METADATA_FILE: {
      if (Record.size() != 3)
        return error("Invalid record");

      IsDistinct = Record[0];
      MetadataList.assignValue(
          GET_OR_DISTINCT(DIFile, (Context, getMDString(Record[1]),
                                   getMDString(Record[2]))),
          NextMetadataNo++);
      break;
    }
    case bitc::METADATA_COMPILE_UNIT: {
      if (Record.size() < 14 || Record.size() > 17)
        return error("Invalid record");

      // Ignore Record[0], which indicates whether this compile unit is
      // distinct.  It's always distinct.
      IsDistinct = true;
      auto *CU = DICompileUnit::getDistinct(
          Context, Record[1], getMDOrNull(Record[2]), getMDString(Record[3]),
          Record[4], getMDString(Record[5]), Record[6], getMDString(Record[7]),
          Record[8], getMDOrNull(Record[9]), getMDOrNull(Record[10]),
          getMDOrNull(Record[12]), getMDOrNull(Record[13]),
          Record.size() <= 15 ? nullptr : getMDOrNull(Record[15]),
          Record.size() <= 14 ? 0 : Record[14],
          Record.size() <= 16 ? true : Record[16]);

      MetadataList.assignValue(CU, NextMetadataNo++);

      // Move the Upgrade the list of subprograms.
      if (Metadata *SPs = getMDOrNullWithoutPlaceholders(Record[11]))
        CUSubprograms.push_back({CU, SPs});
      break;
    }
    case bitc::METADATA_SUBPROGRAM: {
      if (Record.size() < 18 || Record.size() > 20)
        return error("Invalid record");

      IsDistinct =
          (Record[0] & 1) || Record[8]; // All definitions should be distinct.
      // Version 1 has a Function as Record[15].
      // Version 2 has removed Record[15].
      // Version 3 has the Unit as Record[15].
      // Version 4 added thisAdjustment.
      bool HasUnit = Record[0] >= 2;
      if (HasUnit && Record.size() < 19)
        return error("Invalid record");
      Metadata *CUorFn = getMDOrNull(Record[15]);
      unsigned Offset = Record.size() >= 19 ? 1 : 0;
      bool HasFn = Offset && !HasUnit;
      bool HasThisAdj = Record.size() >= 20;
      DISubprogram *SP = GET_OR_DISTINCT(
          DISubprogram, (Context,
                         getDITypeRefOrNull(Record[1]),  // scope
                         getMDString(Record[2]),         // name
                         getMDString(Record[3]),         // linkageName
                         getMDOrNull(Record[4]),         // file
                         Record[5],                      // line
                         getMDOrNull(Record[6]),         // type
                         Record[7],                      // isLocal
                         Record[8],                      // isDefinition
                         Record[9],                      // scopeLine
                         getDITypeRefOrNull(Record[10]), // containingType
                         Record[11],                     // virtuality
                         Record[12],                     // virtualIndex
                         HasThisAdj ? Record[19] : 0,    // thisAdjustment
                         static_cast<DINode::DIFlags>(Record[13] // flags
                                                      ),
                         Record[14],                       // isOptimized
                         HasUnit ? CUorFn : nullptr,       // unit
                         getMDOrNull(Record[15 + Offset]), // templateParams
                         getMDOrNull(Record[16 + Offset]), // declaration
                         getMDOrNull(Record[17 + Offset])  // variables
                         ));
      MetadataList.assignValue(SP, NextMetadataNo++);

      // Upgrade sp->function mapping to function->sp mapping.
      if (HasFn) {
        if (auto *CMD = dyn_cast_or_null<ConstantAsMetadata>(CUorFn))
          if (auto *F = dyn_cast<Function>(CMD->getValue())) {
            if (F->isMaterializable())
              // Defer until materialized; unmaterialized functions may not have
              // metadata.
              FunctionsWithSPs[F] = SP;
            else if (!F->empty())
              F->setSubprogram(SP);
          }
      }
      break;
    }
    case bitc::METADATA_LEXICAL_BLOCK: {
      if (Record.size() != 5)
        return error("Invalid record");

      IsDistinct = Record[0];
      MetadataList.assignValue(
          GET_OR_DISTINCT(DILexicalBlock,
                          (Context, getMDOrNull(Record[1]),
                           getMDOrNull(Record[2]), Record[3], Record[4])),
          NextMetadataNo++);
      break;
    }
    case bitc::METADATA_LEXICAL_BLOCK_FILE: {
      if (Record.size() != 4)
        return error("Invalid record");

      IsDistinct = Record[0];
      MetadataList.assignValue(
          GET_OR_DISTINCT(DILexicalBlockFile,
                          (Context, getMDOrNull(Record[1]),
                           getMDOrNull(Record[2]), Record[3])),
          NextMetadataNo++);
      break;
    }
    case bitc::METADATA_NAMESPACE: {
      if (Record.size() != 5)
        return error("Invalid record");

      IsDistinct = Record[0] & 1;
      bool ExportSymbols = Record[0] & 2;
      MetadataList.assignValue(
          GET_OR_DISTINCT(DINamespace,
                          (Context, getMDOrNull(Record[1]),
                           getMDOrNull(Record[2]), getMDString(Record[3]),
                           Record[4], ExportSymbols)),
          NextMetadataNo++);
      break;
    }
    case bitc::METADATA_MACRO: {
      if (Record.size() != 5)
        return error("Invalid record");

      IsDistinct = Record[0];
      MetadataList.assignValue(
          GET_OR_DISTINCT(DIMacro,
                          (Context, Record[1], Record[2],
                           getMDString(Record[3]), getMDString(Record[4]))),
          NextMetadataNo++);
      break;
    }
    case bitc::METADATA_MACRO_FILE: {
      if (Record.size() != 5)
        return error("Invalid record");

      IsDistinct = Record[0];
      MetadataList.assignValue(
          GET_OR_DISTINCT(DIMacroFile,
                          (Context, Record[1], Record[2],
                           getMDOrNull(Record[3]), getMDOrNull(Record[4]))),
          NextMetadataNo++);
      break;
    }
    case bitc::METADATA_TEMPLATE_TYPE: {
      if (Record.size() != 3)
        return error("Invalid record");

      IsDistinct = Record[0];
      MetadataList.assignValue(GET_OR_DISTINCT(DITemplateTypeParameter,
                                               (Context, getMDString(Record[1]),
                                                getDITypeRefOrNull(Record[2]))),
                               NextMetadataNo++);
      break;
    }
    case bitc::METADATA_TEMPLATE_VALUE: {
      if (Record.size() != 5)
        return error("Invalid record");

      IsDistinct = Record[0];
      MetadataList.assignValue(
          GET_OR_DISTINCT(DITemplateValueParameter,
                          (Context, Record[1], getMDString(Record[2]),
                           getDITypeRefOrNull(Record[3]),
                           getMDOrNull(Record[4]))),
          NextMetadataNo++);
      break;
    }
    case bitc::METADATA_GLOBAL_VAR: {
      if (Record.size() < 11 || Record.size() > 12)
        return error("Invalid record");

      IsDistinct = Record[0];

      // Upgrade old metadata, which stored a global variable reference or a
      // ConstantInt here.
      Metadata *Expr = getMDOrNull(Record[9]);
      uint32_t AlignInBits = 0;
      if (Record.size() > 11) {
        if (Record[11] > (uint64_t)std::numeric_limits<uint32_t>::max())
          return error("Alignment value is too large");
        AlignInBits = Record[11];
      }
      GlobalVariable *Attach = nullptr;
      if (auto *CMD = dyn_cast_or_null<ConstantAsMetadata>(Expr)) {
        if (auto *GV = dyn_cast<GlobalVariable>(CMD->getValue())) {
          Attach = GV;
          Expr = nullptr;
        } else if (auto *CI = dyn_cast<ConstantInt>(CMD->getValue())) {
          Expr = DIExpression::get(Context,
                                   {dwarf::DW_OP_constu, CI->getZExtValue(),
                                    dwarf::DW_OP_stack_value});
        } else {
          Expr = nullptr;
        }
      }

      DIGlobalVariable *DGV = GET_OR_DISTINCT(
          DIGlobalVariable,
          (Context, getMDOrNull(Record[1]), getMDString(Record[2]),
           getMDString(Record[3]), getMDOrNull(Record[4]), Record[5],
           getDITypeRefOrNull(Record[6]), Record[7], Record[8], Expr,
           getMDOrNull(Record[10]), AlignInBits));
      MetadataList.assignValue(DGV, NextMetadataNo++);

      if (Attach)
        Attach->addDebugInfo(DGV);

      break;
    }
    case bitc::METADATA_LOCAL_VAR: {
      // 10th field is for the obseleted 'inlinedAt:' field.
      if (Record.size() < 8 || Record.size() > 10)
        return error("Invalid record");

      IsDistinct = Record[0] & 1;
      bool HasAlignment = Record[0] & 2;
      // 2nd field used to be an artificial tag, either DW_TAG_auto_variable or
      // DW_TAG_arg_variable, if we have alignment flag encoded it means, that
      // this is newer version of record which doesn't have artifical tag.
      bool HasTag = !HasAlignment && Record.size() > 8;
      DINode::DIFlags Flags = static_cast<DINode::DIFlags>(Record[7 + HasTag]);
      uint32_t AlignInBits = 0;
      if (HasAlignment) {
        if (Record[8 + HasTag] > (uint64_t)std::numeric_limits<uint32_t>::max())
          return error("Alignment value is too large");
        AlignInBits = Record[8 + HasTag];
      }
      MetadataList.assignValue(
          GET_OR_DISTINCT(DILocalVariable,
                          (Context, getMDOrNull(Record[1 + HasTag]),
                           getMDString(Record[2 + HasTag]),
                           getMDOrNull(Record[3 + HasTag]), Record[4 + HasTag],
                           getDITypeRefOrNull(Record[5 + HasTag]),
                           Record[6 + HasTag], Flags, AlignInBits)),
          NextMetadataNo++);
      break;
    }
    case bitc::METADATA_EXPRESSION: {
      if (Record.size() < 1)
        return error("Invalid record");

      IsDistinct = Record[0] & 1;
      bool HasOpFragment = Record[0] & 2;
      auto Elts = MutableArrayRef<uint64_t>(Record).slice(1);
      if (!HasOpFragment)
        if (unsigned N = Elts.size())
          if (N >= 3 && Elts[N - 3] == dwarf::DW_OP_bit_piece)
            Elts[N - 3] = dwarf::DW_OP_LLVM_fragment;

      MetadataList.assignValue(
          GET_OR_DISTINCT(DIExpression,
                          (Context, makeArrayRef(Record).slice(1))),
          NextMetadataNo++);
      break;
    }
    case bitc::METADATA_OBJC_PROPERTY: {
      if (Record.size() != 8)
        return error("Invalid record");

      IsDistinct = Record[0];
      MetadataList.assignValue(
          GET_OR_DISTINCT(DIObjCProperty,
                          (Context, getMDString(Record[1]),
                           getMDOrNull(Record[2]), Record[3],
                           getMDString(Record[4]), getMDString(Record[5]),
                           Record[6], getDITypeRefOrNull(Record[7]))),
          NextMetadataNo++);
      break;
    }
    case bitc::METADATA_IMPORTED_ENTITY: {
      if (Record.size() != 6)
        return error("Invalid record");

      IsDistinct = Record[0];
      MetadataList.assignValue(
          GET_OR_DISTINCT(DIImportedEntity,
                          (Context, Record[1], getMDOrNull(Record[2]),
                           getDITypeRefOrNull(Record[3]), Record[4],
                           getMDString(Record[5]))),
          NextMetadataNo++);
      break;
    }
    case bitc::METADATA_STRING_OLD: {
      std::string String(Record.begin(), Record.end());

      // Test for upgrading !llvm.loop.
      HasSeenOldLoopTags |= mayBeOldLoopAttachmentTag(String);

      Metadata *MD = MDString::get(Context, String);
      MetadataList.assignValue(MD, NextMetadataNo++);
      break;
    }
    case bitc::METADATA_STRINGS:
      if (Error Err = parseMetadataStrings(Record, Blob, NextMetadataNo))
        return Err;
      break;
    case bitc::METADATA_GLOBAL_DECL_ATTACHMENT: {
      if (Record.size() % 2 == 0)
        return error("Invalid record");
      unsigned ValueID = Record[0];
      if (ValueID >= ValueList.size())
        return error("Invalid record");
      if (auto *GO = dyn_cast<GlobalObject>(ValueList[ValueID]))
        if (Error Err = parseGlobalObjectAttachment(
                *GO, ArrayRef<uint64_t>(Record).slice(1)))
          return Err;
      break;
    }
    case bitc::METADATA_KIND: {
      // Support older bitcode files that had METADATA_KIND records in a
      // block with METADATA_BLOCK_ID.
      if (Error Err = parseMetadataKindRecord(Record))
        return Err;
      break;
    }
    }
  }
#undef GET_OR_DISTINCT
}

Error MetadataLoader::MetadataLoaderImpl::parseMetadataStrings(
    ArrayRef<uint64_t> Record, StringRef Blob, unsigned &NextMetadataNo) {
  // All the MDStrings in the block are emitted together in a single
  // record.  The strings are concatenated and stored in a blob along with
  // their sizes.
  if (Record.size() != 2)
    return error("Invalid record: metadata strings layout");

  unsigned NumStrings = Record[0];
  unsigned StringsOffset = Record[1];
  if (!NumStrings)
    return error("Invalid record: metadata strings with no strings");
  if (StringsOffset > Blob.size())
    return error("Invalid record: metadata strings corrupt offset");

  StringRef Lengths = Blob.slice(0, StringsOffset);
  SimpleBitstreamCursor R(Lengths);

  StringRef Strings = Blob.drop_front(StringsOffset);
  do {
    if (R.AtEndOfStream())
      return error("Invalid record: metadata strings bad length");

    unsigned Size = R.ReadVBR(6);
    if (Strings.size() < Size)
      return error("Invalid record: metadata strings truncated chars");

    MetadataList.assignValue(MDString::get(Context, Strings.slice(0, Size)),
                             NextMetadataNo++);
    Strings = Strings.drop_front(Size);
  } while (--NumStrings);

  return Error::success();
}

Error MetadataLoader::MetadataLoaderImpl::parseGlobalObjectAttachment(
    GlobalObject &GO, ArrayRef<uint64_t> Record) {
  assert(Record.size() % 2 == 0);
  for (unsigned I = 0, E = Record.size(); I != E; I += 2) {
    auto K = MDKindMap.find(Record[I]);
    if (K == MDKindMap.end())
      return error("Invalid ID");
    MDNode *MD = MetadataList.getMDNodeFwdRefOrNull(Record[I + 1]);
    if (!MD)
      return error("Invalid metadata attachment");
    GO.addMetadata(K->second, *MD);
  }
  return Error::success();
}

/// Parse metadata attachments.
Error MetadataLoader::MetadataLoaderImpl::parseMetadataAttachment(
    Function &F, const SmallVectorImpl<Instruction *> &InstructionList) {
  if (Stream.EnterSubBlock(bitc::METADATA_ATTACHMENT_ID))
    return error("Invalid record");

  SmallVector<uint64_t, 64> Record;

  while (true) {
    BitstreamEntry Entry = Stream.advanceSkippingSubblocks();

    switch (Entry.Kind) {
    case BitstreamEntry::SubBlock: // Handled for us already.
    case BitstreamEntry::Error:
      return error("Malformed block");
    case BitstreamEntry::EndBlock:
      return Error::success();
    case BitstreamEntry::Record:
      // The interesting case.
      break;
    }

    // Read a metadata attachment record.
    Record.clear();
    switch (Stream.readRecord(Entry.ID, Record)) {
    default: // Default behavior: ignore.
      break;
    case bitc::METADATA_ATTACHMENT: {
      unsigned RecordLength = Record.size();
      if (Record.empty())
        return error("Invalid record");
      if (RecordLength % 2 == 0) {
        // A function attachment.
        if (Error Err = parseGlobalObjectAttachment(F, Record))
          return Err;
        continue;
      }

      // An instruction attachment.
      Instruction *Inst = InstructionList[Record[0]];
      for (unsigned i = 1; i != RecordLength; i = i + 2) {
        unsigned Kind = Record[i];
        DenseMap<unsigned, unsigned>::iterator I = MDKindMap.find(Kind);
        if (I == MDKindMap.end())
          return error("Invalid ID");
        Metadata *Node = MetadataList.getMetadataFwdRef(Record[i + 1]);
        if (isa<LocalAsMetadata>(Node))
          // Drop the attachment.  This used to be legal, but there's no
          // upgrade path.
          break;
        MDNode *MD = dyn_cast_or_null<MDNode>(Node);
        if (!MD)
          return error("Invalid metadata attachment");

        if (HasSeenOldLoopTags && I->second == LLVMContext::MD_loop)
          MD = upgradeInstructionLoopAttachment(*MD);

        if (I->second == LLVMContext::MD_tbaa) {
          assert(!MD->isTemporary() && "should load MDs before attachments");
          MD = UpgradeTBAANode(*MD);
        }
        Inst->setMetadata(I->second, MD);
      }
      break;
    }
    }
  }
}

/// Parse a single METADATA_KIND record, inserting result in MDKindMap.
Error MetadataLoader::MetadataLoaderImpl::parseMetadataKindRecord(
    SmallVectorImpl<uint64_t> &Record) {
  if (Record.size() < 2)
    return error("Invalid record");

  unsigned Kind = Record[0];
  SmallString<8> Name(Record.begin() + 1, Record.end());

  unsigned NewKind = TheModule.getMDKindID(Name.str());
  if (!MDKindMap.insert(std::make_pair(Kind, NewKind)).second)
    return error("Conflicting METADATA_KIND records");
  return Error::success();
}

/// Parse the metadata kinds out of the METADATA_KIND_BLOCK.
Error MetadataLoader::MetadataLoaderImpl::parseMetadataKinds() {
  if (Stream.EnterSubBlock(bitc::METADATA_KIND_BLOCK_ID))
    return error("Invalid record");

  SmallVector<uint64_t, 64> Record;

  // Read all the records.
  while (true) {
    BitstreamEntry Entry = Stream.advanceSkippingSubblocks();

    switch (Entry.Kind) {
    case BitstreamEntry::SubBlock: // Handled for us already.
    case BitstreamEntry::Error:
      return error("Malformed block");
    case BitstreamEntry::EndBlock:
      return Error::success();
    case BitstreamEntry::Record:
      // The interesting case.
      break;
    }

    // Read a record.
    Record.clear();
    unsigned Code = Stream.readRecord(Entry.ID, Record);
    switch (Code) {
    default: // Default behavior: ignore.
      break;
    case bitc::METADATA_KIND: {
      if (Error Err = parseMetadataKindRecord(Record))
        return Err;
      break;
    }
    }
  }
}

MetadataLoader &MetadataLoader::operator=(MetadataLoader &&RHS) {
  Pimpl = std::move(RHS.Pimpl);
  return *this;
}
MetadataLoader::MetadataLoader(MetadataLoader &&RHS)
    : Pimpl(std::move(RHS.Pimpl)) {}

MetadataLoader::~MetadataLoader() = default;
MetadataLoader::MetadataLoader(BitstreamCursor &Stream, Module &TheModule,
                               BitcodeReaderValueList &ValueList,
                               std::function<Type *(unsigned)> getTypeByID)
    : Pimpl(llvm::make_unique<MetadataLoaderImpl>(Stream, TheModule, ValueList,
                                                  getTypeByID)) {}

Error MetadataLoader::parseMetadata(bool ModuleLevel) {
  return Pimpl->parseMetadata(ModuleLevel);
}

bool MetadataLoader::hasFwdRefs() const { return Pimpl->hasFwdRefs(); }

/// Return the given metadata, creating a replaceable forward reference if
/// necessary.
Metadata *MetadataLoader::getMetadataFwdRef(unsigned Idx) {
  return Pimpl->getMetadataFwdRef(Idx);
}

MDNode *MetadataLoader::getMDNodeFwdRefOrNull(unsigned Idx) {
  return Pimpl->getMDNodeFwdRefOrNull(Idx);
}

DISubprogram *MetadataLoader::lookupSubprogramForFunction(Function *F) {
  return Pimpl->lookupSubprogramForFunction(F);
}

Error MetadataLoader::parseMetadataAttachment(
    Function &F, const SmallVectorImpl<Instruction *> &InstructionList) {
  return Pimpl->parseMetadataAttachment(F, InstructionList);
}

Error MetadataLoader::parseMetadataKinds() {
  return Pimpl->parseMetadataKinds();
}

unsigned MetadataLoader::size() const { return Pimpl->size(); }
void MetadataLoader::shrinkTo(unsigned N) { return Pimpl->shrinkTo(N); }
