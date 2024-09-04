//===- Utility.cpp ------ Collection of generic offloading utilities ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/BinaryFormat/AMDGPUMetadataVerifier.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Value.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;
using namespace llvm::offloading;

StructType *offloading::getEntryTy(Module &M) {
  LLVMContext &C = M.getContext();
  StructType *EntryTy =
      StructType::getTypeByName(C, "struct.__tgt_offload_entry");
  if (!EntryTy)
    EntryTy = StructType::create(
        "struct.__tgt_offload_entry", PointerType::getUnqual(C),
        PointerType::getUnqual(C), M.getDataLayout().getIntPtrType(C),
        Type::getInt32Ty(C), Type::getInt32Ty(C));
  return EntryTy;
}

// TODO: Rework this interface to be more generic.
std::pair<Constant *, GlobalVariable *>
offloading::getOffloadingEntryInitializer(Module &M, Constant *Addr,
                                          StringRef Name, uint64_t Size,
                                          int32_t Flags, int32_t Data) {
  llvm::Triple Triple(M.getTargetTriple());
  Type *Int8PtrTy = PointerType::getUnqual(M.getContext());
  Type *Int32Ty = Type::getInt32Ty(M.getContext());
  Type *SizeTy = M.getDataLayout().getIntPtrType(M.getContext());

  Constant *AddrName = ConstantDataArray::getString(M.getContext(), Name);

  StringRef Prefix =
      Triple.isNVPTX() ? "$offloading$entry_name" : ".offloading.entry_name";

  // Create the constant string used to look up the symbol in the device.
  auto *Str =
      new GlobalVariable(M, AddrName->getType(), /*isConstant=*/true,
                         GlobalValue::InternalLinkage, AddrName, Prefix);
  Str->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

  // Construct the offloading entry.
  Constant *EntryData[] = {
      ConstantExpr::getPointerBitCastOrAddrSpaceCast(Addr, Int8PtrTy),
      ConstantExpr::getPointerBitCastOrAddrSpaceCast(Str, Int8PtrTy),
      ConstantInt::get(SizeTy, Size),
      ConstantInt::get(Int32Ty, Flags),
      ConstantInt::get(Int32Ty, Data),
  };
  Constant *EntryInitializer = ConstantStruct::get(getEntryTy(M), EntryData);
  return {EntryInitializer, Str};
}

void offloading::emitOffloadingEntry(Module &M, Constant *Addr, StringRef Name,
                                     uint64_t Size, int32_t Flags, int32_t Data,
                                     StringRef SectionName) {
  llvm::Triple Triple(M.getTargetTriple());

  auto [EntryInitializer, NameGV] =
      getOffloadingEntryInitializer(M, Addr, Name, Size, Flags, Data);

  StringRef Prefix =
      Triple.isNVPTX() ? "$offloading$entry$" : ".offloading.entry.";
  auto *Entry = new GlobalVariable(
      M, getEntryTy(M),
      /*isConstant=*/true, GlobalValue::WeakAnyLinkage, EntryInitializer,
      Prefix + Name, nullptr, GlobalValue::NotThreadLocal,
      M.getDataLayout().getDefaultGlobalsAddressSpace());

  // The entry has to be created in the section the linker expects it to be.
  if (Triple.isOSBinFormatCOFF())
    Entry->setSection((SectionName + "$OE").str());
  else
    Entry->setSection(SectionName);
  Entry->setAlignment(Align(1));
}

std::pair<GlobalVariable *, GlobalVariable *>
offloading::getOffloadEntryArray(Module &M, StringRef SectionName) {
  llvm::Triple Triple(M.getTargetTriple());

  auto *ZeroInitilaizer =
      ConstantAggregateZero::get(ArrayType::get(getEntryTy(M), 0u));
  auto *EntryInit = Triple.isOSBinFormatCOFF() ? ZeroInitilaizer : nullptr;
  auto *EntryType = ArrayType::get(getEntryTy(M), 0);
  auto Linkage = Triple.isOSBinFormatCOFF() ? GlobalValue::WeakODRLinkage
                                            : GlobalValue::ExternalLinkage;

  auto *EntriesB =
      new GlobalVariable(M, EntryType, /*isConstant=*/true, Linkage, EntryInit,
                         "__start_" + SectionName);
  EntriesB->setVisibility(GlobalValue::HiddenVisibility);
  auto *EntriesE =
      new GlobalVariable(M, EntryType, /*isConstant=*/true, Linkage, EntryInit,
                         "__stop_" + SectionName);
  EntriesE->setVisibility(GlobalValue::HiddenVisibility);

  if (Triple.isOSBinFormatELF()) {
    // We assume that external begin/end symbols that we have created above will
    // be defined by the linker. This is done whenever a section name with a
    // valid C-identifier is present. We define a dummy variable here to force
    // the linker to always provide these symbols.
    auto *DummyEntry = new GlobalVariable(
        M, ZeroInitilaizer->getType(), true, GlobalVariable::InternalLinkage,
        ZeroInitilaizer, "__dummy." + SectionName);
    DummyEntry->setSection(SectionName);
    appendToCompilerUsed(M, DummyEntry);
  } else {
    // The COFF linker will merge sections containing a '$' together into a
    // single section. The order of entries in this section will be sorted
    // alphabetically by the characters following the '$' in the name. Set the
    // sections here to ensure that the beginning and end symbols are sorted.
    EntriesB->setSection((SectionName + "$OA").str());
    EntriesE->setSection((SectionName + "$OZ").str());
  }

  return std::make_pair(EntriesB, EntriesE);
}

bool llvm::offloading::amdgpu::isImageCompatibleWithEnv(StringRef ImageArch,
                                                        uint32_t ImageFlags,
                                                        StringRef EnvTargetID) {
  using namespace llvm::ELF;
  StringRef EnvArch = EnvTargetID.split(":").first;

  // Trivial check if the base processors match.
  if (EnvArch != ImageArch)
    return false;

  // Check if the image is requesting xnack on or off.
  switch (ImageFlags & EF_AMDGPU_FEATURE_XNACK_V4) {
  case EF_AMDGPU_FEATURE_XNACK_OFF_V4:
    // The image is 'xnack-' so the environment must be 'xnack-'.
    if (!EnvTargetID.contains("xnack-"))
      return false;
    break;
  case EF_AMDGPU_FEATURE_XNACK_ON_V4:
    // The image is 'xnack+' so the environment must be 'xnack+'.
    if (!EnvTargetID.contains("xnack+"))
      return false;
    break;
  case EF_AMDGPU_FEATURE_XNACK_UNSUPPORTED_V4:
  case EF_AMDGPU_FEATURE_XNACK_ANY_V4:
  default:
    break;
  }

  // Check if the image is requesting sramecc on or off.
  switch (ImageFlags & EF_AMDGPU_FEATURE_SRAMECC_V4) {
  case EF_AMDGPU_FEATURE_SRAMECC_OFF_V4:
    // The image is 'sramecc-' so the environment must be 'sramecc-'.
    if (!EnvTargetID.contains("sramecc-"))
      return false;
    break;
  case EF_AMDGPU_FEATURE_SRAMECC_ON_V4:
    // The image is 'sramecc+' so the environment must be 'sramecc+'.
    if (!EnvTargetID.contains("sramecc+"))
      return false;
    break;
  case EF_AMDGPU_FEATURE_SRAMECC_UNSUPPORTED_V4:
  case EF_AMDGPU_FEATURE_SRAMECC_ANY_V4:
    break;
  }

  return true;
}

namespace {
/// Reads the AMDGPU specific per-kernel-metadata from an image.
class KernelInfoReader {
public:
  KernelInfoReader(StringMap<offloading::amdgpu::AMDGPUKernelMetaData> &KIM)
      : KernelInfoMap(KIM) {}

  /// Process ELF note to read AMDGPU metadata from respective information
  /// fields.
  Error processNote(const llvm::object::ELF64LE::Note &Note, size_t Align) {
    if (Note.getName() != "AMDGPU")
      return Error::success(); // We are not interested in other things

    assert(Note.getType() == ELF::NT_AMDGPU_METADATA &&
           "Parse AMDGPU MetaData");
    auto Desc = Note.getDesc(Align);
    StringRef MsgPackString =
        StringRef(reinterpret_cast<const char *>(Desc.data()), Desc.size());
    msgpack::Document MsgPackDoc;
    if (!MsgPackDoc.readFromBlob(MsgPackString, /*Multi=*/false))
      return Error::success();

    AMDGPU::HSAMD::V3::MetadataVerifier Verifier(true);
    if (!Verifier.verify(MsgPackDoc.getRoot()))
      return Error::success();

    auto RootMap = MsgPackDoc.getRoot().getMap(true);

    if (auto Err = iterateAMDKernels(RootMap))
      return Err;

    return Error::success();
  }

private:
  /// Extracts the relevant information via simple string look-up in the msgpack
  /// document elements.
  Error
  extractKernelData(msgpack::MapDocNode::MapTy::value_type V,
                    std::string &KernelName,
                    offloading::amdgpu::AMDGPUKernelMetaData &KernelData) {
    if (!V.first.isString())
      return Error::success();

    const auto IsKey = [](const msgpack::DocNode &DK, StringRef SK) {
      return DK.getString() == SK;
    };

    const auto GetSequenceOfThreeInts = [](msgpack::DocNode &DN,
                                           uint32_t *Vals) {
      assert(DN.isArray() && "MsgPack DocNode is an array node");
      auto DNA = DN.getArray();
      assert(DNA.size() == 3 && "ArrayNode has at most three elements");

      int I = 0;
      for (auto DNABegin = DNA.begin(), DNAEnd = DNA.end(); DNABegin != DNAEnd;
           ++DNABegin) {
        Vals[I++] = DNABegin->getUInt();
      }
    };

    if (IsKey(V.first, ".name")) {
      KernelName = V.second.toString();
    } else if (IsKey(V.first, ".sgpr_count")) {
      KernelData.SGPRCount = V.second.getUInt();
    } else if (IsKey(V.first, ".sgpr_spill_count")) {
      KernelData.SGPRSpillCount = V.second.getUInt();
    } else if (IsKey(V.first, ".vgpr_count")) {
      KernelData.VGPRCount = V.second.getUInt();
    } else if (IsKey(V.first, ".vgpr_spill_count")) {
      KernelData.VGPRSpillCount = V.second.getUInt();
    } else if (IsKey(V.first, ".agpr_count")) {
      KernelData.AGPRCount = V.second.getUInt();
    } else if (IsKey(V.first, ".private_segment_fixed_size")) {
      KernelData.PrivateSegmentSize = V.second.getUInt();
    } else if (IsKey(V.first, ".group_segment_fixed_size")) {
      KernelData.GroupSegmentList = V.second.getUInt();
    } else if (IsKey(V.first, ".reqd_workgroup_size")) {
      GetSequenceOfThreeInts(V.second, KernelData.RequestedWorkgroupSize);
    } else if (IsKey(V.first, ".workgroup_size_hint")) {
      GetSequenceOfThreeInts(V.second, KernelData.WorkgroupSizeHint);
    } else if (IsKey(V.first, ".wavefront_size")) {
      KernelData.WavefrontSize = V.second.getUInt();
    } else if (IsKey(V.first, ".max_flat_workgroup_size")) {
      KernelData.MaxFlatWorkgroupSize = V.second.getUInt();
    }

    return Error::success();
  }

  /// Get the "amdhsa.kernels" element from the msgpack Document
  Expected<msgpack::ArrayDocNode> getAMDKernelsArray(msgpack::MapDocNode &MDN) {
    auto Res = MDN.find("amdhsa.kernels");
    if (Res == MDN.end())
      return createStringError(inconvertibleErrorCode(),
                               "Could not find amdhsa.kernels key");

    auto Pair = *Res;
    assert(Pair.second.isArray() &&
           "AMDGPU kernel entries are arrays of entries");

    return Pair.second.getArray();
  }

  /// Iterate all entries for one "amdhsa.kernels" entry. Each entry is a
  /// MapDocNode that either maps a string to a single value (most of them) or
  /// to another array of things. Currently, we only handle the case that maps
  /// to scalar value.
  Error generateKernelInfo(msgpack::ArrayDocNode::ArrayTy::iterator It) {
    offloading::amdgpu::AMDGPUKernelMetaData KernelData;
    std::string KernelName;
    auto Entry = (*It).getMap();
    for (auto MI = Entry.begin(), E = Entry.end(); MI != E; ++MI)
      if (auto Err = extractKernelData(*MI, KernelName, KernelData))
        return Err;

    KernelInfoMap.insert({KernelName, KernelData});
    return Error::success();
  }

  /// Go over the list of AMD kernels in the "amdhsa.kernels" entry
  Error iterateAMDKernels(msgpack::MapDocNode &MDN) {
    auto KernelsOrErr = getAMDKernelsArray(MDN);
    if (auto Err = KernelsOrErr.takeError())
      return Err;

    auto KernelsArr = *KernelsOrErr;
    for (auto It = KernelsArr.begin(), E = KernelsArr.end(); It != E; ++It) {
      if (!It->isMap())
        continue; // we expect <key,value> pairs

      // Obtain the value for the different entries. Each array entry is a
      // MapDocNode
      if (auto Err = generateKernelInfo(It))
        return Err;
    }
    return Error::success();
  }

  // Kernel names are the keys
  StringMap<offloading::amdgpu::AMDGPUKernelMetaData> &KernelInfoMap;
};
} // namespace

Error llvm::offloading::amdgpu::getAMDGPUMetaDataFromImage(
    MemoryBufferRef MemBuffer,
    StringMap<offloading::amdgpu::AMDGPUKernelMetaData> &KernelInfoMap,
    uint16_t &ELFABIVersion) {
  Error Err = Error::success(); // Used later as out-parameter

  auto ELFOrError = object::ELF64LEFile::create(MemBuffer.getBuffer());
  if (auto Err = ELFOrError.takeError())
    return Err;

  const object::ELF64LEFile ELFObj = ELFOrError.get();
  Expected<ArrayRef<object::ELF64LE::Shdr>> Sections = ELFObj.sections();
  if (!Sections)
    return Sections.takeError();
  KernelInfoReader Reader(KernelInfoMap);

  // Read the code object version from ELF image header
  auto Header = ELFObj.getHeader();
  ELFABIVersion = (uint8_t)(Header.e_ident[ELF::EI_ABIVERSION]);
  for (const auto &S : *Sections) {
    if (S.sh_type != ELF::SHT_NOTE)
      continue;

    for (const auto N : ELFObj.notes(S, Err)) {
      if (Err)
        return Err;
      // Fills the KernelInfoTabel entries in the reader
      if ((Err = Reader.processNote(N, S.sh_addralign)))
        return Err;
    }
  }
  return Error::success();
}
