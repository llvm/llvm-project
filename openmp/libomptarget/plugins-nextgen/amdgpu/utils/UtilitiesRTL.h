//===----RTLs/amdgpu/utils/UtilitiesRTL.h ------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL Utilities for AMDGPU plugins
//
//===----------------------------------------------------------------------===//

#include <cstdint>

#include "Debug.h"
#include "omptarget.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "llvm/BinaryFormat/AMDGPUMetadataVerifier.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Support/MemoryBufferRef.h"

#include "llvm/Support/YAMLTraits.h"

namespace llvm {
namespace omp {
namespace target {
namespace plugin {
namespace utils {

// The implicit arguments of AMDGPU kernels.
struct AMDGPUImplicitArgsTy {
  uint64_t OffsetX;
  uint64_t OffsetY;
  uint64_t OffsetZ;
  uint64_t HostcallPtr;
  uint64_t Unused0;
  uint64_t Unused1;
  uint64_t Unused2;
};

static_assert(sizeof(AMDGPUImplicitArgsTy) == 56,
              "Unexpected size of implicit arguments");

/// Parse a TargetID to get processor arch and feature map.
/// Returns processor subarch.
/// Returns TargetID features in \p FeatureMap argument.
/// If the \p TargetID contains feature+, FeatureMap it to true.
/// If the \p TargetID contains feature-, FeatureMap it to false.
/// If the \p TargetID does not contain a feature (default), do not map it.
StringRef parseTargetID(StringRef TargetID, StringMap<bool> &FeatureMap) {
  if (TargetID.empty())
    return llvm::StringRef();

  auto ArchFeature = TargetID.split(":");
  auto Arch = ArchFeature.first;
  auto Features = ArchFeature.second;
  if (Features.empty())
    return Arch;

  if (Features.contains("sramecc+")) {
    FeatureMap.insert(std::pair<StringRef, bool>("sramecc", true));
  } else if (Features.contains("sramecc-")) {
    FeatureMap.insert(std::pair<StringRef, bool>("sramecc", false));
  }
  if (Features.contains("xnack+")) {
    FeatureMap.insert(std::pair<StringRef, bool>("xnack", true));
  } else if (Features.contains("xnack-")) {
    FeatureMap.insert(std::pair<StringRef, bool>("xnack", false));
  }

  return Arch;
}

/// Check if an image is compatible with current system's environment.
bool isImageCompatibleWithEnv(const __tgt_image_info *Info,
                              StringRef EnvTargetID) {
  llvm::StringRef ImageTargetID(Info->Arch);
  // Compatible in case of exact match.
  if (ImageTargetID == EnvTargetID) {
    DP("Compatible: Exact match \t[Image: %s]\t:\t[Env: %s]\n",
       ImageTargetID.data(), EnvTargetID.data());
    return true;
  }

  // Incompatible if Archs mismatch.
  StringMap<bool> ImgMap, EnvMap;
  StringRef ImgArch = utils::parseTargetID(ImageTargetID, ImgMap);
  StringRef EnvArch = utils::parseTargetID(EnvTargetID, EnvMap);

  // Both EnvArch and ImgArch can't be empty here.
  if (EnvArch.empty() || ImgArch.empty() || !ImgArch.contains(EnvArch)) {
    DP("Incompatible: Processor mismatch \t[Image: %s]\t:\t[Env: %s]\n",
       ImageTargetID.data(), EnvTargetID.data());
    return false;
  }

  // Incompatible if image has more features than the environment,
  // irrespective of type or sign of features.
  if (ImgMap.size() > EnvMap.size()) {
    DP("Incompatible: Image has more features than the Environment \t[Image: "
       "%s]\t:\t[Env: %s]\n",
       ImageTargetID.data(), EnvTargetID.data());
    return false;
  }

  // Compatible if each target feature specified by the environment is
  // compatible with target feature of the image. The target feature is
  // compatible if the iamge does not specify it (meaning Any), or if it
  // specifies it with the same value (meaning On or Off).
  for (const auto &ImgFeature : ImgMap) {
    auto EnvFeature = EnvMap.find(ImgFeature.first());
    if (EnvFeature == EnvMap.end() ||
        (EnvFeature->first() == ImgFeature.first() &&
         EnvFeature->second != ImgFeature.second)) {
      DP("Incompatible: Value of Image's non-ANY feature is not matching with "
         "the Environment's non-ANY feature \t[Image: %s]\t:\t[Env: %s]\n",
         ImageTargetID.data(), EnvTargetID.data());
      return false;
    }
  }

  // Image is compatible if all features of Environment are:
  //   - either, present in the Image's features map with the same sign,
  //   - or, the feature is missing from Image's features map i.e. it is
  //   set to ANY
  DP("Compatible: Target IDs are compatible \t[Image: %s]\t:\t[Env: %s]\n",
     ImageTargetID.data(), EnvTargetID.data());

  return true;
}

struct KernelMetaDataTy {
  uint64_t KernelObject;
  uint32_t GroupSegmentList;
  uint32_t PrivateSegmentSize;
  uint32_t SGPRCount;
  uint32_t VGPRCount;
  uint32_t SGPRSpillCount;
  uint32_t VGPRSpillCount;
  uint32_t KernelSegmentSize;
  uint32_t ExplicitArgumentCount;
  uint32_t ImplicitArgumentCount;
  uint32_t RequestedWorkgroupSize[3];
  uint32_t WorkgroupSizeHint[3];
  uint32_t WavefronSize;
  uint32_t MaxFlatWorkgroupSize;
};
namespace {

/// Reads the AMDGPU specific per-kernel-metadata from an image.
class KernelInfoReader {
public:
  KernelInfoReader(StringMap<KernelMetaDataTy> &KIM) : KernelInfoMap(KIM) {}

  /// Process ELF note to read AMDGPU metadata from respective information
  /// fields.
  Error processNote(const object::ELF64LE::Note &Note) {
    if (Note.getName() != "AMDGPU")
      return Error::success(); // We are not interested in other things

    assert(Note.getType() == ELF::NT_AMDGPU_METADATA &&
           "Parse AMDGPU MetaData");
    auto Desc = Note.getDesc();
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
  Error extractKernelData(msgpack::MapDocNode::MapTy::value_type V,
                          std::string &KernelName,
                          KernelMetaDataTy &KernelData) {
    if (!V.first.isString())
      return Error::success();

    const auto isKey = [](const msgpack::DocNode &DK, StringRef SK) {
      return DK.getString() == SK;
    };

    const auto getSequenceOfThreeInts = [](msgpack::DocNode &DN,
                                           uint32_t *Vals) {
      assert(DN.isArray() && "MsgPack DocNode is an array node");
      auto DNA = DN.getArray();
      assert(DNA.size() == 3 && "ArrayNode has at most three elements");

      int i = 0;
      for (auto DNABegin = DNA.begin(), DNAEnd = DNA.end(); DNABegin != DNAEnd;
           ++DNABegin) {
        Vals[i++] = DNABegin->getUInt();
      }
    };

    if (isKey(V.first, ".name")) {
      KernelName = V.second.toString();
    } else if (isKey(V.first, ".sgpr_count")) {
      KernelData.SGPRCount = V.second.getUInt();
    } else if (isKey(V.first, ".sgpr_spill_count")) {
      KernelData.SGPRSpillCount = V.second.getUInt();
    } else if (isKey(V.first, ".vgpr_count")) {
      KernelData.VGPRCount = V.second.getUInt();
    } else if (isKey(V.first, ".vgpr_spill_count")) {
      KernelData.VGPRSpillCount = V.second.getUInt();
    } else if (isKey(V.first, ".private_segment_fixed_size")) {
      KernelData.PrivateSegmentSize = V.second.getUInt();
    } else if (isKey(V.first, ".group_segment_fixed_size")) {
      KernelData.GroupSegmentList = V.second.getUInt();
    } else if (isKey(V.first, ".reqd_workgroup_size")) {
      getSequenceOfThreeInts(V.second, KernelData.RequestedWorkgroupSize);
    } else if (isKey(V.first, ".workgroup_size_hint")) {
      getSequenceOfThreeInts(V.second, KernelData.WorkgroupSizeHint);
    } else if (isKey(V.first, ".wavefront_size")) {
      KernelData.WavefronSize = V.second.getUInt();
    } else if (isKey(V.first, ".max_flat_workgroup_size")) {
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
    KernelMetaDataTy KernelData;
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
  StringMap<KernelMetaDataTy> &KernelInfoMap;
};
} // namespace

/// Reads the AMDGPU specific metadata from the ELF file and propagates the
/// KernelInfoMap
Error readAMDGPUMetaDataFromImage(MemoryBufferRef MemBuffer,
                                  StringMap<KernelMetaDataTy> &KernelInfoMap,
                                  uint16_t &ELFABIVersion) {

  Error Err = Error::success(); // Used later as out-parameter

  auto ELFOrError = object::ELF64LEFile::create(MemBuffer.getBuffer());
  if (auto Err = ELFOrError.takeError())
    return Err;

  const object::ELF64LEFile ELFObj = ELFOrError.get();
  ArrayRef<object::ELF64LE::Shdr> Sections = cantFail(ELFObj.sections());
  KernelInfoReader Reader(KernelInfoMap);

  auto Header = ELFObj.getHeader();
  ELFABIVersion = (uint8_t)(Header.e_ident[ELF::EI_ABIVERSION]);
  DP("ELFABIVERSION Version: %u\n", ELFABIVersion);
  printf("ELFABIVERSION Version: %u\n", ELFABIVersion);

  for (const auto &S : Sections) {
    if (S.sh_type != ELF::SHT_NOTE)
      continue;

    for (const auto N : ELFObj.notes(S, Err)) {
      if (Err)
        return Err;
      // Fills the KernelInfoTabel entries in the reader
      if ((Err = Reader.processNote(N)))
        return Err;
    }
  }

  return Error::success();
}

} // namespace utils
} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm
