//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Level Zero Program abstraction
//
//===----------------------------------------------------------------------===//

#include <fstream>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
#endif // !_WIN32

#include "L0Plugin.h"
#include "L0Program.h"

namespace llvm::omp::target::plugin {

Error L0GlobalHandlerTy::getGlobalMetadataFromDevice(GenericDeviceTy &Device,
                                                     DeviceImageTy &Image,
                                                     GlobalTy &DeviceGlobal) {
  const char *GlobalName = DeviceGlobal.getName().data();

  L0ProgramTy &Program = L0ProgramTy::makeL0Program(Image);
  void *Addr = Program.getOffloadVarDeviceAddr(GlobalName);

  // Save the pointer to the symbol allowing nullptr.
  DeviceGlobal.setPtr(Addr);

  if (Addr == nullptr)
    return Plugin::error(ErrorCode::UNKNOWN, "Failed to load global '%s'",
                         GlobalName);

  return Plugin::success();
}

inline L0DeviceTy &L0ProgramTy::getL0Device() const {
  return L0DeviceTy::makeL0Device(getDevice());
}

L0ProgramTy::~L0ProgramTy() {
  for (auto *Kernel : Kernels) {
    // We need explicit destructor and deallocate calls to release the kernels
    // created by `GenericDeviceTy::constructKernel()`.
    Kernel->~L0KernelTy();
    getL0Device().getPlugin().free(Kernel);
  }
  for (auto Module : Modules) {
    CALL_ZE_RET_VOID(zeModuleDestroy, Module);
  }
}

void L0ProgramTy::setLibModule() {
#if _WIN32
  return;
#else
  // Check if the image belongs to a dynamic library
  Dl_info DLI{nullptr, nullptr, nullptr, nullptr};
  if (dladdr(getStart(), &DLI) && DLI.dli_fname) {
    std::vector<uint8_t> FileBin;
    auto Size = readFile(DLI.dli_fname, FileBin);
    if (Size) {
      auto MB = MemoryBuffer::getMemBuffer(
          StringRef(reinterpret_cast<const char *>(FileBin.data()), Size),
          /*BufferName=*/"", /*RequiresNullTerminator=*/false);
      auto ELF = ELFObjectFileBase::createELFObjectFile(MB->getMemBufferRef());
      if (ELF) {
        if (auto *Obj = dyn_cast<ELF64LEObjectFile>((*ELF).get())) {
          const auto Header = Obj->getELFFile().getHeader();
          if (Header.e_type == ELF::ET_DYN) {
            DP("Processing current image as library\n");
            IsLibModule = true;
          }
        }
      }
    }
  }
#endif // _WIN32
}

int32_t L0ProgramTy::addModule(size_t Size, const uint8_t *Image,
                               const std::string_view CommonBuildOptions,
                               ze_module_format_t Format) {
  const ze_module_constants_t SpecConstants =
      LevelZeroPluginTy::getOptions().CommonSpecConstants.getModuleConstants();
  auto &l0Device = getL0Device();
  std::string BuildOptions(CommonBuildOptions);

  // Add required flag to enable dynamic linking.
  if (IsLibModule)
    BuildOptions += " -library-compilation ";

  ze_module_desc_t ModuleDesc{};
  ModuleDesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  ModuleDesc.pNext = nullptr;
  ModuleDesc.format = Format;
  ze_module_handle_t Module = nullptr;
  ze_module_build_log_handle_t BuildLog = nullptr;
  ze_result_t RC;

  // Build a single module from a single image
  ModuleDesc.inputSize = Size;
  ModuleDesc.pInputModule = Image;
  ModuleDesc.pBuildFlags = BuildOptions.c_str();
  ModuleDesc.pConstants = &SpecConstants;
  CALL_ZE_RC(RC, zeModuleCreate, l0Device.getZeContext(),
             l0Device.getZeDevice(), &ModuleDesc, &Module, &BuildLog);

  const bool BuildFailed = (RC != ZE_RESULT_SUCCESS);

  if (BuildFailed)
    return OFFLOAD_FAIL;

  // Check if module link is required. We do not need this check for
  // library module
  if (!RequiresModuleLink && !IsLibModule) {
    ze_module_properties_t Properties = {ZE_STRUCTURE_TYPE_MODULE_PROPERTIES,
                                         nullptr, 0};
    CALL_ZE_RET_FAIL(zeModuleGetProperties, Module, &Properties);
    RequiresModuleLink = Properties.flags & ZE_MODULE_PROPERTY_FLAG_IMPORTS;
  }
  // For now, assume the first module contains libraries, globals.
  if (Modules.empty())
    GlobalModule = Module;
  Modules.push_back(Module);
  l0Device.addGlobalModule(Module);
  return OFFLOAD_SUCCESS;
}

int32_t L0ProgramTy::linkModules() {
  auto &l0Device = getL0Device();
  if (!RequiresModuleLink) {
    DP("Module link is not required\n");
    return OFFLOAD_SUCCESS;
  }

  if (Modules.empty()) {
    DP("Invalid number of modules when linking modules\n");
    return OFFLOAD_FAIL;
  }

  ze_result_t RC;
  ze_module_build_log_handle_t LinkLog = nullptr;
  CALL_ZE_RC(RC, zeModuleDynamicLink,
             static_cast<uint32_t>(l0Device.getNumGlobalModules()),
             l0Device.getGlobalModulesArray(), &LinkLog);
  const bool LinkFailed = (RC != ZE_RESULT_SUCCESS);
  return LinkFailed ? OFFLOAD_FAIL : OFFLOAD_SUCCESS;
}

size_t L0ProgramTy::readFile(const char *FileName,
                             std::vector<uint8_t> &OutFile) const {
  std::ifstream IFS(FileName, std::ios::binary);
  if (!IFS.good())
    return 0;
  IFS.seekg(0, IFS.end);
  auto FileSize = static_cast<size_t>(IFS.tellg());
  OutFile.resize(FileSize);
  IFS.seekg(0);
  if (!IFS.read(reinterpret_cast<char *>(OutFile.data()), FileSize)) {
    OutFile.clear();
    return 0;
  }
  return FileSize;
}

void L0ProgramTy::replaceDriverOptsWithBackendOpts(const L0DeviceTy &Device,
                                                   std::string &Options) const {
  // Options that need to be replaced with backend-specific options
  static const struct {
    std::string Option;
    std::string BackendOption;
  } OptionTranslationTable[] = {
      {"-ftarget-compile-fast",
       "-igc_opts 'PartitionUnit=1,SubroutineThreshold=50000'"},
      {"-foffload-fp32-prec-div", "-ze-fp32-correctly-rounded-divide-sqrt"},
      {"-foffload-fp32-prec-sqrt", "-ze-fp32-correctly-rounded-divide-sqrt"},
  };

  for (const auto &OptPair : OptionTranslationTable) {
    const size_t Pos = Options.find(OptPair.Option);
    if (Pos != std::string::npos) {
      Options.replace(Pos, OptPair.Option.length(), OptPair.BackendOption);
    }
  }
}

// FIXME: move this to llvm/BinaryFormat/ELF.h and elf.h:
#define NT_INTEL_ONEOMP_OFFLOAD_VERSION 1
#define NT_INTEL_ONEOMP_OFFLOAD_IMAGE_COUNT 2
#define NT_INTEL_ONEOMP_OFFLOAD_IMAGE_AUX 3

bool isValidOneOmpImage(StringRef Image, uint64_t &MajorVer,
                        uint64_t &MinorVer) {
  const auto MB = MemoryBuffer::getMemBuffer(Image,
                                             /*BufferName=*/"",
                                             /*RequiresNullTerminator=*/false);
  auto ExpectedNewE =
      ELFObjectFileBase::createELFObjectFile(MB->getMemBufferRef());
  if (!ExpectedNewE) {
    DP("Warning: unable to get ELF handle!\n");
    return false;
  }
  bool Res = false;
  auto processObjF = [&](const auto ELFObjF) {
    if (!ELFObjF) {
      DP("Warning: Unexpected ELF type!\n");
      return false;
    }
    const auto &ELFF = ELFObjF->getELFFile();
    auto Sections = ELFF.sections();
    if (!Sections) {
      DP("Warning: unable to get ELF sections!\n");
      return false;
    }
    bool SeenOffloadSection = false;
    for (auto Sec : *Sections) {
      if (Sec.sh_type != ELF::SHT_NOTE)
        continue;
      Error Err = Plugin::success();
      for (auto Note : ELFF.notes(Sec, Err)) {
        if (Err) {
          DP("Warning: unable to get ELF notes handle!\n");
          return false;
        }
        if (Note.getName() != "INTELONEOMPOFFLOAD")
          continue;
        SeenOffloadSection = true;
        if (Note.getType() != NT_INTEL_ONEOMP_OFFLOAD_VERSION)
          continue;

        std::string DescStr(std::move(Note.getDescAsStringRef(4).str()));
        const auto DelimPos = DescStr.find('.');
        if (DelimPos == std::string::npos) {
          // The version has to look like "Major#.Minor#".
          DP("Invalid NT_INTEL_ONEOMP_OFFLOAD_VERSION: '%s'\n",
             DescStr.c_str());
          return false;
        }
        const std::string MajorVerStr = DescStr.substr(0, DelimPos);
        DescStr.erase(0, DelimPos + 1);
        MajorVer = std::stoull(MajorVerStr);
        MinorVer = std::stoull(DescStr);
        return (MajorVer == 1 && MinorVer == 0);
      }
    }
    return SeenOffloadSection;
  };
  if (const auto *O = dyn_cast<ELF64LEObjectFile>((*ExpectedNewE).get())) {
    Res = processObjF(O);
  } else if (const auto *O =
                 dyn_cast<ELF32LEObjectFile>((*ExpectedNewE).get())) {
    Res = processObjF(O);
  } else {
    assert(false && "Unexpected ELF format");
  }
  return Res;
}

int32_t L0ProgramTy::buildModules(const std::string_view BuildOptions) {
  auto &l0Device = getL0Device();
  auto Image = getMemoryBuffer();
  if (identify_magic(Image.getBuffer()) == file_magic::spirv_object) {
    // Handle legacy plain SPIR-V image.
    const uint8_t *ImgBegin = reinterpret_cast<const uint8_t *>(getStart());
    return addModule(getSize(), ImgBegin, BuildOptions,
                     ZE_MODULE_FORMAT_IL_SPIRV);
  }

  uint64_t MajorVer, MinorVer;
  if (!isValidOneOmpImage(Image.getBuffer(), MajorVer, MinorVer)) {
    DP("Warning: image is not a valid oneAPI OpenMP image.\n");
    return OFFLOAD_FAIL;
  }

  setLibModule();

  // Iterate over the images and pick the first one that fits.
  uint64_t ImageCount = 0;
  struct V1ImageInfo {
    // 0 - native, 1 - SPIR-V
    uint64_t Format = (std::numeric_limits<uint64_t>::max)();
    std::string CompileOpts;
    std::string LinkOpts;
    // We may have multiple sections created from split-kernel mode
    std::vector<const uint8_t *> PartBegin;
    std::vector<uint64_t> PartSize;

    V1ImageInfo(uint64_t Format, std::string CompileOpts, std::string LinkOpts)
        : Format(Format), CompileOpts(std::move(CompileOpts)),
          LinkOpts(std::move(LinkOpts)) {}
  };
  std::unordered_map<uint64_t, V1ImageInfo> AuxInfo;

  auto ExpectedNewE = ELFObjectFileBase::createELFObjectFile(Image);
  assert(ExpectedNewE &&
         "isValidOneOmpImage() returns true for invalid ELF image");
  auto processELF = [&](auto *EObj) {
    assert(EObj && "isValidOneOmpImage() returns true for invalid ELF image.");
    const auto &E = EObj->getELFFile();
    // Collect auxiliary information.
    uint64_t MaxImageIdx = 0;

    auto Sections = E.sections();
    assert(Sections && "isValidOneOmpImage() returns true for ELF image with "
                       "invalid sections.");

    for (auto Sec : *Sections) {
      if (Sec.sh_type != ELF::SHT_NOTE)
        continue;
      Error Err = Plugin::success();
      for (auto Note : E.notes(Sec, Err)) {
        assert(!Err && "isValidOneOmpImage() returns true for ELF image with "
                       "invalid notes.");
        if (Note.getName().str() != "INTELONEOMPOFFLOAD")
          continue;

        const uint64_t Type = Note.getType();
        auto DescStrRef = Note.getDescAsStringRef(4);
        switch (Type) {
        default:
          DP("Warning: unrecognized INTELONEOMPOFFLOAD note.\n");
          break;
        case NT_INTEL_ONEOMP_OFFLOAD_VERSION:
          break;
        case NT_INTEL_ONEOMP_OFFLOAD_IMAGE_COUNT:
          if (DescStrRef.getAsInteger(10, ImageCount)) {
            DP("Warning: invalid NT_INTEL_ONEOMP_OFFLOAD_IMAGE_COUNT: '%s'\n",
               DescStrRef.str().c_str());
            ImageCount = 0;
          }
          break;
        case NT_INTEL_ONEOMP_OFFLOAD_IMAGE_AUX:
          llvm::SmallVector<llvm::StringRef, 4> Parts;
          DescStrRef.split(Parts, '\0', /* MaxSplit = */ 4,
                           /* KeepEmpty = */ true);

          // Ignore records with less than 4 strings.
          if (Parts.size() != 4) {
            DP("Warning: short NT_INTEL_ONEOMP_OFFLOAD_IMAGE_AUX "
               "record is ignored.\n");
            continue;
          }

          uint64_t Idx = 0;
          if (Parts[0].getAsInteger(10, Idx)) {
            DP("Warning: ignoring auxiliary information (invalid index "
               "'%s').\n",
               Parts[0].str().c_str());
            continue;
          }
          MaxImageIdx = (std::max)(MaxImageIdx, Idx);
          if (AuxInfo.find(Idx) != AuxInfo.end()) {
            DP("Warning: duplicate auxiliary information for image %" PRIu64
               " is ignored.\n",
               Idx);
            continue;
          }

          uint64_t Part1Id;
          if (Parts[1].getAsInteger(10, Part1Id)) {
            DP("Warning: ignoring auxiliary information (invalid part id "
               "'%s').\n",
               Parts[1].str().c_str());
            continue;
          }

          AuxInfo.emplace(
              std::piecewise_construct, std::forward_as_tuple(Idx),
              std::forward_as_tuple(Part1Id, Parts[2].str(), Parts[3].str()));
          // Image pointer and size
          // will be initialized later.
        }
      }
    }

    if (MaxImageIdx >= ImageCount)
      DP("Warning: invalid image index found in auxiliary information.\n");

    for (auto Sec : *Sections) {
      const char *Prefix = "__openmp_offload_spirv_";
      auto ExpectedSectionName = E.getSectionName(Sec);
      assert(ExpectedSectionName && "isValidOneOmpImage() returns true for ELF "
                                    "image with invalid section names");
      auto &SectionNameRef = *ExpectedSectionName;
      if (!SectionNameRef.consume_front(Prefix))
        continue;

      // Expected section name in split-kernel mode:
      // __openmp_offload_spirv_<image_id>_<part_id>
      auto Parts = SectionNameRef.split('_');
      // It seems that we do not need part ID as long as they are ordered
      // in the image and we keep the ordering in the runtime.
      SectionNameRef = Parts.first;
      if (Parts.second.empty()) {
        DP("Found a single section in the image\n");
      } else {
        DP("Found a split section in the image\n");
      }

      uint64_t Idx = 0;
      if (SectionNameRef.getAsInteger(10, Idx)) {
        DP("Warning: ignoring image section (invalid index '%s').\n",
           SectionNameRef.str().c_str());
        continue;
      }
      if (Idx >= ImageCount) {
        DP("Warning: ignoring image section (index %" PRIu64
           " is out of range).\n",
           Idx);
        continue;
      }

      auto AuxInfoIt = AuxInfo.find(Idx);
      if (AuxInfoIt == AuxInfo.end()) {
        DP("Warning: ignoring image section (no aux info).\n");
        continue;
      }
      auto Contents = E.getSectionContents(Sec);
      assert(Contents);
      AuxInfoIt->second.PartBegin.push_back((*Contents).data());
      AuxInfoIt->second.PartSize.push_back(Sec.sh_size);
    }
  };

  if (auto *O = dyn_cast<ELF64LEObjectFile>((*ExpectedNewE).get())) {
    processELF(O);
  } else if (auto *O = dyn_cast<ELF32LEObjectFile>((*ExpectedNewE).get())) {
    processELF(O);
  } else {
    assert(false && "Unexpected ELF format");
  }

  for (uint64_t Idx = 0; Idx < ImageCount; ++Idx) {
    const auto It = AuxInfo.find(Idx);
    if (It == AuxInfo.end()) {
      DP("Warning: image %" PRIu64
         " without auxiliary information is ingored.\n",
         Idx);
      continue;
    }

    const auto NumParts = It->second.PartBegin.size();
    // Split-kernel is not supported in SPIRV format
    if (NumParts > 1 && It->second.Format != 0) {
      DP("Warning: split-kernel images are not supported in SPIRV format\n");
      continue;
    }

    // Skip unknown image format
    if (It->second.Format != 0 && It->second.Format != 1) {
      DP("Warning: image %" PRIu64 "is ignored due to unknown format.\n", Idx);
      continue;
    }

    const bool IsBinary = (It->second.Format == 0);
    const auto ModuleFormat =
        IsBinary ? ZE_MODULE_FORMAT_NATIVE : ZE_MODULE_FORMAT_IL_SPIRV;
    std::string Options(BuildOptions);
    {
      Options += " " + It->second.CompileOpts + " " + It->second.LinkOpts;
      replaceDriverOptsWithBackendOpts(l0Device, Options);
    }

    for (size_t I = 0; I < NumParts; I++) {
      const unsigned char *ImgBegin =
          reinterpret_cast<const unsigned char *>(It->second.PartBegin[I]);
      size_t ImgSize = It->second.PartSize[I];

      auto RC = addModule(ImgSize, ImgBegin, Options, ModuleFormat);

      if (RC != OFFLOAD_SUCCESS) {
        DP("Error: failed to create program from %s "
           "(%" PRIu64 "-%zu).\n",
           IsBinary ? "Binary" : "SPIR-V", Idx, I);
        return OFFLOAD_FAIL;
      }
    }
    DP("Created module from image #%" PRIu64 ".\n", Idx);

    return OFFLOAD_SUCCESS;
  }

  return OFFLOAD_FAIL;
}

void *L0ProgramTy::getOffloadVarDeviceAddr(const char *CName) const {
  DP("Looking up OpenMP global variable '%s'.\n", CName);

  if (!GlobalModule || !CName)
    return nullptr;

  std::string Name(CName);
  size_t SizeDummy = 0;
  void *DevicePtr = nullptr;
  ze_result_t RC;
  for (auto Module : Modules) {
    CALL_ZE(RC, zeModuleGetGlobalPointer, Module, Name.c_str(), &SizeDummy,
            &DevicePtr);
    if (RC == ZE_RESULT_SUCCESS && DevicePtr)
      return DevicePtr;
  }
  DP("Warning: global variable '%s' was not found in the device.\n",
     Name.c_str());
  return nullptr;
}

int32_t L0ProgramTy::readGlobalVariable(const char *Name, size_t Size,
                                        void *HostPtr) {
  size_t SizeDummy = 0;
  void *DevicePtr = nullptr;
  ze_result_t RC;
  CALL_ZE(RC, zeModuleGetGlobalPointer, GlobalModule, Name, &SizeDummy,
          &DevicePtr);
  if (RC != ZE_RESULT_SUCCESS || !DevicePtr) {
    DP("Warning: cannot read from device global variable %s\n", Name);
    return OFFLOAD_FAIL;
  }
  return getL0Device().enqueueMemCopy(HostPtr, DevicePtr, Size);
}

int32_t L0ProgramTy::writeGlobalVariable(const char *Name, size_t Size,
                                         const void *HostPtr) {
  size_t SizeDummy = 0;
  void *DevicePtr = nullptr;
  ze_result_t RC;
  CALL_ZE(RC, zeModuleGetGlobalPointer, GlobalModule, Name, &SizeDummy,
          &DevicePtr);
  if (RC != ZE_RESULT_SUCCESS || !DevicePtr) {
    DP("Warning: cannot write to device global variable %s\n", Name);
    return OFFLOAD_FAIL;
  }
  return getL0Device().enqueueMemCopy(DevicePtr, HostPtr, Size);
}

int32_t L0ProgramTy::loadModuleKernels() {
  // We need to build kernels here before filling the offload entries since we
  // don't know which module contains a specific kernel with a name.
  for (auto Module : Modules) {
    uint32_t Count = 0;
    CALL_ZE_RET_FAIL(zeModuleGetKernelNames, Module, &Count, nullptr);
    if (Count == 0)
      continue;

    llvm::SmallVector<const char *> Names(Count);
    CALL_ZE_RET_FAIL(zeModuleGetKernelNames, Module, &Count, Names.data());

    for (auto *Name : Names) {
      KernelsToModuleMap.emplace(Name, Module);
    }
  }

  return OFFLOAD_SUCCESS;
}

} // namespace llvm::omp::target::plugin
