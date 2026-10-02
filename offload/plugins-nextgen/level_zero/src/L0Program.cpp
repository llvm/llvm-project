//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Level Zero Program abstraction.
//
//===----------------------------------------------------------------------===//

#include "L0Plugin.h"
#include "L0Program.h"

namespace llvm::omp::target::plugin {

Error L0GlobalHandlerTy::getGlobalMetadataFromDevice(GenericDeviceTy &Device,
                                                     DeviceImageTy &Image,
                                                     GlobalTy &DeviceGlobal) {
  const char *GlobalName = DeviceGlobal.getName().data();
  size_t SymbolSize = 0;
  void *SymbolAddr = nullptr;

  L0ProgramTy &Program = L0ProgramTy::makeL0Program(Image);
  if (auto Err =
          Program.getSymbolMetadata(GlobalName, &SymbolAddr, &SymbolSize))
    return Err;

  // Save the pointer to the symbol allowing nullptr.
  DeviceGlobal.setPtr(SymbolAddr);
  DeviceGlobal.setSize(SymbolSize);

  return Plugin::success();
}

bool L0GlobalHandlerTy::isExportedSymbol(uint32_t Flags) {
  // Images returned by the Level Zero runtime do not correctly expose kernel
  // functions as global symbols. Bypass the normal ELF handling.here.
  uint32_t Ignored = SymbolRef::SF_Undefined | SymbolRef::SF_Hidden |
                     SymbolRef::SF_FormatSpecific;
  return !(Flags & Ignored);
}

inline L0DeviceTy &L0ProgramTy::getL0Device() const {
  return L0DeviceTy::makeL0Device(getDevice());
}

Error L0ProgramTy::deinit() {
  for (auto *Kernel : Kernels) {
    if (auto Err = Kernel->deinit())
      return Err;
    getL0Device().getPlugin().free(Kernel);
  }
  for (auto Module : Modules) {
    CALL_ZE_RET_ERROR(zeModuleDestroy, Module);
  }
  return Plugin::success();
}

Error L0ProgramBuilderTy::addModule(size_t Size, const uint8_t *Image,
                                    const std::string_view CommonBuildOptions,
                                    ze_module_format_t Format) {
  auto &L0Device = getL0Device();
  const ze_module_constants_t SpecConstants =
      L0Device.getPlugin()
          .getOptions()
          .CommonSpecConstants.getModuleConstants();

  std::string BuildOptions(CommonBuildOptions);

  bool IsLibModule =
      BuildOptions.find("-library-compilation") != std::string::npos;

  ze_module_desc_t ModuleDesc{};
  ModuleDesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  ModuleDesc.pNext = nullptr;
  ModuleDesc.format = Format;
  ze_module_handle_t Module = nullptr;
  ze_module_build_log_handle_t BuildLog = nullptr;

  // Build a single module from a single image.
  ModuleDesc.inputSize = Size;
  ModuleDesc.pInputModule = Image;
  ModuleDesc.pBuildFlags = BuildOptions.c_str();
  ModuleDesc.pConstants = &SpecConstants;
  ze_result_t RC;
  CALL_ZE(RC, zeModuleCreate, getZeContext(), L0Device.getZeDevice(),
          &ModuleDesc, &Module, &BuildLog);
  if (BuildLog)
    zeModuleBuildLogDestroy(BuildLog);
  if (RC != ZE_RESULT_SUCCESS) {
    // zeModuleCreate compiles/loads the provided image, so a build failure here
    // means the image itself could not be loaded for this device (e.g. a
    // truncated or malformed binary) rather than a generic JIT failure of an
    // otherwise valid program. Report it as INVALID_BINARY in that case (as
    // opposed to the default mapping of ZE_RESULT_ERROR_MODULE_BUILD_FAILURE
    // to ErrorCode::COMPILE_FAILURE).
    const auto ErrCode = RC == ZE_RESULT_ERROR_MODULE_BUILD_FAILURE
                             ? ErrorCode::INVALID_BINARY
                             : getOffloadErrorCode(RC);
    return Plugin::error(ErrCode, "zeModuleCreate failed with error %d, %s", RC,
                         getZeErrorName(RC));
  }

  // Check if module link is required. We do not need this check for
  // library module.
  if (!RequiresModuleLink && !IsLibModule) {
    ze_module_properties_t Properties = {ZE_STRUCTURE_TYPE_MODULE_PROPERTIES,
                                         nullptr, 0};
    ze_result_t RC;
    CALL_ZE(RC, zeModuleGetProperties, Module, &Properties);
    if (RC == ZE_RESULT_SUCCESS)
      RequiresModuleLink = Properties.flags & ZE_MODULE_PROPERTY_FLAG_IMPORTS;
  }
  // For now, assume the first module contains libraries, globals.
  if (Modules.empty())
    GlobalModule = Module;
  Modules.push_back(Module);
  L0Device.addGlobalModule(Module);
  return Plugin::success();
}

Error L0ProgramBuilderTy::linkModules() {
  auto &L0Device = getL0Device();
  if (!RequiresModuleLink) {
    ODBG(OLDT_Module) << "Module link is not required";
    return Plugin::success();
  }

  if (Modules.empty())
    return Plugin::error(ErrorCode::UNKNOWN,
                         "Invalid number of modules when linking modules");

  ze_module_build_log_handle_t LinkLog = nullptr;
  CALL_ZE_RET_ERROR(zeModuleDynamicLink,
                    static_cast<uint32_t>(L0Device.getNumGlobalModules()),
                    L0Device.getGlobalModulesArray(), &LinkLog);
  return Plugin::success();
}

static void replaceDriverOptsWithBackendOpts(const L0DeviceTy &Device,
                                             std::string &Options) {
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
    if (Pos != std::string::npos)
      Options.replace(Pos, OptPair.Option.length(), OptPair.BackendOption);
  }
}

Error L0ProgramBuilderTy::buildModules(const std::string_view BuildOptions) {
  auto &L0Device = getL0Device();
  auto Image = getMemoryBuffer();

  // Check if image is an inner OffloadBinary (nested format)
  if (identify_magic(Image.getBuffer()) == file_magic::offload_binary) {
    ODBG(OLDT_Module) << "Processing nested OffloadBinary image";

    // Parse inner OffloadBinary
    auto InnerBinariesOrErr = llvm::object::OffloadBinary::create(Image);
    if (!InnerBinariesOrErr)
      return Plugin::error(
          ErrorCode::INVALID_BINARY, "Failed to parse inner OffloadBinary: %s",
          llvm::toString(InnerBinariesOrErr.takeError()).c_str());

    auto &InnerBinaries = *InnerBinariesOrErr;

    // Should contain exactly one image
    if (InnerBinaries.size() != 1)
      return Plugin::error(ErrorCode::INVALID_BINARY,
                           "Expected single inner OffloadBinary entry, got %zu",
                           InnerBinaries.size());

    const llvm::object::OffloadBinary *InnerBinary = InnerBinaries[0].get();
    llvm::object::ImageKind ImageKind = InnerBinary->getImageKind();

    // Extract image data from inner binary
    llvm::StringRef ImageData = InnerBinary->getImage();
    const uint8_t *ImgBegin =
        reinterpret_cast<const uint8_t *>(ImageData.data());

    // Read metadata from inner binary
    llvm::StringRef Version = InnerBinary->getString("version");
    llvm::StringRef CompileOpts = InnerBinary->getString("compile-opts");
    llvm::StringRef LinkOpts = InnerBinary->getString("link-opts");

    ODBG(OLDT_Module) << "Inner OffloadBinary metadata: version=" << Version
                      << ", kind=" << ImageKind;

    // Build options string combining BuildOptions with compile/link opts
    std::string Options(BuildOptions);
    if (!CompileOpts.empty() || !LinkOpts.empty()) {
      if (!CompileOpts.empty())
        Options += " " + CompileOpts.str();
      if (!LinkOpts.empty())
        Options += " " + LinkOpts.str();
      replaceDriverOptsWithBackendOpts(L0Device, Options);
      ODBG(OLDT_Module) << "Using compile options: " << CompileOpts
                        << ", link options: " << LinkOpts;
    }

    // Determine module format based on image kind
    ze_module_format_t ModuleFormat;
    if (ImageKind == llvm::object::IMG_SPIRV) {
      // SPIR-V intermediate language
      ODBG(OLDT_Module) << "Loading SPIR-V module";
      ModuleFormat = ZE_MODULE_FORMAT_IL_SPIRV;
    } else if (ImageKind == llvm::object::IMG_Object) {
      // Native binary format
      ODBG(OLDT_Module) << "Loading native binary module";
      ModuleFormat = ZE_MODULE_FORMAT_NATIVE;
    } else {
      return Plugin::error(ErrorCode::INVALID_BINARY,
                           "Unsupported image kind %d in inner OffloadBinary",
                           static_cast<int>(ImageKind));
    }

    // Load module into Level Zero
    auto Err = addModule(ImageData.size(), ImgBegin, Options, ModuleFormat);
    if (Err)
      return Err;

    if (RequiresModuleLink) {
      ODBG(OLDT_Module) << "Linking modules after adding OffloadBinary image";
      if (auto Err = linkModules())
        return Err;
    }
    return Plugin::success();
  }

  if (identify_magic(Image.getBuffer()) == file_magic::spirv_object) {
    ODBG(OLDT_Module) << "Processing raw SPIR-V image";
    const uint8_t *ImgBegin =
        reinterpret_cast<const uint8_t *>(Image.getBufferStart());
    auto Err = addModule(Image.getBufferSize(), ImgBegin, BuildOptions,
                         ZE_MODULE_FORMAT_IL_SPIRV);
    if (Err)
      return Err;

    if (RequiresModuleLink) {
      ODBG(OLDT_Module) << "Linking modules after adding SPIR-V image";
      if (auto Err = linkModules())
        return Err;
    }
    return Plugin::success();
  }

  return Plugin::error(ErrorCode::INVALID_BINARY,
                       "Unsupported image format for L0 plugin");
}

Expected<std::unique_ptr<MemoryBuffer>> L0ProgramBuilderTy::getELF() {
  assert(GlobalModule != nullptr && "GlobalModule is null");

  size_t Size = 0;

  CALL_ZE_RET_ERROR(zeModuleGetNativeBinary, GlobalModule, &Size, nullptr);
  std::vector<uint8_t> ELFData(Size);
  CALL_ZE_RET_ERROR(zeModuleGetNativeBinary, GlobalModule, &Size,
                    ELFData.data());
  return MemoryBuffer::getMemBufferCopy(
      StringRef(reinterpret_cast<const char *>(ELFData.data()), Size),
      /*BufferName=*/"L0Program ELF");
}

Error L0ProgramTy::getSymbolMetadata(const char *Name, void **AddrPtr,
                                     size_t *SizePtr) const {
  if (!Name || !AddrPtr || !SizePtr)
    return Plugin::error(ErrorCode::INVALID_ARGUMENT,
                         "Invalid arguments to getSymbolDeviceAddr");

  size_t SymbolSize = 0;
  void *SymbolAddr = nullptr;
  ze_result_t RC;
  for (auto Module : Modules) {
    CALL_ZE(RC, zeModuleGetGlobalPointer, Module, Name, &SymbolSize,
            &SymbolAddr);
    if (RC == ZE_RESULT_SUCCESS && SymbolAddr) {
      *AddrPtr = SymbolAddr;
      *SizePtr = SymbolSize;
      return Plugin::success();
    }
    CALL_ZE(RC, zeModuleGetFunctionPointer, Module, Name, &SymbolAddr);
    if (RC == ZE_RESULT_SUCCESS && SymbolAddr) {
      *AddrPtr = SymbolAddr;
      *SizePtr = 0;
      return Plugin::success();
    }
  }
  return Plugin::error(ErrorCode::NOT_FOUND, "symbol '%s' not found on device",
                       Name);
}

Error L0ProgramTy::readGlobalVariable(const char *Name, size_t Size,
                                      void *HostPtr) {
  size_t SizeDummy = 0;
  void *DevicePtr = nullptr;
  ze_result_t RC;
  CALL_ZE(RC, zeModuleGetGlobalPointer, GlobalModule, Name, &SizeDummy,
          &DevicePtr);
  if (RC != ZE_RESULT_SUCCESS || !DevicePtr) {
    return Plugin::error(ErrorCode::INVALID_ARGUMENT,
                         "Cannot read from device global variable %s", Name);
  }
  return getL0Device().enqueueMemCopyAndSync(HostPtr, DevicePtr, Size);
}

Error L0ProgramTy::writeGlobalVariable(const char *Name, size_t Size,
                                       const void *HostPtr) {
  size_t SizeDummy = 0;
  void *DevicePtr = nullptr;
  ze_result_t RC;
  CALL_ZE(RC, zeModuleGetGlobalPointer, GlobalModule, Name, &SizeDummy,
          &DevicePtr);
  if (RC != ZE_RESULT_SUCCESS || !DevicePtr) {
    return Plugin::error(ErrorCode::INVALID_ARGUMENT,
                         "Cannot write to device global variable %s", Name);
  }
  return getL0Device().enqueueMemCopyAndSync(DevicePtr, HostPtr, Size);
}

Error L0ProgramTy::loadModuleKernels() {
  // We need to build kernels here before filling the offload entries since we
  // don't know which module contains a specific kernel with a name.
  for (auto Module : Modules) {
    uint32_t Count = 0;
    CALL_ZE_RET_ERROR(zeModuleGetKernelNames, Module, &Count,
                      /*Names=*/nullptr);
    if (Count == 0)
      continue;

    llvm::SmallVector<const char *> Names(Count);
    CALL_ZE_RET_ERROR(zeModuleGetKernelNames, Module, &Count, Names.data());

    for (auto *Name : Names) {
      KernelsToModuleMap.emplace(Name, Module);
    }
  }

  return Plugin::success();
}

} // namespace llvm::omp::target::plugin
