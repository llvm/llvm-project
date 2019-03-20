//==------ program_manager.cpp --- SYCL program manager---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/program_manager/program_manager.hpp>
#include <CL/sycl/detail/util.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/stl.hpp>

#include <assert.h>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>

namespace cl {
namespace sycl {
namespace detail {

static constexpr int DbgProgMgr = 0;

ProgramManager &ProgramManager::getInstance() {
  // The singleton ProgramManager instance, uses the "magic static" idiom.
  static ProgramManager Instance;
  return Instance;
}

static cl_device_id getFirstDevice(cl_context Context) {
  cl_uint NumDevices = 0;
  cl_int Err = clGetContextInfo(Context, CL_CONTEXT_NUM_DEVICES,
                                sizeof(NumDevices), &NumDevices,
                                /*param_value_size_ret=*/nullptr);
  CHECK_OCL_CODE(Err);
  assert(NumDevices > 0 && "Context without devices?");

  vector_class<cl_device_id> Devices(NumDevices);
  size_t ParamValueSize = 0;
  Err = clGetContextInfo(Context, CL_CONTEXT_DEVICES,
                         sizeof(cl_device_id) * NumDevices, &Devices[0],
                         &ParamValueSize);
  CHECK_OCL_CODE(Err);
  assert(ParamValueSize == sizeof(cl_device_id) * NumDevices &&
         "Number of CL_CONTEXT_DEVICES should match CL_CONTEXT_NUM_DEVICES.");
  return Devices[0];
}

static cl_program createBinaryProgram(const cl_context Context,
                                      const unsigned char *Data,
                                      size_t DataLen) {
  // FIXME: we don't yet support multiple devices with a single binary.
#ifndef _NDEBUG
  cl_uint NumDevices = 0;
  CHECK_OCL_CODE(clGetContextInfo(Context, CL_CONTEXT_NUM_DEVICES,
                                  sizeof(NumDevices), &NumDevices,
                                  /*param_value_size_ret=*/nullptr));
  assert(NumDevices > 0 &&
         "Only a single device is supported for AOT compilation");
#endif

  cl_device_id Device = getFirstDevice(Context);
  cl_int Err = CL_SUCCESS;
  cl_int BinaryStatus = CL_SUCCESS;
  cl_program Program = clCreateProgramWithBinary(
      Context, 1 /*one binary*/, &Device, &DataLen, &Data, &BinaryStatus, &Err);
  CHECK_OCL_CODE(Err);

  return Program;
}

static cl_program createSpirvProgram(const cl_context Context,
                                     const unsigned char *Data,
                                     size_t DataLen) {
  cl_int Err = CL_SUCCESS;
  cl_program ClProgram = clCreateProgramWithIL(Context, Data, DataLen, &Err);
  CHECK_OCL_CODE(Err);
  return ClProgram;
}

cl_program ProgramManager::getBuiltOpenCLProgram(OSModuleHandle M,
                                                 const context &Context) {
  cl_program &ClProgram = m_CachedSpirvPrograms[std::make_pair(Context, M)];
  if (!ClProgram) {
    DeviceImage *Img = nullptr;
    ClProgram = loadProgram(M, Context, &Img);
    build(ClProgram, Img->BuildOptions);
  }
  return ClProgram;
}

cl_kernel ProgramManager::getOrCreateKernel(OSModuleHandle M,
                                            const context &Context,
                                            const string_class &KernelName) {
  if (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::getOrCreateKernel(" << M << ", "
              << getRawSyclObjImpl(Context) << ", " << KernelName << ")\n";
  }
  cl_program Program = getBuiltOpenCLProgram(M, Context);
  std::map<string_class, cl_kernel> &KernelsCache = m_CachedKernels[Program];
  cl_kernel &Kernel = KernelsCache[KernelName];
  if (!Kernel) {
    cl_int Err = CL_SUCCESS;
    Kernel = clCreateKernel(Program, KernelName.c_str(), &Err);
    CHECK_OCL_CODE(Err);
  }
  return Kernel;
}

cl_program ProgramManager::getClProgramFromClKernel(cl_kernel ClKernel) {
  cl_program ClProgram;
  CHECK_OCL_CODE(clGetKernelInfo(ClKernel, CL_KERNEL_PROGRAM,
                                 sizeof(cl_program), &ClProgram, nullptr));
  return ClProgram;
}

void ProgramManager::build(cl_program &ClProgram, const string_class &Options,
                           std::vector<cl_device_id> ClDevices) {

  if (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::build(" << ClProgram << ", " << Options
              << ", ... " << ClDevices.size() << ")\n";
  }
  const char *Opts = std::getenv("SYCL_PROGRAM_BUILD_OPTIONS");

  if (!Opts)
    Opts = Options.c_str();
  if (clBuildProgram(ClProgram, ClDevices.size(), ClDevices.data(),
                     Opts, nullptr, nullptr) == CL_SUCCESS)
    return;

  // Get OpenCL build log and add it to the exception message.
  size_t Size = 0;
  CHECK_OCL_CODE(
      clGetProgramInfo(ClProgram, CL_PROGRAM_DEVICES, 0, nullptr, &Size));

  std::vector<cl_device_id> DevIds(Size / sizeof(cl_device_id));
  CHECK_OCL_CODE(clGetProgramInfo(ClProgram, CL_PROGRAM_DEVICES, Size,
                                  DevIds.data(), nullptr));
  std::string Log;
  for (cl_device_id &DevId : DevIds) {
    CHECK_OCL_CODE(clGetProgramBuildInfo(ClProgram, DevId, CL_PROGRAM_BUILD_LOG,
                                         0, nullptr, &Size));
    std::vector<char> BuildLog(Size);
    CHECK_OCL_CODE(clGetProgramBuildInfo(ClProgram, DevId, CL_PROGRAM_BUILD_LOG,
                                         Size, BuildLog.data(), nullptr));
    device Dev(DevId);
    Log += "\nBuild program fail log for '" +
           Dev.get_info<info::device::name>() + "':\n" + BuildLog.data();
  }
  throw compile_program_error(Log.c_str());
}

bool ProgramManager::ContextAndModuleLess::
operator()(const std::pair<context, OSModuleHandle> &LHS,
           const std::pair<context, OSModuleHandle> &RHS) const {
  if (LHS.first != RHS.first)
    return getRawSyclObjImpl(LHS.first) < getRawSyclObjImpl(RHS.first);
  return reinterpret_cast<intptr_t>(LHS.second) <
         reinterpret_cast<intptr_t>(RHS.second);
}

void ProgramManager::addImages(cnri_bin_desc *DeviceImages) {
  std::lock_guard<std::mutex> Guard(Sync::getGlobalLock());

  for (int I = 0; I < DeviceImages->NumDeviceImages; I++) {
    cnri_device_image *Img = &(DeviceImages->DeviceImages[I]);
    OSModuleHandle M = OSUtil::getOSModuleHandle(Img);
    auto &Imgs = m_DeviceImages[M];

    if (Imgs == nullptr)
      Imgs.reset(new std::vector<DeviceImage *>());
    Imgs->push_back(Img);
  }
}

void ProgramManager::debugDumpBinaryImage(const DeviceImage *Img) const {
  std::cerr << "  --- Image " << Img << "\n";
  if (!Img)
    return;
  std::cerr << "    Version  : " << (int)Img->Version << "\n";
  std::cerr << "    Kind     : " << (int)Img->Kind << "\n";
  std::cerr << "    Format   : " << (int)Img->Format << "\n";
  std::cerr << "    Target   : " << Img->DeviceTargetSpec << "\n";
  std::cerr << "    Options  : "
            << (Img->BuildOptions ? Img->BuildOptions : "NULL") << "\n";
  std::cerr << "    Bin size : "
            << ((intptr_t)Img->ImageEnd - (intptr_t)Img->ImageStart) << "\n";
}

void ProgramManager::debugDumpBinaryImages() const {
  for (const auto &ModImgvec : m_DeviceImages) {
    std::cerr << "  ++++++ Module: " << ModImgvec.first << "\n";
    for (const auto *Img : *(ModImgvec.second)) {
      debugDumpBinaryImage(Img);
    }
  }
}

struct ImageDeleter {
  void operator()(DeviceImage *I) {
    delete[] I->ImageStart;
    delete I;
  }
};

cnri_program ProgramManager::loadProgram(OSModuleHandle M,
                                         const context &Context,
                                         DeviceImage **I) {
  std::lock_guard<std::mutex> Guard(Sync::getGlobalLock());

  if (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::loadProgram(" << M << ","
              << getRawSyclObjImpl(Context) << ")\n";
  }

  DeviceImage *Img = nullptr;
  bool UseKernelSpv = false;
  const std::string UseSpvEnv("SYCL_USE_KERNEL_SPV");

  if (const char *Spv = std::getenv(UseSpvEnv.c_str())) {
    // The env var requests that the program is loaded from a SPIRV file on disk
    UseKernelSpv = true;
    std::string Fname(Spv);
    std::ifstream File(Fname, std::ios::binary);

    if (!File.is_open()) {
      throw runtime_error(std::string("Can't open file specified via ") +
                          UseSpvEnv + ": " + Fname);
    }
    File.seekg(0, std::ios::end);
    size_t Size = File.tellg();
    auto *Data = new unsigned char[Size];
    File.seekg(0);
    File.read(reinterpret_cast<char *>(Data), Size);
    File.close();

    if (!File.good()) {
      delete[] Data;
      throw runtime_error(std::string("read from ") + Fname +
                          std::string(" failed"));
    }
    Img = new DeviceImage();
    Img->Version = CNRI_DEVICE_IMAGE_STRUCT_VERSION;
    Img->Kind = SYCL_OFFLOAD_KIND;
    Img->Format = CNRI_IMG_NONE;
    Img->DeviceTargetSpec = CNRI_TGT_STR_UNKNOWN;
    Img->BuildOptions = nullptr;
    Img->ManifestStart = nullptr;
    Img->ManifestEnd = nullptr;
    Img->ImageStart = Data;
    Img->ImageEnd = Data + Size;
    Img->EntriesBegin = nullptr;
    Img->EntriesEnd = nullptr;

    std::unique_ptr<DeviceImage, ImageDeleter> ImgPtr(Img, ImageDeleter());
    m_OrphanDeviceImages.emplace_back(std::move(ImgPtr));

    if (DbgProgMgr > 0) {
      std::cerr << "loaded device image from " << Fname << "\n";
    }
  } else {
    // Take all device images in module M and ask the native runtime under the
    // given context to choose one it prefers.
    auto ImgIt = m_DeviceImages.find(M);

    if (ImgIt == m_DeviceImages.end()) {
      throw runtime_error("No device program image found");
    }
    std::vector<DeviceImage *> *Imgs = (ImgIt->second).get();
    const cnri_context &Ctx = getRawSyclObjImpl(Context)->getHandleRef();

    if (cnriSelectDeviceImage(Ctx, Imgs->data(), (cl_uint)Imgs->size(), &Img) !=
        CNRI_SUCCESS) {
      throw device_error("cnriSelectDeviceImage failed");
    }
    if (DbgProgMgr > 0) {
      std::cerr << "available device images:\n";
      debugDumpBinaryImages();
      std::cerr << "selected device image: " << Img << "\n";
      debugDumpBinaryImage(Img);
    }
  }
  assert(Img->ImageEnd >= Img->ImageStart);
  size_t ImgSize = static_cast<size_t>(Img->ImageEnd - Img->ImageStart);

  // Determine the kind of the image if not set already
  if (Img->Kind == CNRI_IMG_NONE) {
    struct {
      cnri_device_image_format Fmt;
      const int32_t Magic;
    } Fmts[] = {{CNRI_IMG_SPIRV, 0x07230203},
                {CNRI_IMG_LLVMIR_BITCODE, 0x4243C0DE}};
    if (ImgSize >= sizeof(Fmts[0].Magic)) {
      std::remove_const<decltype(Fmts[0].Magic)>::type Hdr = 0;
      std::copy(Img->ImageStart, Img->ImageStart + sizeof(Hdr),
                reinterpret_cast<char *>(&Hdr));

      for (const auto &Fmt : Fmts) {
        if (Hdr == Fmt.Magic) {
          Img->Format = Fmt.Fmt;

          if (DbgProgMgr > 1) {
            std::cerr << "determined image format: " << Img->Format;
          }
          break;
        }
      }
    }
  }
  // Dump program image if requested
  if (std::getenv("SYCL_DUMP_IMAGES") && !UseKernelSpv) {
    std::string Fname("sycl_");
    Fname += Img->DeviceTargetSpec;
    std::string Ext;

    if (Img->Kind == CNRI_IMG_SPIRV) {
      Ext = ".spv";
    } else if (Img->Kind == CNRI_IMG_LLVMIR_BITCODE) {
      Ext = ".bc";
    } else {
      Ext = ".bin";
    }
    Fname += Ext;

    std::ofstream F(Fname, std::ios::binary);

    if (!F.is_open()) {
      throw runtime_error(std::string("Can not write ") + Fname);
    }
    F.write(reinterpret_cast<const char *>(Img->ImageStart), ImgSize);
    F.close();
  }
  // Load the selected image
  const cnri_context &Ctx = getRawSyclObjImpl(Context)->getHandleRef();
  cnri_program Res = nullptr;
  Res = Img->Format == CNRI_IMG_SPIRV
            ? createSpirvProgram(Ctx, Img->ImageStart, ImgSize)
            : createBinaryProgram(Ctx, Img->ImageStart, ImgSize);

  if (I)
    *I = Img;
  if (DbgProgMgr > 1) {
    std::cerr << "created native program: " << Res << "\n";
  }
  return Res;
}
} // namespace detail
} // namespace sycl
} // namespace cl

extern "C" void __tgt_register_lib(cnri_bin_desc *desc) {
  cl::sycl::detail::ProgramManager::getInstance().addImages(desc);
}

// Executed as a part of current module's (.exe, .dll) static initialization
extern "C" void __tgt_unregister_lib(cnri_bin_desc *desc) {
  // TODO implement the function
}
