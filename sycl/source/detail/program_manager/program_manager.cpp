//==------ program_manager.cpp --- SYCL program manager---------------------==//
//
// The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/program_manager/program_manager.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/stl.hpp>

#include <assert.h>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace cl {
namespace sycl {
namespace detail {

ProgramManager &ProgramManager::getInstance() {
  // The singleton ProgramManager instance, uses the "magic static" idiom.
  static ProgramManager Instance;
  return Instance;
}

static cl_device_id getFirstDevice(cl_context Context) {
  cl_uint NumDevices = 0;
  cl_int Err = clGetContextInfo(Context, CL_CONTEXT_NUM_DEVICES,
                                sizeof(NumDevices), &NumDevices,
                                /*param_value_size_ret=*/ nullptr);
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
                                      const vector_class<char> &BinProg) {
  // FIXME: we don't yet support multiple device binaries or multiple devices
  // with a single binary.
#ifndef _NDEBUG
  cl_uint NumDevices = 0;
  CHECK_OCL_CODE(clGetContextInfo(Context, CL_CONTEXT_NUM_DEVICES,
                                  sizeof(NumDevices), &NumDevices,
                                  /*param_value_size_ret=*/ nullptr));
  assert(NumDevices > 0 &&
         "Only a single device is supported for AOT compilation");
#endif

  cl_device_id Device = getFirstDevice(Context);
  cl_int Err = CL_SUCCESS;
  cl_int BinaryStatus = CL_SUCCESS;
  size_t BinarySize = BinProg.size();
  const unsigned char *Binary = (const unsigned char *) &BinProg[0];
  cl_program Program =
    clCreateProgramWithBinary(Context, 1, &Device, &BinarySize, &Binary,
                              &BinaryStatus, &Err);
  CHECK_OCL_CODE(Err);

  return Program;
}

cl_program createSpirvProgram(const cl_context Context,
                              const vector_class<char> &SpirvProg) {
  cl_int Err = CL_SUCCESS;
  cl_program ClProgram = clCreateProgramWithIL(Context, SpirvProg.data(),
                                               SpirvProg.size(), &Err);
  CHECK_OCL_CODE(Err);
  return ClProgram;
}

static cl_program createProgram(cl_context Context,
                                const vector_class<char> &DeviceProg) {
  cl_program Program = nullptr;
  int32_t SpirvMagic = 0;
  const int32_t ValidSpirvMagic = 0x07230203;
  if (DeviceProg.size() > sizeof(SpirvMagic)) {
    std::copy(DeviceProg.begin(),
              DeviceProg.begin() + sizeof(SpirvMagic),
              (char*)&SpirvMagic);

    if (SpirvMagic == ValidSpirvMagic) {
      Program = createSpirvProgram(Context, DeviceProg);
    }
  }

  // Program is not a SPIR-V, assume a device binary
  if (!Program) {
    Program = createBinaryProgram(Context, DeviceProg);
  }

  return Program;
}

cl_program ProgramManager::getBuiltOpenCLProgram(const context &Context) {
  cl_program &ClProgram = m_CachedSpirvPrograms[Context];
  if (!ClProgram) {
    vector_class<char> DeviceProg = getSpirvSource();

    cl_context ClContext = Context.get();
    ClProgram = createProgram(ClContext, DeviceProg);
    clReleaseContext(ClContext);

    build(ClProgram);
  }
  return ClProgram;
}

cl_kernel ProgramManager::getOrCreateKernel(const context &Context,
                                            const char *KernelName) {
  cl_program Program = getBuiltOpenCLProgram(Context);
  auto &KernelsCache = m_CachedKernels[Program];
  cl_kernel &Kernel = KernelsCache[string_class(KernelName)];
  if (!Kernel) {
    cl_int Err = CL_SUCCESS;
    Kernel = clCreateKernel(Program, KernelName, &Err);
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

const vector_class<char> ProgramManager::getSpirvSource() {
  // TODO FIXME make this function thread-safe
  if (!m_SpirvSource) {
    vector_class<char> *DataPtr = nullptr;

    if (DeviceImages && !std::getenv("SYCL_USE_KERNEL_SPV")) {
      assert(DeviceImages->NumDeviceImages == 1 &&
             "only single image is supported for now");
      const __tgt_device_image &Img = DeviceImages->DeviceImages[0];
      auto *BegPtr = reinterpret_cast<const char *>(Img.ImageStart);
      auto *EndPtr = reinterpret_cast<const char *>(Img.ImageEnd);
      ptrdiff_t ImgSize = EndPtr - BegPtr;
      DataPtr = new vector_class<char>(static_cast<size_t>(ImgSize));
      // TODO this code is expected to be heavily refactored, this copying
      // might be redundant (unless we don't want to work on live .rodata)
      std::copy(BegPtr, EndPtr, DataPtr->begin());

      if (std::getenv("SYCL_DUMP_IMAGES")) {
        std::ofstream F("kernel.spv", std::ios::binary);

        if (!F.is_open())
          throw compile_program_error("Can not write kernel.spv\n");
        F.write(BegPtr, ImgSize);
        F.close();
      }
    } else {
      std::ifstream File("kernel.spv", std::ios::binary);
      if (!File.is_open()) {
        throw compile_program_error("Can not open kernel.spv\n");
      }
      File.seekg(0, std::ios::end);
      DataPtr = new vector_class<char>(File.tellg());
      File.seekg(0);
      File.read(DataPtr->data(), DataPtr->size());
      File.close();
    }
    m_SpirvSource.reset(DataPtr);
  }
  // TODO makes unnecessary copy of the data
  return *m_SpirvSource.get();
}

void ProgramManager::build(cl_program &ClProgram, const string_class &Options,
                           std::vector<cl_device_id> ClDevices) {

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

bool ProgramManager::ContextLess::operator()(const context &LHS,
                                             const context &RHS) const {
  return std::hash<context>()(LHS) < std::hash<context>()(RHS);
}

} // namespace detail
} // namespace sycl
} // namespace cl

extern "C" void __tgt_register_lib(__tgt_bin_desc *desc) {
  // TODO FIXME POC hacky implementation to replace the "kernel.spv" dirtier
  // hack and enable separate compilation of device code.
  // Major TODOs:
  // - support (images for) multiple devices - depends on the native runtime
  //   interface adoption
  // - add synchronization to avoid races when multiple modules (.exe and .dlls)
  //   try to do image registration at the same time
  // - merge with program and kernel management infrastructure (requires more
  //   design work)
  cl::sycl::detail::ProgramManager::getInstance().setDeviceImages(desc);
}

// Executed as a part of current module's (.exe, .dll) static initialization
extern "C" void __tgt_unregister_lib(__tgt_bin_desc *desc) {
  // TODO implement the function
}
