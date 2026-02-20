//===-- Main entry into the loader interface ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This utility is used to launch standard programs onto the GPU in conjunction
// with the LLVM 'libc' project. It is designed to mimic a standard emulator
// workflow, allowing for unit tests to be run on the GPU directly.
//
//===----------------------------------------------------------------------===//

#include "llvm-gpu-loader.h"

#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"
#include "llvm/TargetParser/Triple.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/file.h>

using namespace llvm;

static cl::OptionCategory LoaderCategory("loader options");

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden,
                          cl::cat(LoaderCategory));

static cl::opt<unsigned>
    ThreadsX("threads-x", cl::desc("Number of threads in the 'x' dimension"),
             cl::init(1), cl::cat(LoaderCategory));
static cl::opt<unsigned>
    ThreadsY("threads-y", cl::desc("Number of threads in the 'y' dimension"),
             cl::init(1), cl::cat(LoaderCategory));
static cl::opt<unsigned>
    ThreadsZ("threads-z", cl::desc("Number of threads in the 'z' dimension"),
             cl::init(1), cl::cat(LoaderCategory));
static cl::alias threads("threads", cl::aliasopt(ThreadsX),
                         cl::desc("Alias for --threads-x"),
                         cl::cat(LoaderCategory));

static cl::opt<unsigned>
    BlocksX("blocks-x", cl::desc("Number of blocks in the 'x' dimension"),
            cl::init(1), cl::cat(LoaderCategory));
static cl::opt<unsigned>
    BlocksY("blocks-y", cl::desc("Number of blocks in the 'y' dimension"),
            cl::init(1), cl::cat(LoaderCategory));
static cl::opt<unsigned>
    BlocksZ("blocks-z", cl::desc("Number of blocks in the 'z' dimension"),
            cl::init(1), cl::cat(LoaderCategory));
static cl::alias Blocks("blocks", cl::aliasopt(BlocksX),
                        cl::desc("Alias for --blocks-x"),
                        cl::cat(LoaderCategory));

static cl::opt<std::string> File(cl::Positional, cl::Required,
                                 cl::desc("<gpu executable>"),
                                 cl::cat(LoaderCategory));
static cl::list<std::string> Args(cl::ConsumeAfter,
                                  cl::desc("<program arguments>..."),
                                  cl::cat(LoaderCategory));

// The arguments to the '_begin' kernel.
struct BeginArgs {
  int Argc;
  void *Argv;
  void *Envp;
};

// The arguments to the '_start' kernel.
struct StartArgs {
  int Argc;
  void *Argv;
  void *Envp;
  void *Ret;
};

// The arguments to the '_end' kernel.
struct EndArgs {};

[[noreturn]] static void handleError(Error E) {
  outs().flush();
  logAllUnhandledErrors(std::move(E), WithColor::error(errs(), "loader"));
  exit(EXIT_FAILURE);
}

[[noreturn]] static void handleError(ol_result_t Err, unsigned Line) {
  fprintf(stderr, "%s:%d %s\n", __FILE__, Line, Err->Details);
  exit(EXIT_FAILURE);
}

#define OFFLOAD_ERR(X)                                                         \
  if (ol_result_t Err = X)                                                     \
    handleError(Err, __LINE__);

static void *copyArgumentVector(int Argc, const char **Argv,
                                ol_device_handle_t Device) {
  size_t ArgSize = sizeof(char *) * (Argc + 1);
  size_t StringLen = 0;
  for (int i = 0; i < Argc; ++i)
    StringLen += strlen(Argv[i]) + 1;

  // We allocate enough space for a null terminated array and all the strings.
  void *DevArgv;
  OFFLOAD_ERR(
      olMemAlloc(Device, OL_ALLOC_TYPE_HOST, ArgSize + StringLen, &DevArgv));
  if (!DevArgv)
    handleError(
        createStringError("Failed to allocate memory for environment."));

  // Store the strings linerally in the same memory buffer.
  void *DevString = reinterpret_cast<uint8_t *>(DevArgv) + ArgSize;
  for (int i = 0; i < Argc; ++i) {
    size_t size = strlen(Argv[i]) + 1;
    std::memcpy(DevString, Argv[i], size);
    static_cast<void **>(DevArgv)[i] = DevString;
    DevString = reinterpret_cast<uint8_t *>(DevString) + size;
  }

  // Ensure the vector is null terminated.
  reinterpret_cast<void **>(DevArgv)[Argc] = nullptr;
  return DevArgv;
}

void *copyEnvironment(const char **Envp, ol_device_handle_t Device) {
  int Envc = 0;
  for (const char **Env = Envp; *Env != 0; ++Env)
    ++Envc;

  return copyArgumentVector(Envc, Envp, Device);
}

ol_device_handle_t findDevice(MemoryBufferRef Binary) {
  ol_device_handle_t Device;
  std::tuple Data = std::make_tuple(&Device, &Binary);
  OFFLOAD_ERR(olIterateDevices(
      [](ol_device_handle_t Device, void *UserData) {
        auto &[Output, Binary] = *reinterpret_cast<decltype(Data) *>(UserData);
        bool IsValid = false;
        OFFLOAD_ERR(olIsValidBinary(Device, Binary->getBufferStart(),
                                    Binary->getBufferSize(), &IsValid));
        if (!IsValid)
          return true;

        *Output = Device;
        return false;
      },
      &Data));
  return Device;
}

ol_device_handle_t getHostDevice() {
  ol_device_handle_t Device;
  OFFLOAD_ERR(olIterateDevices(
      [](ol_device_handle_t Device, void *UserData) {
        ol_platform_handle_t Platform;
        olGetDeviceInfo(Device, OL_DEVICE_INFO_PLATFORM, sizeof(Platform),
                        &Platform);
        ol_platform_backend_t Backend;
        olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND, sizeof(Backend),
                          &Backend);

        auto &Output = *reinterpret_cast<decltype(Device) *>(UserData);
        if (Backend == OL_PLATFORM_BACKEND_HOST) {
          Output = Device;
          return false;
        }
        return true;
      },
      &Device));
  return Device;
}

template <typename Args>
void launchKernel(ol_queue_handle_t Queue, ol_device_handle_t Device,
                  ol_program_handle_t Program, const char *Name,
                  ol_kernel_launch_size_args_t LaunchArgs, Args &KernelArgs) {
  ol_symbol_handle_t Kernel;
  OFFLOAD_ERR(olGetSymbol(Program, Name, OL_SYMBOL_KIND_KERNEL, &Kernel));

  OFFLOAD_ERR(olLaunchKernel(Queue, Device, Kernel, &KernelArgs,
                             std::is_empty_v<Args> ? 0 : sizeof(Args),
                             &LaunchArgs));
}

int main(int argc, const char **argv, const char **envp) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  cl::HideUnrelatedOptions(LoaderCategory);
  cl::ParseCommandLineOptions(
      argc, argv,
      "A utility used to launch unit tests built for a GPU target. This is\n"
      "intended to provide an intrface simular to cross-compiling emulators\n");

  if (Help) {
    cl::PrintHelpMessage();
    return EXIT_SUCCESS;
  }

  if (Error Err = loadLLVMOffload())
    handleError(std::move(Err));

  ErrorOr<std::unique_ptr<MemoryBuffer>> ImageOrErr =
      MemoryBuffer::getFileOrSTDIN(File);
  if (std::error_code EC = ImageOrErr.getError())
    handleError(errorCodeToError(EC));
  MemoryBufferRef Image = **ImageOrErr;

  ol_platform_backend_t Backend;
  ol_init_args_t InitArgs = OL_INIT_ARGS_INIT;

  file_magic Magic = identify_magic(Image.getBuffer());
  if (Magic >= file_magic::elf && Magic <= file_magic::elf_core) {
    Expected<object::ELFFile<object::ELF64LE>> ElfOrErr =
        object::ELFFile<object::ELF64LE>::create(Image.getBuffer());
    if (!ElfOrErr)
      handleError(ElfOrErr.takeError());

    switch (ElfOrErr->getHeader().e_machine) {
    case ELF::EM_AMDGPU:
      Backend = OL_PLATFORM_BACKEND_AMDGPU;
      break;
    case ELF::EM_CUDA:
      Backend = OL_PLATFORM_BACKEND_CUDA;
      break;
    default:
      handleError(createStringError(
          "unhandled ELF architecture: %s",
          ELF::convertEMachineToArchName(ElfOrErr->getHeader().e_machine)
              .data()));
    }
    InitArgs.NumPlatforms = 1;
    InitArgs.Platforms = &Backend;
  }

  SmallVector<const char *> NewArgv = {File.c_str()};
  llvm::transform(Args, std::back_inserter(NewArgv),
                  [](const std::string &Arg) { return Arg.c_str(); });

  OFFLOAD_ERR(olInit(&InitArgs));
  ol_device_handle_t Device = findDevice(Image);
  ol_device_handle_t Host = getHostDevice();

  ol_program_handle_t Program;
  OFFLOAD_ERR(olCreateProgram(Device, Image.getBufferStart(),
                              Image.getBufferSize(), &Program));

  ol_queue_handle_t Queue;
  OFFLOAD_ERR(olCreateQueue(Device, &Queue));

  int DevArgc = static_cast<int>(NewArgv.size());
  void *DevArgv = copyArgumentVector(NewArgv.size(), NewArgv.begin(), Device);
  void *DevEnvp = copyEnvironment(envp, Device);

  void *DevRet;
  OFFLOAD_ERR(olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, sizeof(int), &DevRet));

  ol_kernel_launch_size_args_t BeginLaunch{1, {1, 1, 1}, {1, 1, 1}, 0};
  BeginArgs BeginArgs = {DevArgc, DevArgv, DevEnvp};
  launchKernel(Queue, Device, Program, "_begin", BeginLaunch, BeginArgs);
  OFFLOAD_ERR(olSyncQueue(Queue));

  uint32_t Dims = (BlocksZ > 1) ? 3 : (BlocksY > 1) ? 2 : 1;
  ol_kernel_launch_size_args_t StartLaunch{Dims,
                                           {BlocksX, BlocksY, BlocksZ},
                                           {ThreadsX, ThreadsY, ThreadsZ},
                                           /*SharedMemBytes=*/0};
  StartArgs StartArgs = {DevArgc, DevArgv, DevEnvp, DevRet};
  launchKernel(Queue, Device, Program, "_start", StartLaunch, StartArgs);

  ol_kernel_launch_size_args_t EndLaunch{1, {1, 1, 1}, {1, 1, 1}, 0};
  EndArgs EndArgs = {};
  launchKernel(Queue, Device, Program, "_end", EndLaunch, EndArgs);

  int Ret;
  OFFLOAD_ERR(olMemcpy(Queue, &Ret, Host, DevRet, Device, sizeof(int)));
  OFFLOAD_ERR(olSyncQueue(Queue));

  OFFLOAD_ERR(olMemFree(DevArgv));
  OFFLOAD_ERR(olMemFree(DevEnvp));
  OFFLOAD_ERR(olDestroyQueue(Queue));
  OFFLOAD_ERR(olDestroyProgram(Program));
  OFFLOAD_ERR(olShutDown());

  return Ret;
}
