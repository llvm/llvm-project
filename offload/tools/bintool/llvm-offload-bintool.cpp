
#include <OffloadAPI.h>

#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include "OffloadError.h"

#include <iostream>
#include <vector>

using namespace llvm;

#define OFFLOAD_ERR(X)                                                         \
  if (auto Err = X) {                                                          \
    return make_error<error::OffloadError>(                                    \
        static_cast<error::ErrorCode>(Err->Code), Err->Details);               \
  }

struct OffloadEnvTy {
  SmallVector<ol_device_handle_t> FoundDevices;
};

struct BinEnvTy {
  StringRef Name;
  MemoryBufferRef Data;

  BinEnvTy(StringRef Name, MemoryBufferRef Data) : Name(Name), Data(Data) {}
};

Error init(OffloadEnvTy &Data) {
  OFFLOAD_ERR(olInit());
  OFFLOAD_ERR(olIterateDevices(
      [](ol_device_handle_t Device, void *UserData) {
        reinterpret_cast<decltype(Data.FoundDevices) *>(UserData)->push_back(
            Device);
        return true;
      },
      &Data.FoundDevices));

  return Error::success();
}

Error deinit() {
  OFFLOAD_ERR(olShutDown());
  return Error::success();
}

int dumpBuffer(StringRef Base, StringRef Label, int BufferId, StringRef Ext,
               const char *Buffer, size_t Size) {
  std::string Filename =
      formatv("{0}_{1}_{2}.{3}", Base.str(), Label.str(), BufferId, Ext.str());
  std::error_code EC;
  raw_fd_ostream FS(Filename, EC, llvm::sys::fs::OF_None);
  if (EC) {
    std::cout << "Error saving " << Label.str() << " image : " << EC.message();
    return 1;
  }
  FS.write(Buffer, Size);
  FS.close();
  std::cout << "Saved " << Filename << "\n";
  return 0;
}

Error processIntelBinary(const OffloadEnvTy &OffloadEnv, const BinEnvTy &BinEnv,
                         size_t FileId, const object::OffloadBinary &Binary) {
  StringRef Image = Binary.getImage();
  const char *Data = Image.data();
  size_t Size = Image.size();

  std::cout << "Processing SPIR-V binary #" << FileId
            << " for target " << Binary.getTriple().str() << "/"
            << Binary.getArch().str() << ".\n";

  /* TODO: extract spv files from Binary*/

  dumpBuffer(BinEnv.Name, "spirv", FileId, "spv", Data, Size);
  return Error::success();
}

Error processOffloadBinary(const OffloadEnvTy &OffloadEnv, const BinEnvTy &BinEnv,
                         size_t FileId, const object::OffloadBinary &Binary) {
  StringRef Image = Binary.getImage();
  const char *Data = Image.data();
  size_t Size = Image.size();

  std::cout << "Processing offloading binary #" << FileId << " with image kind "
            << Binary.getImageKind() << " and offload kind "
            << Binary.getOffloadKind() << " for target "
            << Binary.getTriple().str() << "/" << Binary.getArch().str() << ".\n"
            << ".\n";

  dumpBuffer(BinEnv.Name, "image", FileId, "bin", Data, Size);

  if (Binary.getTriple() == "spirv64-intel")
    if (auto Err = processIntelBinary(OffloadEnv, BinEnv, FileId, Binary))
      return Err;

  for (auto &D : OffloadEnv.FoundDevices) {
    bool isValid = false;
    OFFLOAD_ERR(olIsValidBinary(D, Data, Size, &isValid));
    if (!isValid) {
      continue;
    }

    ol_program_handle_t Program;
    OFFLOAD_ERR(olCreateProgram(D, Data, Size, &Program));

    const char *ImageData;
    size_t ImageSize;
    OFFLOAD_ERR(olGetProgramDeviceImage(Program, &ImageData, &ImageSize));
    dumpBuffer(BinEnv.Name, "device", FileId, "bin", ImageData, ImageSize);
    OFFLOAD_ERR(olDestroyProgram(Program));
    return Error::success();
  }
  std::cout << "No compatible devices found for the provided binary.\n";
  return Error::success();
}

Error processBinary(const OffloadEnvTy &OffloadEnv, const BinEnvTy &BinEnv) {
  llvm::SmallVector<object::OffloadFile> Files;
  if (auto Err = extractOffloadBinaries(BinEnv.Data, Files)) {
    return Err;
  }

  std::cout << "Found " << Files.size()
            << " offloading binary(ies) in the file.\n";

  for (size_t FileId = 0; FileId < Files.size(); ++FileId) {
    const auto &OffloadFile = Files[FileId];
    if (auto Err = processOffloadBinary(OffloadEnv, BinEnv, FileId,
                                      *OffloadFile.getBinary()))
      return Err;
  }

  return Error::success();
}

int main(int argc, char **argv) {
  cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<binary>"),
                                     cl::Required);

  cl::ParseCommandLineOptions(argc, argv, "llvm-offload-bintool\n");
  auto BinOrErr = MemoryBuffer::getFile(InputFilename, /*isText=*/false,
                                        /*RequiresNullTerminator=*/false);
  if (!BinOrErr) {
    llvm::errs() << "Error reading the binary file: "
                 << BinOrErr.getError().message() << "\n";
    return 1;
  }

  OffloadEnvTy OffloadEnv;
  BinEnvTy BinEnv(llvm::sys::path::filename(InputFilename).str(),
                  BinOrErr.get()->getMemBufferRef());

  if (auto Err = init(OffloadEnv)) {
    llvm::errs() << "Error initializing liboffload: "
                 << toString(std::move(Err)) << "\n";
    return 1;
  }

  if (auto Err = processBinary(OffloadEnv, BinEnv)) {
    llvm::errs() << "Error processing the binary: " << toString(std::move(Err))
                 << "\n";
    return 1;
  }

  if (auto Err = deinit()) {
    llvm::errs() << "Error shutting down liboffload: "
                 << toString(std::move(Err)) << "\n";
    return 1;
  }

  return 0;
}
