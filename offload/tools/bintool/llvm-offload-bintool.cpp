
#include <OffloadAPI.h>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/ELFObjectFile.h"
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

int dumpBuffer(StringRef Filename, const char *Buffer, size_t Size) {
  std::error_code EC;
  raw_fd_ostream FS(Filename, EC, llvm::sys::fs::OF_None);
  if (EC) {
    std::cout << "Error saving " << Filename.str()
              << " image : " << EC.message();
    return 1;
  }
  FS.write(Buffer, Size);
  FS.close();
  std::cout << "Saved " << Filename.str() << "\n";
  return 0;
}

int dumpBuffer(StringRef Filename, StringRef Buffer) {
  return dumpBuffer(Filename, Buffer.data(), Buffer.size());
}

template <typename... Args>
std::string makeFilename(StringRef ext, Args &&...args) {
  return join_items("_", std::forward<Args>(args)...) + "." + ext.str();
}

#define NT_INTEL_ONEOMP_OFFLOAD_VERSION 1
#define NT_INTEL_ONEOMP_OFFLOAD_IMAGE_COUNT 2
#define NT_INTEL_ONEOMP_OFFLOAD_IMAGE_AUX 3

struct IntelKernelImageInfo {
  // 0 - native, 1 - SPIR-V.
  uint64_t Format = std::numeric_limits<uint64_t>::max();
  std::string CompileOpts;
  std::string LinkOpts;
  // We may have multiple sections created from split-kernel mode.
  std::vector<StringRef> Parts;

  IntelKernelImageInfo() = default;
  IntelKernelImageInfo(uint64_t Format, std::string CompileOpts,
                       std::string LinkOpts)
      : Format(Format), CompileOpts(std::move(CompileOpts)),
        LinkOpts(std::move(LinkOpts)) {}
};

template <typename T>
Error getIntelKernelImages(const T *Object,
                           llvm::SmallVector<IntelKernelImageInfo> &Images) {
  const auto &E = Object->getELFFile();
  auto Sections = E.sections();
  if (!Sections)
    return Sections.takeError();

  for (auto Sec : *Sections) {
    if (Sec.sh_type != ELF::SHT_NOTE)
      continue;
    Error Err = Error::success();
    for (auto Note : E.notes(Sec, Err)) {
      if (Err)
        return Err;
      if (Note.getName().str() != "INTELONEOMPOFFLOAD")
        continue;

      const uint64_t Type = Note.getType();
      auto DescStrRef = Note.getDescAsStringRef(4);
      switch (Type) {
      default:
      case NT_INTEL_ONEOMP_OFFLOAD_VERSION:
      case NT_INTEL_ONEOMP_OFFLOAD_IMAGE_COUNT:
        break;
      case NT_INTEL_ONEOMP_OFFLOAD_IMAGE_AUX:
        uint64_t Idx = 0;
        uint64_t Part1Id = 0;
        llvm::SmallVector<llvm::StringRef, 4> Parts;

        DescStrRef.split(Parts, '\0', /* MaxSplit = */ 4,
                         /* KeepEmpty = */ true);

        // Ignore records with less than 4 strings or invalid index/part id.
        if (Parts.size() != 4 || Parts[0].getAsInteger(10, Idx) ||
            Parts[1].getAsInteger(10, Part1Id))
          continue;

        if (Idx >= Images.size())
          Images.resize(Idx + 1);

        Images[Idx] =
            IntelKernelImageInfo(Part1Id, Parts[2].str(), Parts[3].str());
        break;
      }
    }
  }

  for (auto Sec : *Sections) {
    const char *Prefix = "__openmp_offload_spirv_";
    auto ExpectedSectionName = E.getSectionName(Sec);
    if (!ExpectedSectionName)
      return ExpectedSectionName.takeError();
    auto &SectionNameRef = *ExpectedSectionName;
    if (!SectionNameRef.consume_front(Prefix))
      continue;

    // Expected section name in split-kernel mode with the following pattern:
    // __openmp_offload_spirv_<image_id>_<part_id>
    auto Parts = SectionNameRef.split('_');
    // It seems that we do not need part ID as long as they are ordered
    // in the image and we keep the ordering in the runtime.
    SectionNameRef = Parts.first;

    uint64_t Idx = 0;
    if (SectionNameRef.getAsInteger(10, Idx))
      continue;
    if (Idx >= Images.size())
      continue;

    auto Contents = E.getSectionContents(Sec);
    if (!Contents)
      return Contents.takeError();

    Images[Idx].Parts.push_back(StringRef(
        reinterpret_cast<const char *>((*Contents).data()), Sec.sh_size));
  }
  return Error::success();
}

Error processIntelBinary(const OffloadEnvTy &OffloadEnv, const BinEnvTy &BinEnv,
                         size_t FileId, const object::OffloadBinary &Binary) {
  StringRef Image = Binary.getImage();
  auto ImageBuffer = MemoryBuffer::getMemBuffer(Image, "", false);

  llvm::SmallVector<IntelKernelImageInfo> ImageList;

  auto ExpectedNewE = object::ELFObjectFileBase::createELFObjectFile(
      ImageBuffer->getMemBufferRef());
  if (!ExpectedNewE)
    return ExpectedNewE.takeError();

  if (auto *O = dyn_cast<object::ELF64LEObjectFile>((*ExpectedNewE).get())) {
    if (auto Err = getIntelKernelImages(O, ImageList))
      return Err;
  } else if (auto *O =
                 dyn_cast<object::ELF32LEObjectFile>((*ExpectedNewE).get())) {
    if (auto Err = getIntelKernelImages(O, ImageList))
      return Err;
  } else
    return make_error<error::OffloadError>(
        error::ErrorCode::INVALID_ARGUMENT,
        "Unsupported ELF type in the provided binary");

  constexpr static struct {
    StringRef Name;
    bool supportSplitKernel;
    StringRef Ext;
  } supportedImages[] = {
      {"binary", true, "bin"}, // Native binary format.
      {"spirv", false, "spv"}, // SPIR-V format.
  };
  constexpr size_t NumSupportedFormats =
      sizeof(supportedImages) / sizeof(supportedImages[0]);

  for (uint64_t Idx = 0; Idx < ImageList.size(); ++Idx) {
    auto &ImageInfo = ImageList[Idx];
    const auto NumParts = ImageInfo.Parts.size();

    // Skip unknown image format.
    if (ImageInfo.Format >= NumSupportedFormats ||
        (NumParts > 1 && !supportedImages[ImageInfo.Format].supportSplitKernel))
      continue;

    for (size_t I = 0; I < NumParts; I++)
      dumpBuffer(makeFilename(supportedImages[ImageInfo.Format].Ext,
                              BinEnv.Name, "kernels", std::to_string(Idx),
                              std::to_string(I)),
                 ImageInfo.Parts[I]);
  }

  return Error::success();
}

Error processOffloadBinary(const OffloadEnvTy &OffloadEnv,
                           const BinEnvTy &BinEnv, size_t FileId,
                           const object::OffloadBinary &Binary) {
  StringRef Image = Binary.getImage();

  dumpBuffer(makeFilename("bin", BinEnv.Name, "image", std::to_string(FileId)),
             Image);

  if (Binary.getTriple() == "spirv64-intel")
    if (auto Err = processIntelBinary(OffloadEnv, BinEnv, FileId, Binary))
      return Err;

  const char *Data = Image.data();
  const size_t Size = Image.size();

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
    dumpBuffer(
        makeFilename("bin", BinEnv.Name, "device", std::to_string(FileId)),
        ImageData, ImageSize);
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

  const size_t NumFiles = Files.size();
  std::cout << "Found " << NumFiles << " offloading "
            << (NumFiles == 1 ? "binary" : "binaries") << " in the file.\n";

  for (size_t FileId = 0; FileId < NumFiles; ++FileId) {
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
