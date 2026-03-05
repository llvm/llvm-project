//===-- llvm-offload-binary.cpp - offload binary management utility -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tool takes several device object files and bundles them into a single
// binary image using a custom binary format. This is intended to be used to
// embed many device files into an application to create a fat binary. It also
// supports extracting these files from a known location.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;
using namespace llvm::object;

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

static cl::OptionCategory OffloadBinaryCategory("llvm-offload-binary options");

static cl::opt<std::string> OutputFile("o", cl::desc("Write output to <file>."),
                                       cl::value_desc("file"),
                                       cl::cat(OffloadBinaryCategory));

static cl::opt<std::string> InputFile(cl::Positional,
                                      cl::desc("Extract from <file>."),
                                      cl::value_desc("file"),
                                      cl::cat(OffloadBinaryCategory));

static cl::list<std::string>
    DeviceImages("image",
                 cl::desc("List of key and value arguments. Required keywords "
                          "are 'file' and 'triple'."),
                 cl::value_desc("<key>=<value>,..."),
                 cl::cat(OffloadBinaryCategory));

static cl::opt<bool>
    CreateArchive("archive",
                  cl::desc("Write extracted files to a static archive"),
                  cl::cat(OffloadBinaryCategory));

/// Path of the current binary.
static const char *PackagerExecutable;

// Get a map containing all the arguments for the image. Repeated arguments will
// be placed in a comma separated list.
static DenseMap<StringRef, StringRef> getImageArguments(StringRef Image,
                                                        StringSaver &Saver) {
  DenseMap<StringRef, StringRef> Args;
  for (StringRef Arg : llvm::split(Image, ",")) {
    auto [Key, Value] = Arg.split("=");
    auto [It, Inserted] = Args.try_emplace(Key, Value);
    if (!Inserted)
      It->second = Saver.save(It->second + "," + Value);
  }

  return Args;
}

static Error writeFile(StringRef Filename, StringRef Data) {
  Expected<std::unique_ptr<FileOutputBuffer>> OutputOrErr =
      FileOutputBuffer::create(Filename, Data.size());
  if (!OutputOrErr)
    return OutputOrErr.takeError();
  std::unique_ptr<FileOutputBuffer> Output = std::move(*OutputOrErr);
  llvm::copy(Data, Output->getBufferStart());
  if (Error E = Output->commit())
    return E;
  return Error::success();
}

static Error bundleImages() {
  SmallVector<char, 1024> BinaryData;
  raw_svector_ostream OS(BinaryData);
  for (StringRef Image : DeviceImages) {
    BumpPtrAllocator Alloc;
    StringSaver Saver(Alloc);
    DenseMap<StringRef, StringRef> Args = getImageArguments(Image, Saver);

    if (!Args.count("file"))
      return createStringError(inconvertibleErrorCode(),
                               "'file' is a required image arguments");

    // Permit using multiple instances of `file` in a single string.
    for (auto &File : llvm::split(Args["file"], ",")) {
      OffloadBinary::OffloadingImage ImageBinary{};

      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> ObjectOrErr =
          llvm::MemoryBuffer::getFileOrSTDIN(File);
      if (std::error_code EC = ObjectOrErr.getError())
        return errorCodeToError(EC);

      // Clang uses the '.o' suffix for LTO bitcode.
      if (identify_magic((*ObjectOrErr)->getBuffer()) == file_magic::bitcode)
        ImageBinary.TheImageKind = object::IMG_Bitcode;
      else if (sys::path::has_extension(File))
        ImageBinary.TheImageKind =
            getImageKind(sys::path::extension(File).drop_front());
      else
        ImageBinary.TheImageKind = IMG_None;
      ImageBinary.Image = std::move(*ObjectOrErr);
      for (const auto &[Key, Value] : Args) {
        if (Key == "kind") {
          ImageBinary.TheOffloadKind = getOffloadKind(Value);
        } else if (Key != "file") {
          ImageBinary.StringData[Key] = Value;
        }
      }
      llvm::SmallString<0> Buffer = OffloadBinary::write(ImageBinary);
      if (Buffer.size() % OffloadBinary::getAlignment() != 0)
        return createStringError(inconvertibleErrorCode(),
                                 "Offload binary has invalid size alignment");
      OS << Buffer;
    }
  }

  if (Error E = writeFile(OutputFile,
                          StringRef(BinaryData.begin(), BinaryData.size())))
    return E;
  return Error::success();
}

// Extract SPIR-V binaries from an ELF image with triple "spirv64-intel" or
// "spirv32-intel". These ELF images contain SPIR-V binaries in sections named
// "__openmp_offload_spirv_*".
static Expected<SmallVector<StringRef>>
extractSPIRVFromELF(StringRef ImageData) {
  SmallVector<StringRef> SPIRVBinaries;

  // Try to parse as ELF object file
  Expected<std::unique_ptr<ObjectFile>> ObjOrErr =
      ObjectFile::createObjectFile(MemoryBufferRef(ImageData, "spirv-elf"));
  if (!ObjOrErr)
    return ObjOrErr.takeError();

  ObjectFile &Obj = *ObjOrErr->get();
  if (!Obj.isELF())
    return createStringError(inconvertibleErrorCode(),
                             "Expected ELF format for Intel SPIR-V image");

  // Extract all sections with name matching "__openmp_offload_spirv_*"
  for (const SectionRef &Sec : Obj.sections()) {
    Expected<StringRef> NameOrErr = Sec.getName();
    if (!NameOrErr)
      continue;

    if (!NameOrErr->starts_with("__openmp_offload_spirv_"))
      continue;

    Expected<StringRef> ContentsOrErr = Sec.getContents();
    if (!ContentsOrErr)
      return ContentsOrErr.takeError();

    SPIRVBinaries.push_back(*ContentsOrErr);
  }

  if (SPIRVBinaries.empty())
    return createStringError(inconvertibleErrorCode(),
                             "No SPIR-V sections found in ELF image");

  return SPIRVBinaries;
}

// Helper function to extract a single binary image, with SPIR-V support.
// Returns Error on failure.
static Error extractBinary(const OffloadBinary *Binary, StringRef InputFile,
                           uint64_t Idx, StringSaver &Saver) {
  // Check if this is a SPIR-V image that needs special handling
  if (Binary->getTriple().starts_with("spirv64-intel")) {
    StringRef ImageData = Binary->getImage();
    std::string BaseFilename =
        sys::path::stem(InputFile).str() + "-" + Binary->getTriple().str();
    StringRef Arch = Binary->getArch();
    if (!Arch.empty())
      BaseFilename += "-" + Arch.str();
    BaseFilename += "." + std::to_string(Idx);

    // Check if the image is already raw SPIR-V (not ELF-wrapped)
    if (identify_magic(ImageData) == file_magic::spirv_object) {
      // Image is already SPIR-V, just extract it with .spv extension
      StringRef Filename = Saver.save(BaseFilename + ".spv");
      if (Error E = writeFile(Filename, ImageData))
        return E;
      outs() << "Extracted SPIR-V: " << Filename << "\n";
      return Error::success();
    }

    // Try to parse as ELF and extract SPIR-V from sections
    auto SPIRVBinariesOrErr = extractSPIRVFromELF(ImageData);
    if (!SPIRVBinariesOrErr) {
      // Not ELF or no SPIR-V sections, extract as-is with .bin extension
      StringRef Filename = Saver.save(BaseFilename + ".bin");
      if (Error E = writeFile(Filename, ImageData))
        return E;
      outs() << "Extracted (unknown format): " << Filename << "\n";
      return Error::success();
    }

    // Successfully extracted SPIR-V from ELF
    // Extract the ELF wrapper
    StringRef ELFFilename = Saver.save(BaseFilename + ".elf");
    if (Error E = writeFile(ELFFilename, ImageData))
      return E;
    outs() << "Extracted (ELF wrapper): " << ELFFilename << "\n";

    // Extract each SPIR-V binary found in the ELF
    uint64_t SPIRVIdx = 0;
    for (StringRef SPIRVBinary : *SPIRVBinariesOrErr) {
      StringRef Filename =
          Saver.save(BaseFilename + "_" + std::to_string(SPIRVIdx++) + ".spv");
      if (Error E = writeFile(Filename, SPIRVBinary))
        return E;
      outs() << "Extracted SPIR-V: " << Filename << "\n";
    }
  } else {
    // Regular extraction (non-SPIR-V)
    std::string Filename =
        sys::path::stem(InputFile).str() + "-" + Binary->getTriple().str();
    StringRef Arch = Binary->getArch();
    if (!Arch.empty())
      Filename += "-" + Arch.str();
    Filename += "." + std::to_string(Idx) + "." +
                getImageKindName(Binary->getImageKind()).str();

    if (Error E = writeFile(Saver.save(Filename), Binary->getImage()))
      return E;
    outs() << "Extracted: " << Filename << "\n";
  }

  return Error::success();
}

static Error unbundleImages() {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFileOrSTDIN(InputFile);
  if (std::error_code EC = BufferOrErr.getError())
    return createFileError(InputFile, EC);
  std::unique_ptr<MemoryBuffer> Buffer = std::move(*BufferOrErr);

  // This data can be misaligned if extracted from an archive.
  if (!isAddrAligned(Align(OffloadBinary::getAlignment()),
                     Buffer->getBufferStart()))
    Buffer = MemoryBuffer::getMemBufferCopy(Buffer->getBuffer(),
                                            Buffer->getBufferIdentifier());

  SmallVector<OffloadFile> Binaries;
  if (Error Err = extractOffloadBinaries(*Buffer, Binaries))
    return Err;

  // If no filters specified, extract all images
  if (DeviceImages.empty()) {
    BumpPtrAllocator Alloc;
    StringSaver Saver(Alloc);
    uint64_t Idx = 0;
    for (const OffloadFile &File : Binaries) {
      if (Error E = extractBinary(File.getBinary(), InputFile, Idx++, Saver))
        return E;
    }
    return Error::success();
  }

  // Try to extract each device image specified by the user from the input file.
  for (StringRef Image : DeviceImages) {
    BumpPtrAllocator Alloc;
    StringSaver Saver(Alloc);
    auto Args = getImageArguments(Image, Saver);

    SmallVector<const OffloadBinary *> Extracted;
    for (const OffloadFile &File : Binaries) {
      const auto *Binary = File.getBinary();
      // We handle the 'file' and 'kind' identifiers differently.
      bool Match = llvm::all_of(Args, [&](auto &Arg) {
        const auto [Key, Value] = Arg;
        if (Key == "file")
          return true;
        if (Key == "kind")
          return Binary->getOffloadKind() == getOffloadKind(Value);
        return Binary->getString(Key) == Value;
      });
      if (Match)
        Extracted.push_back(Binary);
    }

    if (Extracted.empty())
      continue;

    if (CreateArchive) {
      if (!Args.count("file"))
        return createStringError(inconvertibleErrorCode(),
                                 "Image must have a 'file' argument.");

      SmallVector<NewArchiveMember> Members;
      for (const OffloadBinary *Binary : Extracted)
        Members.emplace_back(MemoryBufferRef(
            Binary->getImage(),
            Binary->getMemoryBufferRef().getBufferIdentifier()));

      if (Error E = writeArchive(
              Args["file"], Members, SymtabWritingMode::NormalSymtab,
              Archive::getDefaultKind(), true, false, nullptr))
        return E;
    } else if (auto It = Args.find("file"); It != Args.end()) {
      if (Extracted.size() > 1)
        WithColor::warning(errs(), PackagerExecutable)
            << "Multiple inputs match to a single file, '" << It->second
            << "'\n";
      if (Error E = writeFile(It->second, Extracted.back()->getImage()))
        return E;
    } else {
      uint64_t Idx = 0;
      for (const OffloadBinary *Binary : Extracted) {
        if (Error E = extractBinary(Binary, InputFile, Idx++, Saver))
          return E;
      }
    }
  }

  return Error::success();
}

int main(int argc, const char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  cl::HideUnrelatedOptions(OffloadBinaryCategory);
  cl::ParseCommandLineOptions(
      argc, argv,
      "A utility for bundling several object files into a single binary.\n"
      "The output binary can then be embedded into the host section table\n"
      "to create a fatbinary containing offloading code.\n");

  if (sys::path::stem(argv[0]).ends_with("clang-offload-packager"))
    WithColor::warning(errs(), PackagerExecutable)
        << "'clang-offload-packager' is deprecated. Use 'llvm-offload-binary' "
           "instead.\n";

  if (Help || (OutputFile.empty() && InputFile.empty())) {
    cl::PrintHelpMessage();
    return EXIT_SUCCESS;
  }

  PackagerExecutable = argv[0];
  auto reportError = [argv](Error E) {
    logAllUnhandledErrors(std::move(E), WithColor::error(errs(), argv[0]));
    return EXIT_FAILURE;
  };

  if (!InputFile.empty() && !OutputFile.empty())
    return reportError(
        createStringError(inconvertibleErrorCode(),
                          "Packaging to an output file and extracting from an "
                          "input file are mutually exclusive."));

  if (!OutputFile.empty()) {
    if (Error Err = bundleImages())
      return reportError(std::move(Err));
  } else if (!InputFile.empty()) {
    if (Error Err = unbundleImages())
      return reportError(std::move(Err));
  }

  return EXIT_SUCCESS;
}
