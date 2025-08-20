//===- llvm-offload-wrapper: Create runtime registration code for devices -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides a utility for generating runtime registration code for device code.
// We take a binary image (CUDA fatbinary, HIP offload bundle, LLVM binary) and
// create a new IR module that calls the respective runtime to load it on the
// device.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Frontend/Offloading/OffloadWrapper.h"
#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/WithColor.h"
#include "llvm/TargetParser/Host.h"

using namespace llvm;

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

static cl::OptionCategory
    OffloadWrapeprCategory("llvm-offload-wrapper options");

static cl::opt<object::OffloadKind> Kind(
    "kind", cl::desc("Wrap for offload kind:"), cl::cat(OffloadWrapeprCategory),
    cl::Required,
    cl::values(clEnumValN(object::OFK_OpenMP, "openmp", "Wrap OpenMP binaries"),
               clEnumValN(object::OFK_Cuda, "cuda", "Wrap CUDA binaries"),
               clEnumValN(object::OFK_HIP, "hip", "Wrap HIP binaries")));

static cl::opt<std::string> OutputFile("o", cl::desc("Write output to <file>."),
                                       cl::value_desc("file"),
                                       cl::cat(OffloadWrapeprCategory));

static cl::list<std::string> InputFiles(cl::Positional,
                                        cl::desc("Wrap input from <file>"),
                                        cl::value_desc("file"), cl::OneOrMore,
                                        cl::cat(OffloadWrapeprCategory));

static cl::opt<std::string>
    TheTriple("triple", cl::desc("Target triple for the wrapper module"),
              cl::init(sys::getDefaultTargetTriple()),
              cl::cat(OffloadWrapeprCategory));

static Error wrapImages(ArrayRef<ArrayRef<char>> BuffersToWrap) {
  if (BuffersToWrap.size() > 1 &&
      (Kind == llvm::object::OFK_Cuda || Kind == llvm::object::OFK_HIP))
    return createStringError(
        "CUDA / HIP offloading uses a single fatbinary or offload bundle");

  LLVMContext Context;
  Module M("offload.wrapper.module", Context);
  M.setTargetTriple(Triple());

  switch (Kind) {
  case llvm::object::OFK_OpenMP:
    if (Error Err = offloading::wrapOpenMPBinaries(
            M, BuffersToWrap, offloading::getOffloadEntryArray(M),
            /*Suffix=*/"", /*Relocatable=*/false))
      return Err;
    break;
  case llvm::object::OFK_Cuda:
    if (Error Err = offloading::wrapCudaBinary(
            M, BuffersToWrap.front(), offloading::getOffloadEntryArray(M),
            /*Suffix=*/"", /*EmitSurfacesAndTextures=*/false))
      return Err;
    break;
  case llvm::object::OFK_HIP:
    if (Error Err = offloading::wrapHIPBinary(
            M, BuffersToWrap.front(), offloading::getOffloadEntryArray(M)))
      return Err;
    break;
  default:
    return createStringError(getOffloadKindName(Kind) +
                             " wrapping is not supported");
  }

  int FD = -1;
  if (std::error_code EC = sys::fs::openFileForWrite(OutputFile, FD))
    return errorCodeToError(EC);
  llvm::raw_fd_ostream OS(FD, true);
  WriteBitcodeToFile(M, OS);

  return Error::success();
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  cl::HideUnrelatedOptions(OffloadWrapeprCategory);
  cl::ParseCommandLineOptions(
      argc, argv,
      "Generate runtime registration code for a device binary image\n");

  if (Help) {
    cl::PrintHelpMessage();
    return EXIT_SUCCESS;
  }

  auto ReportError = [argv](Error E) {
    logAllUnhandledErrors(std::move(E), WithColor::error(errs(), argv[0]));
    exit(EXIT_FAILURE);
  };

  SmallVector<std::unique_ptr<MemoryBuffer>> Buffers;
  SmallVector<ArrayRef<char>> BuffersToWrap;
  for (StringRef Input : InputFiles) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
        MemoryBuffer::getFileOrSTDIN(Input);
    if (std::error_code EC = BufferOrErr.getError())
      ReportError(createFileError(Input, EC));
    std::unique_ptr<MemoryBuffer> &Buffer =
        Buffers.emplace_back(std::move(*BufferOrErr));
    BuffersToWrap.emplace_back(
        ArrayRef<char>(Buffer->getBufferStart(), Buffer->getBufferSize()));
  }

  if (Error Err = wrapImages(BuffersToWrap))
    ReportError(std::move(Err));

  return EXIT_SUCCESS;
}
