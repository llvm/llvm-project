//===- llvm-mt.cpp - Merge .manifest files ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// Merge .manifest files.  This is intended to be a platform-independent port
// of Microsoft's mt.exe.
//
//===---------------------------------------------------------------------===//

#include "llvm/Config/llvm-config.h" // for LLVM_ON_UNIX
#include "llvm/ObjCopy/COFF/COFFConfig.h"
#include "llvm/ObjCopy/COFF/COFFObjcopy.h"
#include "llvm/ObjCopy/CommonConfig.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/WindowsResource.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Driver.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/WindowsManifest/WindowsManifestMerger.h"

#include <optional>
#include <system_error>

using namespace llvm;

namespace {

enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

using namespace llvm::opt;
#define OPTTABLE_CODE
#include "Opts.inc"

class CvtResOptTable : public opt::OptTable {
public:
  CvtResOptTable() : opt::OptTable(optionTables(), true) {}
};

// The type of manifest resources.
constexpr uint32_t RT_MANIFEST = 24;

// A manifest embedded in a PE image as a resource, as specified by the
// "<file>[;[#]<id>]" argument of the /inputresource, /outputresource and
// /updateresource options.
struct ManifestResource {
  std::string File;
  uint32_t ID = 1; // CREATEPROCESS_MANIFEST_RESOURCE_ID
};

} // namespace

[[noreturn]] static void reportError(Twine Msg) {
  WithColor::error(errs(), "llvm-mt") << Msg << '\n';
  exit(1);
}

static void reportError(StringRef Input, std::error_code EC) {
  reportError(Twine(Input) + ": " + EC.message());
}

static void error(Error EC) {
  if (EC)
    handleAllErrors(std::move(EC), [&](const ErrorInfoBase &EI) {
      reportError(EI.message());
    });
}

static ManifestResource parseManifestResource(StringRef Arg) {
  auto [File, ID] = Arg.rsplit(';');
  if (File.empty())
    reportError("missing file name in '" + Arg + "'");
  ManifestResource Resource;
  Resource.File = std::string(File);
  if (Arg.contains(';')) {
    ID.consume_front("#");
    if (ID.getAsInteger(10, Resource.ID))
      reportError("invalid resource ID in '" + Arg + "'");
  }
  return Resource;
}

static Expected<object::OwningBinary<object::Binary>>
openImage(StringRef File) {
  Expected<object::OwningBinary<object::Binary>> BinaryOrErr =
      object::createBinary(File);
  if (!BinaryOrErr)
    return createFileError(File, BinaryOrErr.takeError());
  auto *Obj = dyn_cast<object::COFFObjectFile>(BinaryOrErr->getBinary());
  if (!Obj || !(Obj->getPE32Header() || Obj->getPE32PlusHeader()))
    return createFileError(
        File, createStringError(errc::invalid_argument, "not a PE image"));
  return BinaryOrErr;
}

// Returns the manifest embedded in a PE image as the given resource, or
// std::nullopt if the image does not contain it.
static Expected<std::optional<std::string>>
readManifestResource(const ManifestResource &Resource) {
  Expected<object::OwningBinary<object::Binary>> BinaryOrErr =
      openImage(Resource.File);
  if (!BinaryOrErr)
    return BinaryOrErr.takeError();
  auto *Obj = cast<object::COFFObjectFile>(BinaryOrErr->getBinary());
  const object::data_directory *Dir =
      Obj->getDataDirectory(COFF::RESOURCE_TABLE);
  if (!Dir || Dir->RelativeVirtualAddress == 0 || Dir->Size == 0)
    return std::nullopt;

  object::ResourceSectionRef RSR;
  if (Error E = RSR.load(Obj))
    return createFileError(Resource.File, std::move(E));
  object::WindowsResourceParser Parser;
  std::vector<std::string> Duplicates;
  if (Error E = Parser.parse(RSR, Resource.File, Duplicates))
    return createFileError(Resource.File, std::move(E));
  if (!Duplicates.empty())
    return createFileError(Resource.File,
                           createStringError(object::object_error::parse_failed,
                                             "%s", Duplicates.front().c_str()));

  const object::WindowsResourceParser::TreeNode *Node =
      Parser.findResource(RT_MANIFEST, Resource.ID);
  if (!Node || Node->getIDChildren().empty())
    return std::nullopt;
  // Manifests are language-neutral in practice, so use the first language.
  const object::WindowsResourceParser::TreeNode &Language =
      *Node->getIDChildren().begin()->second;
  if (!Language.checkIsDataNode())
    return std::nullopt;
  ArrayRef<uint8_t> Data = Parser.getData()[Language.getDataIndex()];
  return std::string(Data.begin(), Data.end());
}

// Embeds a manifest in a PE image as the given resource.
static Error writeManifestResource(const ManifestResource &Resource,
                                   StringRef Manifest) {
  Expected<FilePermissionsApplier> PermsApplierOrErr =
      FilePermissionsApplier::create(Resource.File);
  if (!PermsApplierOrErr)
    return PermsApplierOrErr.takeError();
  Expected<object::OwningBinary<object::Binary>> BinaryOrErr =
      openImage(Resource.File);
  if (!BinaryOrErr)
    return BinaryOrErr.takeError();
  auto *Obj = cast<object::COFFObjectFile>(BinaryOrErr->getBinary());

  objcopy::CommonConfig Config;
  Config.InputFilename = Resource.File;
  Config.OutputFilename = Resource.File;
  objcopy::COFFConfig COFFConfig;
  COFFConfig.UpdateResource.push_back(
      {{RT_MANIFEST, Resource.ID, std::nullopt},
       MemoryBuffer::getMemBufferCopy(Manifest, "manifest")});
  if (Error E = writeToOutput(Resource.File, [&](raw_ostream &OS) {
        return objcopy::coff::executeObjcopyOnBinary(Config, COFFConfig, *Obj,
                                                     OS);
      }))
    return E;
  return PermsApplierOrErr->apply(Resource.File);
}

int llvm_mt_main(int Argc, char **Argv, const llvm::ToolContext &) {
  CvtResOptTable T;
  unsigned MAI, MAC;
  ArrayRef<const char *> ArgsArr = ArrayRef(Argv + 1, Argc - 1);
  opt::InputArgList InputArgs = T.ParseArgs(ArgsArr, MAI, MAC);

  for (auto *Arg : InputArgs.filtered(OPT_INPUT)) {
    auto ArgString = Arg->getAsString(InputArgs);
    std::string Diag;
    raw_string_ostream OS(Diag);
    OS << "invalid option '" << ArgString << "'";

    std::string Nearest;
    if (T.findNearest(ArgString, Nearest) < 2)
      OS << ", did you mean '" << Nearest << "'?";

    reportError(OS.str());
  }

  for (auto &Arg : InputArgs) {
    if (Arg->getOption().matches(OPT_unsupported)) {
      outs() << "llvm-mt: ignoring unsupported '" << Arg->getOption().getName()
             << "' option\n";
    }
  }

  if (InputArgs.hasArg(OPT_help)) {
    T.printHelp(outs(), "llvm-mt [options] file...", "Manifest Tool", false);
    return 0;
  }

  std::vector<std::string> InputFiles = InputArgs.getAllArgValues(OPT_manifest);
  std::vector<ManifestResource> InputResources;
  for (auto *Arg : InputArgs.filtered(OPT_input_resource, OPT_update_resource))
    InputResources.push_back(parseManifestResource(Arg->getValue()));
  std::vector<ManifestResource> OutputResources;
  for (auto *Arg : InputArgs.filtered(OPT_output_resource, OPT_update_resource))
    OutputResources.push_back(parseManifestResource(Arg->getValue()));

  if (InputFiles.empty() && InputResources.empty())
    reportError("no input file specified");

  StringRef OutputFile;
  if (InputArgs.hasArg(OPT_out)) {
    OutputFile = InputArgs.getLastArgValue(OPT_out);
  } else if (OutputResources.empty()) {
    if (InputFiles.size() == 1 && InputResources.empty())
      OutputFile = InputFiles[0];
    else
      reportError("no output file specified");
  }

  windows_manifest::WindowsManifestMerger Merger;

  for (const ManifestResource &Resource : InputResources) {
    Expected<std::optional<std::string>> ManifestOrErr =
        readManifestResource(Resource);
    if (!ManifestOrErr)
      error(ManifestOrErr.takeError());
    if (!*ManifestOrErr)
      reportError(Twine(Resource.File) + ": manifest resource with ID " +
                  Twine(Resource.ID) + " not found");
    error(Merger.merge(MemoryBufferRef(**ManifestOrErr, Resource.File)));
  }

  for (const auto &File : InputFiles) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> ManifestOrErr =
        MemoryBuffer::getFile(File);
    if (!ManifestOrErr)
      reportError(File, ManifestOrErr.getError());
    error(Merger.merge(*ManifestOrErr.get()));
  }

  std::unique_ptr<MemoryBuffer> OutputBuffer = Merger.getMergedManifest();
  if (!OutputBuffer)
    reportError("empty manifest not written");
  StringRef Output = OutputBuffer->getBuffer();

  int ExitCode = 0;
  if (InputArgs.hasArg(OPT_notify_update)) {
    bool Same = true;
    if (!OutputFile.empty()) {
      ErrorOr<std::unique_ptr<MemoryBuffer>> OutBuffOrErr =
          MemoryBuffer::getFile(OutputFile);
      // Assume if we couldn't open the output file then it doesn't exist
      // meaning there was a change.
      Same = OutBuffOrErr && (*OutBuffOrErr)->getBuffer() == Output;
    }
    for (const ManifestResource &Resource : OutputResources) {
      Expected<std::optional<std::string>> ExistingOrErr =
          readManifestResource(Resource);
      if (!ExistingOrErr)
        error(ExistingOrErr.takeError());
      if (!*ExistingOrErr || **ExistingOrErr != Output)
        Same = false;
    }
    if (!Same) {
#if LLVM_ON_UNIX
      ExitCode = 0xbb;
#elif defined(_WIN32)
      ExitCode = 0x41020001;
#endif
    }
  }

  if (!OutputFile.empty()) {
    Expected<std::unique_ptr<FileOutputBuffer>> FileOrErr =
        FileOutputBuffer::create(OutputFile, Output.size());
    if (!FileOrErr)
      reportError(OutputFile, errorToErrorCode(FileOrErr.takeError()));
    std::unique_ptr<FileOutputBuffer> FileBuffer = std::move(*FileOrErr);
    llvm::copy(Output, FileBuffer->getBufferStart());
    error(FileBuffer->commit());
  }

  for (const ManifestResource &Resource : OutputResources)
    error(writeManifestResource(Resource, Output));

  return ExitCode;
}
