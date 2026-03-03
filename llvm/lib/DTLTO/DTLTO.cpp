//===- DTLTO.cpp - Distributed ThinLTO implementation ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This file implements support functions for Distributed ThinLTO, focusing on
// preparing input files for distribution.
//
//===----------------------------------------------------------------------===//

#include "llvm/DTLTO/DTLTO.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/raw_ostream.h"
#ifdef _WIN32
#include "llvm/Support/Windows/WindowsSupport.h"
#endif

#include <string>

using namespace llvm;

namespace {

// Saves the content of Buffer to Path overwriting any existing file.
Error save(StringRef Buffer, StringRef Path) {
  std::error_code EC;
  raw_fd_ostream OS(Path.str(), EC, sys::fs::OpenFlags::OF_None);
  if (EC)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create file %s: %s", Path.data(),
                             EC.message().c_str());
  OS.write(Buffer.data(), Buffer.size());
  if (OS.has_error())
    return createStringError(inconvertibleErrorCode(),
                             "Failed writing to file %s", Path.data());
  return Error::success();
}

// Saves the content of Input to Path overwriting any existing file.
Error save(lto::InputFile *Input, StringRef Path) {
  MemoryBufferRef MB = Input->getFileBuffer();
  return save(MB.getBuffer(), Path);
}

// Normalize and save a path. Aside from expanding Windows 8.3 short paths,
// no other normalization is currently required here. These paths are
// machine-local and break distribution systems; other normalization is
// handled by the DTLTO distributors.
Expected<StringRef> normalizePath(StringRef Path, StringSaver &Saver) {
#if defined(_WIN32)
  if (Path.empty())
    return Path;
  SmallString<256> Expanded;
  if (std::error_code EC = sys::windows::makeLongFormPath(Path, Expanded))
    return createStringError(inconvertibleErrorCode(),
                             "Normalization failed for path %s: %s",
                             Path.str().c_str(), EC.message().c_str());
  return Saver.save(Expanded.str());
#else
  return Saver.save(Path);
#endif
}

// Compute the file path for a thin archive member.
//
// For thin archives, an archive member name is typically a file path relative
// to the archive file's directory. This function resolves that path.
SmallString<256> computeThinArchiveMemberPath(StringRef ArchivePath,
                                              StringRef MemberName) {
  assert(!ArchivePath.empty() && "An archive file path must be non empty.");
  SmallString<256> MemberPath;
  if (sys::path::is_relative(MemberName)) {
    MemberPath = sys::path::parent_path(ArchivePath);
    sys::path::append(MemberPath, MemberName);
  } else {
    MemberPath = MemberName;
  }
  sys::path::remove_dots(MemberPath, /*remove_dot_dot=*/true);
  return MemberPath;
}

} // namespace

// Determines if a file at the given path is a thin archive file.
Expected<bool> lto::DTLTO::isThinArchive(const StringRef ArchivePath) {
  // Return cached result if available.
  auto Cached = ArchiveIsThinCache.find(ArchivePath);
  if (Cached != ArchiveIsThinCache.end())
    return Cached->second;

  uint64_t FileSize = -1;
  std::error_code EC = sys::fs::file_size(ArchivePath, FileSize);
  if (EC)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to get file size from archive %s: %s",
                             ArchivePath.data(), EC.message().c_str());
  if (FileSize < sizeof(object::ThinArchiveMagic))
    return createStringError(inconvertibleErrorCode(),
                             "Archive file size is too small %s",
                             ArchivePath.data());

  // Read only the first few bytes containing the magic signature.
  ErrorOr<std::unique_ptr<MemoryBuffer>> MBOrErr = MemoryBuffer::getFileSlice(
      ArchivePath, sizeof(object::ThinArchiveMagic), 0);
  if ((EC = MBOrErr.getError()))
    return createStringError(inconvertibleErrorCode(),
                             "Failed to read from archive %s: %s",
                             ArchivePath.data(), EC.message().c_str());

  StringRef Buf = (*MBOrErr)->getBuffer();
  if (file_magic::archive != identify_magic(Buf))
    return createStringError(inconvertibleErrorCode(),
                             "Unknown format for archive %s",
                             ArchivePath.data());

  bool IsThin = Buf.starts_with(object::ThinArchiveMagic);

  // Cache the result.
  ArchiveIsThinCache[ArchivePath] = IsThin;

  return IsThin;
}

// Add an input file and prepare it for distribution.
Expected<std::shared_ptr<lto::InputFile>>
lto::DTLTO::addInput(std::unique_ptr<InputFile> InputPtr) {
  TimeTraceScope TimeScope("Add input for DTLTO");

  // Add the input file to the LTO object.
  InputFiles.emplace_back(InputPtr.release());
  auto &Input = InputFiles.back();
  BitcodeModule &BM = Input->getPrimaryBitcodeModule();

  auto setIdFromPath = [&](StringRef Path) -> Error {
    auto N = normalizePath(Path, Saver);
    if (!N)
      return N.takeError();
    BM.setModuleIdentifier(*N);
    return Error::success();
  };

  StringRef ArchivePath = Input->getArchivePath();

  // In most cases, the module ID already points to an individual bitcode file
  // on disk, so no further preparation for distribution is required. However,
  // on Windows we overwite the module ID to expand Windows 8.3 short form
  // paths. These paths are machine-local and break distribution systems; other
  // normalization is handled by the DTLTO distributors.
  if (ArchivePath.empty() && !Input->isFatLTOObject()) {
#if defined(_WIN32)
    if (Error E = setIdFromPath(Input->getName()))
      return std::move(E);
#endif
    return Input;
  }

  // For a member of a thin archive that is not a FatLTO object, there is an
  // existing file on disk that can be used, so we can avoid having to
  // serialize.
  Expected<bool> UseThinMember =
      Input->isFatLTOObject() ? false : isThinArchive(ArchivePath);
  if (!UseThinMember)
    return UseThinMember.takeError();
  if (*UseThinMember) {
    // For thin archives, use the path to the actual member file on disk.
    auto MemberPath =
        computeThinArchiveMemberPath(ArchivePath, Input->getMemberName());
    if (Error E = setIdFromPath(MemberPath))
      return std::move(E);
    return Input;
  }

  // A new file on disk will be needed for archive members and FatLTO objects.
  Input->setSerializeForDistribution(true);

  // Get the normalized output directory, if we haven't already.
  if (LinkerOutputDir.empty()) {
    auto N = normalizePath(
        sys::path::parent_path(DistributorParams.LinkerOutputFile), Saver);
    if (!N)
      return N.takeError();
    LinkerOutputDir = *N;
  }

  // Create a unique path by including the process ID and sequence number in the
  // filename.
  SmallString<256> Id(LinkerOutputDir);
  sys::path::append(Id,
                    Twine(sys::path::filename(Input->getName())) + "." +
                        std::to_string(InputFiles.size()) /*Sequence number*/ +
                        "." + utohexstr(sys::Process::getProcessId()) + ".o");
  BM.setModuleIdentifier(Saver.save(Id.str()));
  return Input;
}

// Save the contents of ThinLTO-enabled input files that must be serialized for
// distribution.
Error lto::DTLTO::handleArchiveInputs() {
  for (auto &Input : InputFiles) {
    if (!Input->isThinLTO() || !Input->getSerializeForDistribution())
      continue;
    // Save the content of the input file to a file named after the module ID.
    StringRef ModuleId = Input->getName();
    TimeTraceScope TimeScope("Serialize bitcode input for DTLTO", ModuleId);
    MemoryBufferRef Buf = Input->getFileBuffer();
    if (Error Err = save(Buf.getBuffer(), ModuleId))
      return Err;
    // Cleanup this file on abnormal process exit.
    if (!SaveTemps)
      addToCleanup(ModuleId);
  }
  return Error::success();
}

// Remove temporary files created to enable distribution.
void lto::DTLTO::cleanup() {
  if (!SaveTemps) {
    // Remove one file, report error if any.
    auto removeFile = [](StringRef FileName) -> void {
      std::error_code EC = sys::fs::remove(FileName, true);
      if (EC &&
          EC != std::make_error_code(std::errc::no_such_file_or_directory))
        errs() << "warning: could not remove the file '" << FileName
               << "': " << EC.message() << "\n";
    };

    TimeTraceScope JobScope("Remove DTLTO temporary files");
    for (const auto &Name : CleanupList) {
      removeFile(Name);
    }
  }
  // Base::cleanup();
}

// Runs the DTLTO thin link phase, producing per-module summary indices,
// import lists, and cache keys for distribution.
Error lto::DTLTO::performThinLink() {
  auto ThinIndexBackend = lto::createWriteIndexesThinBackend(
      hardware_concurrency(), "", "", "", true, nullptr, nullptr);
  setThinBackend(ThinIndexBackend);
  setLTOMode(lto::LTO::LTOKind::LTOK_UnifiedThin);

  size_t NumTasks = getMaxTasks();
  SummaryIndexFiles.resize(NumTasks);
  ImportsFilesLists.resize(NumTasks);
  CacheKeysList.resize(NumTasks);

  lto::Config &Cfg = getConfig();
  Cfg.OnSummaryIndexStoreCb =
      [&](size_t task) -> std::unique_ptr<raw_svector_ostream> {
    return std::make_unique<raw_svector_ostream>(SummaryIndexFiles[task]);
  };
  Cfg.OnCacheKeyStoreCb = [&](size_t task) -> std::string & {
    return CacheKeysList[task];
  };
  Cfg.OnImportsListStoreCb = [&](size_t task) -> std::vector<std::string> & {
    return ImportsFilesLists[task];
  };

  return Base::run(AddStreamFunc, {});
}

// Runs the DTLTO pipeline.
LLVM_ABI Error lto::DTLTO::run(AddStreamFn AddStream, FileCache CacheParam) {
  scope_exit CleanUp([this]() { cleanup(); });

  AddStreamFunc = AddStream;
  Cache = std::move(CacheParam);
  Conf.Dtlto = 1;
  UID = itostr(sys::Process::getProcessId());

  if (Error Err = performThinLink())
    return Err;

  ThinLTOTaskOffset = RegularLTO.ParallelCodeGenParallelismLevel;
  DistributorParams.TargetTriple = RegularLTO.CombinedModule->getTargetTriple();

  if (Error Err = prepareDtltoJobs())
    return Err;
  if (Error Err = handleArchiveInputs())
    return Err;
  if (Error Err = performCodegen())
    return Err;
  if (Error Err = addObjectFilesToLink())
    return Err;
  return Error::success();
}

// Probes the LTO cache for a compiled native object for the given job.
Error lto::DTLTO::checkCacheHit(Job &J) {
  if (!Cache.isValid())
    return Error::success();

  auto CacheAddStreamExp = Cache(J.Task, J.CacheKey, J.ModuleID);
  if (Error Err = CacheAddStreamExp.takeError())
    return Err;
  AddStreamFn &CacheAddStream = *CacheAddStreamExp;
  // If CacheAddStream is null, we have a cache hit and at this point
  // object file is already passed back to the linker.
  if (!CacheAddStream) {
    J.Cached = true; // Cache hit, mark the job as cached.
    CachedJobs.fetch_add(1);
  } else {
    // If CacheAddStream is not null, we have a cache miss and we need to
    // run the backend for codegen. Save cache 'add stream'
    // function for a later use.
    J.CacheAddStream = std::move(CacheAddStream);
  }
  return Error::success();
}

// Prepares a single DTLTO backend compilation job for a ThinLTO module.
Error lto::DTLTO::prepareDtltoJob(StringRef ModulePath, unsigned Task) {
  assert(Task >= ThinLTOTaskOffset && Task - ThinLTOTaskOffset < Jobs.size() &&
         "Task index out of range for Jobs");
  assert(Task < SummaryIndexFiles.size() && "Task index out of range");

  SString ObjFilePath =
      sys::path::parent_path(DistributorParams.LinkerOutputFile);
  sys::path::append(ObjFilePath, sys::path::stem(ModulePath) + "." +
                                     itostr(Task) + "." + UID + ".native.o");

  SString SummaryIndexPathStr = ObjFilePath;
  SummaryIndexPathStr += ".thinlto.bc";
  SString ImportsPathStr = ModulePath;
  ImportsPathStr += ".imports";

  Job &J = Jobs[Task - ThinLTOTaskOffset];
  J = {Task,
       ModulePath,
       Saver.save(ObjFilePath.str()),
       Saver.save(SummaryIndexPathStr.str()),
       Saver.save(ImportsPathStr.str()),
       ImportsFilesLists[Task],
       CacheKeysList[Task],
       nullptr,
       false};

  if (Error Err = checkCacheHit(J))
    return Err;
  if (!J.Cached) {
    TimeTraceScope JobScope("Emit individual index for DTLTO",
                            J.SummaryIndexPath);
    if (Error Err = save(SummaryIndexFiles[Task], J.SummaryIndexPath))
      return Err;
  }
  if (OnWriteCb)
    OnWriteCb(J.SummaryIndexPath.str());

  if (ShouldEmitImportFiles)
    if (Error Err = save(join(ImportsFilesLists[Task], "\n"), J.ImportsPath))
      return Err;

  if (!SaveTemps) {
    if (!J.Cached)
      addToCleanup(J.NativeObjectPath.str());
    if (!ShouldEmitIndexFiles)
      addToCleanup(J.SummaryIndexPath.str());
    if (!ShouldEmitImportFiles)
      addToCleanup(J.ImportsPath.str());
  }
  return Error::success();
}

// Derive a set of Clang options that will be shared/common for all DTLTO
// backend compilations.
void lto::DTLTO::buildCommonRemoteCompilerOptions() {
  const lto::Config &C = getConfig();
  auto &Ops = DistributorParams.CodegenOptions;

  Ops.push_back(Saver.save("-O" + Twine(C.OptLevel)));

  if (C.Options.EmitAddrsig)
    Ops.push_back("-faddrsig");
  if (C.Options.FunctionSections)
    Ops.push_back("-ffunction-sections");
  if (C.Options.DataSections)
    Ops.push_back("-fdata-sections");

  if (C.RelocModel == Reloc::PIC_)
    // Clang doesn't have -fpic for all triples.
    if (!DistributorParams.TargetTriple.isOSBinFormatCOFF())
      Ops.push_back("-fpic");

  // Turn on/off warnings about profile cfg mismatch (default on)
  // --lto-pgo-warn-mismatch.
  if (!C.PGOWarnMismatch) {
    Ops.push_back("-mllvm");
    Ops.push_back("-no-pgo-warn-mismatch");
  }

  // Enable sample-based profile guided optimizations.
  // Sample profile file path --lto-sample-profile=<value>.
  if (!C.SampleProfile.empty()) {
    Ops.push_back(Saver.save("-fprofile-sample-use=" + Twine(C.SampleProfile)));
    DistributorParams.CommonInputs.insert(C.SampleProfile);
  }

  // We don't know which of options will be used by Clang.
  Ops.push_back("-Wno-unused-command-line-argument");

  // Forward any supplied options.
  if (!DistributorParams.RemoteCompilerArgs.empty())
    for (auto &a : DistributorParams.RemoteCompilerArgs)
      Ops.push_back(a);
}

// Initializes DTLTO state and prepares a job for each ThinLTO module.
Error lto::DTLTO::prepareDtltoJobs() {
  auto &ModuleMap =
      ThinLTO.ModulesToCompile ? *ThinLTO.ModulesToCompile : ThinLTO.ModuleMap;

  if (ModuleMap.empty())
    return Error::success();

  Jobs.resize(ModuleMap.size());

  for (auto [I, Mod] : enumerate(ModuleMap))
    if (Error E = prepareDtltoJob(Mod.first, ThinLTOTaskOffset + I))
      return E;

  return Error::success();
}

// Runs the DTLTO code generation phase. Must be invoked after thinLink().
Error lto::DTLTO::performCodegen() {
  if (Jobs.empty())
    return Error::success();
  // Build common remote compiler options.
  buildCommonRemoteCompilerOptions();

  DistributionDriver Distributor(DistributorParams, Jobs, SaveTemps,
                                 [&](StringRef S) { addToCleanup(S); });

  if (CachedJobs.load() < Jobs.size()) {
    if (Error E = Distributor())
      return E;
  }
  return Error::success();
}

// Adds compiled object files to the link for each non-cached job.
Error lto::DTLTO::addObjectFilesToLink() {
  TimeTraceScope FilesScope("Add DTLTO files to the link");
  for (auto &Job : Jobs) {
    if (!Job.CacheKey.empty() && Job.Cached) {
      assert(Cache.isValid());
      continue;
    }
    // Load the native object from a file into a memory buffer
    // and store its contents in the output buffer.
    auto ObjFileMbOrErr =
        MemoryBuffer::getFile(Job.NativeObjectPath, /*IsText=*/false,
                              /*RequiresNullTerminator=*/false);
    if (std::error_code EC = ObjFileMbOrErr.getError())
      return make_error<StringError>(
          BCError + "cannot open native object file: " + Job.NativeObjectPath +
              ": " + EC.message(),
          inconvertibleErrorCode());

    MemoryBufferRef ObjFileMbRef = ObjFileMbOrErr->get()->getMemBufferRef();
    if (Cache.isValid()) {
      // Cache hits are taken care of earlier. At this point, we could only
      // have cache misses.
      assert(Job.CacheAddStream);
      // Obtain a file stream for a storing a cache entry.
      auto CachedFileStreamOrErr = Job.CacheAddStream(Job.Task, Job.ModuleID);
      if (!CachedFileStreamOrErr)
        return joinErrors(
            CachedFileStreamOrErr.takeError(),
            createStringError(inconvertibleErrorCode(),
                              "Cannot get a cache file stream: %s",
                              Job.NativeObjectPath.data()));
      // Store a file buffer into the cache stream.
      auto &CacheStream = *(CachedFileStreamOrErr->get());
      *(CacheStream.OS) << ObjFileMbRef.getBuffer();
      if (Error Err = CacheStream.commit())
        return Err;
    } else {
      if (AddBuffer) {
        AddBuffer(Job.Task, Job.ModuleID, std::move(ObjFileMbOrErr.get()));
      } else {
        auto StreamOrErr = AddStreamFunc(Job.Task, Job.ModuleID);
        if (Error Err = StreamOrErr.takeError())
          return Err;
        auto &Stream = *StreamOrErr->get();
        *Stream.OS << ObjFileMbRef.getBuffer();
        if (Error Err = Stream.commit())
          return Err;
      }
    }
  }
  return Error::success();
}

// Generates a JSON file describing the backend compilations, for the
// distributor.
Error lto::DistributionDriver::emitJson() {
  using json::Array;
  std::error_code EC;
  raw_fd_ostream OS(DistributorJsonFile, EC);
  if (EC)
    return createStringError(EC, "Error while creating Json file");

  json::OStream JOS(OS);
  JOS.object([&]() {
    // Information common to all jobs.
    JOS.attributeObject("common", [&]() {
      JOS.attribute("linker_output", Params.LinkerOutputFile);

      JOS.attributeArray("args", [&]() {
        JOS.value(Params.RemoteCompiler);

        // Forward any supplied prepend options.
        if (!Params.RemoteCompilerPrependArgs.empty())
          for (auto &A : Params.RemoteCompilerPrependArgs)
            JOS.value(A);

        JOS.value("-c");

        JOS.value(std::string("--target=") + Params.TargetTriple.str());

        for (const auto &A : Params.CodegenOptions)
          JOS.value(A);
      });

      JOS.attribute("inputs", Array(Params.CommonInputs));
    });

    // Per-compilation-job information.
    JOS.attributeArray("jobs", [&]() {
      for (const auto &J : Jobs) {
        assert(J.Task != 0);
        if (J.Cached) {
          continue;
        }

        SmallVector<StringRef, 2> Inputs;
        SmallVector<StringRef, 1> Outputs;

        JOS.object([&]() {
          JOS.attributeArray("args", [&]() {
            JOS.value(J.ModuleID);
            Inputs.push_back(J.ModuleID);

            JOS.value(
                std::string("-fthinlto-index=" + J.SummaryIndexPath.str()));
            Inputs.push_back(J.SummaryIndexPath);

            JOS.value("-o");
            JOS.value(J.NativeObjectPath);
            Outputs.push_back(J.NativeObjectPath);
          });

          // Add the bitcode files from which imports will be made. These do
          // not explicitly appear on the backend compilation command lines
          // but are recorded in the summary index shards.
          append_range(Inputs, J.ImportsFiles);
          JOS.attribute("inputs", Array(Inputs));

          JOS.attribute("outputs", Array(Outputs));
        });
      }
    });
  });

  return Error::success();
}

// Saves JSON file on a filesystem.
Error lto::DistributionDriver::saveJson() {
  DistributorJsonFile = sys::path::parent_path(Params.LinkerOutputFile);
  TimeTraceScope TimeScope("Emit DTLTO JSON");
  sys::path::append(DistributorJsonFile,
                    sys::path::stem(Params.LinkerOutputFile) + "." +
                        itostr(sys::Process::getProcessId()) +
                        ".dist-file.json");
  if (Error E = emitJson())
    return make_error<StringError>(
        BCError + "failed to generate distributor JSON script: " +
            DistributorJsonFile,
        errorToErrorCode(std::move(E)));

  // Add JSON file to the cleanup files list.
  if (!SaveTemps)
    AddToCleanup(DistributorJsonFile);
  return Error::success();
}

// Invokes the distributor to compile uncached ThinLTO modules remotely.
Error lto::DistributionDriver::operator()() {
  if (Error E = saveJson())
    return E;

  TimeTraceScope TimeScope("Execute DTLTO distributor", Params.DistributorPath);
  SmallVector<StringRef, 3> Args = {Params.DistributorPath};
  append_range(Args, Params.DistributorArgs);
  Args.push_back(DistributorJsonFile);
  std::string ErrMsg;
  if (sys::ExecuteAndWait(Args[0], Args,
                          /*Env=*/std::nullopt, /*Redirects=*/{},
                          /*SecondsToWait=*/0, /*MemoryLimit=*/0, &ErrMsg)) {
    return make_error<StringError>(
        BCError + "distributor execution failed" +
            (!ErrMsg.empty() ? ": " + ErrMsg + Twine(".") : Twine(".")),
        inconvertibleErrorCode());
  }
  return Error::success();
}
