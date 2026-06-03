//===- DTLTO.cpp - Integrated Distributed ThinLTO implementation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// \file
// This file implements support functions for Integrated Distributed ThinLTO,
// focusing on preparing complilation jobs for distribution.
//
//===----------------------------------------------------------------------===//

#include "llvm/DTLTO/DTLTO.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace llvm;

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
    for (const auto &Name : CleanupList)
      removeFile(Name);
    // Clean the CleanupList for safety.
    CleanupList.clear();
  }
}

// Runs the DTLTO thin link phase, producing per-module summary indices,
// import lists, and cache keys for distribution.
Error lto::DTLTO::performThinLink() {
  size_t NumTasks = getMaxTasks();
  SummaryIndexFiles.resize(NumTasks);
  ImportsFilesList.resize(NumTasks);
  CacheKeysList.resize(NumTasks);

  lto::Config &Cfg = getConfig();
  Cfg.GetSummaryIndexOutputStream =
      [&](size_t task) -> std::unique_ptr<raw_svector_ostream> {
    return std::make_unique<raw_svector_ostream>(SummaryIndexFiles[task]);
  };
  Cfg.GetCacheKeyOutputString = [&](size_t task) -> std::string & {
    return CacheKeysList[task];
  };
  Cfg.GetImportsListOutputArray =
      [&](size_t task) -> std::vector<std::string> & {
    return ImportsFilesList[task];
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
  if (Error Err = serializeLTOInputs())
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
       ImportsFilesList[Task],
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
  if (OnIndexWriteCb)
    OnIndexWriteCb(J.SummaryIndexPath.str());

  if (ShouldEmitImportFiles)
    if (Error Err = save(join(J.ImportsFilesList, "\n"), J.ImportsPath))
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
