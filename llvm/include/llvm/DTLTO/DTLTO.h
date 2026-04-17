//===- DTLTO.h - Distributed ThinLTO implementation -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// Declarations for Distributed ThinLTO, including the DTLTO class and the
// distribution driver. The implementation focuses on preparing input files for
// distribution.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DTLTO_DTLTO_H
#define LLVM_DTLTO_DTLTO_H

#include "llvm/ADT/SmallString.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Signals.h"

#include <functional>
#include <vector>

namespace llvm {
namespace lto {

class DTLTO;

/// Prepares inputs for Distributed ThinLTO so that backend compilations can use
/// individual bitcode paths and consistent module IDs.
///
/// Each input must exist as an individual bitcode file on disk and be loadable
/// via its ModuleID. Archive members and FatLTO objects do not satisfy that by
/// default; this class writes bitcode out when needed and updates ModuleID.
/// On Windows, module IDs are normalized to remove short 8.3 path components
/// that are machine-local and break distribution; other normalization is left
/// to DTLTO distributors.
///
/// Input files are kept until the pipeline has determined per-module ThinLTO
/// participation. addInput() performs: (1) register the input; (2) on Windows,
/// normalize module ID for standalone bitcode; (3) for thin archive members,
/// set module ID to the on-disk member path; (4) for other archives and FatLTO,
/// set module ID to a unique path and serialize content in
/// handleArchiveInputs().
class DTLTO : public LTO {
  using Base = LTO;

public:
  LLVM_ABI DTLTO(Config Conf, ThinBackend Backend,
                 unsigned ParallelCodeGenParallelismLevel, LTOKind LTOMode,
                 IndexWriteCallback OnWrite, bool EmitIndexFiles,
                 bool EmitImportsFiles, StringRef LinkerOutputFile,
                 StringRef Distributor, ArrayRef<StringRef> DistributorArgs,
                 StringRef RemoteCompiler,
                 ArrayRef<StringRef> RemoteCompilerPrependArgs,
                 ArrayRef<StringRef> RemoteCompilerArgs,
                 AddBufferFn AddBufferArg, bool SaveTempsArg)
      : Base(std::move(Conf), Backend, ParallelCodeGenParallelismLevel,
             LTOMode),
        AddBuffer(AddBufferArg), SaveTemps(SaveTempsArg),
        ShouldEmitIndexFiles(EmitIndexFiles),
        ShouldEmitImportFiles(EmitImportsFiles), OnWriteCb(OnWrite),
        DistributorParams{Distributor,        DistributorArgs,
                          RemoteCompiler,     RemoteCompilerPrependArgs,
                          RemoteCompilerArgs, LinkerOutputFile} {
    assert(!LinkerOutputFile.empty() && "expected a valid linker output file");
  }

  /// Add an input file and prepare it for distribution.
  ///
  /// This function performs the following tasks:
  /// 1. Add the input file to the LTO object's list of input files.
  /// 2. For individual bitcode file inputs on Windows only, overwrite the
  /// module
  ///    ID with a normalized path to remove short 8.3 form components.
  /// 3. For thin archive members, overwrite the module ID with the path
  ///    (normalized on Windows) to the member file on disk.
  /// 4. For archive members and FatLTO objects, overwrite the module ID with a
  ///    unique path (normalized on Windows) naming a file that will contain the
  ///    member content. The file is created and populated later (see
  ///    serializeInputs()).
  LLVM_ABI Expected<std::shared_ptr<InputFile>>
  addInput(std::unique_ptr<InputFile> InputPtr) override;

  /// Runs the DTLTO pipeline. This function calls the supplied AddStream
  /// function to add native object files to the link.
  ///
  /// The Cache parameter is optional. If supplied, it will be used to cache
  /// native object files and add them to the link.
  ///
  /// The client will receive at most one callback (via either AddStream or
  /// Cache) for each task identifier.
  LLVM_ABI virtual Error run(AddStreamFn AddStream,
                             FileCache Cache = {}) override;

private:
  /// DTLTO archives support.
  ///
  /// Save the contents of ThinLTO-enabled input files that must be serialized
  /// for distribution, such as archive members and FatLTO objects, to
  /// individual bitcode files named after the module ID.
  ///
  /// Must be called after all input files are added but before optimization
  /// begins. If a file with that name already exists, it is likely a leftover
  /// from a previously terminated linker process and can be safely overwritten.
  LLVM_ABI Error handleArchiveInputs();

  // Remove temporary files created to enable distribution.
  LLVM_ABI void cleanup() override;

public:
  // Mutable and const accessors to the LTO configuration object.
  Config &getConfig() { return Conf; }
  const Config &getConfig() const { return Conf; }

  // Set the LTO kind.
  void setLTOMode(LTOKind Knd) { LTOMode = Knd; }
  // Replace the ThinLTO backend (e.g. WriteIndexesThinBackend for the thin
  // link).
  void setThinBackend(ThinBackend Backend) { ThinLTO.Backend = Backend; }

private:
  // Bump allocator for saving updated module IDs.
  BumpPtrAllocator PtrAlloc;
  // String saver backed by PtrAlloc.
  StringSaver Saver{PtrAlloc};

  using SString = SmallString<128>;

  // Function pointer that defines the callback to add a pre-existing file.
  AddBufferFn AddBuffer;
  // Count of jobs that hit the cache.
  std::atomic<size_t> CachedJobs{0};
  // Normalized output directory from LinkerOutputFile.
  SString LinkerOutputDir;
  // Keep temporary files when true.
  bool SaveTemps = false;

public:
  struct Job {
    // Task index (combines RegularLTO parallel codegen offset with module
    // index).
    unsigned Task;
    // Module identifier (bitcode path) for the ThinLTO module.
    StringRef ModuleID;
    // Native object path.
    StringRef NativeObjectPath;
    // Per-module summary index path.
    StringRef SummaryIndexPath;
    // Per-module imports list path.
    StringRef ImportsPath;
    // Bitcode files from which this module imports.
    ArrayRef<std::string> ImportsFiles;
    // Cache key from thin link.
    std::string CacheKey;
    // On cache miss, stream used to store the compiled object in the cache.
    AddStreamFn CacheAddStream;
    // Set when the object was already supplied via the cache callback.
    bool Cached = false;
  };

private:
  // Backend compilation jobs, one per module.
  SmallVector<Job> Jobs;
  // Task index offset for first ThinLTO job.
  unsigned ThinLTOTaskOffset;
  // Optional cache for native objects.
  FileCache Cache;
  // Keep summary index files when true.
  bool ShouldEmitIndexFiles = false;
  // Keep summary import files when true.
  bool ShouldEmitImportFiles = false;
  // On index file write callback.
  IndexWriteCallback OnWriteCb;

  /// Probes the LTO cache for a compiled native object for the given job.
  ///
  /// If no cache is configured (Cache.isValid() is false), returns immediately
  /// without modifying the job.
  ///
  /// Otherwise, looks up the cache using J.CacheKey. On a cache hit, the cached
  /// object has already been passed to the linker via the Cache callback, so
  /// J.Cached is set to true, CachedJobs is incremented, and the distributor
  /// can skip this job. On a cache miss, the cache returns an AddStreamFn; we
  /// store it in J.CacheAddStream for use when storing the freshly compiled
  /// object after the distributor runs.
  ///
  /// \param J The job to check. Must have Task, CacheKey, and ModuleID set.
  ///          On return, J.Cached and J.CacheAddStream may be updated.
  ///
  /// \returns Error::success() on success, or an Error from the cache lookup.
  Error checkCacheHit(Job &J);

  /// Prepares a single DTLTO backend compilation job for a ThinLTO module.
  ///
  /// Called once per module during performCodegen(). This function:
  ///
  /// 1. Computes output paths for the native object and summary index files.
  ///    Both are placed in the linker output directory with names of the form
  ///    stem.Task.UID.native.o and stem.Task.UID.thinlto.bc, where stem is
  ///    derived from ModulePath.
  ///
  /// 2. Initializes the Job struct with Task, ModuleID (ModulePath), paths,
  ///    ImportsFiles and CacheKey from thin link results, and default values
  ///    for CacheAddStream and Cached.
  ///
  /// 3. Calls checkCacheHit() to probe the cache. On a cache hit, J.Cached is
  ///    set and the cached object has already been passed to the linker; the
  ///    distributor will skip this job. On a cache miss, J.CacheAddStream is
  ///    set for later use when storing the compiled object.
  ///
  /// 4. Writes the per-module summary index to disk only on cache miss. The
  ///    remote compiler will read this via -fthinlto-index=.
  ///
  /// 5. Registers the job's temporary files for removal on abnormal process
  ///    exit when SaveTemps is false (only for files that will be created).
  ///
  /// \param ModulePath The module identifier (bitcode path) for the ThinLTO
  ///                   module.
  /// \param Task       The task index (combines RegularLTO.ParallelCodeGen
  ///                   parallelism offset with the module index).
  ///
  /// \returns Error::success() on success, or an Error from saveBuffer() or
  ///          checkCacheHit().
  Error prepareDtltoJob(StringRef ModulePath, unsigned Task);

  /// Initializes DTLTO state and prepares a job for each ThinLTO module.
  ///
  /// Sets task offset, target triple, UID, and Jobs. For each module, calls
  /// prepareDtltoJob() to assign output paths, check the cache, and write
  /// summary index shards to disk when needed.
  ///
  /// \returns Error::success() on success, or an Error from prepareDtltoJob.
  Error prepareDtltoJobs();

  /// Runs the DTLTO code generation phase. Must be invoked after thinLink().
  ///
  /// Builds Clang options, emits a JSON manifest describing compilation jobs,
  /// and invokes the distributor to compile ThinLTO modules remotely. Cache
  /// hits are skipped; the distributor runs only when there are uncached jobs.
  ///
  /// \returns Error::success() on success, or an Error on manifest or
  /// distributor failure.
  Error performCodegen();

  /// Adds compiled object files to the link for each non-cached job.
  ///
  /// Loads each native object from disk, then either writes it to the cache
  /// (which adds it to the link via the cache callback) or passes it to
  /// AddStreamFunc directly when caching is disabled.
  ///
  /// \returns Error::success() on success, or an Error if a file cannot be read
  /// or a cache stream cannot be obtained.
  Error addObjectFilesToLink();

  // Determines if a file at the given path is a thin archive file.
  //
  // Uses a cache to avoid repeatedly reading the same file; reads only the
  // header (magic bytes) to identify the archive type.
  Expected<bool> isThinArchive(const StringRef ArchivePath);

  // Unique ID for this link (process ID as string).
  std::string UID;

  // Input files registered for this link (same order as addInput).
  std::vector<std::shared_ptr<lto::InputFile>> InputFiles;
  // Cache for isThinArchive() results keyed by archive path.
  StringMap<bool> ArchiveIsThinCache;
  // Callback used by run() to add native objects to the link.
  AddStreamFn AddStreamFunc = nullptr;
  // Per-task summary index shards from the thin link (in-memory buffers).
  std::vector<SmallString<0>> SummaryIndexFiles;
  // Per-task imported bitcode paths from the thin link.
  std::vector<std::vector<std::string>> ImportsFilesLists;
  // Per-task cache keys for incremental builds from the thin link.
  std::vector<std::string> CacheKeysList;

  /// Runs the DTLTO thin link phase, producing per-module summary indices,
  /// import lists, and cache keys for distribution.
  ///
  /// This function configures a WriteIndexesThinBackend and invokes the base
  /// LTO run, which performs the thin link. The thin link resolves cross-module
  /// references and produces:
  ///
  /// - SummaryIndexFiles: per-module summary index shards (in-memory buffers)
  /// - ImportsFilesLists: per-module lists of imported bitcode files
  /// - CacheKeysList: per-module cache keys for incremental builds
  /// - ModuleNames: per-module identifiers
  ///
  /// The Config callbacks (GetSummaryIndexStreamFunc, GetCacheKeysListRefFunc,
  /// GetImportsListRefFunc) are installed so the WriteIndexesThinBackend
  /// populates these arrays. performCodegen() later uses them to prepare
  /// backend jobs.
  ///
  /// \returns Error::success() if the thin link completes, or an Error from
  ///          Base::run().
  Error performThinLink();

  /// Derive a set of Clang options that will be shared/common for all DTLTO
  /// backend compilations. We are intentionally minimal here as these options
  /// must remain synchronized with the behavior of Clang. DTLTO does not
  /// support all the features available with in-process LTO. More features are
  /// expected to be added over time. Users can specify Clang options directly
  /// if a feature is not supported. Note that explicitly specified options that
  /// imply additional input or output file dependencies must be communicated to
  /// the distribution system, potentially by setting extra options on the
  /// distributor program.
  void buildCommonRemoteCompilerOptions();

public:
  // Parameters and shared state for DistributorDriver class.
  struct DistributionDriverParams {
    LLVM_ABI
    DistributionDriverParams() = default;
    DistributionDriverParams(StringRef DistributorArg,
                             ArrayRef<StringRef> DistributorArgsArg,
                             StringRef RemoteCompilerArg,
                             ArrayRef<StringRef> RemoteCompilerPrependArgsArg,
                             ArrayRef<StringRef> RemoteCompilerArgsArg,
                             StringRef LinkerOutputFileArg)
        : LinkerOutputFile(LinkerOutputFileArg),
          DistributorPath(DistributorArg), DistributorArgs(DistributorArgsArg),
          RemoteCompiler(RemoteCompilerArg),
          RemoteCompilerPrependArgs(RemoteCompilerPrependArgsArg),
          RemoteCompilerArgs(RemoteCompilerArgsArg) {}

    // Output linker file path.
    SString LinkerOutputFile;
    // Path to the distributor executable.
    SString DistributorPath;
    // Arguments passed to the distributor.
    ArrayRef<StringRef> DistributorArgs;
    // Compiler executabl invoked by the distributor (e.g., Clang).
    SString RemoteCompiler;
    // Options prepended to remote compiler args.
    ArrayRef<StringRef> RemoteCompilerPrependArgs;
    // User-supplied options passed to remote compiler.
    ArrayRef<StringRef> RemoteCompilerArgs;

    // Common Clang options for all compilation jobs.
    SmallVector<StringRef, 0> CodegenOptions;
    // Input paths shared across compilation jobs.
    DenseSet<StringRef> CommonInputs;
    // Target triple for compilations.
    Triple TargetTriple;
  };

private:
  // Distributor configuration class instance.
  DistributionDriverParams DistributorParams;

  // Cleanup files list.
  std::vector<std::string> CleanupList;

  // Record a file for cleanup and register signal-time removal if requested.
  void addToCleanup(StringRef Filename) {
    CleanupList.push_back(Filename.str());
    sys::RemoveFileOnSignal(Filename);
  }
};

namespace {
constexpr StringRef BCError = "DTLTO backend compilation: ";
}

class DistributionDriver {
public:
  LLVM_ABI
  DistributionDriver(DTLTO::DistributionDriverParams &ParamsArg,
                     ArrayRef<DTLTO::Job> JobsArg, bool SaveTempsArg,
                     std::function<void(StringRef)> AddToClenupArg)
      : Params{ParamsArg}, SaveTemps{SaveTempsArg},
        AddToCleanup{AddToClenupArg}, Jobs{JobsArg} {};

private:
  DTLTO::DistributionDriverParams &Params;
  // Keep temporary files when true.
  bool SaveTemps = false;
  std::function<void(StringRef)> AddToCleanup;
  ArrayRef<DTLTO::Job> Jobs;
  SmallString<128> DistributorJsonFile;

  // Generates a JSON file describing the compilations
  Error emitJson();
  // Saves JSON file on a filesystem.
  Error saveJson();

public:
  /// Invokes the distributor to compile bitcode modules remotely.
  ///
  /// Runs the distributor with the
  /// JSON manifest path; the distributor spawns remote compiler processes.
  ///
  /// \returns Error::success() on success, or an Error if the distributor
  /// fails.
  Error operator()();
};

} // namespace lto
} // namespace llvm

#endif // LLVM_DTLTO_DTLTO_H
