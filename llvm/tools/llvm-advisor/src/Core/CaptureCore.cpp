//===------------------- CaptureCore.cpp - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Core/CaptureCore.h"
#include "Analysis/AnalyzerBase.h"
#include "Analysis/Clang/ClangAnalyzerUtils.h"
#include "Capability/CapabilityExecutor.h"
#include "Capability/CapabilityPlanner.h"
#include "Capability/CapabilityScheduler.h"
#include "Utils/Hashing.h"
#include "Utils/Normalization.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <chrono>

using namespace llvm;
using namespace llvm::advisor;

namespace {

// Capability categories used to determine which artifacts must be synthesized.
constexpr StringLiteral IRCapabilities[] = {
    "llvm.ir.summary",     "llvm.ir.function_stats",
    "llvm.ir.view",        "llvm.ir.diff",
    "llvm.cfg",            "llvm.dom_tree",
    "llvm.call_graph",     "llvm.loop_info",
    "llvm.ir.similarity",  "llvm.ir.passes.list",
};

constexpr StringLiteral RemarksCapabilities[] = {
    "llvm.remarks.summary",     "llvm.remarks.instruction_mix",
    "llvm.remarks.size_diff",     "llvm.remarks.detail",
    "llvm.inlining.tree",
};

constexpr StringLiteral ObjectCapabilities[] = {
    "llvm.obj.summary",       "llvm.obj.readobj",
    "llvm.debug.summary",     "llvm.debug.detail",
    "llvm.debug.consistency", "llvm.asm.view",
    "llvm.mca.report",        "llvm.cgdata",
};

template <size_t N>
bool scheduleNeeds(ArrayRef<CapabilityNode> Schedule,
                   const StringLiteral (&IDs)[N]) {
  for (const CapabilityNode &Node : Schedule)
    for (StringRef ID : IDs)
      if (Node.Spec.ID == ID)
        return true;
  return false;
}

ArrayRef<std::string> defaultCaptureCapabilities() {
  static const std::string Defaults[] = {
      "build.command.meta",   "build.dependency.headers",
      "clang.diag.summary",   "clang.ast.summary",
      "llvm.ir.summary",      "llvm.ir.function_stats",
      "llvm.remarks.summary", "llvm.remarks.detail",
      "llvm.obj.summary",     "llvm.debug.summary",
  };
  return ArrayRef<std::string>(Defaults);
}

// Finds the IR file adjacent to ObjectPath or SourcePath.
std::string findIRPath(const UnitRecord &Unit, StringRef ObjectPath) {
  if (!Unit.IRPath.empty() && sys::fs::exists(Unit.IRPath))
    return Unit.IRPath;

  SmallVector<std::string, 8> Candidates;
  auto addCandidates = [&](StringRef Path) {
    if (Path.empty())
      return;
    SmallString<256> LL(Path);
    SmallString<256> BC(Path);
    sys::path::replace_extension(LL, "ll");
    sys::path::replace_extension(BC, "bc");
    Candidates.push_back(LL.str().str());
    Candidates.push_back(BC.str().str());
  };
  addCandidates(ObjectPath);
  addCandidates(Unit.SourcePath);
  for (const std::string &Candidate : Candidates) {
    if (sys::fs::exists(Candidate))
      return Candidate;
  }
  return {};
}

// Returns the per-unit artifact cache directory, creating it if needed.
Expected<SmallString<256>> artifactDir(StringRef StoreRoot, StringRef UnitID) {
  SmallString<256> Dir(StoreRoot);
  sys::path::append(Dir, "artifacts");
  // Use full unit ID as directory name — it's a stable content hash.
  sys::path::append(Dir, UnitID);
  if (std::error_code EC = sys::fs::create_directories(Dir))
    return createStringError(EC, "failed to create artifact directory: %s",
                             Dir.c_str());
  return Dir;
}

CapabilityContext makeContext(const UnitRecord &Unit) {
  CapabilityContext Context;
  Context.Unit = Unit;
  Context.SourcePath = Unit.SourcePath;
  Context.ObjectPath = Unit.ObjectPath;
  Context.IRPath = Unit.IRPath;
  Context.RemarksPath = Unit.RemarksPath;
  Context.ToolchainVersion = Unit.ToolchainVersion;
  Context.WorkingDirectory = Unit.Directory;
  return Context;
}

// Auto-generate artifacts that are missing but needed by the schedule.
// Generation failures are non-fatal — the capability reports 'available:
// false' and the user can fix their environment and re-capture.
Error synthesizeArtifacts(CapabilityContext &Context, StringRef StoreRoot,
                          bool NeedsObject, bool NeedsIR, bool NeedsRemarks) {
  if (!NeedsObject && !NeedsIR && !NeedsRemarks)
    return Error::success();

  Expected<SmallString<256>> ArtDirOrErr = artifactDir(StoreRoot, Context.Unit.ID);
  if (!ArtDirOrErr)
    return ArtDirOrErr.takeError();

  const SmallString<256> &ArtDir = *ArtDirOrErr;

  if (NeedsObject && Context.ObjectPath.empty()) {
    SmallString<256> ObjOut(ArtDir);
    sys::path::append(ObjOut, "unit.o");
    if (auto Path = emitObject(Context, ObjOut))
      Context.ObjectPath = *Path;
    else
      consumeError(Path.takeError());
  }

  if (NeedsIR && Context.IRPath.empty()) {
    SmallString<256> IROut(ArtDir);
    sys::path::append(IROut, "module.ll");
    if (auto Path = emitLLVMIR(Context, IROut))
      Context.IRPath = *Path;
    else
      consumeError(Path.takeError());
  }

  if (NeedsRemarks && findRemarksPath(Context).empty()) {
    SmallString<256> RemarksOut(ArtDir);
    sys::path::append(RemarksOut, "remarks.opt.yaml");
    if (auto Path = emitOptRemarks(Context, RemarksOut))
      Context.RemarksPath = *Path;
    else
      consumeError(Path.takeError());
  }

  return Error::success();
}

} // namespace

Expected<SnapshotRecord>
CaptureCore::initializeSnapshot(StringRef SourceRoot, StringRef BuildRoot) {
  uint64_t Now = std::chrono::duration_cast<std::chrono::seconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count();

  SnapshotRecord Snapshot;
  Expected<std::string> CanonSourceRoot =
      canonicalizePath(SourceRoot, {SourceRoot});
  if (!CanonSourceRoot)
    return CanonSourceRoot.takeError();
  Expected<std::string> CanonBuildRoot =
      canonicalizePath(BuildRoot, {BuildRoot});
  if (!CanonBuildRoot)
    return CanonBuildRoot.takeError();

  Snapshot.SourceRoot = *CanonSourceRoot;
  Snapshot.BuildRoot = *CanonBuildRoot;
  Snapshot.CreatedUnix = Now;
  Snapshot.ID = computeSnapshotID(Snapshot.SourceRoot, Snapshot.BuildRoot, Now);
  if (Error Err = Storage.metadata().putSnapshot(Snapshot))
    return std::move(Err);

  return Snapshot;
}

Expected<UnitRecord>
CaptureCore::prepareUnit(const CompileCommand &Command,
                         const SnapshotRecord &Snapshot) {
  UnitRecord Unit;
  Unit.SnapshotID = Snapshot.ID;
  Unit.Directory = normalizePath(Command.Directory);
  std::string SourcePath = normalizePath(Command.File, Unit.Directory);
  Expected<std::string> CanonSource =
      canonicalizePath(SourcePath, {Snapshot.SourceRoot, Snapshot.BuildRoot});
  if (!CanonSource)
    return CanonSource.takeError();
  Unit.SourcePath = *CanonSource;
  Unit.Language = inferLanguage(Command.File);
  Unit.Arguments = normalizeCommand(Command.Arguments);
  Unit.TargetTriple = inferTargetTriple(Unit.Arguments);
  Unit.ObjectPath = resolveOutputPath(Unit.Arguments, Unit.Directory);
  if (!Unit.ObjectPath.empty() && !sys::fs::exists(Unit.ObjectPath))
    Unit.ObjectPath.clear();
  Unit.IRPath = findIRPath(Unit, Unit.ObjectPath);
  Expected<std::string> SourceHash = hashFile(Unit.SourcePath);
  if (!SourceHash)
    return SourceHash.takeError();
  Unit.SourceContentHash = *SourceHash;
  Unit.CommandFingerprint = hashJSON(toJSON(Command));
  Unit.ID = Identity.compute(Unit);
  return Unit;
}

Expected<SnapshotRecord>
CaptureCore::createSnapshot(StringRef SourceRoot, StringRef BuildRoot,
                            ArrayRef<std::string> Capabilities) {
  Expected<SnapshotRecord> Snapshot = initializeSnapshot(SourceRoot, BuildRoot);
  if (!Snapshot)
    return Snapshot.takeError();

  ArrayRef<std::string> Planned =
      Capabilities.empty() ? defaultCaptureCapabilities() : Capabilities;

  CapabilityPlanner Planner(Registry);
  Expected<SmallVector<CapabilityNode, 16>> Plan = Planner.plan(Planned);
  if (!Plan)
    return Plan.takeError();

  CapabilityScheduler Scheduler;
  SmallVector<CapabilityNode, 16> Schedule = Scheduler.schedule(*Plan);
  CapabilityExecutor Executor(Registry, Storage);

  bool NeedsIR = scheduleNeeds(Schedule, IRCapabilities);
  bool NeedsRemarks = scheduleNeeds(Schedule, RemarksCapabilities);
  bool NeedsObject = scheduleNeeds(Schedule, ObjectCapabilities);

  Expected<SmallVector<CompileCommand, 64>> Commands =
      Builds.loadCompileCommands(BuildRoot);
  if (!Commands)
    return Commands.takeError();

  for (const CompileCommand &Command : *Commands) {
    Expected<UnitRecord> Unit = prepareUnit(Command, *Snapshot);
    if (!Unit)
      return Unit.takeError();

    CapabilityContext Context = makeContext(*Unit);

    if (Error Err = synthesizeArtifacts(Context, Storage.root(), NeedsObject,
                                        NeedsIR, NeedsRemarks))
      return std::move(Err);

    Unit->ObjectPath = Context.ObjectPath;
    Unit->IRPath = Context.IRPath;
    Unit->RemarksPath = Context.RemarksPath;
    Context.Unit = *Unit;

    if (Error Err = Storage.metadata().putUnit(*Unit))
      return std::move(Err);
    if (Error Err =
            Storage.indexes().add("snapshot_units", Snapshot->ID, Unit->ID))
      return std::move(Err);

    Expected<json::Array> Results = Executor.execute(Schedule, Context);
    if (!Results)
      return Results.takeError();
  }

  return *Snapshot;
}
