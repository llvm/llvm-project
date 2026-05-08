//===--- HostRegionsAnalyzer.cpp - LLVM Advisor ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Offload/HostRegionsAnalyzer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::advisor;

static bool isSourceFile(StringRef Name) {
  return Name.ends_with_insensitive(".c") ||
         Name.ends_with_insensitive(".cpp") ||
         Name.ends_with_insensitive(".cc") ||
         Name.ends_with_insensitive(".cxx") ||
         Name.ends_with_insensitive(".cu") ||
         Name.ends_with_insensitive(".cuh") ||
         Name.ends_with_insensitive(".h") || Name.ends_with_insensitive(".hpp");
}

struct FileSummary {
  std::string Path;
  int64_t OmpPragmas = 0;
  int64_t DeviceKernels = 0;
  int64_t GlobalKernels = 0;
  int64_t HostAnnotations = 0;
};

static FileSummary scanFile(StringRef Path) {
  FileSummary Summary;
  Summary.Path = Path.str();
  ErrorOr<std::unique_ptr<MemoryBuffer>> MB = MemoryBuffer::getFile(Path);
  if (!MB)
    return Summary;

  SmallVector<StringRef, 1024> Lines;
  (*MB)->getBuffer().split(Lines, '\n');
  for (StringRef Line : Lines) {
    Line = Line.trim();
    if (Line.starts_with("#pragma omp"))
      ++Summary.OmpPragmas;
    if (Line.contains("__device__"))
      ++Summary.DeviceKernels;
    if (Line.contains("__global__"))
      ++Summary.GlobalKernels;
    if (Line.contains("__host__"))
      ++Summary.HostAnnotations;
    if (Line.contains("__kernel"))
      ++Summary.GlobalKernels;
  }
  return Summary;
}

Expected<std::unique_ptr<CapabilityResult>>
HostRegionsAnalyzer::run(const CapabilityContext &Context) {
  StringRef CapID = getCapabilityID();
  StringRef UnitID = Context.Unit.ID;
  if (Context.WorkingDirectory.empty())
    return makeUnavailableResult(CapID, UnitID, "no working directory");

  SmallVector<FileSummary, 8> Files;
  std::error_code EC;
  for (sys::fs::directory_iterator It(Context.WorkingDirectory, EC), End;
       !EC && It != End; It.increment(EC)) {
    StringRef Name = sys::path::filename(It->path());
    if (isSourceFile(Name))
      Files.push_back(scanFile(It->path()));
  }

  if (Files.empty())
    return makeUnavailableResult(
        CapID, UnitID, "no source files found in working directory");

  json::Array FileArr;
  int64_t TotalOmp = 0, TotalDevice = 0, TotalGlobal = 0, TotalHost = 0;
  for (const FileSummary &F : Files) {
    if (F.OmpPragmas == 0 && F.DeviceKernels == 0 && F.GlobalKernels == 0 &&
        F.HostAnnotations == 0)
      continue;
    json::Object Obj;
    Obj["path"] = sys::path::filename(F.Path);
    Obj["omp_pragmas"] = F.OmpPragmas;
    Obj["device_kernels"] = F.DeviceKernels;
    Obj["global_kernels"] = F.GlobalKernels;
    Obj["host_annotations"] = F.HostAnnotations;
    FileArr.push_back(std::move(Obj));
    TotalOmp += F.OmpPragmas;
    TotalDevice += F.DeviceKernels;
    TotalGlobal += F.GlobalKernels;
    TotalHost += F.HostAnnotations;
  }

  return makeJSONResult(CapID, UnitID, json::Object{
      {"file_count", static_cast<int64_t>(Files.size())},
      {"total_omp_pragmas", TotalOmp},
      {"total_device_kernels", TotalDevice},
      {"total_global_kernels", TotalGlobal},
      {"total_host_annotations", TotalHost},
      {"files", std::move(FileArr)},
  });
}
