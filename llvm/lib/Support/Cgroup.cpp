//===- Cgroup.cpp - Cgroup support ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Threading.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

using namespace llvm;

namespace {

#ifdef __linux__

enum class CgroupVersion { V1, V2 };

struct CgroupMembership {
  CgroupVersion Version;
  std::string Path;
};

struct CgroupMount {
  std::string Root;
  std::string MountPoint;
};

static std::unique_ptr<MemoryBuffer> readFile(StringRef Path) {
  if (Path.empty())
    return nullptr;
  // procfs and cgroupfs files commonly report a size of zero and cannot be
  // read through the mmap-based path used by MemoryBuffer::getFile().
  auto Buffer = MemoryBuffer::getFileAsStream(Path);
  if (!Buffer)
    return nullptr;
  return std::move(*Buffer);
}

static bool hasController(StringRef Controllers, StringRef Expected) {
  SmallVector<StringRef, 4> Values;
  Controllers.split(Values, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  return llvm::is_contained(Values, Expected);
}

static std::string unescapeMountInfoPath(StringRef Value) {
  std::string Result;
  Result.reserve(Value.size());
  for (size_t I = 0; I < Value.size(); ++I) {
    if (Value[I] != '\\' || I + 3 >= Value.size()) {
      Result.push_back(Value[I]);
      continue;
    }
    StringRef Escape = Value.substr(I + 1, 3);
    unsigned C;
    if (Escape.getAsInteger(8, C) || C > 255) {
      Result.push_back(Value[I]);
      continue;
    }
    Result.push_back(static_cast<char>(C));
    I += 3;
  }
  return Result;
}

static std::optional<CgroupMembership>
findCgroupMembership(StringRef Contents, CgroupVersion Version) {
  SmallVector<StringRef, 16> Lines;
  Contents.split(Lines, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef Line : Lines) {
    auto [Hierarchy, Rest] = Line.split(':');
    auto [Controllers, Path] = Rest.split(':');
    if (Rest.empty() || Path.empty())
      continue;
    if (Version == CgroupVersion::V2 && Hierarchy == "0" && Controllers.empty())
      return CgroupMembership{Version, Path.str()};
    if (Version == CgroupVersion::V1 && hasController(Controllers, "cpu"))
      return CgroupMembership{Version, Path.str()};
  }
  return std::nullopt;
}

static SmallVector<CgroupMount, 2> findCgroupMounts(StringRef Contents,
                                                    CgroupVersion Version) {
  SmallVector<CgroupMount, 2> Mounts;
  SmallVector<StringRef, 16> Lines;
  Contents.split(Lines, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef Line : Lines) {
    SmallVector<StringRef, 16> Fields;
    Line.split(Fields, ' ', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    auto Separator = llvm::find(Fields, "-");
    if (Separator == Fields.end())
      continue;
    size_t SeparatorIndex = std::distance(Fields.begin(), Separator);
    if (SeparatorIndex < 5 || SeparatorIndex + 3 >= Fields.size())
      continue;

    StringRef FileSystemType = Fields[SeparatorIndex + 1];
    StringRef SuperOptions = Fields[SeparatorIndex + 3];
    bool Matches =
        Version == CgroupVersion::V2
            ? FileSystemType == "cgroup2"
            : FileSystemType == "cgroup" && hasController(SuperOptions, "cpu");
    if (!Matches)
      continue;
    Mounts.push_back(CgroupMount{unescapeMountInfoPath(Fields[3]),
                                 unescapeMountInfoPath(Fields[4])});
  }
  return Mounts;
}

static std::optional<StringRef> relativeTo(StringRef Child, StringRef Parent) {
  while (Parent.size() > 1 && Parent.ends_with('/'))
    Parent = Parent.drop_back();
  while (Child.size() > 1 && Child.ends_with('/'))
    Child = Child.drop_back();
  if (Parent == "/")
    return Child.consume_front("/") ? std::optional<StringRef>(Child)
                                    : std::nullopt;
  if (Child == Parent)
    return StringRef();
  if (!Child.consume_front(Parent) || !Child.consume_front("/"))
    return std::nullopt;
  return Child;
}

static std::optional<unsigned> cpuCountFromQuota(uint64_t Quota,
                                                 uint64_t Period) {
  if (Quota == 0 || Period == 0)
    return std::nullopt;
  uint64_t Count = Quota / Period + (Quota % Period != 0);
  return static_cast<unsigned>(
      std::min<uint64_t>(Count, std::numeric_limits<unsigned>::max()));
}

static std::optional<unsigned> readCgroupV2Limit(StringRef Path) {
  auto Buffer = readFile(Path);
  if (!Buffer)
    return std::nullopt;
  SmallVector<StringRef, 3> Fields;
  Buffer->getBuffer().trim().split(Fields, ' ', /*MaxSplit=*/-1,
                                   /*KeepEmpty=*/false);
  if (Fields.size() != 2 || Fields[0] == "max")
    return std::nullopt;
  uint64_t Quota, Period;
  if (Fields[0].getAsInteger(10, Quota) || Fields[1].getAsInteger(10, Period))
    return std::nullopt;
  return cpuCountFromQuota(Quota, Period);
}

static std::optional<unsigned> readCgroupV1Limit(StringRef QuotaPath,
                                                 StringRef PeriodPath) {
  auto QuotaBuffer = readFile(QuotaPath);
  auto PeriodBuffer = readFile(PeriodPath);
  if (!QuotaBuffer || !PeriodBuffer)
    return std::nullopt;
  int64_t Quota;
  uint64_t Period;
  if (QuotaBuffer->getBuffer().trim().getAsInteger(10, Quota) || Quota <= 0 ||
      PeriodBuffer->getBuffer().trim().getAsInteger(10, Period))
    return std::nullopt;
  return cpuCountFromQuota(static_cast<uint64_t>(Quota), Period);
}

static std::optional<unsigned>
readHierarchyLimit(const CgroupMembership &Membership,
                   const CgroupMount &Mount) {
  std::optional<StringRef> Relative = relativeTo(Membership.Path, Mount.Root);
  if (!Relative)
    return std::nullopt;

  SmallString<256> MountPoint(Mount.MountPoint);
  sys::path::remove_dots(MountPoint, /*remove_dot_dot=*/true);
  SmallString<256> Current(MountPoint);
  if (!Relative->empty())
    sys::path::append(Current, *Relative);
  sys::path::remove_dots(Current, /*remove_dot_dot=*/true);

  std::optional<unsigned> Result;
  for (;;) {
    std::optional<unsigned> Limit;
    if (Membership.Version == CgroupVersion::V2) {
      SmallString<256> CpuMax(Current);
      sys::path::append(CpuMax, "cpu.max");
      Limit = readCgroupV2Limit(CpuMax);
    } else {
      SmallString<256> Quota(Current), Period(Current);
      sys::path::append(Quota, "cpu.cfs_quota_us");
      sys::path::append(Period, "cpu.cfs_period_us");
      Limit = readCgroupV1Limit(Quota, Period);
    }
    if (Limit)
      Result = Result ? std::min(*Result, *Limit) : Limit;

    if (Current == MountPoint)
      break;
    sys::path::remove_filename(Current);
    while (Current.size() > 1 && sys::path::is_separator(Current.back()))
      Current.pop_back();
    if (Current.empty() || !relativeTo(Current, MountPoint))
      break;
  }
  return Result;
}

#endif

} // namespace

std::optional<unsigned>
llvm::detail::get_cgroup_cpu_count(const CgroupFilePaths &Paths) {
#ifdef __linux__
  auto Cgroup = readFile(Paths.ProcSelfCgroup);
  auto MountInfo = readFile(Paths.ProcSelfMountInfo);
  if (Cgroup && MountInfo) {
    for (CgroupVersion Version : {CgroupVersion::V2, CgroupVersion::V1}) {
      auto Membership = findCgroupMembership(Cgroup->getBuffer(), Version);
      if (!Membership)
        continue;
      for (const CgroupMount &Mount :
           findCgroupMounts(MountInfo->getBuffer(), Version))
        if (auto Limit = readHierarchyLimit(*Membership, Mount))
          return Limit;
    }
  }

  if (auto Limit = readCgroupV2Limit(Paths.V2CpuMax))
    return Limit;
  if (auto Limit = readCgroupV1Limit(Paths.V1CpuQuota, Paths.V1CpuPeriod))
    return Limit;
  return readCgroupV1Limit(Paths.V1CpuAcctQuota, Paths.V1CpuAcctPeriod);
#else
  return std::nullopt;
#endif
}
