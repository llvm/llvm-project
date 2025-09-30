//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Level Zero RTL Options support
//
//===----------------------------------------------------------------------===//

#include "omptarget.h"

#include "L0Defs.h"
#include "L0Options.h"
#include "L0Trace.h"

namespace llvm::omp::target::plugin {

/// Read environment variables
void L0OptionsTy::processEnvironmentVars() {
  // Compilation options for IGC
  UserCompilationOptions +=
      std::string(" ") +
      StringEnvar("LIBOMPTARGET_LEVEL_ZERO_COMPILATION_OPTIONS", "").get();

  // Memory pool
  // LIBOMPTARGET_LEVEL_ZERO_MEMORY_POOL=<Option>
  //  <Option>       := 0 | <PoolInfoList>
  //  <PoolInfoList> := <PoolInfo>[,<PoolInfoList>]
  //  <PoolInfo>     := <MemType>[,<AllocMax>[,<Capacity>[,<PoolSize>]]]
  //  <MemType>      := all | device | host | shared
  //  <AllocMax>     := non-negative integer or empty, max allocation size in
  //                    MB (default: 1)
  //  <Capacity>     := positive integer or empty, number of allocations from
  //                    a single block (default: 4)
  //  <PoolSize>     := positive integer or empty, max pool size in MB
  //                    (default: 256)
  const StringEnvar MemoryPoolVar("LIBOMPTARGET_LEVEL_ZERO_MEMORY_POOL", "");
  if (MemoryPoolVar.isPresent()) {
    if (MemoryPoolVar.get() == "0") {
      Flags.UseMemoryPool = 0;
      MemPoolInfo.clear();
    } else {
      std::istringstream Str(MemoryPoolVar.get());
      int32_t MemType = -1;
      int32_t Offset = 0;
      int32_t Valid = 1;
      const std::array<int32_t, 3> DefaultValue{1, 4, 256};
      const int32_t AllMemType = INT32_MAX;
      std::array<int32_t, 3> AllInfo{1, 4, 256};
      std::map<int32_t, std::array<int32_t, 3>> PoolInfo;
      for (std::string Token; std::getline(Str, Token, ',') && Valid > 0;) {
        if (Token == "device") {
          MemType = TARGET_ALLOC_DEVICE;
          PoolInfo.emplace(MemType, DefaultValue);
          Offset = 0;
        } else if (Token == "host") {
          MemType = TARGET_ALLOC_HOST;
          PoolInfo.emplace(MemType, DefaultValue);
          Offset = 0;
        } else if (Token == "shared") {
          MemType = TARGET_ALLOC_SHARED;
          PoolInfo.emplace(MemType, DefaultValue);
          Offset = 0;
        } else if (Token == "all") {
          MemType = AllMemType;
          Offset = 0;
          Valid = 2;
        } else if (Offset < 3 && MemType >= 0) {
          int32_t Num = std::atoi(Token.c_str());
          bool ValidNum = (Num >= 0 && Offset == 0) || (Num > 0 && Offset > 0);
          if (ValidNum && MemType == AllMemType)
            AllInfo[Offset++] = Num;
          else if (ValidNum)
            PoolInfo[MemType][Offset++] = Num;
          else if (Token.size() == 0)
            Offset++;
          else
            Valid = 0;
        } else {
          Valid = 0;
        }
      }
      if (Valid > 0) {
        if (Valid == 2) {
          // "all" is specified -- ignore other inputs
          if (AllInfo[0] > 0) {
            MemPoolInfo[TARGET_ALLOC_DEVICE] = AllInfo;
            MemPoolInfo[TARGET_ALLOC_HOST] = AllInfo;
            MemPoolInfo[TARGET_ALLOC_SHARED] = std::move(AllInfo);
          } else {
            MemPoolInfo.clear();
          }
        } else {
          // Use user-specified configuration
          for (auto &I : PoolInfo) {
            if (I.second[0] > 0)
              MemPoolInfo[I.first] = I.second;
            else
              MemPoolInfo.erase(I.first);
          }
        }
      } else {
        DP("Ignoring incorrect memory pool configuration "
           "LIBOMPTARGET_LEVEL_ZERO_MEMORY_POOL=%s\n",
           MemoryPoolVar.get().c_str());
        DP("LIBOMPTARGET_LEVEL_ZERO_MEMORY_POOL=<Option>\n");
        DP("  <Option>       := 0 | <PoolInfoList>\n");
        DP("  <PoolInfoList> := <PoolInfo>[,<PoolInfoList>]\n");
        DP("  <PoolInfo>     := "
           "<MemType>[,<AllocMax>[,<Capacity>[,<PoolSize>]]]\n");
        DP("  <MemType>      := all | device | host | shared\n");
        DP("  <AllocMax>     := non-negative integer or empty, "
           "max allocation size in MB (default: 1)\n");
        DP("  <Capacity>     := positive integer or empty, "
           "number of allocations from a single block (default: 4)\n");
        DP("  <PoolSize>     := positive integer or empty, "
           "max pool size in MB (default: 256)\n");
      }
    }
  }

  if (StringEnvar("INTEL_ENABLE_OFFLOAD_ANNOTATIONS").isPresent()) {
    // To match SYCL RT behavior, we just need to check whether
    // INTEL_ENABLE_OFFLOAD_ANNOTATIONS is set. The actual value
    // does not matter.
    CommonSpecConstants.addConstant<char>(0xFF747469, 1);
  }

  // LIBOMPTARGET_LEVEL_ZERO_STAGING_BUFFER_SIZE=<SizeInKB>
  const Envar<size_t> StagingBufferSizeVar(
      "LIBOMPTARGET_LEVEL_ZERO_STAGING_BUFFER_SIZE");
  if (StagingBufferSizeVar.isPresent()) {
    size_t SizeInKB = StagingBufferSizeVar;
    if (SizeInKB > (16 << 10)) {
      SizeInKB = (16 << 10);
      DP("Staging buffer size is capped at %zu KB\n", SizeInKB);
    }
    StagingBufferSize = SizeInKB << 10;
  }

  // LIBOMPTARGET_LEVEL_ZERO_COMMAND_MODE=<Fmt>
  // <Fmt> := sync | async | async_ordered
  // sync: perform synchronization after each command
  // async: perform synchronization when it is required
  // async_ordered: same as "async", but command is ordered
  // This option is ignored unless IMM is fully enabled on compute and copy.
  // On Intel PVC GPU, when used with immediate command lists over Level Zero
  // backend, a target region may involve multiple command submissions to the
  // L0 copy queue and compute queue. L0 events are used for each submission
  // (data transfer of a single item or kernel execution). When "async" is
  // specified, a) each data transfer to device is submitted with an event.
  // b) The kernel is submitted next with a dependence on all the previous
  // data transfer events. The kernel also has an event associated with it.
  // c) The data transfer from device will be submitted with a dependence on
  // the kernel event. d) Finally wait on the host for all the events
  // associated with the data transfer from device.
  // The env-var also affects any "target update" constructs as well.
  // The env-var only affects the L0 copy/compute commands issued from a
  // single target construct execution, not across multiple invocations.
  const StringEnvar CommandModeVar("LIBOMPTARGET_LEVEL_ZERO_COMMAND_MODE");
  if (CommandModeVar.isPresent()) {
    if (match(CommandModeVar, "sync"))
      CommandMode = CommandModeTy::Sync;
    else if (match(CommandModeVar, "async"))
      CommandMode = CommandModeTy::Async;
    else if (match(CommandModeVar, "async_ordered"))
      CommandMode = CommandModeTy::AsyncOrdered;
    else
      INVALID_OPTION(LIBOMPTARGET_LEVEL_ZERO_COMMAND_MODE,
                     CommandModeVar.get().c_str());
  }

  // Detect if we need to enable compatibility with Level Zero debug mode.
  ZeDebugEnabled = BoolEnvar("ZET_ENABLE_PROGRAM_DEBUGGING", false);
}

} // namespace llvm::omp::target::plugin
