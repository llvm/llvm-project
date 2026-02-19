//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Level Zero RTL Options support.
//
//===----------------------------------------------------------------------===//

#include "omptarget.h"

#include "L0Defs.h"
#include "L0Options.h"
#include "L0Trace.h"

namespace llvm::omp::target::plugin {

/// Read environment variables.
void L0OptionsTy::processEnvironmentVars() {
  // Compilation options for IGC.
  UserCompilationOptions +=
      std::string(" ") +
      StringEnvar("LIBOMPTARGET_LEVEL_ZERO_COMPILATION_OPTIONS", "").get();

  // Memory pool syntax:
  // LIBOMPTARGET_LEVEL_ZERO_MEMORY_POOL=<Option>
  //  <Option>       := 0 | <PoolInfoList>
  //  <PoolInfoList> := <PoolInfo>[,<PoolInfoList>]
  //  <PoolInfo>     := <MemType>[,<AllocMax>[,<Capacity>[,<PoolSize>]]]
  //  <MemType>      := all | device | host | shared
  //  <AllocMax>     := non-negative integer or empty, max allocation size in
  //                    MB (default: 1).
  //  <Capacity>     := positive integer or empty, number of allocations from
  //                    a single block (default: 4).
  //  <PoolSize>     := positive integer or empty, max pool size in MB
  //                    (default: 256).
  const StringEnvar MemoryPoolVar("LIBOMPTARGET_LEVEL_ZERO_MEMORY_POOL", "");
  if (MemoryPoolVar.isPresent()) {
    if (MemoryPoolVar.get() == "0") {
      Flags.UseMemoryPool = 0;
      MemPoolConfig.fill({false, 0, 0, 0});
    } else {
      std::istringstream Str(MemoryPoolVar.get());
      int32_t MemType = -1;
      int32_t Offset = 0;
      int32_t Valid = 1;
      constexpr std::array<int32_t, 3> DefaultValue{1, 4, 256};
      constexpr int32_t AllMemType =
          std::numeric_limits<decltype(AllMemType)>::max();
      std::array<int32_t, 3> AllInfo{1, 4, 256};
      std::array<std::array<int32_t, 3>, 3> PoolInfo;
      PoolInfo.fill({-1, 0, 0});
      for (std::string Token; std::getline(Str, Token, ',') && Valid > 0;) {
        if (Token == "device") {
          MemType = TARGET_ALLOC_DEVICE;
          PoolInfo[TARGET_ALLOC_DEVICE] = DefaultValue;
          Offset = 0;
        } else if (Token == "host") {
          MemType = TARGET_ALLOC_HOST;
          PoolInfo[TARGET_ALLOC_HOST] = DefaultValue;
          Offset = 0;
        } else if (Token == "shared") {
          MemType = TARGET_ALLOC_SHARED;
          PoolInfo[TARGET_ALLOC_SHARED] = DefaultValue;
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
          // "all" is specified -- ignore other inputs.
          if (AllInfo[0] > 0) {
            MemPoolConfig[TARGET_ALLOC_DEVICE] = {true, AllInfo[0], AllInfo[1],
                                                  AllInfo[2]};
            MemPoolConfig[TARGET_ALLOC_HOST] = {true, AllInfo[0], AllInfo[1],
                                                AllInfo[2]};
            MemPoolConfig[TARGET_ALLOC_SHARED] = {true, AllInfo[0], AllInfo[1],
                                                  AllInfo[2]};
          } else {
            MemPoolConfig.fill({false, 0, 0, 0});
          }
        } else {
          for (size_t Pool = 0; Pool < PoolInfo.size(); ++Pool) {
            switch (PoolInfo[Pool][0]) {
            case -1:
              // No value was specified, keep the default.
              break;
            case 0:
              // Pool was disabled.
              MemPoolConfig[Pool] = {false, 0, 0, 0};
              break;
            default:
              // Use the user specified values.
              MemPoolConfig[Pool] = {true, PoolInfo[Pool][0], PoolInfo[Pool][1],
                                     PoolInfo[Pool][2]};
              break;
            }
          }
        }
      } else {
        ODBG_OS(OLDT_Init, [&](llvm::raw_ostream &O) {
          O << "Ignoring incorrect memory pool configuration "
               "LIBOMPTARGET_LEVEL_ZERO_MEMORY_POOL="
            << MemoryPoolVar.get() << "\n";
          O << "LIBOMPTARGET_LEVEL_ZERO_MEMORY_POOL=<Option>\n";
          O << "  <Option>       := 0 | <PoolInfoList>\n";
          O << "  <PoolInfoList> := <PoolInfo>[,<PoolInfoList>]\n";
          O << "  <PoolInfo>     := "
               "<MemType>[,<AllocMax>[,<Capacity>[,<PoolSize>]]]\n";
          O << "  <MemType>      := all | device | host | shared\n";
          O << "  <AllocMax>     := non-negative integer or empty, "
               "max allocation size in MB (default: 1)\n";
          O << "  <Capacity>     := positive integer or empty, "
               "number of allocations from a single block (default: 4)\n";
          O << "  <PoolSize>     := positive integer or empty, "
               "max pool size in MB (default: 256)\n";
        });
      }
    }
  }

  if (StringEnvar("INTEL_ENABLE_OFFLOAD_ANNOTATIONS").isPresent()) {
    // To match SYCL RT behavior, we just need to check whether
    // INTEL_ENABLE_OFFLOAD_ANNOTATIONS is set. The actual value
    // does not matter.
    CommonSpecConstants.addConstant<char>(0xFF747469, 1);
  }

  // LIBOMPTARGET_LEVEL_ZERO_STAGING_BUFFER_SIZE=<SizeInKB>.
  const Envar<size_t> StagingBufferSizeVar(
      "LIBOMPTARGET_LEVEL_ZERO_STAGING_BUFFER_SIZE");
  if (StagingBufferSizeVar.isPresent()) {
    size_t SizeInKB = StagingBufferSizeVar;
    if (SizeInKB > (16 << 10)) {
      SizeInKB = (16 << 10);
      ODBG(OLDT_Init) << "Staging buffer size is capped at " << SizeInKB
                      << " KB";
    }
    StagingBufferSize = SizeInKB << 10;
  }

  // LIBOMPTARGET_LEVEL_ZERO_COMMAND_MODE=<Fmt>.
  // <Fmt> := sync | async | async_ordered
  // sync: perform synchronization after each command.
  // async: perform synchronization when it is required.
  // async_ordered: same as "async", but command is ordered.
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
  // The env-var only affects the L0 copy/  compute commands issued from a
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
      MESSAGE("Warning: Ignoring invalid value for "
              "LIBOMPTARGET_LEVEL_ZERO_COMMAND_MODE=%s\n",
              CommandModeVar.get().c_str());
  }

  // Detect if we need to enable compatibility with Level Zero debug mode.
  ZeDebugEnabled = BoolEnvar("ZET_ENABLE_PROGRAM_DEBUGGING", false);
}

} // namespace llvm::omp::target::plugin
