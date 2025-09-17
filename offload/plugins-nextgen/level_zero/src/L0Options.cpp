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

/// Is the given RootID, SubID, CcsID specified in ONEAPI_DEVICE_SELECTOR
bool L0OptionsTy::shouldAddDevice(int32_t RootID, int32_t SubID,
                                  int32_t CCSID) const {
  if (ExplicitRootDevices.empty())
    return false;
  for (const auto &RootDev : ExplicitRootDevices) {
    const auto ErootID = std::get<1>(RootDev);
    if (ErootID != -2 && RootID != ErootID)
      continue;
    const auto EsubID = std::get<2>(RootDev);
    if (((EsubID != -2) || (SubID == -1)) && (EsubID != SubID))
      continue;
    const auto ECCSID = std::get<3>(RootDev);
    if (((ECCSID != -2) || (CCSID == -1)) && (ECCSID != CCSID))
      continue;
    // Check if isDiscard
    if (!std::get<0>(RootDev))
      return false;
    return true;
  }
  return false;
}

/// Read environment variables
void L0OptionsTy::processEnvironmentVars() {
  // Compilation options for IGC
  UserCompilationOptions +=
      std::string(" ") +
      StringEnvar("LIBOMPTARGET_LEVEL_ZERO_COMPILATION_OPTIONS", "").get();

  // Explicit Device mode if ONEAPI_DEVICE_SELECTOR is set
  const StringEnvar DeviceSelectorVar("ONEAPI_DEVICE_SELECTOR", "");
  if (DeviceSelectorVar.isPresent()) {
    std::string EnvStr(std::move(DeviceSelectorVar.get()));
    uint32_t numDiscard = 0;
    std::transform(EnvStr.begin(), EnvStr.end(), EnvStr.begin(),
                   [](unsigned char C) { return std::tolower(C); });

    std::vector<std::string_view> Entries = tokenize(EnvStr, ";", true);
    for (const auto &Term : Entries) {
      bool isDiscard = false;
      std::vector<std::string_view> Pair = tokenize(Term, ":", true);
      if (Pair.empty()) {
        FAILURE_MESSAGE(
            "Incomplete selector! Pair and device must be specified.\n");
      } else if (Pair.size() == 1) {
        FAILURE_MESSAGE("Incomplete selector!  Try '%s:*'if all devices "
                        "under the Pair was original intention.\n",
                        Pair[0].data());
      } else if (Pair.size() > 2) {
        FAILURE_MESSAGE(
            "Error parsing selector string \"%s\" Too many colons (:)\n",
            Term.data());
      }
      if (!((Pair[0][0] == '*') ||
            (!strncmp(Pair[0].data(), "level_zero", Pair[0].length())) ||
            (!strncmp(Pair[0].data(), "!level_zero", Pair[0].length()))))
        break;
      isDiscard = Pair[0][0] == '!';
      if (isDiscard)
        numDiscard++;
      else if (numDiscard > 0)
        FAILURE_MESSAGE("All negative(discarding) filters must appear after "
                        "all positive(accepting) filters!");

      std::vector<std::string_view> Targets = tokenize(Pair[1], ",", true);
      for (const auto &TargetStr : Targets) {
        bool HasDeviceWildCard = false;
        bool HasSubDeviceWildCard = false;
        bool DeviceNum = false;
        std::vector<std::string_view> DeviceSubTuple =
            tokenize(TargetStr, ".", true);
        int32_t RootD[3] = {-1, -1, -1};
        if (DeviceSubTuple.empty()) {
          FAILURE_MESSAGE(
              "ONEAPI_DEVICE_SELECTOR parsing error. Device must be "
              "specified.");
        }

        std::string_view TopDeviceStr = DeviceSubTuple[0];
        static const std::array<std::string, 7> DeviceStr = {
            "host", "cpu", "gpu", "acc", "*"};
        auto It =
            find_if(DeviceStr.begin(), DeviceStr.end(),
                    [&](auto DeviceStr) { return TopDeviceStr == DeviceStr; });
        if (It != DeviceStr.end()) {
          if (TopDeviceStr[0] == '*') {
            HasDeviceWildCard = true;
            RootD[0] = -2;
          } else if (!strncmp(DeviceSubTuple[0].data(), "gpu", 3))
            continue;
        } else {
          std::string TDS(TopDeviceStr);
          if (!isDigits(TDS)) {
            FAILURE_MESSAGE("error parsing device number: %s",
                            DeviceSubTuple[0].data());
          } else {
            RootD[0] = std::stoi(TDS);
            DeviceNum = true;
          }
        }
        if (DeviceSubTuple.size() >= 2) {
          if (!DeviceNum && !HasDeviceWildCard)
            FAILURE_MESSAGE("sub-devices can only be requested when parent "
                            "device is specified by number or wildcard, not a "
                            "device type like \'gpu\'");
          std::string_view SubDeviceStr = DeviceSubTuple[1];
          if (SubDeviceStr[0] == '*') {
            HasSubDeviceWildCard = true;
            RootD[1] = -2;
          } else {
            if (HasDeviceWildCard) // subdevice is a number and device is a *
              FAILURE_MESSAGE(
                  "sub-device can't be requested by number if parent "
                  "device is specified by a wildcard.");

            std::string SDS(SubDeviceStr);
            if (!isDigits(SDS)) {
              FAILURE_MESSAGE("error parsing subdevice index: %s",
                              DeviceSubTuple[1].data());
            } else
              RootD[1] = std::stoi(SDS);
          }
        }
        if (DeviceSubTuple.size() == 3) {
          std::string_view SubSubDeviceStr = DeviceSubTuple[2];
          if (SubSubDeviceStr[0] == '*') {
            RootD[2] = -2;
          } else {
            if (HasSubDeviceWildCard)
              FAILURE_MESSAGE("sub-sub-device can't be requested by number if "
                              "sub-device before is specified by a wildcard.");
            std::string SSDS(SubSubDeviceStr);
            if (!isDigits(SSDS)) {
              FAILURE_MESSAGE("error parsing sub-sub-device index: %s",
                              DeviceSubTuple[2].data());
            } else
              RootD[2] = std::stoi(SSDS);
          }
        } else if (DeviceSubTuple.size() > 3) {
          FAILURE_MESSAGE("error parsing %s Only two levels of sub-devices "
                          "supported at this time ",
                          TargetStr.data());
        }
        if (isDiscard)
          ExplicitRootDevices.insert(
              ExplicitRootDevices.begin(),
              std::tuple<bool, int32_t, int32_t, int32_t>(!isDiscard, RootD[0],
                                                          RootD[1], RootD[2]));
        else
          ExplicitRootDevices.push_back(
              std::tuple<bool, int32_t, int32_t, int32_t>(!isDiscard, RootD[0],
                                                          RootD[1], RootD[2]));
      }
    }
  }

  DP("ONEAPI_DEVICE_SELECTOR specified %zu root devices\n",
     ExplicitRootDevices.size());
  DP("  (Accept/Discard [T/F] DeviceID[.SubID[.CCSID]]) -2(all), "
     "-1(ignore)\n");
  for (auto &T : ExplicitRootDevices) {
    DP(" %c %d.%d.%d\n", (std::get<0>(T) == true) ? 'T' : 'F', std::get<1>(T),
       std::get<2>(T), std::get<3>(T));
    (void)T; // silence warning
  }

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
}
/// Parse String  and split into tokens of string_views based on the
/// Delim character.
std::vector<std::string_view>
L0OptionsTy::tokenize(const std::string_view &Filter, const std::string &Delim,
                      bool ProhibitEmptyTokens) {
  std::vector<std::string_view> Tokens;
  size_t Pos = 0;
  size_t LastPos = 0;
  while ((Pos = Filter.find(Delim, LastPos)) != std::string::npos) {
    std::string_view Tok(Filter.data() + LastPos, (Pos - LastPos));

    if (!Tok.empty()) {
      Tokens.push_back(Tok);
    } else if (ProhibitEmptyTokens) {
      FAILURE_MESSAGE("ONEAPI_DEVICE_SELECTOR parsing error. Empty input "
                      "before '%s'delimiter is not allowed.",
                      Delim.c_str());
    }
    // move the search starting index
    LastPos = Pos + 1;
  }

  // Add remainder if any
  if (LastPos < Filter.size()) {
    std::string_view Tok(Filter.data() + LastPos, Filter.size() - LastPos);
    Tokens.push_back(Tok);
  } else if ((LastPos != 0) && ProhibitEmptyTokens) {
    // if delimiter is the last sybmol in the string.
    FAILURE_MESSAGE("ONEAPI_DEVICE_SELECTOR parsing error. Empty input after "
                    "'%s' delimiter is not allowed.",
                    Delim.c_str());
  }
  return Tokens;
}

} // namespace llvm::omp::target::plugin
