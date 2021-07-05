//===------------------ Client.h - Client Implementation ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// gRPC Client for the remote plugin.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPENMP_LIBOMPTARGET_PLUGINS_REMOTE_SRC_CLIENT_H
#define LLVM_OPENMP_LIBOMPTARGET_PLUGINS_REMOTE_SRC_CLIENT_H

#include "Utils.h"
#include "omptarget.h"
#include <google/protobuf/arena.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/security/credentials.h>
#include <grpcpp/support/channel_arguments.h>
#include <memory>
#include <mutex>
#include <numeric>

using grpc::Channel;
using openmp::libomptarget::remote::RemoteOffload;
using namespace RemoteOffloading;

using namespace google;

class RemoteOffloadClient {
  int DebugLevel;
  const int Timeout;
  const uint64_t MaxSize;
  const int64_t BlockSize;

  std::unique_ptr<RemoteOffload::Stub> Stub;
  std::unique_ptr<protobuf::Arena> Arena;

  std::unique_ptr<std::mutex> ArenaAllocatorLock;

  std::map<int32_t, std::unordered_map<void *, void *>> RemoteEntries;
  std::map<int32_t, std::unique_ptr<__tgt_target_table>> DevicesToTables;

  template <typename Fn1, typename Fn2, typename TReturn>
  auto remoteCall(Fn1 Preprocessor, Fn2 Postprocessor, TReturn ErrorValue,
                  bool CanTimeOut = true);

public:
  RemoteOffloadClient(std::shared_ptr<Channel> Channel, int Timeout,
                      uint64_t MaxSize, int64_t BlockSize)
      : Timeout(Timeout), MaxSize(MaxSize), BlockSize(BlockSize),
        Stub(RemoteOffload::NewStub(Channel)) {
    DebugLevel = getDebugLevel();
    Arena = std::make_unique<protobuf::Arena>();
    ArenaAllocatorLock = std::make_unique<std::mutex>();
  }

  RemoteOffloadClient(RemoteOffloadClient &&C) = default;

  ~RemoteOffloadClient() {
    for (auto &TableIt : DevicesToTables)
      freeTargetTable(TableIt.second.get());
  }

  int32_t shutdown(void);

  int32_t registerLib(__tgt_bin_desc *Desc);
  int32_t unregisterLib(__tgt_bin_desc *Desc);

  int32_t isValidBinary(__tgt_device_image *Image);
  int32_t getNumberOfDevices();

  int32_t initDevice(int32_t DeviceId);
  int32_t initRequires(int64_t RequiresFlags);

  __tgt_target_table *loadBinary(int32_t DeviceId, __tgt_device_image *Image);

  void *dataAlloc(int32_t DeviceId, int64_t Size, void *HstPtr);
  int32_t dataDelete(int32_t DeviceId, void *TgtPtr);

  int32_t dataSubmit(int32_t DeviceId, void *TgtPtr, void *HstPtr,
                     int64_t Size);
  int32_t dataRetrieve(int32_t DeviceId, void *HstPtr, void *TgtPtr,
                       int64_t Size);

  int32_t isDataExchangeable(int32_t SrcDevId, int32_t DstDevId);
  int32_t dataExchange(int32_t SrcDevId, void *SrcPtr, int32_t DstDevId,
                       void *DstPtr, int64_t Size);

  int32_t runTargetRegion(int32_t DeviceId, void *TgtEntryPtr, void **TgtArgs,
                          ptrdiff_t *TgtOffsets, int32_t ArgNum);
  int32_t runTargetTeamRegion(int32_t DeviceId, void *TgtEntryPtr,
                              void **TgtArgs, ptrdiff_t *TgtOffsets,
                              int32_t ArgNum, int32_t TeamNum,
                              int32_t ThreadLimit, uint64_t LoopTripCount);
};

class RemoteClientManager {
private:
  std::vector<RemoteOffloadClient> Clients;
  std::vector<int> Devices;

  std::pair<int32_t, int32_t> mapDeviceId(int32_t DeviceId);
  int DebugLevel;

public:
  RemoteClientManager() {
    ClientManagerConfigTy Config;

    grpc::ChannelArguments ChArgs;
    ChArgs.SetMaxReceiveMessageSize(-1);
    DebugLevel = getDebugLevel();
    for (auto Address : Config.ServerAddresses) {
      Clients.push_back(RemoteOffloadClient(
          grpc::CreateChannel(Address, grpc::InsecureChannelCredentials()),
          Config.Timeout, Config.MaxSize, Config.BlockSize));
    }
  }

  int32_t shutdown(void);

  int32_t registerLib(__tgt_bin_desc *Desc);
  int32_t unregisterLib(__tgt_bin_desc *Desc);

  int32_t isValidBinary(__tgt_device_image *Image);
  int32_t getNumberOfDevices();

  int32_t initDevice(int32_t DeviceId);
  int32_t initRequires(int64_t RequiresFlags);

  __tgt_target_table *loadBinary(int32_t DeviceId, __tgt_device_image *Image);

  void *dataAlloc(int32_t DeviceId, int64_t Size, void *HstPtr);
  int32_t dataDelete(int32_t DeviceId, void *TgtPtr);

  int32_t dataSubmit(int32_t DeviceId, void *TgtPtr, void *HstPtr,
                     int64_t Size);
  int32_t dataRetrieve(int32_t DeviceId, void *HstPtr, void *TgtPtr,
                       int64_t Size);

  int32_t isDataExchangeable(int32_t SrcDevId, int32_t DstDevId);
  int32_t dataExchange(int32_t SrcDevId, void *SrcPtr, int32_t DstDevId,
                       void *DstPtr, int64_t Size);

  int32_t runTargetRegion(int32_t DeviceId, void *TgtEntryPtr, void **TgtArgs,
                          ptrdiff_t *TgtOffsets, int32_t ArgNum);
  int32_t runTargetTeamRegion(int32_t DeviceId, void *TgtEntryPtr,
                              void **TgtArgs, ptrdiff_t *TgtOffsets,
                              int32_t ArgNum, int32_t TeamNum,
                              int32_t ThreadLimit, uint64_t LoopTripCount);
};

#endif
