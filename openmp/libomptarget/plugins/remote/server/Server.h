//===-------------------------- Server.h - Server -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Offloading gRPC server for remote host.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPENMP_LIBOMPTARGET_PLUGINS_REMOTE_SERVER_SERVER_H
#define LLVM_OPENMP_LIBOMPTARGET_PLUGINS_REMOTE_SERVER_SERVER_H

#include <grpcpp/server_context.h>

#include "Utils.h"
#include "device.h"
#include "omptarget.h"
#include "openmp.grpc.pb.h"
#include "openmp.pb.h"
#include "rtl.h"

using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerWriter;
using grpc::Status;

using namespace openmp::libomptarget::remote;
using namespace RemoteOffloading;

using namespace google;

extern PluginManager *PM;

class RemoteOffloadImpl final : public RemoteOffload::Service {
private:
  int32_t mapHostRTLDeviceId(int32_t RTLDeviceID);

  std::unordered_map<const void *, __tgt_device_image *>
      HostToRemoteDeviceImage;
  std::unordered_map<const void *, __tgt_offload_entry *>
      HostToRemoteOffloadEntry;
  std::unordered_map<const void *, std::unique_ptr<__tgt_bin_desc>>
      Descriptions;
  __tgt_target_table *Table = nullptr;

  int DebugLevel;
  uint64_t MaxSize;
  uint64_t BlockSize;
  std::unique_ptr<protobuf::Arena> Arena;

public:
  RemoteOffloadImpl(uint64_t MaxSize, uint64_t BlockSize)
      : MaxSize(MaxSize), BlockSize(BlockSize) {
    DebugLevel = getDebugLevel();
    Arena = std::make_unique<protobuf::Arena>();
  }

  Status Shutdown(ServerContext *Context, const Null *Request,
                  I32 *Reply) override;

  Status RegisterLib(ServerContext *Context,
                     const TargetBinaryDescription *Description,
                     I32 *Reply) override;
  Status UnregisterLib(ServerContext *Context, const Pointer *Request,
                       I32 *Reply) override;

  Status IsValidBinary(ServerContext *Context,
                       const TargetDeviceImagePtr *Image,
                       I32 *IsValid) override;
  Status GetNumberOfDevices(ServerContext *Context, const Null *Null,
                            I32 *NumberOfDevices) override;

  Status InitDevice(ServerContext *Context, const I32 *DeviceNum,
                    I32 *Reply) override;
  Status InitRequires(ServerContext *Context, const I64 *RequiresFlag,
                      I32 *Reply) override;

  Status LoadBinary(ServerContext *Context, const Binary *Binary,
                    TargetTable *Reply) override;
  Status Synchronize(ServerContext *Context, const SynchronizeDevice *Info,
                     I32 *Reply) override;
  Status IsDataExchangeable(ServerContext *Context, const DevicePair *Request,
                            I32 *Reply) override;

  Status DataAlloc(ServerContext *Context, const AllocData *Request,
                   Pointer *Reply) override;

  Status DataSubmitAsync(ServerContext *Context,
                         ServerReader<SubmitDataAsync> *Reader,
                         I32 *Reply) override;
  Status DataRetrieveAsync(ServerContext *Context,
                           const RetrieveDataAsync *Request,
                           ServerWriter<Data> *Writer) override;

  Status DataExchangeAsync(ServerContext *Context,
                           const ExchangeDataAsync *Request,
                           I32 *Reply) override;

  Status DataDelete(ServerContext *Context, const DeleteData *Request,
                    I32 *Reply) override;

  Status RunTargetRegionAsync(ServerContext *Context,
                              const TargetRegionAsync *Request,
                              I32 *Reply) override;

  Status RunTargetTeamRegionAsync(ServerContext *Context,
                                  const TargetTeamRegionAsync *Request,
                                  I32 *Reply) override;
};

#endif
