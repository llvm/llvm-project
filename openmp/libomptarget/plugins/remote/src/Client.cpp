//===----------------- Client.cpp - Client Implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// gRPC (Client) for the remote plugin.
//
//===----------------------------------------------------------------------===//

#include <cmath>

#include "Client.h"
#include "omptarget.h"
#include "openmp.pb.h"

using namespace std::chrono;

using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientWriter;
using grpc::Status;

template <typename Fn1, typename Fn2, typename TReturn>
auto RemoteOffloadClient::remoteCall(Fn1 Preprocessor, Fn2 Postprocessor,
                                     TReturn ErrorValue, bool CanTimeOut) {
  ArenaAllocatorLock->lock();
  if (Arena->SpaceAllocated() >= MaxSize)
    Arena->Reset();
  ArenaAllocatorLock->unlock();

  ClientContext Context;
  if (CanTimeOut) {
    auto Deadline =
        std::chrono::system_clock::now() + std::chrono::seconds(Timeout);
    Context.set_deadline(Deadline);
  }

  Status RPCStatus;
  auto Reply = Preprocessor(RPCStatus, Context);

  if (!RPCStatus.ok()) {
    CLIENT_DBG("%s", RPCStatus.error_message().c_str())
  } else {
    return Postprocessor(Reply);
  }

  CLIENT_DBG("Failed")
  return ErrorValue;
}

int32_t RemoteOffloadClient::shutdown(void) {
  ClientContext Context;
  Null Request;
  I32 Reply;
  CLIENT_DBG("Shutting down server.")
  auto Status = Stub->Shutdown(&Context, Request, &Reply);
  if (Status.ok())
    return Reply.number();
  return 1;
}

int32_t RemoteOffloadClient::registerLib(__tgt_bin_desc *Desc) {
  return remoteCall(
      /* Preprocessor */
      [&](auto &RPCStatus, auto &Context) {
        auto *Request = protobuf::Arena::CreateMessage<TargetBinaryDescription>(
            Arena.get());
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());
        loadTargetBinaryDescription(Desc, *Request);
        Request->set_bin_ptr((uint64_t)Desc);

        RPCStatus = Stub->RegisterLib(&Context, *Request, Reply);
        return Reply;
      },
      /* Postprocessor */
      [&](const auto &Reply) {
        if (Reply->number() == 0) {
          CLIENT_DBG("Registered library")
          return 0;
        }
        return 1;
      },
      /* Error Value */ 1);
}

int32_t RemoteOffloadClient::unregisterLib(__tgt_bin_desc *Desc) {
  return remoteCall(
      /* Preprocessor */
      [&](auto &RPCStatus, auto &Context) {
        auto *Request = protobuf::Arena::CreateMessage<Pointer>(Arena.get());
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());

        Request->set_number((uint64_t)Desc);

        RPCStatus = Stub->UnregisterLib(&Context, *Request, Reply);
        return Reply;
      },
      /* Postprocessor */
      [&](const auto &Reply) {
        if (Reply->number() == 0) {
          CLIENT_DBG("Unregistered library")
          return 0;
        }
        CLIENT_DBG("Failed to unregister library")
        return 1;
      },
      /* Error Value */ 1);
}

int32_t RemoteOffloadClient::isValidBinary(__tgt_device_image *Image) {
  return remoteCall(
      /* Preprocessor */
      [&](auto &RPCStatus, auto &Context) {
        auto *Request =
            protobuf::Arena::CreateMessage<TargetDeviceImagePtr>(Arena.get());
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());

        Request->set_image_ptr((uint64_t)Image->ImageStart);

        auto *EntryItr = Image->EntriesBegin;
        while (EntryItr != Image->EntriesEnd)
          Request->add_entry_ptrs((uint64_t)EntryItr++);

        RPCStatus = Stub->IsValidBinary(&Context, *Request, Reply);
        return Reply;
      },
      /* Postprocessor */
      [&](const auto &Reply) {
        if (Reply->number()) {
          CLIENT_DBG("Validated binary")
        } else {
          CLIENT_DBG("Could not validate binary")
        }
        return Reply->number();
      },
      /* Error Value */ 0);
}

int32_t RemoteOffloadClient::getNumberOfDevices() {
  return remoteCall(
      /* Preprocessor */
      [&](Status &RPCStatus, ClientContext &Context) {
        auto *Request = protobuf::Arena::CreateMessage<Null>(Arena.get());
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());

        RPCStatus = Stub->GetNumberOfDevices(&Context, *Request, Reply);

        return Reply;
      },
      /* Postprocessor */
      [&](const auto &Reply) {
        if (Reply->number()) {
          CLIENT_DBG("Found %d devices", Reply->number())
        } else {
          CLIENT_DBG("Could not get the number of devices")
        }
        return Reply->number();
      },
      /*Error Value*/ -1);
}

int32_t RemoteOffloadClient::initDevice(int32_t DeviceId) {
  return remoteCall(
      /* Preprocessor */
      [&](auto &RPCStatus, auto &Context) {
        auto *Request = protobuf::Arena::CreateMessage<I32>(Arena.get());
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());

        Request->set_number(DeviceId);

        RPCStatus = Stub->InitDevice(&Context, *Request, Reply);

        return Reply;
      },
      /* Postprocessor */
      [&](const auto &Reply) {
        if (!Reply->number()) {
          CLIENT_DBG("Initialized device %d", DeviceId)
        } else {
          CLIENT_DBG("Could not initialize device %d", DeviceId)
        }
        return Reply->number();
      },
      /* Error Value */ -1);
}

int32_t RemoteOffloadClient::initRequires(int64_t RequiresFlags) {
  return remoteCall(
      /* Preprocessor */
      [&](auto &RPCStatus, auto &Context) {
        auto *Request = protobuf::Arena::CreateMessage<I64>(Arena.get());
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());
        Request->set_number(RequiresFlags);
        RPCStatus = Stub->InitRequires(&Context, *Request, Reply);
        return Reply;
      },
      /* Postprocessor */
      [&](const auto &Reply) {
        if (Reply->number()) {
          CLIENT_DBG("Initialized requires")
        } else {
          CLIENT_DBG("Could not initialize requires")
        }
        return Reply->number();
      },
      /* Error Value */ -1);
}

__tgt_target_table *RemoteOffloadClient::loadBinary(int32_t DeviceId,
                                                    __tgt_device_image *Image) {
  return remoteCall(
      /* Preprocessor */
      [&](auto &RPCStatus, auto &Context) {
        auto *ImageMessage =
            protobuf::Arena::CreateMessage<Binary>(Arena.get());
        auto *Reply = protobuf::Arena::CreateMessage<TargetTable>(Arena.get());
        ImageMessage->set_image_ptr((uint64_t)Image->ImageStart);
        ImageMessage->set_device_id(DeviceId);

        RPCStatus = Stub->LoadBinary(&Context, *ImageMessage, Reply);
        return Reply;
      },
      /* Postprocessor */
      [&](auto &Reply) {
        if (Reply->entries_size() == 0) {
          CLIENT_DBG("Could not load image %p onto device %d", Image, DeviceId)
          return (__tgt_target_table *)nullptr;
        }
        DevicesToTables[DeviceId] = std::make_unique<__tgt_target_table>();
        unloadTargetTable(*Reply, DevicesToTables[DeviceId].get(),
                          RemoteEntries[DeviceId]);

        CLIENT_DBG("Loaded Image %p to device %d with %d entries", Image,
                   DeviceId, Reply->entries_size())

        return DevicesToTables[DeviceId].get();
      },
      /* Error Value */ (__tgt_target_table *)nullptr,
      /* CanTimeOut */ false);
}

int32_t RemoteOffloadClient::isDataExchangeable(int32_t SrcDevId,
                                                int32_t DstDevId) {
  return remoteCall(
      /* Preprocessor */
      [&](auto &RPCStatus, auto &Context) {
        auto *Request = protobuf::Arena::CreateMessage<DevicePair>(Arena.get());
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());

        Request->set_src_dev_id(SrcDevId);
        Request->set_dst_dev_id(DstDevId);

        RPCStatus = Stub->IsDataExchangeable(&Context, *Request, Reply);
        return Reply;
      },
      /* Postprocessor */
      [&](auto &Reply) {
        if (Reply->number()) {
          CLIENT_DBG("Data is exchangeable between %d, %d", SrcDevId, DstDevId)
        } else {
          CLIENT_DBG("Data is not exchangeable between %d, %d", SrcDevId,
                     DstDevId)
        }
        return Reply->number();
      },
      /* Error Value */ -1);
}

void *RemoteOffloadClient::dataAlloc(int32_t DeviceId, int64_t Size,
                                     void *HstPtr) {
  return remoteCall(
      /* Preprocessor */
      [&](auto &RPCStatus, auto &Context) {
        auto *Reply = protobuf::Arena::CreateMessage<Pointer>(Arena.get());
        auto *Request = protobuf::Arena::CreateMessage<AllocData>(Arena.get());

        Request->set_device_id(DeviceId);
        Request->set_size(Size);
        Request->set_hst_ptr((uint64_t)HstPtr);

        RPCStatus = Stub->DataAlloc(&Context, *Request, Reply);
        return Reply;
      },
      /* Postprocessor */
      [&](auto &Reply) {
        if (Reply->number()) {
          CLIENT_DBG("Allocated %ld bytes on device %d at %p", Size, DeviceId,
                     (void *)Reply->number())
        } else {
          CLIENT_DBG("Could not allocate %ld bytes on device %d at %p", Size,
                     DeviceId, (void *)Reply->number())
        }
        return (void *)Reply->number();
      },
      /* Error Value */ (void *)nullptr);
}

int32_t RemoteOffloadClient::dataSubmit(int32_t DeviceId, void *TgtPtr,
                                        void *HstPtr, int64_t Size) {

  return remoteCall(
      /* Preprocessor */
      [&](auto &RPCStatus, auto &Context) {
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());
        std::unique_ptr<ClientWriter<SubmitData>> Writer(
            Stub->DataSubmit(&Context, Reply));

        if (Size > BlockSize) {
          int64_t Start = 0, End = BlockSize;
          for (auto I = 0; I < ceil((float)Size / BlockSize); I++) {
            auto *Request =
                protobuf::Arena::CreateMessage<SubmitData>(Arena.get());

            Request->set_device_id(DeviceId);
            Request->set_data((char *)HstPtr + Start, End - Start);
            Request->set_hst_ptr((uint64_t)HstPtr);
            Request->set_tgt_ptr((uint64_t)TgtPtr);
            Request->set_start(Start);
            Request->set_size(Size);

            if (!Writer->Write(*Request)) {
              CLIENT_DBG("Broken stream when submitting data")
              Reply->set_number(0);
              return Reply;
            }

            Start += BlockSize;
            End += BlockSize;
            if (End >= Size)
              End = Size;
          }
        } else {
          auto *Request =
              protobuf::Arena::CreateMessage<SubmitData>(Arena.get());

          Request->set_device_id(DeviceId);
          Request->set_data(HstPtr, Size);
          Request->set_hst_ptr((uint64_t)HstPtr);
          Request->set_tgt_ptr((uint64_t)TgtPtr);
          Request->set_start(0);
          Request->set_size(Size);

          if (!Writer->Write(*Request)) {
            CLIENT_DBG("Broken stream when submitting data")
            Reply->set_number(0);
            return Reply;
          }
        }

        Writer->WritesDone();
        RPCStatus = Writer->Finish();

        return Reply;
      },
      /* Postprocessor */
      [&](auto &Reply) {
        if (!Reply->number()) {
          CLIENT_DBG(" submitted %ld bytes on device %d at %p", Size, DeviceId,
                     TgtPtr)
        } else {
          CLIENT_DBG("Could not async submit %ld bytes on device %d at %p",
                     Size, DeviceId, TgtPtr)
        }
        return Reply->number();
      },
      /* Error Value */ -1,
      /* CanTimeOut */ false);
}

int32_t RemoteOffloadClient::dataRetrieve(int32_t DeviceId, void *HstPtr,
                                          void *TgtPtr, int64_t Size) {
  return remoteCall(
      /* Preprocessor */
      [&](auto &RPCStatus, auto &Context) {
        auto *Request =
            protobuf::Arena::CreateMessage<RetrieveData>(Arena.get());

        Request->set_device_id(DeviceId);
        Request->set_size(Size);
        Request->set_hst_ptr((int64_t)HstPtr);
        Request->set_tgt_ptr((int64_t)TgtPtr);

        auto *Reply = protobuf::Arena::CreateMessage<Data>(Arena.get());
        std::unique_ptr<ClientReader<Data>> Reader(
            Stub->DataRetrieve(&Context, *Request));
        Reader->WaitForInitialMetadata();
        while (Reader->Read(Reply)) {
          if (Reply->ret()) {
            CLIENT_DBG("Could not async retrieve %ld bytes on device %d at %p "
                       "for %p",
                       Size, DeviceId, TgtPtr, HstPtr)
            return Reply;
          }

          if (Reply->start() == 0 && Reply->size() == Reply->data().size()) {
            memcpy(HstPtr, Reply->data().data(), Reply->data().size());

            return Reply;
          }

          memcpy((void *)((char *)HstPtr + Reply->start()),
                 Reply->data().data(), Reply->data().size());
        }
        RPCStatus = Reader->Finish();

        return Reply;
      },
      /* Postprocessor */
      [&](auto &Reply) {
        if (!Reply->ret()) {
          CLIENT_DBG("Retrieved %ld bytes on Device %d", Size, DeviceId)
        } else {
          CLIENT_DBG("Could not async retrieve %ld bytes on Device %d", Size,
                     DeviceId)
        }
        return Reply->ret();
      },
      /* Error Value */ -1,
      /* CanTimeOut */ false);
}

int32_t RemoteOffloadClient::dataExchange(int32_t SrcDevId, void *SrcPtr,
                                          int32_t DstDevId, void *DstPtr,
                                          int64_t Size) {
  return remoteCall(
      /* Preprocessor */
      [&](auto &RPCStatus, auto &Context) {
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());
        auto *Request =
            protobuf::Arena::CreateMessage<ExchangeData>(Arena.get());

        Request->set_src_dev_id(SrcDevId);
        Request->set_src_ptr((uint64_t)SrcPtr);
        Request->set_dst_dev_id(DstDevId);
        Request->set_dst_ptr((uint64_t)DstPtr);
        Request->set_size(Size);

        RPCStatus = Stub->DataExchange(&Context, *Request, Reply);
        return Reply;
      },
      /* Postprocessor */
      [&](auto &Reply) {
        if (Reply->number()) {
          CLIENT_DBG(
              "Exchanged %ld bytes on device %d at %p for %p on device %d",
              Size, SrcDevId, SrcPtr, DstPtr, DstDevId)
        } else {
          CLIENT_DBG("Could not exchange %ld bytes on device %d at %p for %p "
                     "on device %d",
                     Size, SrcDevId, SrcPtr, DstPtr, DstDevId)
        }
        return Reply->number();
      },
      /* Error Value */ -1);
}

int32_t RemoteOffloadClient::dataDelete(int32_t DeviceId, void *TgtPtr) {
  return remoteCall(
      /* Preprocessor */
      [&](auto &RPCStatus, auto &Context) {
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());
        auto *Request = protobuf::Arena::CreateMessage<DeleteData>(Arena.get());

        Request->set_device_id(DeviceId);
        Request->set_tgt_ptr((uint64_t)TgtPtr);

        RPCStatus = Stub->DataDelete(&Context, *Request, Reply);
        return Reply;
      },
      /* Postprocessor */
      [&](auto &Reply) {
        if (!Reply->number()) {
          CLIENT_DBG("Deleted data at %p on device %d", TgtPtr, DeviceId)
        } else {
          CLIENT_DBG("Could not delete data at %p on device %d", TgtPtr,
                     DeviceId)
        }
        return Reply->number();
      },
      /* Error Value */ -1);
}

int32_t RemoteOffloadClient::runTargetRegion(int32_t DeviceId,
                                             void *TgtEntryPtr, void **TgtArgs,
                                             ptrdiff_t *TgtOffsets,
                                             int32_t ArgNum) {
  return remoteCall(
      /* Preprocessor */
      [&](auto &RPCStatus, auto &Context) {
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());
        auto *Request =
            protobuf::Arena::CreateMessage<TargetRegion>(Arena.get());

        Request->set_device_id(DeviceId);

        Request->set_tgt_entry_ptr(
            (uint64_t)RemoteEntries[DeviceId][TgtEntryPtr]);

        char **ArgPtr = (char **)TgtArgs;
        for (auto I = 0; I < ArgNum; I++, ArgPtr++)
          Request->add_tgt_args((uint64_t)*ArgPtr);

        char *OffsetPtr = (char *)TgtOffsets;
        for (auto I = 0; I < ArgNum; I++, OffsetPtr++)
          Request->add_tgt_offsets((uint64_t)*OffsetPtr);

        Request->set_arg_num(ArgNum);

        RPCStatus = Stub->RunTargetRegion(&Context, *Request, Reply);
        return Reply;
      },
      /* Postprocessor */
      [&](auto &Reply) {
        if (!Reply->number()) {
          CLIENT_DBG("Ran target region async on device %d", DeviceId)
        } else {
          CLIENT_DBG("Could not run target region async on device %d", DeviceId)
        }
        return Reply->number();
      },
      /* Error Value */ -1,
      /* CanTimeOut */ false);
}

int32_t RemoteOffloadClient::runTargetTeamRegion(
    int32_t DeviceId, void *TgtEntryPtr, void **TgtArgs, ptrdiff_t *TgtOffsets,
    int32_t ArgNum, int32_t TeamNum, int32_t ThreadLimit,
    uint64_t LoopTripcount) {
  return remoteCall(
      /* Preprocessor */
      [&](auto &RPCStatus, auto &Context) {
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());
        auto *Request =
            protobuf::Arena::CreateMessage<TargetTeamRegion>(Arena.get());

        Request->set_device_id(DeviceId);

        Request->set_tgt_entry_ptr(
            (uint64_t)RemoteEntries[DeviceId][TgtEntryPtr]);

        char **ArgPtr = (char **)TgtArgs;
        for (auto I = 0; I < ArgNum; I++, ArgPtr++) {
          Request->add_tgt_args((uint64_t)*ArgPtr);
        }

        char *OffsetPtr = (char *)TgtOffsets;
        for (auto I = 0; I < ArgNum; I++, OffsetPtr++)
          Request->add_tgt_offsets((uint64_t)*OffsetPtr);

        Request->set_arg_num(ArgNum);
        Request->set_team_num(TeamNum);
        Request->set_thread_limit(ThreadLimit);
        Request->set_loop_tripcount(LoopTripcount);

        RPCStatus = Stub->RunTargetTeamRegion(&Context, *Request, Reply);
        return Reply;
      },
      /* Postprocessor */
      [&](auto &Reply) {
        if (!Reply->number()) {
          CLIENT_DBG("Ran target team region async on device %d", DeviceId)
        } else {
          CLIENT_DBG("Could not run target team region async on device %d",
                     DeviceId)
        }
        return Reply->number();
      },
      /* Error Value */ -1,
      /* CanTimeOut */ false);
}

int32_t RemoteClientManager::shutdown(void) {
  int32_t Ret = 0;
  for (auto &Client : Clients)
    Ret &= Client.shutdown();
  return Ret;
}

int32_t RemoteClientManager::registerLib(__tgt_bin_desc *Desc) {
  int32_t Ret = 0;
  for (auto &Client : Clients)
    Ret &= Client.registerLib(Desc);
  return Ret;
}

int32_t RemoteClientManager::unregisterLib(__tgt_bin_desc *Desc) {
  int32_t Ret = 0;
  for (auto &Client : Clients)
    Ret &= Client.unregisterLib(Desc);
  return Ret;
}

int32_t RemoteClientManager::isValidBinary(__tgt_device_image *Image) {
  int32_t ClientIdx = 0;
  for (auto &Client : Clients) {
    if (auto Ret = Client.isValidBinary(Image))
      return Ret;
    ClientIdx++;
  }
  return 0;
}

int32_t RemoteClientManager::getNumberOfDevices() {
  auto ClientIdx = 0;
  for (auto &Client : Clients) {
    if (auto NumDevices = Client.getNumberOfDevices()) {
      Devices.push_back(NumDevices);
    }
    ClientIdx++;
  }

  return std::accumulate(Devices.begin(), Devices.end(), 0);
}

std::pair<int32_t, int32_t> RemoteClientManager::mapDeviceId(int32_t DeviceId) {
  for (size_t ClientIdx = 0; ClientIdx < Devices.size(); ClientIdx++) {
    if (DeviceId < Devices[ClientIdx])
      return {ClientIdx, DeviceId};
    DeviceId -= Devices[ClientIdx];
  }
  return {-1, -1};
}

int32_t RemoteClientManager::initDevice(int32_t DeviceId) {
  int32_t ClientIdx, DeviceIdx;
  std::tie(ClientIdx, DeviceIdx) = mapDeviceId(DeviceId);
  return Clients[ClientIdx].initDevice(DeviceIdx);
}

int32_t RemoteClientManager::initRequires(int64_t RequiresFlags) {
  for (auto &Client : Clients)
    Client.initRequires(RequiresFlags);

  return RequiresFlags;
}

__tgt_target_table *RemoteClientManager::loadBinary(int32_t DeviceId,
                                                    __tgt_device_image *Image) {
  int32_t ClientIdx, DeviceIdx;
  std::tie(ClientIdx, DeviceIdx) = mapDeviceId(DeviceId);
  return Clients[ClientIdx].loadBinary(DeviceIdx, Image);
}

int32_t RemoteClientManager::isDataExchangeable(int32_t SrcDevId,
                                                int32_t DstDevId) {
  int32_t SrcClientIdx, SrcDeviceIdx, DstClientIdx, DstDeviceIdx;
  std::tie(SrcClientIdx, SrcDeviceIdx) = mapDeviceId(SrcDevId);
  std::tie(DstClientIdx, DstDeviceIdx) = mapDeviceId(DstDevId);
  return Clients[SrcClientIdx].isDataExchangeable(SrcDeviceIdx, DstDeviceIdx);
}

void *RemoteClientManager::dataAlloc(int32_t DeviceId, int64_t Size,
                                     void *HstPtr) {
  int32_t ClientIdx, DeviceIdx;
  std::tie(ClientIdx, DeviceIdx) = mapDeviceId(DeviceId);
  return Clients[ClientIdx].dataAlloc(DeviceIdx, Size, HstPtr);
}

int32_t RemoteClientManager::dataDelete(int32_t DeviceId, void *TgtPtr) {
  int32_t ClientIdx, DeviceIdx;
  std::tie(ClientIdx, DeviceIdx) = mapDeviceId(DeviceId);
  return Clients[ClientIdx].dataDelete(DeviceIdx, TgtPtr);
}

int32_t RemoteClientManager::dataSubmit(int32_t DeviceId, void *TgtPtr,
                                        void *HstPtr, int64_t Size) {
  int32_t ClientIdx, DeviceIdx;
  std::tie(ClientIdx, DeviceIdx) = mapDeviceId(DeviceId);
  return Clients[ClientIdx].dataSubmit(DeviceIdx, TgtPtr, HstPtr, Size);
}

int32_t RemoteClientManager::dataRetrieve(int32_t DeviceId, void *HstPtr,
                                          void *TgtPtr, int64_t Size) {
  int32_t ClientIdx, DeviceIdx;
  std::tie(ClientIdx, DeviceIdx) = mapDeviceId(DeviceId);
  return Clients[ClientIdx].dataRetrieve(DeviceIdx, HstPtr, TgtPtr, Size);
}

int32_t RemoteClientManager::dataExchange(int32_t SrcDevId, void *SrcPtr,
                                          int32_t DstDevId, void *DstPtr,
                                          int64_t Size) {
  int32_t SrcClientIdx, SrcDeviceIdx, DstClientIdx, DstDeviceIdx;
  std::tie(SrcClientIdx, SrcDeviceIdx) = mapDeviceId(SrcDevId);
  std::tie(DstClientIdx, DstDeviceIdx) = mapDeviceId(DstDevId);
  return Clients[SrcClientIdx].dataExchange(SrcDeviceIdx, SrcPtr, DstDeviceIdx,
                                            DstPtr, Size);
}

int32_t RemoteClientManager::runTargetRegion(int32_t DeviceId,
                                             void *TgtEntryPtr, void **TgtArgs,
                                             ptrdiff_t *TgtOffsets,
                                             int32_t ArgNum) {
  int32_t ClientIdx, DeviceIdx;
  std::tie(ClientIdx, DeviceIdx) = mapDeviceId(DeviceId);
  return Clients[ClientIdx].runTargetRegion(DeviceIdx, TgtEntryPtr, TgtArgs,
                                            TgtOffsets, ArgNum);
}

int32_t RemoteClientManager::runTargetTeamRegion(
    int32_t DeviceId, void *TgtEntryPtr, void **TgtArgs, ptrdiff_t *TgtOffsets,
    int32_t ArgNum, int32_t TeamNum, int32_t ThreadLimit,
    uint64_t LoopTripCount) {
  int32_t ClientIdx, DeviceIdx;
  std::tie(ClientIdx, DeviceIdx) = mapDeviceId(DeviceId);
  return Clients[ClientIdx].runTargetTeamRegion(DeviceIdx, TgtEntryPtr, TgtArgs,
                                                TgtOffsets, ArgNum, TeamNum,
                                                ThreadLimit, LoopTripCount);
}
