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
auto RemoteOffloadClient::remoteCall(Fn1 Preprocess, Fn2 Postprocess,
                                     TReturn ErrorValue, bool Timeout) {
  ArenaAllocatorLock->lock();
  if (Arena->SpaceAllocated() >= MaxSize)
    Arena->Reset();
  ArenaAllocatorLock->unlock();

  ClientContext Context;
  if (Timeout) {
    auto Deadline =
        std::chrono::system_clock::now() + std::chrono::seconds(Timeout);
    Context.set_deadline(Deadline);
  }

  Status RPCStatus;
  auto Reply = Preprocess(RPCStatus, Context);

  // TODO: Error handle more appropriately
  if (!RPCStatus.ok()) {
    CLIENT_DBG("%s", RPCStatus.error_message().c_str());
  } else {
    return Postprocess(Reply);
  }

  CLIENT_DBG("Failed");
  return ErrorValue;
}

int32_t RemoteOffloadClient::shutdown(void) {
  ClientContext Context;
  Null Request;
  I32 Reply;
  CLIENT_DBG("Shutting down server.");
  auto Status = Stub->Shutdown(&Context, Request, &Reply);
  if (Status.ok())
    return Reply.number();
  return 1;
}

int32_t RemoteOffloadClient::registerLib(__tgt_bin_desc *Desc) {
  return remoteCall(
      /* Preprocess */
      [&](auto &RPCStatus, auto &Context) {
        auto *Request = protobuf::Arena::CreateMessage<TargetBinaryDescription>(
            Arena.get());
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());
        loadTargetBinaryDescription(Desc, *Request);
        Request->set_bin_ptr((uint64_t)Desc);

        CLIENT_DBG("Registering library");
        RPCStatus = Stub->RegisterLib(&Context, *Request, Reply);
        return Reply;
      },
      /* Postprocess */
      [&](const auto &Reply) {
        if (Reply->number() == 0) {
          CLIENT_DBG("Registered library");
          return 0;
        }
        return 1;
      },
      /* Error Value */ 1);
}

int32_t RemoteOffloadClient::unregisterLib(__tgt_bin_desc *Desc) {
  return remoteCall(
      /* Preprocess */
      [&](auto &RPCStatus, auto &Context) {
        auto *Request = protobuf::Arena::CreateMessage<Pointer>(Arena.get());
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());

        Request->set_number((uint64_t)Desc);

        CLIENT_DBG("Unregistering library");
        RPCStatus = Stub->UnregisterLib(&Context, *Request, Reply);
        return Reply;
      },
      /* Postprocess */
      [&](const auto &Reply) {
        if (Reply->number() == 0) {
          CLIENT_DBG("Unregistered library");
          return 0;
        }
        CLIENT_DBG("Failed to unregister library");
        return 1;
      },
      /* Error Value */ 1);
}

int32_t RemoteOffloadClient::isValidBinary(__tgt_device_image *Image) {
  return remoteCall(
      /* Preprocess */
      [&](auto &RPCStatus, auto &Context) {
        auto *Request =
            protobuf::Arena::CreateMessage<TargetDeviceImagePtr>(Arena.get());
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());

        Request->set_image_ptr((uint64_t)Image->ImageStart);

        auto *EntryItr = Image->EntriesBegin;
        while (EntryItr != Image->EntriesEnd)
          Request->add_entry_ptrs((uint64_t)EntryItr++);

        CLIENT_DBG("Validating binary");
        RPCStatus = Stub->IsValidBinary(&Context, *Request, Reply);
        return Reply;
      },
      /* Postprocess */
      [&](const auto &Reply) {
        if (Reply->number()) {
          CLIENT_DBG("Validated binary");
        } else {
          CLIENT_DBG("Could not validate binary");
        }
        return Reply->number();
      },
      /* Error Value */ 0);
}

int32_t RemoteOffloadClient::getNumberOfDevices() {
  return remoteCall(
      /* Preprocess */
      [&](Status &RPCStatus, ClientContext &Context) {
        auto *Request = protobuf::Arena::CreateMessage<Null>(Arena.get());
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());

        CLIENT_DBG("Getting number of devices");
        RPCStatus = Stub->GetNumberOfDevices(&Context, *Request, Reply);

        return Reply;
      },
      /* Postprocess */
      [&](const auto &Reply) {
        if (Reply->number()) {
          CLIENT_DBG("Found %d devices", Reply->number());
        } else {
          CLIENT_DBG("Could not get the number of devices");
        }
        return Reply->number();
      },
      /*Error Value*/ -1);
}

int32_t RemoteOffloadClient::initDevice(int32_t DeviceId) {
  return remoteCall(
      /* Preprocess */
      [&](auto &RPCStatus, auto &Context) {
        auto *Request = protobuf::Arena::CreateMessage<I32>(Arena.get());
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());

        Request->set_number(DeviceId);

        CLIENT_DBG("Initializing device %d", DeviceId);
        RPCStatus = Stub->InitDevice(&Context, *Request, Reply);

        return Reply;
      },
      /* Postprocess */
      [&](const auto &Reply) {
        if (!Reply->number()) {
          CLIENT_DBG("Initialized device %d", DeviceId);
        } else {
          CLIENT_DBG("Could not initialize device %d", DeviceId);
        }
        return Reply->number();
      },
      /* Error Value */ -1);
}

int32_t RemoteOffloadClient::initRequires(int64_t RequiresFlags) {
  return remoteCall(
      /* Preprocess */
      [&](auto &RPCStatus, auto &Context) {
        auto *Request = protobuf::Arena::CreateMessage<I64>(Arena.get());
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());
        Request->set_number(RequiresFlags);
        CLIENT_DBG("Initializing requires");
        RPCStatus = Stub->InitRequires(&Context, *Request, Reply);
        return Reply;
      },
      /* Postprocess */
      [&](const auto &Reply) {
        if (Reply->number()) {
          CLIENT_DBG("Initialized requires");
        } else {
          CLIENT_DBG("Could not initialize requires");
        }
        return Reply->number();
      },
      /* Error Value */ -1);
}

__tgt_target_table *RemoteOffloadClient::loadBinary(int32_t DeviceId,
                                                    __tgt_device_image *Image) {
  return remoteCall(
      /* Preprocess */
      [&](auto &RPCStatus, auto &Context) {
        auto *ImageMessage =
            protobuf::Arena::CreateMessage<Binary>(Arena.get());
        auto *Reply = protobuf::Arena::CreateMessage<TargetTable>(Arena.get());
        ImageMessage->set_image_ptr((uint64_t)Image->ImageStart);
        ImageMessage->set_device_id(DeviceId);

        CLIENT_DBG("Loading Image %p to device %d", Image, DeviceId);
        RPCStatus = Stub->LoadBinary(&Context, *ImageMessage, Reply);
        return Reply;
      },
      /* Postprocess */
      [&](auto &Reply) {
        if (Reply->entries_size() == 0) {
          CLIENT_DBG("Could not load image %p onto device %d", Image, DeviceId);
          return (__tgt_target_table *)nullptr;
        }
        DevicesToTables[DeviceId] = std::make_unique<__tgt_target_table>();
        unloadTargetTable(*Reply, DevicesToTables[DeviceId].get(),
                          RemoteEntries[DeviceId]);

        CLIENT_DBG("Loaded Image %p to device %d with %d entries", Image,
                   DeviceId, Reply->entries_size());

        return DevicesToTables[DeviceId].get();
      },
      /* Error Value */ (__tgt_target_table *)nullptr,
      /* Timeout */ false);
}

int64_t RemoteOffloadClient::synchronize(int32_t DeviceId,
                                         __tgt_async_info *AsyncInfoPtr) {
  return remoteCall(
      /* Preprocess */
      [&](auto &RPCStatus, auto &Context) {
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());
        auto *Info =
            protobuf::Arena::CreateMessage<SynchronizeDevice>(Arena.get());

        Info->set_device_id(DeviceId);
        Info->set_queue_ptr((uint64_t)AsyncInfoPtr);

        CLIENT_DBG("Synchronizing device %d", DeviceId);
        RPCStatus = Stub->Synchronize(&Context, *Info, Reply);
        return Reply;
      },
      /* Postprocess */
      [&](auto &Reply) {
        if (Reply->number()) {
          CLIENT_DBG("Synchronized device %d", DeviceId);
        } else {
          CLIENT_DBG("Could not synchronize device %d", DeviceId);
        }
        return Reply->number();
      },
      /* Error Value */ -1);
}

int32_t RemoteOffloadClient::isDataExchangeable(int32_t SrcDevId,
                                                int32_t DstDevId) {
  return remoteCall(
      /* Preprocess */
      [&](auto &RPCStatus, auto &Context) {
        auto *Request = protobuf::Arena::CreateMessage<DevicePair>(Arena.get());
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());

        Request->set_src_dev_id(SrcDevId);
        Request->set_dst_dev_id(DstDevId);

        CLIENT_DBG("Asking if data is exchangeable between %d, %d", SrcDevId,
                   DstDevId);
        RPCStatus = Stub->IsDataExchangeable(&Context, *Request, Reply);
        return Reply;
      },
      /* Postprocess */
      [&](auto &Reply) {
        if (Reply->number()) {
          CLIENT_DBG("Data is exchangeable between %d, %d", SrcDevId, DstDevId);
        } else {
          CLIENT_DBG("Data is not exchangeable between %d, %d", SrcDevId,
                     DstDevId);
        }
        return Reply->number();
      },
      /* Error Value */ -1);
}

void *RemoteOffloadClient::dataAlloc(int32_t DeviceId, int64_t Size,
                                     void *HstPtr) {
  return remoteCall(
      /* Preprocess */
      [&](auto &RPCStatus, auto &Context) {
        auto *Reply = protobuf::Arena::CreateMessage<Pointer>(Arena.get());
        auto *Request = protobuf::Arena::CreateMessage<AllocData>(Arena.get());

        Request->set_device_id(DeviceId);
        Request->set_size(Size);
        Request->set_hst_ptr((uint64_t)HstPtr);

        CLIENT_DBG("Allocating %ld bytes on device %d", Size, DeviceId);
        RPCStatus = Stub->DataAlloc(&Context, *Request, Reply);
        return Reply;
      },
      /* Postprocess */
      [&](auto &Reply) {
        if (Reply->number()) {
          CLIENT_DBG("Allocated %ld bytes on device %d at %p", Size, DeviceId,
                     (void *)Reply->number());
        } else {
          CLIENT_DBG("Could not allocate %ld bytes on device %d at %p", Size,
                     DeviceId, (void *)Reply->number());
        }
        return (void *)Reply->number();
      },
      /* Error Value */ (void *)nullptr);
}

int32_t RemoteOffloadClient::dataSubmitAsync(int32_t DeviceId, void *TgtPtr,
                                             void *HstPtr, int64_t Size,
                                             __tgt_async_info *AsyncInfoPtr) {

  return remoteCall(
      /* Preprocess */
      [&](auto &RPCStatus, auto &Context) {
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());
        std::unique_ptr<ClientWriter<SubmitDataAsync>> Writer(
            Stub->DataSubmitAsync(&Context, Reply));

        if (Size > BlockSize) {
          int64_t Start = 0, End = BlockSize;
          for (auto I = 0; I < ceil((float)Size / BlockSize); I++) {
            auto *Request =
                protobuf::Arena::CreateMessage<SubmitDataAsync>(Arena.get());

            Request->set_device_id(DeviceId);
            Request->set_data((char *)HstPtr + Start, End - Start);
            Request->set_hst_ptr((uint64_t)HstPtr);
            Request->set_tgt_ptr((uint64_t)TgtPtr);
            Request->set_start(Start);
            Request->set_size(Size);
            Request->set_queue_ptr((uint64_t)AsyncInfoPtr);

            CLIENT_DBG("Submitting %ld-%ld/%ld bytes async on device %d at %p",
                       Start, End, Size, DeviceId, TgtPtr)

            if (!Writer->Write(*Request)) {
              CLIENT_DBG("Broken stream when submitting data");
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
              protobuf::Arena::CreateMessage<SubmitDataAsync>(Arena.get());

          Request->set_device_id(DeviceId);
          Request->set_data(HstPtr, Size);
          Request->set_hst_ptr((uint64_t)HstPtr);
          Request->set_tgt_ptr((uint64_t)TgtPtr);
          Request->set_start(0);
          Request->set_size(Size);

          CLIENT_DBG("Submitting %ld bytes async on device %d at %p", Size,
                     DeviceId, TgtPtr)
          if (!Writer->Write(*Request)) {
            CLIENT_DBG("Broken stream when submitting data");
            Reply->set_number(0);
            return Reply;
          }
        }

        Writer->WritesDone();
        RPCStatus = Writer->Finish();

        return Reply;
      },
      /* Postprocess */
      [&](auto &Reply) {
        if (!Reply->number()) {
          CLIENT_DBG("Async submitted %ld bytes on device %d at %p", Size,
                     DeviceId, TgtPtr)
        } else {
          CLIENT_DBG("Could not async submit %ld bytes on device %d at %p",
                     Size, DeviceId, TgtPtr)
        }
        return Reply->number();
      },
      /* Error Value */ -1,
      /* Timeout */ false);
}

int32_t RemoteOffloadClient::dataRetrieveAsync(int32_t DeviceId, void *HstPtr,
                                               void *TgtPtr, int64_t Size,
                                               __tgt_async_info *AsyncInfoPtr) {
  return remoteCall(
      /* Preprocess */
      [&](auto &RPCStatus, auto &Context) {
        auto *Request =
            protobuf::Arena::CreateMessage<RetrieveDataAsync>(Arena.get());

        Request->set_device_id(DeviceId);
        Request->set_size(Size);
        Request->set_hst_ptr((int64_t)HstPtr);
        Request->set_tgt_ptr((int64_t)TgtPtr);
        Request->set_queue_ptr((uint64_t)AsyncInfoPtr);

        auto *Reply = protobuf::Arena::CreateMessage<Data>(Arena.get());
        std::unique_ptr<ClientReader<Data>> Reader(
            Stub->DataRetrieveAsync(&Context, *Request));
        Reader->WaitForInitialMetadata();
        while (Reader->Read(Reply)) {
          if (Reply->ret()) {
            CLIENT_DBG("Could not async retrieve %ld bytes on device %d at %p "
                       "for %p",
                       Size, DeviceId, TgtPtr, HstPtr)
            return Reply;
          }

          if (Reply->start() == 0 && Reply->size() == Reply->data().size()) {
            CLIENT_DBG("Async retrieving %ld bytes on device %d at %p for %p",
                       Size, DeviceId, TgtPtr, HstPtr)

            memcpy(HstPtr, Reply->data().data(), Reply->data().size());

            return Reply;
          }
          CLIENT_DBG("Retrieving %lu-%lu/%lu bytes async from (%p) to (%p) "
                     "on Device %d",
                     Reply->start(), Reply->start() + Reply->data().size(),
                     Reply->size(), (void *)Request->tgt_ptr(), HstPtr,
                     Request->device_id());

          memcpy((void *)((char *)HstPtr + Reply->start()),
                 Reply->data().data(), Reply->data().size());
        }
        RPCStatus = Reader->Finish();

        return Reply;
      },
      /* Postprocess */
      [&](auto &Reply) {
        if (!Reply->ret()) {
          CLIENT_DBG("Async retrieve %ld bytes on Device %d", Size, DeviceId);
        } else {
          CLIENT_DBG("Could not async retrieve %ld bytes on Device %d", Size,
                     DeviceId);
        }
        return Reply->ret();
      },
      /* Error Value */ -1,
      /* Timeout */ false);
}

int32_t RemoteOffloadClient::dataExchangeAsync(int32_t SrcDevId, void *SrcPtr,
                                               int32_t DstDevId, void *DstPtr,
                                               int64_t Size,
                                               __tgt_async_info *AsyncInfoPtr) {
  return remoteCall(
      /* Preprocess */
      [&](auto &RPCStatus, auto &Context) {
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());
        auto *Request =
            protobuf::Arena::CreateMessage<ExchangeDataAsync>(Arena.get());

        Request->set_src_dev_id(SrcDevId);
        Request->set_src_ptr((uint64_t)SrcPtr);
        Request->set_dst_dev_id(DstDevId);
        Request->set_dst_ptr((uint64_t)DstPtr);
        Request->set_size(Size);
        Request->set_queue_ptr((uint64_t)AsyncInfoPtr);

        CLIENT_DBG(
            "Exchanging %ld bytes on device %d at %p for %p on device %d", Size,
            SrcDevId, SrcPtr, DstPtr, DstDevId);
        RPCStatus = Stub->DataExchangeAsync(&Context, *Request, Reply);
        return Reply;
      },
      /* Postprocess */
      [&](auto &Reply) {
        if (Reply->number()) {
          CLIENT_DBG(
              "Exchanged %ld bytes on device %d at %p for %p on device %d",
              Size, SrcDevId, SrcPtr, DstPtr, DstDevId);
        } else {
          CLIENT_DBG("Could not exchange %ld bytes on device %d at %p for %p "
                     "on device %d",
                     Size, SrcDevId, SrcPtr, DstPtr, DstDevId);
        }
        return Reply->number();
      },
      /* Error Value */ -1);
}

int32_t RemoteOffloadClient::dataDelete(int32_t DeviceId, void *TgtPtr) {
  return remoteCall(
      /* Preprocess */
      [&](auto &RPCStatus, auto &Context) {
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());
        auto *Request = protobuf::Arena::CreateMessage<DeleteData>(Arena.get());

        Request->set_device_id(DeviceId);
        Request->set_tgt_ptr((uint64_t)TgtPtr);

        CLIENT_DBG("Deleting data at %p on device %d", TgtPtr, DeviceId)
        RPCStatus = Stub->DataDelete(&Context, *Request, Reply);
        return Reply;
      },
      /* Postprocess */
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

int32_t RemoteOffloadClient::runTargetRegionAsync(
    int32_t DeviceId, void *TgtEntryPtr, void **TgtArgs, ptrdiff_t *TgtOffsets,
    int32_t ArgNum, __tgt_async_info *AsyncInfoPtr) {
  return remoteCall(
      /* Preprocess */
      [&](auto &RPCStatus, auto &Context) {
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());
        auto *Request =
            protobuf::Arena::CreateMessage<TargetRegionAsync>(Arena.get());

        Request->set_device_id(DeviceId);
        Request->set_queue_ptr((uint64_t)AsyncInfoPtr);

        Request->set_tgt_entry_ptr(
            (uint64_t)RemoteEntries[DeviceId][TgtEntryPtr]);

        char **ArgPtr = (char **)TgtArgs;
        for (auto I = 0; I < ArgNum; I++, ArgPtr++)
          Request->add_tgt_args((uint64_t)*ArgPtr);

        char *OffsetPtr = (char *)TgtOffsets;
        for (auto I = 0; I < ArgNum; I++, OffsetPtr++)
          Request->add_tgt_offsets((uint64_t)*OffsetPtr);

        Request->set_arg_num(ArgNum);

        CLIENT_DBG("Running target region async on device %d", DeviceId);
        RPCStatus = Stub->RunTargetRegionAsync(&Context, *Request, Reply);
        return Reply;
      },
      /* Postprocess */
      [&](auto &Reply) {
        if (!Reply->number()) {
          CLIENT_DBG("Ran target region async on device %d", DeviceId);
        } else {
          CLIENT_DBG("Could not run target region async on device %d",
                     DeviceId);
        }
        return Reply->number();
      },
      /* Error Value */ -1,
      /* Timeout */ false);
}

int32_t RemoteOffloadClient::runTargetTeamRegionAsync(
    int32_t DeviceId, void *TgtEntryPtr, void **TgtArgs, ptrdiff_t *TgtOffsets,
    int32_t ArgNum, int32_t TeamNum, int32_t ThreadLimit,
    uint64_t LoopTripcount, __tgt_async_info *AsyncInfoPtr) {
  return remoteCall(
      /* Preprocess */
      [&](auto &RPCStatus, auto &Context) {
        auto *Reply = protobuf::Arena::CreateMessage<I32>(Arena.get());
        auto *Request =
            protobuf::Arena::CreateMessage<TargetTeamRegionAsync>(Arena.get());

        Request->set_device_id(DeviceId);
        Request->set_queue_ptr((uint64_t)AsyncInfoPtr);

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

        CLIENT_DBG("Running target team region async on device %d", DeviceId);
        RPCStatus = Stub->RunTargetTeamRegionAsync(&Context, *Request, Reply);
        return Reply;
      },
      /* Postprocess */
      [&](auto &Reply) {
        if (!Reply->number()) {
          CLIENT_DBG("Ran target team region async on device %d", DeviceId);
        } else {
          CLIENT_DBG("Could not run target team region async on device %d",
                     DeviceId);
        }
        return Reply->number();
      },
      /* Error Value */ -1,
      /* Timeout */ false);
}

// TODO: Better error handling for the next three functions
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
    if (!(DeviceId >= Devices[ClientIdx]))
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

int64_t RemoteClientManager::synchronize(int32_t DeviceId,
                                         __tgt_async_info *AsyncInfoPtr) {
  int32_t ClientIdx, DeviceIdx;
  std::tie(ClientIdx, DeviceIdx) = mapDeviceId(DeviceId);
  return Clients[ClientIdx].synchronize(DeviceIdx, AsyncInfoPtr);
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

int32_t RemoteClientManager::dataSubmitAsync(int32_t DeviceId, void *TgtPtr,
                                             void *HstPtr, int64_t Size,
                                             __tgt_async_info *AsyncInfoPtr) {
  int32_t ClientIdx, DeviceIdx;
  std::tie(ClientIdx, DeviceIdx) = mapDeviceId(DeviceId);
  return Clients[ClientIdx].dataSubmitAsync(DeviceIdx, TgtPtr, HstPtr, Size,
                                            AsyncInfoPtr);
}

int32_t RemoteClientManager::dataRetrieveAsync(int32_t DeviceId, void *HstPtr,
                                               void *TgtPtr, int64_t Size,
                                               __tgt_async_info *AsyncInfoPtr) {
  int32_t ClientIdx, DeviceIdx;
  std::tie(ClientIdx, DeviceIdx) = mapDeviceId(DeviceId);
  return Clients[ClientIdx].dataRetrieveAsync(DeviceIdx, HstPtr, TgtPtr, Size,
                                              AsyncInfoPtr);
}

int32_t RemoteClientManager::dataExchangeAsync(int32_t SrcDevId, void *SrcPtr,
                                               int32_t DstDevId, void *DstPtr,
                                               int64_t Size,
                                               __tgt_async_info *AsyncInfoPtr) {
  int32_t SrcClientIdx, SrcDeviceIdx, DstClientIdx, DstDeviceIdx;
  std::tie(SrcClientIdx, SrcDeviceIdx) = mapDeviceId(SrcDevId);
  std::tie(DstClientIdx, DstDeviceIdx) = mapDeviceId(DstDevId);
  return Clients[SrcClientIdx].dataExchangeAsync(
      SrcDeviceIdx, SrcPtr, DstDeviceIdx, DstPtr, Size, AsyncInfoPtr);
}

int32_t RemoteClientManager::runTargetRegionAsync(
    int32_t DeviceId, void *TgtEntryPtr, void **TgtArgs, ptrdiff_t *TgtOffsets,
    int32_t ArgNum, __tgt_async_info *AsyncInfoPtr) {
  int32_t ClientIdx, DeviceIdx;
  std::tie(ClientIdx, DeviceIdx) = mapDeviceId(DeviceId);
  return Clients[ClientIdx].runTargetRegionAsync(
      DeviceIdx, TgtEntryPtr, TgtArgs, TgtOffsets, ArgNum, AsyncInfoPtr);
}

int32_t RemoteClientManager::runTargetTeamRegionAsync(
    int32_t DeviceId, void *TgtEntryPtr, void **TgtArgs, ptrdiff_t *TgtOffsets,
    int32_t ArgNum, int32_t TeamNum, int32_t ThreadLimit,
    uint64_t LoopTripCount, __tgt_async_info *AsyncInfoPtr) {
  int32_t ClientIdx, DeviceIdx;
  std::tie(ClientIdx, DeviceIdx) = mapDeviceId(DeviceId);
  return Clients[ClientIdx].runTargetTeamRegionAsync(
      DeviceIdx, TgtEntryPtr, TgtArgs, TgtOffsets, ArgNum, TeamNum, ThreadLimit,
      LoopTripCount, AsyncInfoPtr);
}
