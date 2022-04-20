//===----------------- Server.cpp - Server Implementation -----------------===//
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

#include <cmath>
#include <future>

#include "Server.h"
#include "omptarget.h"
#include "openmp.grpc.pb.h"
#include "openmp.pb.h"

using grpc::WriteOptions;

extern std::promise<void> ShutdownPromise;

Status RemoteOffloadImpl::Shutdown(ServerContext *Context, const Null *Request,
                                   I32 *Reply) {
  SERVER_DBG("Shutting down the server")

  Reply->set_number(0);
  ShutdownPromise.set_value();
  return Status::OK;
}

Status
RemoteOffloadImpl::RegisterLib(ServerContext *Context,
                               const TargetBinaryDescription *Description,
                               I32 *Reply) {
  auto Desc = std::make_unique<__tgt_bin_desc>();

  unloadTargetBinaryDescription(Description, Desc.get(),
                                HostToRemoteDeviceImage);
  PM->RTLs.RegisterLib(Desc.get());

  if (Descriptions.find((void *)Description->bin_ptr()) != Descriptions.end())
    freeTargetBinaryDescription(
        Descriptions[(void *)Description->bin_ptr()].get());
  else
    Descriptions[(void *)Description->bin_ptr()] = std::move(Desc);

  SERVER_DBG("Registered library")
  Reply->set_number(0);
  return Status::OK;
}

Status RemoteOffloadImpl::UnregisterLib(ServerContext *Context,
                                        const Pointer *Request, I32 *Reply) {
  if (Descriptions.find((void *)Request->number()) == Descriptions.end()) {
    Reply->set_number(1);
    return Status::OK;
  }

  PM->RTLs.UnregisterLib(Descriptions[(void *)Request->number()].get());
  freeTargetBinaryDescription(Descriptions[(void *)Request->number()].get());
  Descriptions.erase((void *)Request->number());

  SERVER_DBG("Unregistered library")
  Reply->set_number(0);
  return Status::OK;
}

Status RemoteOffloadImpl::IsValidBinary(ServerContext *Context,
                                        const TargetDeviceImagePtr *DeviceImage,
                                        I32 *IsValid) {
  __tgt_device_image *Image =
      HostToRemoteDeviceImage[(void *)DeviceImage->image_ptr()];

  IsValid->set_number(0);

  for (auto &RTL : PM->RTLs.AllRTLs)
    if (auto Ret = RTL.is_valid_binary(Image)) {
      IsValid->set_number(Ret);
      break;
    }

  SERVER_DBG("Checked if binary (%p) is valid",
             (void *)(DeviceImage->image_ptr()))
  return Status::OK;
}

Status RemoteOffloadImpl::GetNumberOfDevices(ServerContext *Context,
                                             const Null *Null,
                                             I32 *NumberOfDevices) {
  std::call_once(PM->RTLs.initFlag, &RTLsTy::LoadRTLs, &PM->RTLs);

  int32_t Devices = 0;
  PM->RTLsMtx.lock();
  for (auto &RTL : PM->RTLs.AllRTLs)
    Devices += RTL.NumberOfDevices;
  PM->RTLsMtx.unlock();

  NumberOfDevices->set_number(Devices);

  SERVER_DBG("Got number of devices")
  return Status::OK;
}

Status RemoteOffloadImpl::InitDevice(ServerContext *Context,
                                     const I32 *DeviceNum, I32 *Reply) {
  Reply->set_number(PM->Devices[DeviceNum->number()]->RTL->init_device(
      mapHostRTLDeviceId(DeviceNum->number())));

  SERVER_DBG("Initialized device %d", DeviceNum->number())
  return Status::OK;
}

Status RemoteOffloadImpl::InitRequires(ServerContext *Context,
                                       const I64 *RequiresFlag, I32 *Reply) {
  for (auto &Device : PM->Devices)
    if (Device->RTL->init_requires)
      Device->RTL->init_requires(RequiresFlag->number());
  Reply->set_number(RequiresFlag->number());

  SERVER_DBG("Initialized requires for devices")
  return Status::OK;
}

Status RemoteOffloadImpl::LoadBinary(ServerContext *Context,
                                     const Binary *Binary, TargetTable *Reply) {
  __tgt_device_image *Image =
      HostToRemoteDeviceImage[(void *)Binary->image_ptr()];

  Table = PM->Devices[Binary->device_id()]->RTL->load_binary(
      mapHostRTLDeviceId(Binary->device_id()), Image);
  if (Table)
    loadTargetTable(Table, *Reply, Image);

  SERVER_DBG("Loaded binary (%p) to device %d", (void *)Binary->image_ptr(),
             Binary->device_id())
  return Status::OK;
}

Status RemoteOffloadImpl::IsDataExchangeable(ServerContext *Context,
                                             const DevicePair *Request,
                                             I32 *Reply) {
  Reply->set_number(-1);
  if (PM->Devices[mapHostRTLDeviceId(Request->src_dev_id())]
          ->RTL->is_data_exchangable)
    Reply->set_number(PM->Devices[mapHostRTLDeviceId(Request->src_dev_id())]
                          ->RTL->is_data_exchangable(Request->src_dev_id(),
                                                     Request->dst_dev_id()));

  SERVER_DBG("Checked if data exchangeable between device %d and device %d",
             Request->src_dev_id(), Request->dst_dev_id())
  return Status::OK;
}

Status RemoteOffloadImpl::DataAlloc(ServerContext *Context,
                                    const AllocData *Request, Pointer *Reply) {
  uint64_t TgtPtr =
      (uint64_t)PM->Devices[Request->device_id()]->RTL->data_alloc(
          mapHostRTLDeviceId(Request->device_id()), Request->size(),
          (void *)Request->hst_ptr(), TARGET_ALLOC_DEFAULT);
  Reply->set_number(TgtPtr);

  SERVER_DBG("Allocated at " DPxMOD "", DPxPTR((void *)TgtPtr))

  return Status::OK;
}

Status RemoteOffloadImpl::DataSubmit(ServerContext *Context,
                                     ServerReader<SubmitData> *Reader,
                                     I32 *Reply) {
  SubmitData Request;
  uint8_t *HostCopy = nullptr;
  while (Reader->Read(&Request)) {
    if (Request.start() == 0 && Request.size() == Request.data().size()) {
      Reader->SendInitialMetadata();

      Reply->set_number(PM->Devices[Request.device_id()]->RTL->data_submit(
          mapHostRTLDeviceId(Request.device_id()), (void *)Request.tgt_ptr(),
          (void *)Request.data().data(), Request.data().size()));

      SERVER_DBG("Submitted %lu bytes async to (%p) on device %d",
                 Request.data().size(), (void *)Request.tgt_ptr(),
                 Request.device_id())

      return Status::OK;
    }
    if (!HostCopy) {
      HostCopy = new uint8_t[Request.size()];
      Reader->SendInitialMetadata();
    }

    memcpy((void *)((char *)HostCopy + Request.start()), Request.data().data(),
           Request.data().size());
  }

  Reply->set_number(PM->Devices[Request.device_id()]->RTL->data_submit(
      mapHostRTLDeviceId(Request.device_id()), (void *)Request.tgt_ptr(),
      HostCopy, Request.size()));

  delete[] HostCopy;

  SERVER_DBG("Submitted %lu bytes to (%p) on device %d", Request.data().size(),
             (void *)Request.tgt_ptr(), Request.device_id())

  return Status::OK;
}

Status RemoteOffloadImpl::DataRetrieve(ServerContext *Context,
                                       const RetrieveData *Request,
                                       ServerWriter<Data> *Writer) {
  auto HstPtr = std::make_unique<char[]>(Request->size());

  auto Ret = PM->Devices[Request->device_id()]->RTL->data_retrieve(
      mapHostRTLDeviceId(Request->device_id()), HstPtr.get(),
      (void *)Request->tgt_ptr(), Request->size());

  if (Arena->SpaceAllocated() >= MaxSize)
    Arena->Reset();

  if (Request->size() > BlockSize) {
    uint64_t Start = 0, End = BlockSize;
    for (auto I = 0; I < ceil((float)Request->size() / BlockSize); I++) {
      auto *Reply = protobuf::Arena::CreateMessage<Data>(Arena.get());

      Reply->set_start(Start);
      Reply->set_size(Request->size());
      Reply->set_data((char *)HstPtr.get() + Start, End - Start);
      Reply->set_ret(Ret);

      if (!Writer->Write(*Reply)) {
        CLIENT_DBG("Broken stream when submitting data")
      }

      SERVER_DBG("Retrieved %lu-%lu/%lu bytes from (%p) on device %d", Start,
                 End, Request->size(), (void *)Request->tgt_ptr(),
                 mapHostRTLDeviceId(Request->device_id()))

      Start += BlockSize;
      End += BlockSize;
      if (End >= Request->size())
        End = Request->size();
    }
  } else {
    auto *Reply = protobuf::Arena::CreateMessage<Data>(Arena.get());

    Reply->set_start(0);
    Reply->set_size(Request->size());
    Reply->set_data((char *)HstPtr.get(), Request->size());
    Reply->set_ret(Ret);

    SERVER_DBG("Retrieved %lu bytes from (%p) on device %d", Request->size(),
               (void *)Request->tgt_ptr(),
               mapHostRTLDeviceId(Request->device_id()))

    Writer->WriteLast(*Reply, WriteOptions());
  }

  return Status::OK;
}

Status RemoteOffloadImpl::DataExchange(ServerContext *Context,
                                       const ExchangeData *Request,
                                       I32 *Reply) {
  if (PM->Devices[Request->src_dev_id()]->RTL->data_exchange) {
    int32_t Ret = PM->Devices[Request->src_dev_id()]->RTL->data_exchange(
        mapHostRTLDeviceId(Request->src_dev_id()), (void *)Request->src_ptr(),
        mapHostRTLDeviceId(Request->dst_dev_id()), (void *)Request->dst_ptr(),
        Request->size());
    Reply->set_number(Ret);
  } else
    Reply->set_number(-1);

  SERVER_DBG(
      "Exchanged data asynchronously from device %d (%p) to device %d (%p) of "
      "size %lu",
      mapHostRTLDeviceId(Request->src_dev_id()), (void *)Request->src_ptr(),
      mapHostRTLDeviceId(Request->dst_dev_id()), (void *)Request->dst_ptr(),
      Request->size())
  return Status::OK;
}

Status RemoteOffloadImpl::DataDelete(ServerContext *Context,
                                     const DeleteData *Request, I32 *Reply) {
  auto Ret = PM->Devices[Request->device_id()]->RTL->data_delete(
      mapHostRTLDeviceId(Request->device_id()), (void *)Request->tgt_ptr());
  Reply->set_number(Ret);

  SERVER_DBG("Deleted data from (%p) on device %d", (void *)Request->tgt_ptr(),
             mapHostRTLDeviceId(Request->device_id()))
  return Status::OK;
}

Status RemoteOffloadImpl::RunTargetRegion(ServerContext *Context,
                                          const TargetRegion *Request,
                                          I32 *Reply) {
  std::vector<uint8_t> TgtArgs(Request->arg_num());
  for (auto I = 0; I < Request->arg_num(); I++)
    TgtArgs[I] = (uint64_t)Request->tgt_args()[I];

  std::vector<ptrdiff_t> TgtOffsets(Request->arg_num());
  const auto *TgtOffsetItr = Request->tgt_offsets().begin();
  for (auto I = 0; I < Request->arg_num(); I++, TgtOffsetItr++)
    TgtOffsets[I] = (ptrdiff_t)*TgtOffsetItr;

  void *TgtEntryPtr = ((__tgt_offload_entry *)Request->tgt_entry_ptr())->addr;

  int32_t Ret = PM->Devices[Request->device_id()]->RTL->run_region(
      mapHostRTLDeviceId(Request->device_id()), TgtEntryPtr,
      (void **)TgtArgs.data(), TgtOffsets.data(), Request->arg_num());

  Reply->set_number(Ret);

  SERVER_DBG("Ran TargetRegion on device %d with %d args",
             mapHostRTLDeviceId(Request->device_id()), Request->arg_num())
  return Status::OK;
}

Status RemoteOffloadImpl::RunTargetTeamRegion(ServerContext *Context,
                                              const TargetTeamRegion *Request,
                                              I32 *Reply) {
  std::vector<uint64_t> TgtArgs(Request->arg_num());
  for (auto I = 0; I < Request->arg_num(); I++)
    TgtArgs[I] = (uint64_t)Request->tgt_args()[I];

  std::vector<ptrdiff_t> TgtOffsets(Request->arg_num());
  const auto *TgtOffsetItr = Request->tgt_offsets().begin();
  for (auto I = 0; I < Request->arg_num(); I++, TgtOffsetItr++)
    TgtOffsets[I] = (ptrdiff_t)*TgtOffsetItr;

  void *TgtEntryPtr = ((__tgt_offload_entry *)Request->tgt_entry_ptr())->addr;

  int32_t Ret = PM->Devices[Request->device_id()]->RTL->run_team_region(
      mapHostRTLDeviceId(Request->device_id()), TgtEntryPtr,
      (void **)TgtArgs.data(), TgtOffsets.data(), Request->arg_num(),
      Request->team_num(), Request->thread_limit(), Request->loop_tripcount());

  Reply->set_number(Ret);

  SERVER_DBG("Ran TargetTeamRegion on device %d with %d args",
             mapHostRTLDeviceId(Request->device_id()), Request->arg_num())
  return Status::OK;
}

int32_t RemoteOffloadImpl::mapHostRTLDeviceId(int32_t RTLDeviceID) {
  for (auto &RTL : PM->RTLs.UsedRTLs) {
    if (RTLDeviceID - RTL->NumberOfDevices >= 0)
      RTLDeviceID -= RTL->NumberOfDevices;
    else
      break;
  }
  return RTLDeviceID;
}
