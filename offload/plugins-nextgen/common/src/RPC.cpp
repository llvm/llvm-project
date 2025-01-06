//===- RPC.h - Interface for remote procedure calls from the GPU ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RPC.h"

#include "Shared/Debug.h"
#include "Shared/RPCOpcodes.h"

#include "PluginInterface.h"

#include "shared/rpc.h"
#include "shared/rpc_opcodes.h"

using namespace llvm;
using namespace omp;
using namespace target;

template <uint32_t NumLanes>
rpc::Status handle_offload_opcodes(plugin::GenericDeviceTy &Device,
                                   rpc::Server::Port &Port) {

  switch (Port.get_opcode()) {
  case LIBC_MALLOC: {
    Port.recv_and_send([&](rpc::Buffer *Buffer, uint32_t) {
      Buffer->data[0] = reinterpret_cast<uintptr_t>(Device.allocate(
          Buffer->data[0], nullptr, TARGET_ALLOC_DEVICE_NON_BLOCKING));
    });
    break;
  }
  case LIBC_FREE: {
    Port.recv([&](rpc::Buffer *Buffer, uint32_t) {
      Device.free(reinterpret_cast<void *>(Buffer->data[0]),
                  TARGET_ALLOC_DEVICE_NON_BLOCKING);
    });
    break;
  }
  case OFFLOAD_HOST_CALL: {
    uint64_t Sizes[NumLanes] = {0};
    unsigned long long Results[NumLanes] = {0};
    void *Args[NumLanes] = {nullptr};
    Port.recv_n(Args, Sizes, [&](uint64_t Size) { return new char[Size]; });
    Port.recv([&](rpc::Buffer *buffer, uint32_t ID) {
      using FuncPtrTy = unsigned long long (*)(void *);
      auto Func = reinterpret_cast<FuncPtrTy>(buffer->data[0]);
      Results[ID] = Func(Args[ID]);
    });
    Port.send([&](rpc::Buffer *Buffer, uint32_t ID) {
      Buffer->data[0] = static_cast<uint64_t>(Results[ID]);
      delete[] reinterpret_cast<char *>(Args[ID]);
    });
    break;
  }
  default:
    return rpc::RPC_UNHANDLED_OPCODE;
    break;
  }
  return rpc::RPC_SUCCESS;
}

static rpc::Status handle_offload_opcodes(plugin::GenericDeviceTy &Device,
                                          rpc::Server::Port &Port,
                                          uint32_t NumLanes) {
  if (NumLanes == 1)
    return handle_offload_opcodes<1>(Device, Port);
  else if (NumLanes == 32)
    return handle_offload_opcodes<32>(Device, Port);
  else if (NumLanes == 64)
    return handle_offload_opcodes<64>(Device, Port);
  else
    return rpc::RPC_ERROR;
}

RPCServerTy::RPCServerTy(plugin::GenericPluginTy &Plugin)
    : Buffers(Plugin.getNumDevices()) {}

llvm::Expected<bool>
RPCServerTy::isDeviceUsingRPC(plugin::GenericDeviceTy &Device,
                              plugin::GenericGlobalHandlerTy &Handler,
                              plugin::DeviceImageTy &Image) {
  return Handler.isSymbolInImage(Device, Image, "__llvm_rpc_client");
}

Error RPCServerTy::initDevice(plugin::GenericDeviceTy &Device,
                              plugin::GenericGlobalHandlerTy &Handler,
                              plugin::DeviceImageTy &Image) {
  uint64_t NumPorts =
      std::min(Device.requestedRPCPortCount(), rpc::MAX_PORT_COUNT);
  void *RPCBuffer = Device.allocate(
      rpc::Server::allocation_size(Device.getWarpSize(), NumPorts), nullptr,
      TARGET_ALLOC_HOST);
  if (!RPCBuffer)
    return plugin::Plugin::error(
        "Failed to initialize RPC server for device %d", Device.getDeviceId());

  // Get the address of the RPC client from the device.
  plugin::GlobalTy ClientGlobal("__llvm_rpc_client", sizeof(rpc::Client));
  if (auto Err =
          Handler.getGlobalMetadataFromDevice(Device, Image, ClientGlobal))
    return Err;

  rpc::Client client(NumPorts, RPCBuffer);
  if (auto Err = Device.dataSubmit(ClientGlobal.getPtr(), &client,
                                   sizeof(rpc::Client), nullptr))
    return Err;
  Buffers[Device.getDeviceId()] = RPCBuffer;

  return Error::success();
}

Error RPCServerTy::runServer(plugin::GenericDeviceTy &Device) {
  uint64_t NumPorts =
      std::min(Device.requestedRPCPortCount(), rpc::MAX_PORT_COUNT);
  rpc::Server Server(NumPorts, Buffers[Device.getDeviceId()]);

  auto Port = Server.try_open(Device.getWarpSize());
  if (!Port)
    return Error::success();

  int Status = handle_offload_opcodes(Device, *Port, Device.getWarpSize());

  // Let the `libc` library handle any other unhandled opcodes.
#ifdef LIBOMPTARGET_RPC_SUPPORT
  if (Status == rpc::RPC_UNHANDLED_OPCODE)
    Status = handle_libc_opcodes(*Port, Device.getWarpSize());
#endif

  Port->close();
  if (Status != rpc::RPC_SUCCESS)
    return createStringError("RPC server given invalid opcode!");

  return Error::success();
}

Error RPCServerTy::deinitDevice(plugin::GenericDeviceTy &Device) {
  Device.free(Buffers[Device.getDeviceId()], TARGET_ALLOC_HOST);
  return Error::success();
}
