//===- RPC.h - Interface for remote procedure calls from the GPU ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RPC.h"

#include "Shared/Debug.h"

#include "PluginInterface.h"

// This header file may be present in-tree or from an LLVM installation. The
// installed version lives alongside the GPU headers so we do not want to
// include it directly.
#if __has_include(<gpu-none-llvm/rpc_server.h>)
#include <gpu-none-llvm/rpc_server.h>
#elif defined(LIBOMPTARGET_RPC_SUPPORT)
#include <rpc_server.h>
#endif

using namespace llvm;
using namespace omp;
using namespace target;

RPCServerTy::RPCServerTy(uint32_t NumDevices) {
#ifdef LIBOMPTARGET_RPC_SUPPORT
  // If this fails then something is catastrophically wrong, just exit.
  if (rpc_status_t Err = rpc_init(NumDevices))
    FATAL_MESSAGE(1, "Error initializing the RPC server: %d\n", Err);
#endif
}

llvm::Expected<bool>
RPCServerTy::isDeviceUsingRPC(plugin::GenericDeviceTy &Device,
                              plugin::GenericGlobalHandlerTy &Handler,
                              plugin::DeviceImageTy &Image) {
#ifdef LIBOMPTARGET_RPC_SUPPORT
  return Handler.isSymbolInImage(Device, Image, rpc_client_symbol_name);
#else
  return false;
#endif
}

Error RPCServerTy::initDevice(plugin::GenericDeviceTy &Device,
                              plugin::GenericGlobalHandlerTy &Handler,
                              plugin::DeviceImageTy &Image) {
#ifdef LIBOMPTARGET_RPC_SUPPORT
  uint32_t DeviceId = Device.getDeviceId();
  auto Alloc = [](uint64_t Size, void *Data) {
    plugin::GenericDeviceTy &Device =
        *reinterpret_cast<plugin::GenericDeviceTy *>(Data);
    return Device.allocate(Size, nullptr, TARGET_ALLOC_HOST);
  };
  uint64_t NumPorts =
      std::min(Device.requestedRPCPortCount(), RPC_MAXIMUM_PORT_COUNT);
  if (rpc_status_t Err = rpc_server_init(DeviceId, NumPorts,
                                         Device.getWarpSize(), Alloc, &Device))
    return plugin::Plugin::error(
        "Failed to initialize RPC server for device %d: %d", DeviceId, Err);

  // Register a custom opcode handler to perform plugin specific allocation.
  // FIXME: We need to make sure this uses asynchronous allocations on CUDA.
  auto MallocHandler = [](rpc_port_t Port, void *Data) {
    rpc_recv_and_send(
        Port,
        [](rpc_buffer_t *Buffer, void *Data) {
          plugin::GenericDeviceTy &Device =
              *reinterpret_cast<plugin::GenericDeviceTy *>(Data);
          Buffer->data[0] = reinterpret_cast<uintptr_t>(
              Device.allocate(Buffer->data[0], nullptr, TARGET_ALLOC_DEVICE));
        },
        Data);
  };
  if (rpc_status_t Err =
          rpc_register_callback(DeviceId, RPC_MALLOC, MallocHandler, &Device))
    return plugin::Plugin::error(
        "Failed to register RPC malloc handler for device %d: %d\n", DeviceId,
        Err);

  // Register a custom opcode handler to perform plugin specific deallocation.
  auto FreeHandler = [](rpc_port_t Port, void *Data) {
    rpc_recv(
        Port,
        [](rpc_buffer_t *Buffer, void *Data) {
          plugin::GenericDeviceTy &Device =
              *reinterpret_cast<plugin::GenericDeviceTy *>(Data);
          Device.free(reinterpret_cast<void *>(Buffer->data[0]),
                      TARGET_ALLOC_DEVICE);
        },
        Data);
  };
  if (rpc_status_t Err =
          rpc_register_callback(DeviceId, RPC_FREE, FreeHandler, &Device))
    return plugin::Plugin::error(
        "Failed to register RPC free handler for device %d: %d\n", DeviceId,
        Err);

  // Get the address of the RPC client from the device.
  void *ClientPtr;
  plugin::GlobalTy ClientGlobal(rpc_client_symbol_name, sizeof(void *));
  if (auto Err =
          Handler.getGlobalMetadataFromDevice(Device, Image, ClientGlobal))
    return Err;

  if (auto Err = Device.dataRetrieve(&ClientPtr, ClientGlobal.getPtr(),
                                     sizeof(void *), nullptr))
    return Err;

  const void *ClientBuffer = rpc_get_client_buffer(DeviceId);
  if (auto Err = Device.dataSubmit(ClientPtr, ClientBuffer,
                                   rpc_get_client_size(), nullptr))
    return Err;
#endif
  return Error::success();
}

Error RPCServerTy::runServer(plugin::GenericDeviceTy &Device) {
#ifdef LIBOMPTARGET_RPC_SUPPORT
  if (rpc_status_t Err = rpc_handle_server(Device.getDeviceId()))
    return plugin::Plugin::error(
        "Error while running RPC server on device %d: %d", Device.getDeviceId(),
        Err);
#endif
  return Error::success();
}

Error RPCServerTy::deinitDevice(plugin::GenericDeviceTy &Device) {
#ifdef LIBOMPTARGET_RPC_SUPPORT
  auto Dealloc = [](void *Ptr, void *Data) {
    plugin::GenericDeviceTy &Device =
        *reinterpret_cast<plugin::GenericDeviceTy *>(Data);
    Device.free(Ptr, TARGET_ALLOC_HOST);
  };
  if (rpc_status_t Err =
          rpc_server_shutdown(Device.getDeviceId(), Dealloc, &Device))
    return plugin::Plugin::error(
        "Failed to shut down RPC server for device %d: %d",
        Device.getDeviceId(), Err);
#endif
  return Error::success();
}

RPCServerTy::~RPCServerTy() {
#ifdef LIBOMPTARGET_RPC_SUPPORT
  rpc_shutdown();
#endif
}
