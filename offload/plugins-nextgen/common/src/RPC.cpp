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

#if defined(LIBOMPTARGET_RPC_SUPPORT)
#include "llvm-libc-types/rpc_opcodes_t.h"
#include "llvmlibc_rpc_server.h"
#endif

using namespace llvm;
using namespace omp;
using namespace target;

RPCServerTy::RPCServerTy(plugin::GenericPluginTy &Plugin)
    : Handles(Plugin.getNumDevices()) {}

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
  auto Alloc = [](uint64_t Size, void *Data) {
    plugin::GenericDeviceTy &Device =
        *reinterpret_cast<plugin::GenericDeviceTy *>(Data);
    return Device.allocate(Size, nullptr, TARGET_ALLOC_HOST);
  };
  uint64_t NumPorts =
      std::min(Device.requestedRPCPortCount(), RPC_MAXIMUM_PORT_COUNT);
  rpc_device_t RPCDevice;
  if (rpc_status_t Err = rpc_server_init(&RPCDevice, NumPorts,
                                         Device.getWarpSize(), Alloc, &Device))
    return plugin::Plugin::error(
        "Failed to initialize RPC server for device %d: %d",
        Device.getDeviceId(), Err);

  // Register a custom opcode handler to perform plugin specific allocation.
  auto MallocHandler = [](rpc_port_t Port, void *Data) {
    rpc_recv_and_send(
        Port,
        [](rpc_buffer_t *Buffer, void *Data) {
          plugin::GenericDeviceTy &Device =
              *reinterpret_cast<plugin::GenericDeviceTy *>(Data);
          Buffer->data[0] = reinterpret_cast<uintptr_t>(Device.allocate(
              Buffer->data[0], nullptr, TARGET_ALLOC_DEVICE_NON_BLOCKING));
        },
        Data);
  };
  if (rpc_status_t Err =
          rpc_register_callback(RPCDevice, RPC_MALLOC, MallocHandler, &Device))
    return plugin::Plugin::error(
        "Failed to register RPC malloc handler for device %d: %d\n",
        Device.getDeviceId(), Err);

  // Register a custom opcode handler to perform plugin specific deallocation.
  auto FreeHandler = [](rpc_port_t Port, void *Data) {
    rpc_recv(
        Port,
        [](rpc_buffer_t *Buffer, void *Data) {
          plugin::GenericDeviceTy &Device =
              *reinterpret_cast<plugin::GenericDeviceTy *>(Data);
          Device.free(reinterpret_cast<void *>(Buffer->data[0]),
                      TARGET_ALLOC_DEVICE_NON_BLOCKING);
        },
        Data);
  };
  if (rpc_status_t Err =
          rpc_register_callback(RPCDevice, RPC_FREE, FreeHandler, &Device))
    return plugin::Plugin::error(
        "Failed to register RPC free handler for device %d: %d\n",
        Device.getDeviceId(), Err);

  // Get the address of the RPC client from the device.
  void *ClientPtr;
  plugin::GlobalTy ClientGlobal(rpc_client_symbol_name, sizeof(void *));
  if (auto Err =
          Handler.getGlobalMetadataFromDevice(Device, Image, ClientGlobal))
    return Err;

  if (auto Err = Device.dataRetrieve(&ClientPtr, ClientGlobal.getPtr(),
                                     sizeof(void *), nullptr))
    return Err;

  const void *ClientBuffer = rpc_get_client_buffer(RPCDevice);
  if (auto Err = Device.dataSubmit(ClientPtr, ClientBuffer,
                                   rpc_get_client_size(), nullptr))
    return Err;
  Handles[Device.getDeviceId()] = RPCDevice.handle;
#endif
  return Error::success();
}

Error RPCServerTy::runServer(plugin::GenericDeviceTy &Device) {
#ifdef LIBOMPTARGET_RPC_SUPPORT
  rpc_device_t RPCDevice{Handles[Device.getDeviceId()]};
  if (rpc_status_t Err = rpc_handle_server(RPCDevice))
    return plugin::Plugin::error(
        "Error while running RPC server on device %d: %d", Device.getDeviceId(),
        Err);
#endif
  return Error::success();
}

Error RPCServerTy::deinitDevice(plugin::GenericDeviceTy &Device) {
#ifdef LIBOMPTARGET_RPC_SUPPORT
  rpc_device_t RPCDevice{Handles[Device.getDeviceId()]};
  auto Dealloc = [](void *Ptr, void *Data) {
    plugin::GenericDeviceTy &Device =
        *reinterpret_cast<plugin::GenericDeviceTy *>(Data);
    Device.free(Ptr, TARGET_ALLOC_HOST);
  };
  if (rpc_status_t Err = rpc_server_shutdown(RPCDevice, Dealloc, &Device))
    return plugin::Plugin::error(
        "Failed to shut down RPC server for device %d: %d",
        Device.getDeviceId(), Err);
#endif
  return Error::success();
}
