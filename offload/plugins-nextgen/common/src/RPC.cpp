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
#include "shared/rpc_server.h"

using namespace llvm;
using namespace omp;
using namespace target;

template <uint32_t NumLanes>
rpc::Status handleOffloadOpcodes(plugin::GenericDeviceTy &Device,
                                 rpc::Server::Port &Port) {

  switch (Port.get_opcode()) {
  case LIBC_MALLOC: {
    Port.recv_and_send([&](rpc::Buffer *Buffer, uint32_t) {
      auto PtrOrErr =
          Device.allocate(Buffer->data[0], nullptr, TARGET_ALLOC_DEVICE);
      void *Ptr = nullptr;
      if (!PtrOrErr)
        llvm::consumeError(PtrOrErr.takeError());
      else
        Ptr = *PtrOrErr;
      Buffer->data[0] = reinterpret_cast<uintptr_t>(Ptr);
    });
    break;
  }
  case LIBC_FREE: {
    Port.recv([&](rpc::Buffer *Buffer, uint32_t) {
      if (auto Err = Device.free(reinterpret_cast<void *>(Buffer->data[0]),
                                 TARGET_ALLOC_DEVICE))
        llvm::consumeError(std::move(Err));
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

static rpc::Status handleOffloadOpcodes(plugin::GenericDeviceTy &Device,
                                        rpc::Server::Port &Port,
                                        uint32_t NumLanes) {
  if (NumLanes == 1)
    return handleOffloadOpcodes<1>(Device, Port);
  else if (NumLanes == 32)
    return handleOffloadOpcodes<32>(Device, Port);
  else if (NumLanes == 64)
    return handleOffloadOpcodes<64>(Device, Port);
  else
    return rpc::RPC_ERROR;
}

static rpc::Status runServer(plugin::GenericDeviceTy &Device, void *Buffer,
                             bool &ClientInUse) {
  uint64_t NumPorts =
      std::min(Device.requestedRPCPortCount(), rpc::MAX_PORT_COUNT);
  rpc::Server Server(NumPorts, Buffer);

  auto Port = Server.try_open(Device.getWarpSize());
  if (!Port)
    return rpc::RPC_SUCCESS;

  ClientInUse = true;
  rpc::Status Status =
      handleOffloadOpcodes(Device, *Port, Device.getWarpSize());

  // Let the `libc` library handle any other unhandled opcodes.
  if (Status == rpc::RPC_UNHANDLED_OPCODE)
    Status = LIBC_NAMESPACE::shared::handle_libc_opcodes(*Port,
                                                         Device.getWarpSize());
  Port->close();

  return Status;
}

void RPCServerTy::ServerThread::startThread() {
  if (!Running.fetch_or(true, std::memory_order_acquire))
    Worker = std::thread([this]() { run(); });
}

void RPCServerTy::ServerThread::shutDown() {
  if (!Running.fetch_and(false, std::memory_order_release))
    return;
  {
    std::lock_guard<decltype(Mutex)> Lock(Mutex);
    CV.notify_all();
  }
  if (Worker.joinable())
    Worker.join();
}

void RPCServerTy::ServerThread::run() {
  static constexpr auto IdleTime = std::chrono::microseconds(25);
  static constexpr auto IdleSleep = std::chrono::microseconds(250);
  std::unique_lock<decltype(Mutex)> Lock(Mutex);

  auto LastUse = std::chrono::steady_clock::now();
  for (;;) {
    CV.wait(Lock, [&]() {
      return NumUsers.load(std::memory_order_acquire) > 0 ||
             !Running.load(std::memory_order_acquire);
    });

    if (!Running.load(std::memory_order_acquire))
      return;

    Lock.unlock();
    bool ClientInUse = false;
    while (NumUsers.load(std::memory_order_relaxed) > 0 &&
           Running.load(std::memory_order_relaxed)) {

      // Suspend this thread briefly if there is no current work.
      auto Now = std::chrono::steady_clock::now();
      if (!ClientInUse && Now - LastUse >= IdleTime)
        std::this_thread::sleep_for(IdleSleep);
      else if (ClientInUse)
        LastUse = Now;

      ClientInUse = false;
      std::lock_guard<decltype(Mutex)> Lock(BufferMutex);
      for (const auto &[Buffer, Device] : llvm::zip_equal(Buffers, Devices)) {
        if (!Buffer || !Device)
          continue;

        // If running the server failed, print a message but keep running.
        if (runServer(*Device, Buffer, ClientInUse) != rpc::RPC_SUCCESS)
          FAILURE_MESSAGE("Unhandled or invalid RPC opcode!");
      }
    }
    Lock.lock();
  }
}

RPCServerTy::RPCServerTy(plugin::GenericPluginTy &Plugin)
    : Buffers(std::make_unique<void *[]>(Plugin.getNumDevices())),
      Devices(std::make_unique<plugin::GenericDeviceTy *[]>(
          Plugin.getNumDevices())),
      Thread(new ServerThread(Buffers.get(), Devices.get(),
                              Plugin.getNumDevices(), BufferMutex)) {}

llvm::Error RPCServerTy::startThread() {
  Thread->startThread();
  return Error::success();
}

llvm::Error RPCServerTy::shutDown() {
  Thread->shutDown();
  return Error::success();
}

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
  auto RPCBufferOrErr = Device.allocate(
      rpc::Server::allocation_size(Device.getWarpSize(), NumPorts), nullptr,
      TARGET_ALLOC_HOST);
  if (!RPCBufferOrErr)
    return RPCBufferOrErr.takeError();

  void *RPCBuffer = *RPCBufferOrErr;
  if (!RPCBuffer)
    return plugin::Plugin::error(
        error::ErrorCode::UNKNOWN,
        "failed to initialize RPC server for device %d", Device.getDeviceId());

  // Get the address of the RPC client from the device.
  plugin::GlobalTy ClientGlobal("__llvm_rpc_client", sizeof(rpc::Client));
  if (auto Err =
          Handler.getGlobalMetadataFromDevice(Device, Image, ClientGlobal))
    return Err;

  rpc::Client client(NumPorts, RPCBuffer);
  if (auto Err = Device.dataSubmit(ClientGlobal.getPtr(), &client,
                                   sizeof(rpc::Client), nullptr))
    return Err;
  std::lock_guard<decltype(BufferMutex)> Lock(BufferMutex);
  Buffers[Device.getDeviceId()] = RPCBuffer;
  Devices[Device.getDeviceId()] = &Device;

  return Error::success();
}

Error RPCServerTy::deinitDevice(plugin::GenericDeviceTy &Device) {
  std::lock_guard<decltype(BufferMutex)> Lock(BufferMutex);
  if (auto Err = Device.free(Buffers[Device.getDeviceId()], TARGET_ALLOC_HOST))
    return Err;
  Buffers[Device.getDeviceId()] = nullptr;
  Devices[Device.getDeviceId()] = nullptr;
  return Error::success();
}
