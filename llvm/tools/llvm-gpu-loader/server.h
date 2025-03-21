//===-- Common RPC server handler -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_GPU_LOADER_SERVER_H
#define LLVM_TOOLS_LLVM_GPU_LOADER_SERVER_H

#include <cstddef>
#include <cstdint>

#include "include/llvm-libc-types/test_rpc_opcodes_t.h"

#include "shared/rpc.h"
#include "shared/rpc_opcodes.h"
#include "shared/rpc_server.h"

template <uint32_t num_lanes, typename Alloc, typename Free>
inline uint32_t handle_server(rpc::Server &server, uint32_t index,
                              Alloc &&alloc, Free &&free) {
  auto port = server.try_open(num_lanes, index);
  if (!port)
    return 0;
  index = port->get_index() + 1;

  int status = rpc::RPC_SUCCESS;
  switch (port->get_opcode()) {
  case RPC_TEST_INCREMENT: {
    port->recv_and_send([](rpc::Buffer *buffer, uint32_t) {
      reinterpret_cast<uint64_t *>(buffer->data)[0] += 1;
    });
    break;
  }
  case RPC_TEST_INTERFACE: {
    bool end_with_recv;
    uint64_t cnt;
    port->recv([&](rpc::Buffer *buffer, uint32_t) {
      end_with_recv = buffer->data[0];
    });
    port->recv([&](rpc::Buffer *buffer, uint32_t) { cnt = buffer->data[0]; });
    port->send([&](rpc::Buffer *buffer, uint32_t) {
      buffer->data[0] = cnt = cnt + 1;
    });
    port->recv([&](rpc::Buffer *buffer, uint32_t) { cnt = buffer->data[0]; });
    port->send([&](rpc::Buffer *buffer, uint32_t) {
      buffer->data[0] = cnt = cnt + 1;
    });
    port->recv([&](rpc::Buffer *buffer, uint32_t) { cnt = buffer->data[0]; });
    port->recv([&](rpc::Buffer *buffer, uint32_t) { cnt = buffer->data[0]; });
    port->send([&](rpc::Buffer *buffer, uint32_t) {
      buffer->data[0] = cnt = cnt + 1;
    });
    port->send([&](rpc::Buffer *buffer, uint32_t) {
      buffer->data[0] = cnt = cnt + 1;
    });
    if (end_with_recv)
      port->recv([&](rpc::Buffer *buffer, uint32_t) { cnt = buffer->data[0]; });
    else
      port->send([&](rpc::Buffer *buffer, uint32_t) {
        buffer->data[0] = cnt = cnt + 1;
      });

    break;
  }
  case RPC_TEST_STREAM: {
    uint64_t sizes[num_lanes] = {0};
    void *dst[num_lanes] = {nullptr};
    port->recv_n(dst, sizes,
                 [](uint64_t size) -> void * { return new char[size]; });
    port->send_n(dst, sizes);
    for (uint64_t i = 0; i < num_lanes; ++i) {
      if (dst[i])
        delete[] reinterpret_cast<uint8_t *>(dst[i]);
    }
    break;
  }
  case RPC_TEST_NOOP: {
    port->recv([&](rpc::Buffer *, uint32_t) {});
    break;
  }
  case LIBC_MALLOC: {
    port->recv_and_send([&](rpc::Buffer *buffer, uint32_t) {
      buffer->data[0] = reinterpret_cast<uintptr_t>(alloc(buffer->data[0]));
    });
    break;
  }
  case LIBC_FREE: {
    port->recv([&](rpc::Buffer *buffer, uint32_t) {
      free(reinterpret_cast<void *>(buffer->data[0]));
    });
    break;
  }
  default:
    status = LIBC_NAMESPACE::shared::handle_libc_opcodes(*port, num_lanes);
    break;
  }

  // Handle all of the `libc` specific opcodes.
  if (status != rpc::RPC_SUCCESS)
    handle_error("Error handling RPC server");

  port->close();

  return index;
}

#endif // LLVM_TOOLS_LLVM_GPU_LOADER_SERVER_H
