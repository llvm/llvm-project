// RUN: %libomptarget-compilexx-run-and-check-generic

// REQUIRES: libc
// REQUIRES: gpu

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

// CHECK: PASS

// This should be present in-tree relative to the test directory. If someone is
// using a partial tree just pass the test.
#if !__has_include(<../../libc/shared/rpc.h>)
int main() { printf("PASS\n"); }
#else
#include <../../libc/shared/rpc.h>

extern "C" void __tgt_register_rpc_callback(unsigned (*Callback)(void *,
                                                                 unsigned));
constexpr uint32_t RPC_TEST_OPCODE = 1;

template<uint32_t NumLanes> rpc::Status handleOpcodes(rpc::Server::Port &Port) {
  switch (Port.get_opcode()) {
  case RPC_TEST_OPCODE: {
    Port.recv(
        [&](rpc::Buffer *Buffer, uint32_t) { assert(Buffer->data[0] == 42); });
    Port.send([&](rpc::Buffer *, uint32_t) {});
    break;
  }
  default:
    return rpc::RPC_UNHANDLED_OPCODE;
    break;
  }
  return rpc::RPC_SUCCESS;
}

static uint32_t handleOffloadOpcodes(void *Raw, uint32_t NumLanes) {
  rpc::Server::Port &Port = *reinterpret_cast<rpc::Server::Port *>(Raw);
  if (NumLanes == 1)
    return handleOpcodes<1>(Port);
  else if (NumLanes == 32)
    return handleOpcodes<32>(Port);
  else if (NumLanes == 64)
    return handleOpcodes<64>(Port);
  else
    return rpc::RPC_ERROR;
}

[[gnu::weak]] rpc::Client client asm("__llvm_rpc_client");
#pragma omp declare target to(client) device_type(nohost)

void __tgt_register_rpc_callback(unsigned (*Callback)(void *, unsigned));

int main() {
  __tgt_register_rpc_callback(&handleOffloadOpcodes);
#pragma omp target
  {
    rpc::Client::Port Port = client.open<RPC_TEST_OPCODE>();
    Port.send([=](rpc::Buffer *buffer, uint32_t) { buffer->data[0] = 42; });
    Port.recv([](rpc::Buffer *, uint32_t) {});
    Port.close();
  }
  printf("PASS\n");
}
#endif
