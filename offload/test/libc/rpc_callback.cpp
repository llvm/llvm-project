// RUN: %libomptarget-compilexx-generic -fopenmp-cuda-mode
// RUN: %libomptarget-run-generic
// REQUIRES: libc
// REQUIRES: gpu

#include <assert.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// CHECK: PASS

// If the RPC headers are not present, just pass the test.
#if !__has_include(<../../libc/shared/rpc.h>)
int main() { printf("PASS\n"); }
#else

#include <../../libc/shared/rpc.h>
#include <../../libc/shared/rpc_dispatch.h>

[[gnu::weak]] rpc::Client client asm("__llvm_rpc_client");
#pragma omp declare target to(client) device_type(nohost)

//===------------------------------------------------------------------------===
// Opcodes.
//===------------------------------------------------------------------------===

constexpr uint32_t FOO_OPCODE = 1;
constexpr uint32_t VOID_OPCODE = 2;
constexpr uint32_t WRITEBACK_OPCODE = 3;
constexpr uint32_t CONST_PTR_OPCODE = 4;
constexpr uint32_t STRING_OPCODE = 5;
constexpr uint32_t EMPTY_OPCODE = 6;
constexpr uint32_t DIVERGENT_OPCODE = 7;
constexpr uint32_t ARRAY_SUM_OPCODE = 8;

//===------------------------------------------------------------------------===
// Server-side implementations.
//===------------------------------------------------------------------------===

struct S {
  int arr[4];
};

// 1. Non-pointer arguments, non-void return.
int foo(int x, double d, char c) {
  assert(x == 42);
  assert(d == 0.0);
  assert(c == 'c');
  return -1;
}

// 2. Void return type.
void void_fn(int x) { assert(x == 7); }

// 3. Write-back pointer.
void writeback_fn(int *out) {
  assert(out != nullptr && *out == 42);
  *out = 99;
}

// 4. Const pointer.
int sum_const(const S *p) {
  int s = 0;
  for (int i = 0; i < 4; ++i)
    s += p->arr[i];
  return s;
}

// 5. const char * string.
int c_string(const char *s) {
  assert(s != nullptr);
  assert(strcmp(s, "hello") == 0);
  return strlen(s);
}

// 6. Empty function.
int empty() { return 42; }

// 7. Divergent values.
void divergent(int *p) {
  assert(p);
  *p = *p;
}

// 8. Array argument via span.
int sum_array(const int *arr, int n) {
  int s = 0;
  for (int i = 0; i < n; ++i)
    s += arr[i];
  return s;
}

//===------------------------------------------------------------------------===
// RPC client dispatch.
//===------------------------------------------------------------------------===

#pragma omp begin declare variant match(device = {kind(gpu)})
int foo(int x, double d, char c) {
  return rpc::dispatch<FOO_OPCODE>(client, foo, x, d, c);
}

void void_fn(int x) { rpc::dispatch<VOID_OPCODE>(client, void_fn, x); }

void writeback_fn(int *out) {
  rpc::dispatch<WRITEBACK_OPCODE>(client, writeback_fn, out);
}

int sum_const(const S *p) {
  return rpc::dispatch<CONST_PTR_OPCODE>(client, sum_const, p);
}

int c_string(const char *s) {
  return rpc::dispatch<STRING_OPCODE>(client, c_string, s);
}

int empty() { return rpc::dispatch<EMPTY_OPCODE>(client, empty); }

void divergent(int *p) {
  rpc::dispatch<DIVERGENT_OPCODE>(client, divergent, p);
}

int sum_array(const int *arr, int n) {
  return rpc::dispatch<ARRAY_SUM_OPCODE>(
      client, sum_array, rpc::span<const int>{arr, uint64_t(n)}, n);
}
#pragma omp end declare variant

//===------------------------------------------------------------------------===
// RPC server dispatch.
//===------------------------------------------------------------------------===

template <uint32_t NUM_LANES>
rpc::Status handleOpcodesImpl(rpc::Server::Port &Port) {
  switch (Port.get_opcode()) {
  case FOO_OPCODE:
    rpc::invoke<NUM_LANES>(Port, foo);
    break;
  case VOID_OPCODE:
    rpc::invoke<NUM_LANES>(Port, void_fn);
    break;
  case WRITEBACK_OPCODE:
    rpc::invoke<NUM_LANES>(Port, writeback_fn);
    break;
  case CONST_PTR_OPCODE:
    rpc::invoke<NUM_LANES>(Port, sum_const);
    break;
  case STRING_OPCODE:
    rpc::invoke<NUM_LANES>(Port, c_string);
    break;
  case EMPTY_OPCODE:
    rpc::invoke<NUM_LANES>(Port, empty);
    break;
  case DIVERGENT_OPCODE:
    rpc::invoke<NUM_LANES>(Port, [](int *p) {
      assert(p);
      *p = *p;
    });
    break;
  case ARRAY_SUM_OPCODE:
    rpc::invoke<NUM_LANES>(Port, sum_array);
    break;
  default:
    return rpc::RPC_UNHANDLED_OPCODE;
  }
  return rpc::RPC_SUCCESS;
}

static uint32_t handleOpcodes(void *raw, uint32_t numLanes) {
  rpc::Server::Port &Port = *reinterpret_cast<rpc::Server::Port *>(raw);
  if (numLanes == 1)
    return handleOpcodesImpl<1>(Port);
  else if (numLanes == 32)
    return handleOpcodesImpl<32>(Port);
  else if (numLanes == 64)
    return handleOpcodesImpl<64>(Port);
  else
    return rpc::RPC_ERROR;
}

extern "C" void __tgt_register_rpc_callback(unsigned (*callback)(void *,
                                                                 unsigned));

[[gnu::constructor]] void register_callback() {
  __tgt_register_rpc_callback(&handleOpcodes);
}

int main() {

#pragma omp target
#pragma omp parallel num_threads(32)
  {
    // 1. Non-pointer return.
    assert(foo(42, 0.0, 'c') == -1);

    // 2. Void return.
    void_fn(7);

    // 3. Write-back pointer.
    int value = 42;
    writeback_fn(&value);
    assert(value == 99);

    // 4. Const pointer.
    S s{1, 2, 3, 4};
    int sum = sum_const(&s);
    assert(sum == 10);

    // 5. const char * string.
    const char *msg = "hello";
    int len = c_string(msg);
    assert(len == 5);

    // 6. No arguments.
    int ret = empty();
    assert(ret == 42);

    // 7. Divergent values.
    int id = omp_get_thread_num();
    if (id % 2)
      divergent(&id);
    assert(id == omp_get_thread_num());

    // 8. Array sum via span.
    int arr[4] = {1, 2, 3, 4};
    int total = sum_array(arr, 4);
    assert(total == 10);
  }

  printf("PASS\n");
}

#endif
