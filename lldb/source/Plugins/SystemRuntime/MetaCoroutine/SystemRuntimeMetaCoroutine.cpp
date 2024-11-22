//===-- SystemRuntimeMetaCoroutine.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Process/Utility/HistoryThread.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/ValueObject/ValueObject.h"
#include "lldb/lldb-private-enumerations.h"
#include "lldb/lldb-private.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "SystemRuntimeMetaCoroutine.h"

using namespace lldb;
using namespace lldb_private;

namespace {

// This file implements a SystemRuntimePlugin to provide stack trace support for
// Meta Coroutines. The stack trace is composed of two parts:
//  1. Normal stack frames
//  2. Async stack frames
//
// The normal stack frames are the stack frames that are normally walked by the
// lldb stack trace implementation. The async stack frames are the stack frames
// that are stored in the AsyncStackRoots linked-list.
//
// Below is a diagram of the stack frames:
//
//  Stack Register
//      |
//      V
//  Stack Frame   currentStackRoot (TLS)
//      |               |
//      V               V
//  Stack Frame <- AsyncStackRoot  -> AsyncStackFrame -> AsyncStackFrame -> ...
//      |               |
//      V               |
//  Stack Frame         |
//      :               |
//      V               V
//  Stack Frame <- AsyncStackRoot  -> AsyncStackFrame -> AsyncStackFrame -> ...
//      |               |
//      V               X
//  Stack Frame
//      :
//      V
//
// To retrive the coroutine stack trace, we start at the currentStackRoot
// (which is stored in the thread's TLS) and walk the linked-list of
// AsyncStackRoots until we reach the end of the normal stack. We then
// transition to the async stack and walk the linked-list of AsyncStackFrames
// until we reach the end of the async stack. This will result in a stack
// trace that looks like:
//  [reg_pc]
//  [normal stack frames]
//  [async stack frames]
//
// For more information on the data-structures used to represent the async
// stack, see folly/tracing/AsyncStack.h

static constexpr char kMetaCoroutineType[] = "Meta Coroutine Backtrace";
// Key used in pthread thread local storage to hold a pointer to
// AsyncStackRootHolder
static constexpr char kAsyncStackRootTlsKey[] =
    "folly_async_stack_root_tls_key";

// These are memory representations of the types folly uses for tracking async
// stack traces. See folly/tracing/AsyncStack.h
struct AsyncStackRootHolder {
  addr_t async_stack_root_addr;
};

struct AsyncStackRoot {
  addr_t top_frame;
  addr_t next_root;
  addr_t stack_frame_ptr;
  addr_t stack_root;
};

struct StackFrame {
  addr_t stack_frame;
  addr_t return_address;
};

struct AsyncStackFrame {
  addr_t parent_frame;
  addr_t instruction_pointer;
  addr_t stack_root;
};

struct WalkAsyncStackResult {
  // # Normal stack frame to start the next normal stack walk
  addr_t normal_stack_frame_addr;
  addr_t normal_stack_frame_stop_addr;
  // # Async stack frame to start the next async stack walk after the next
  // # normal stack walk
  addr_t async_stack_frame_addr;
};

bool GetPointerValue(const ThreadSP &real_thread, addr_t addr,
                     addr_t &output_value) {
  Address address;
  address.SetRawAddress(addr);
  Status error;
  constexpr size_t size = sizeof(addr_t);
  if (real_thread->CalculateTarget()->ReadMemory(
          address, static_cast<void *>(&output_value), size, error) == size)
    return true;
  else
    return false;
}

bool GetVariable(const ThreadSP &real_thread, llvm::StringRef expression,
                 uint64_t &output_value) {
  const TargetSP target_sp = real_thread->CalculateTarget();
  SymbolContextList ctx_list;
  target_sp->GetImages().FindSymbolsWithNameAndType(
      ConstString(expression), SymbolType::eSymbolTypeData, ctx_list);
  if (ctx_list.IsEmpty())
    return false;

  ExecutionContextScope *exe_scope = target_sp->GetProcessSP().get();
  if (exe_scope == nullptr)
    exe_scope = target_sp.get();

  SymbolContext ctx;
  if (!ctx_list.GetContextAtIndex(0, ctx))
    return false;

  Address addr = ctx.symbol->GetAddress();
  uint16_t value = 0;
  Status error;
  const auto read = real_thread->CalculateTarget()->ReadMemory(
      addr, static_cast<void *>(&value), 8, error);
  if (read != 8)
    return false;

  output_value = value;
  return true;
}

template <typename buffer_t>
auto make(const ThreadSP &real_thread, addr_t addr) -> buffer_t {
  buffer_t buffer;
  constexpr size_t size = sizeof(buffer);
  Address address;
  address.SetRawAddress(addr);
  Status error;
  if (real_thread->CalculateTarget()->ReadMemory(
          address, static_cast<void *>(&buffer), size, error) == size)
    return buffer;
  else
    return buffer_t{};
}

bool HasCoroutines(Process *process) {
  SymbolContextList sc_list;
  const static ConstString coro_cookie_store_name(
      "__folly_suspended_frame_cookie");
  // TODO: Implement FindFirstSymbolWithNameAndType upstream in lldb
  process->GetTarget().GetImages().FindSymbolsWithNameAndType(
      coro_cookie_store_name, SymbolType::eSymbolTypeData, sc_list);
  return sc_list.GetSize() > 0;
}

bool GetRegisterAsUint64(const ThreadSP &real_thread, llvm::StringRef reg_name,
                         uint64_t &output_value) {
  const auto reg_context = real_thread->GetRegisterContext();
  const RegisterInfo *reg_info = reg_context->GetRegisterInfoByName(reg_name);
  if (!reg_info)
    return false;

  RegisterValue reg_value;
  const bool read = reg_context->ReadRegister(reg_info, reg_value);
  if (!read || reg_value.GetType() != RegisterValue::eTypeUInt64)
    return false;

  output_value = reg_value.GetAsUInt64();
  return true;
}

void WalkNormalStack(const ThreadSP &real_thread,
                     addr_t normal_stack_frame_stop_addr,
                     addr_t normal_stack_frame_addr,
                     std::vector<addr_t> &addrs) {
  while (normal_stack_frame_addr) {
    const auto normal_stack_frame =
        make<StackFrame>(real_thread, normal_stack_frame_addr);
    if (normal_stack_frame_stop_addr &&
        normal_stack_frame.stack_frame == normal_stack_frame_stop_addr) {
      // Reached end of normal stack, transition to the async stack
      // Do not include the return address in the stack trace that points
      // to the frame that registered the AsyncStackRoot.
      break;
    }
    addrs.push_back(normal_stack_frame.return_address);
    normal_stack_frame_addr = normal_stack_frame.stack_frame;
  }
}

WalkAsyncStackResult WalkAsyncStack(const ThreadSP &real_thread,
                                    std::vector<addr_t> &addrs,
                                    addr_t async_stack_frame_addr) {
  addr_t normal_stack_frame_addr = 0;
  addr_t normal_stack_frame_stop_addr = 0;
  addr_t async_stack_frame_next_addr = 0;

  while (async_stack_frame_addr) {
    const auto async_stack_frame =
        make<AsyncStackFrame>(real_thread, async_stack_frame_addr);
    addrs.push_back(async_stack_frame.instruction_pointer);
    if (!async_stack_frame.parent_frame) {
      // Reached end of async stack
      addr_t async_stack_root_addr = async_stack_frame.stack_root;
      if (async_stack_root_addr == 0) {
        // This is a detached async stack. We are done
        break;
      }
      auto async_stack_root =
          make<AsyncStackRoot>(real_thread, async_stack_root_addr);
      normal_stack_frame_addr = async_stack_root.stack_frame_ptr;
      if (normal_stack_frame_addr == 0) {
        // No associated normal stack frame for this async stack root.
        break;
      }
      const auto normal_stack_frame =
          make<StackFrame>(real_thread, normal_stack_frame_addr);
      normal_stack_frame_addr = normal_stack_frame.stack_frame;

      async_stack_root_addr = async_stack_root.next_root;
      if (async_stack_root_addr) {
        async_stack_root =
            make<AsyncStackRoot>(real_thread, async_stack_root_addr);
        normal_stack_frame_stop_addr = async_stack_root.stack_frame_ptr;
        async_stack_frame_next_addr = async_stack_root.top_frame;
      }
    }

    async_stack_frame_addr = async_stack_frame.parent_frame;
  }

  return WalkAsyncStackResult{normal_stack_frame_addr,
                              normal_stack_frame_stop_addr,
                              async_stack_frame_next_addr};
}

std::vector<addr_t> GetAsyncStackAddrs(const ThreadSP &real_thread,
                                       addr_t async_stack_root_holder_addr) {
  Log *log = GetLog(LLDBLog::SystemRuntime);
  std::vector<addr_t> addrs;
  const auto leaf_frame = real_thread->GetStackFrameAtIndex(0);
  if (!leaf_frame) {
    LLDB_LOGF(log,
              "SystemRuntimeMetaCoroutine::GetExtendedBacktraceThread failed "
              "to get leaf frame for thread %lu",
              real_thread->GetID());
    return addrs;
  }

  Scalar rbp_value;
  llvm::Error error = leaf_frame->GetFrameBaseValue(rbp_value);
  if (error) {
    LLDB_LOG_ERROR(
        log, std::move(error),
        "SystemRuntimeMetaCoroutine::GetExtendedBacktraceThread failed "
        "to get leaf frame base value for thread %lu",
        real_thread->GetID());
    return addrs;
  }
  if (rbp_value.GetType() != Scalar::Type::e_int) {
    LLDB_LOGF(log,
              "SystemRuntimeMetaCoroutine::GetExtendedBacktraceThread failed "
              "to get leaf frame base value for thread %lu",
              real_thread->GetID());
    return addrs;
  }

  uint64_t normal_stack_frame_addr = rbp_value.ULongLong();
  const auto async_stack_root_holder =
      make<AsyncStackRootHolder>(real_thread, async_stack_root_holder_addr);
  const auto async_stack_root = make<AsyncStackRoot>(
      real_thread, async_stack_root_holder.async_stack_root_addr);

  addr_t reg_pc = -1;
  auto read_ok = GetRegisterAsUint64(real_thread, "pc", reg_pc);
  if (!read_ok) {
    LLDB_LOGF(log,
              "SystemRuntimeMetaCoroutine::GetExtendedBacktraceThread failed "
              "to read register 'pc' for thread %lu",
              real_thread->GetID());
    return addrs;
  }
  addrs.push_back(reg_pc);

  addr_t normal_stack_frame_stop_addr = async_stack_root.stack_frame_ptr;
  addr_t async_stack_frame_addr = async_stack_root.top_frame;
  while (normal_stack_frame_addr && async_stack_frame_addr) {
    WalkNormalStack(real_thread, normal_stack_frame_stop_addr,
                    normal_stack_frame_addr, addrs);
    WalkAsyncStackResult walk_async_stack_result =
        WalkAsyncStack(real_thread, addrs, async_stack_frame_addr);

    normal_stack_frame_addr = walk_async_stack_result.normal_stack_frame_addr;
    normal_stack_frame_stop_addr =
        walk_async_stack_result.normal_stack_frame_stop_addr;
    async_stack_frame_addr = walk_async_stack_result.async_stack_frame_addr;
  }
  if (addrs.size() == 1)
    // No coroutine frames found, remove reg_pc
    addrs.clear();

  return addrs;
}

bool GetAsyncStackRootHolderAddr(const ThreadSP &real_thread,
                                 uint64_t &output_value) {
  // The StackRootHolder address is located in the 'data' pointer of the
  // pthread_key_data structure. This structure is part of the 'specific' array
  // within the pthread structure, which is stored in the fs_base virtual
  // register. You can see the structures below:
  //
  // struct pthread_key_data {
  //   uint32_t seq;
  //   void *data;
  // };
  //
  // struct pthread {
  //   ...
  //   pthread_key_data *specific [32];
  // };
  //
  // The StackRootHolder address could be retrived using the following
  // expression:
  //   ((pthread*){pthread_addr)->specific"[tls_key / 32][tls_key % 32].data
  //
  //   where 'pthread_addr' is retrieved from the 'fs_base' register and
  //   'tls_key' is the value of the 'folly_async_stack_root_tls_key' variable.
  //
  // The following code implements the above expression but in a more efficient
  // manner by directly reading the variables and memory addresses from the
  // process memory rather than using the lldb expression evaluator.
  constexpr uint64_t offset_specific = 1296;
  constexpr uint64_t size_of_pointer = 8;
  constexpr uint64_t size_of_pthread_key_data = 16;
  constexpr uint64_t offset_data = 8;
  // 0xFFFFFFFF is used to signal invalid tls key. See
  // folly/tracing/AsyncStack.cpp
  constexpr uint64_t kInvalidPthreadValue = 0xFFFFFFFF;
  Log *log = GetLog(LLDBLog::SystemRuntime);

  uint64_t tls_key = kInvalidPthreadValue;
  if (!GetVariable(real_thread, kAsyncStackRootTlsKey, tls_key) ||
      tls_key == kInvalidPthreadValue) {
    LLDB_LOGF(log,
              "SystemRuntimeMetaCoroutine::GetExtendedBacktraceThread failed "
              "to get tls_key for %s",
              kAsyncStackRootTlsKey);
    return false;
  }

  if ((tls_key / 32) >= 32) {
    LLDB_LOGF(log,
              "SystemRuntimeMetaCoroutine::GetExtendedBacktraceThread tls_key "
              "%lu is outside of the 'specific' array range of 32",
              tls_key);
    return false;
  }

  const uint64_t offset_to_pthread_key_data =
      offset_specific + ((tls_key / 32) * size_of_pointer);
  const uint64_t offset_to_data =
      ((tls_key % 32) * size_of_pthread_key_data) + offset_data;

  const uint64_t pthread_value = real_thread->GetThreadPointer();
  if (pthread_value == LLDB_INVALID_ADDRESS) {
    // No coroutines found. This is expected for threads that are not currently
    // executing a coroutine.
    return false;
  }
  addr_t addr = 0;
  if (!GetPointerValue(real_thread, pthread_value, addr)) {
    LLDB_LOGF(log,
              "SystemRuntimeMetaCoroutine::GetExtendedBacktraceThread failed "
              "to read pthread value");
    return false;
  }

  const addr_t pthread_key_data_addr = addr + offset_to_pthread_key_data;
  if (!GetPointerValue(real_thread, pthread_key_data_addr, addr)) {
    LLDB_LOGF(log,
              "SystemRuntimeMetaCoroutine::GetExtendedBacktraceThread failed "
              "to read pthread_key_data_addr");
    return false;
  }

  return GetPointerValue(real_thread, addr + offset_to_data, output_value);
}

} // namespace

LLDB_PLUGIN_DEFINE(SystemRuntimeMetaCoroutine)

SystemRuntime *SystemRuntimeMetaCoroutine::CreateInstance(Process *process) {
  if (!HasCoroutines(process))
    return nullptr;

  Module *exe_module = process->GetTarget().GetExecutableModulePointer();
  if (exe_module) {
    ObjectFile *object_file = exe_module->GetObjectFile();
    if (object_file && object_file->GetStrata() != ObjectFile::eStrataUser)
      return nullptr;
  }

  const llvm::Triple &triple_ref =
      process->GetTarget().GetArchitecture().GetTriple();
  if (!triple_ref.isOSLinux() || !triple_ref.isX86()) {
    // TODO: Implement support for other architectures (e.g. aarch64)
    return nullptr;
  }
  return new SystemRuntimeMetaCoroutine(process);
}

SystemRuntimeMetaCoroutine::SystemRuntimeMetaCoroutine(
    lldb_private::Process *process)
    : SystemRuntime(process) {}

SystemRuntimeMetaCoroutine::~SystemRuntimeMetaCoroutine() {}

void SystemRuntimeMetaCoroutine::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(),
      "System runtime plugin for Meta Coroutine libraries.", CreateInstance);
}

void SystemRuntimeMetaCoroutine::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

ThreadSP
SystemRuntimeMetaCoroutine::GetExtendedBacktraceThread(ThreadSP real_thread,
                                                       ConstString type) {
  if (type != ConstString(kMetaCoroutineType))
    return {};

  uint64_t async_stack_root_holder_addr = 0;
  if (!GetAsyncStackRootHolderAddr(real_thread, async_stack_root_holder_addr))
    return {};

  std::vector<addr_t> addrs =
      GetAsyncStackAddrs(real_thread, async_stack_root_holder_addr);
  if (addrs.empty())
    return {};

  std::shared_ptr<HistoryThread> coro_thread_sp =
      std::make_shared<HistoryThread>(*m_process, real_thread->GetIndexID(),
                                      std::move(addrs), HistoryPCType::Calls);
  coro_thread_sp->SetQueueName("[Async Stack]");
  return coro_thread_sp;
}

const std::vector<ConstString> &
SystemRuntimeMetaCoroutine::GetExtendedBacktraceTypes() {
  if (m_types.size() == 0) {
    m_types.emplace_back(kMetaCoroutineType);
  }
  return m_types;
}
