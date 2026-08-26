//===---- GDBJITRegistrar.cpp - Register objects via GDB JIT iface -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GDBJITRegistrar.h"
#include "orc-rt/bedrock/Compiler.h"

#include <cstdint>
#include <mutex>

using namespace orc_rt;

// Keep in sync with gdb/gdb/jit.h.
extern "C" {

typedef enum {
  JIT_NOACTION = 0,
  JIT_REGISTER_FN,
  JIT_UNREGISTER_FN
} jit_actions_t;

struct jit_code_entry {
  struct jit_code_entry *next_entry;
  struct jit_code_entry *prev_entry;
  const char *symfile_addr;
  uint64_t symfile_size;
};

struct jit_descriptor {
  uint32_t version;
  // This should be jit_actions_t, but we want to be specific about the
  // bit-width.
  uint32_t action_flag;
  struct jit_code_entry *relevant_entry;
  struct jit_code_entry *first_entry;
};

// First version as landed in GDB, August 2009.
static constexpr uint32_t JitDescriptorVersion = 1;

// We put information about the JIT'd object in this global, which the
// debugger reads. Make sure to specify the version statically, because the
// debugger checks the version before we can set it during runtime.
ORC_RT_INTERFACE struct jit_descriptor __jit_debug_descriptor = {
    JitDescriptorVersion, JIT_NOACTION, nullptr, nullptr};

// Debuggers that implement the GDB JIT interface put a special breakpoint in
// this function.
#if defined(_MSC_VER)
ORC_RT_INTERFACE void __jit_debug_register_code() {}
#else
ORC_RT_INTERFACE __attribute__((noinline)) void __jit_debug_register_code() {
  // The noinline attribute above and the asm volatile below prevent calls to
  // this function from being optimized out.
  asm volatile("" ::: "memory");
}
#endif

} // extern "C"

namespace {
// Serializes rendezvous with the debugger, as well as access to the
// __jit_debug_descriptor list.
std::mutex JITDebugLock;
} // namespace

namespace orc_rt::gdb_jit {

Error registerObject(span<char> Obj) {
  auto *E = new jit_code_entry;
  E->symfile_addr = Obj.data();
  E->symfile_size = Obj.size();
  E->prev_entry = nullptr;

  std::scoped_lock<std::mutex> Lock(JITDebugLock);

  // Insert this entry at the head of the list.
  jit_code_entry *NextEntry = __jit_debug_descriptor.first_entry;
  E->next_entry = NextEntry;
  if (NextEntry)
    NextEntry->prev_entry = E;

  __jit_debug_descriptor.first_entry = E;
  __jit_debug_descriptor.relevant_entry = E;
  __jit_debug_descriptor.action_flag = JIT_REGISTER_FN;

  // Run into the rendezvous breakpoint.
  __jit_debug_register_code();

  return Error::success();
}

Error deregisterObject(span<char> Obj) {
  std::scoped_lock<std::mutex> Lock(JITDebugLock);

  jit_code_entry *E = __jit_debug_descriptor.first_entry;
  while (E && (E->symfile_addr != Obj.data() || E->symfile_size != Obj.size()))
    E = E->next_entry;

  if (!E)
    return make_error<StringError>(
        "No GDB JIT debug object registered for range");

  // Unlink E from the list.
  if (E->next_entry)
    E->next_entry->prev_entry = E->prev_entry;
  if (E->prev_entry)
    E->prev_entry->next_entry = E->next_entry;
  else {
    assert(__jit_debug_descriptor.first_entry == E &&
           "Entry has no prev_entry, but is not the first entry");
    __jit_debug_descriptor.first_entry = E->next_entry;
  }

  __jit_debug_descriptor.relevant_entry = E;
  __jit_debug_descriptor.action_flag = JIT_UNREGISTER_FN;

  // Run into the rendezvous breakpoint. Debuggers read symfile_addr out of E
  // here, so E must still be live for this call.
  __jit_debug_register_code();

  // Reset the descriptor rather than leaving relevant_entry dangling once E is
  // freed below. Debuggers treat JIT_NOACTION as a no-op, and those that attach
  // later walk first_entry instead of reading these fields.
  __jit_debug_descriptor.relevant_entry = nullptr;
  __jit_debug_descriptor.action_flag = JIT_NOACTION;

  delete E;

  return Error::success();
}

} // namespace orc_rt::gdb_jit
