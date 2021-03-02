//===- JITLoaderGDB.h - Register objects via GDB JIT interface -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderGDB.h"

#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/ManagedStatic.h"

#include <cstdint>
#include <mutex>
#include <utility>

#define DEBUG_TYPE "orc"

// First version as landed in August 2009
static constexpr uint32_t JitDescriptorVersion = 1;

// Keep in sync with gdb/gdb/jit.h
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

// We put information about the JITed function in this global, which the
// debugger reads.  Make sure to specify the version statically, because the
// debugger checks the version before we can set it during runtime.
struct jit_descriptor __jit_debug_descriptor = {JitDescriptorVersion, 0,
                                                nullptr, nullptr};

// Debuggers that implement the GDB JIT interface put a special breakpoint in
// this function.
LLVM_ATTRIBUTE_NOINLINE void __jit_debug_register_code() {
  // The noinline and the asm prevent calls to this function from being
  // optimized out.
#if !defined(_MSC_VER)
  asm volatile("" ::: "memory");
#endif
}
}

using namespace llvm;

// Serialize rendezvous with the debugger as well as access to shared data.
ManagedStatic<std::mutex> JITDebugLock;

static std::pair<const char *, uint64_t> readDebugObjectInfo(uint8_t *ArgData,
                                                             uint64_t ArgSize) {
  BinaryStreamReader ArgReader(ArrayRef<uint8_t>(ArgData, ArgSize),
                               support::endianness::big);
  uint64_t Addr, Size;
  cantFail(ArgReader.readInteger(Addr));
  cantFail(ArgReader.readInteger(Size));

  return std::make_pair(jitTargetAddressToPointer<const char *>(Addr), Size);
}

extern "C" orc::tpctypes::CWrapperFunctionResult
llvm_orc_registerJITLoaderGDBWrapper(uint8_t *Data, uint64_t Size) {
  if (Size != sizeof(uint64_t) + sizeof(uint64_t))
    return orc::tpctypes::WrapperFunctionResult::from(
               "Invalid arguments to llvm_orc_registerJITLoaderGDBWrapper")
        .release();

  jit_code_entry *E = new jit_code_entry;
  std::tie(E->symfile_addr, E->symfile_size) = readDebugObjectInfo(Data, Size);
  E->prev_entry = nullptr;

  std::lock_guard<std::mutex> Lock(*JITDebugLock);

  // Insert this entry at the head of the list.
  jit_code_entry *NextEntry = __jit_debug_descriptor.first_entry;
  E->next_entry = NextEntry;
  if (NextEntry) {
    NextEntry->prev_entry = E;
  }

  __jit_debug_descriptor.first_entry = E;
  __jit_debug_descriptor.relevant_entry = E;

  // Run into the rendezvous breakpoint.
  __jit_debug_descriptor.action_flag = JIT_REGISTER_FN;
  __jit_debug_register_code();

  return orc::tpctypes::WrapperFunctionResult().release();
}
