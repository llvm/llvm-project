//===---- ExecutionUtils.cpp - Utilities for executing functions in lli ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExecutionUtils.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <vector>

// Declarations follow the GDB JIT interface (version 1, 2009) and must match
// those of the DYLD used for testing. See:
//
//   llvm/lib/ExecutionEngine/Orc/TargetProcess/JITLoaderGDB.cpp
//   llvm/lib/ExecutionEngine/GDBRegistrationListener.cpp
//
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

namespace llvm {

template <typename... Ts> static void outsv(const char *Fmt, Ts &&...Vals) {
  outs() << formatv(Fmt, Vals...);
}

static const char *actionFlagToStr(uint32_t ActionFlag) {
  switch (ActionFlag) {
  case JIT_NOACTION:
    return "JIT_NOACTION";
  case JIT_REGISTER_FN:
    return "JIT_REGISTER_FN";
  case JIT_UNREGISTER_FN:
    return "JIT_UNREGISTER_FN";
  }
  return "<invalid action_flag>";
}

// Sample output:
//
//   Reading __jit_debug_descriptor at 0x0000000000404048
//
//   Version: 0
//   Action: JIT_REGISTER_FN
//
//         Entry               Symbol File         Size  Previous Entry
//   [ 0]  0x0000000000451290  0x0000000000002000   200  0x0000000000000000
//   [ 1]  0x0000000000451260  0x0000000000001000   100  0x0000000000451290
//   ...
//
static void dumpDebugDescriptor(void *Addr) {
  outsv("Reading __jit_debug_descriptor at {0}\n\n", Addr);

  jit_descriptor *Descriptor = reinterpret_cast<jit_descriptor *>(Addr);
  outsv("Version: {0}\n", Descriptor->version);
  outsv("Action: {0}\n\n", actionFlagToStr(Descriptor->action_flag));
  outsv("{0,11}  {1,24}  {2,15}  {3,14}\n", "Entry", "Symbol File", "Size",
        "Previous Entry");

  unsigned Idx = 0;
  for (auto *Entry = Descriptor->first_entry; Entry; Entry = Entry->next_entry)
    outsv("[{0,2}]  {1:X16}  {2:X16}  {3,8:D}  {4}\n", Idx++, Entry,
          reinterpret_cast<const void *>(Entry->symfile_addr),
          Entry->symfile_size, Entry->prev_entry);
}

static LLIBuiltinFunctionGenerator *Generator = nullptr;

static void dumpDebugObjects(void *Addr) {
  jit_descriptor *Descriptor = reinterpret_cast<jit_descriptor *>(Addr);
  for (auto *Entry = Descriptor->first_entry; Entry; Entry = Entry->next_entry)
    Generator->appendDebugObject(Entry->symfile_addr, Entry->symfile_size);
}

LLIBuiltinFunctionGenerator::LLIBuiltinFunctionGenerator(
    std::vector<BuiltinFunctionKind> Enabled, orc::MangleAndInterner &Mangle)
    : TestOut(nullptr) {
  Generator = this;
  for (BuiltinFunctionKind F : Enabled) {
    switch (F) {
    case BuiltinFunctionKind::DumpDebugDescriptor:
      expose(Mangle("__dump_jit_debug_descriptor"), &dumpDebugDescriptor);
      break;
    case BuiltinFunctionKind::DumpDebugObjects:
      expose(Mangle("__dump_jit_debug_objects"), &dumpDebugObjects);
      TestOut = createToolOutput();
      break;
    }
  }
}

Error LLIBuiltinFunctionGenerator::tryToGenerate(
    orc::LookupState &LS, orc::LookupKind K, orc::JITDylib &JD,
    orc::JITDylibLookupFlags JDLookupFlags,
    const orc::SymbolLookupSet &Symbols) {
  orc::SymbolMap NewSymbols;
  for (const auto &NameFlags : Symbols) {
    auto It = BuiltinFunctions.find(NameFlags.first);
    if (It != BuiltinFunctions.end())
      NewSymbols.insert(*It);
  }

  if (NewSymbols.empty())
    return Error::success();

  return JD.define(absoluteSymbols(std::move(NewSymbols)));
}

// static
std::unique_ptr<ToolOutputFile>
LLIBuiltinFunctionGenerator::createToolOutput() {
  std::error_code EC;
  auto TestOut = std::make_unique<ToolOutputFile>("-", EC, sys::fs::OF_None);
  if (EC) {
    errs() << "Error creating tool output file: " << EC.message() << '\n';
    exit(1);
  }
  return TestOut;
}

} // namespace llvm
