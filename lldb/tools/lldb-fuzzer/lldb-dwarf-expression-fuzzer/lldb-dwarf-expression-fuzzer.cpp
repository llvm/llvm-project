//===-- lldb-target-fuzzer.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "utils/TempFile.h"

#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Value.h"
#include "lldb/Expression/DWARFExpression.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::plugin::dwarf;
using namespace lldb_fuzzer;

extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) {
  FileSystem::Initialize();
  HostInfo::Initialize();
  platform_linux::PlatformLinux::Initialize();
  return 0;
}

static void Evaluate(llvm::ArrayRef<uint8_t> expr,
                     lldb::ModuleSP module_sp = {}, DWARFUnit *unit = nullptr,
                     ExecutionContext *exe_ctx = nullptr) {
  DataExtractor extractor(expr.data(), expr.size(), lldb::eByteOrderLittle,
                          /*addr_size*/ 4);

  llvm::Expected<Value> result =
      DWARFExpression::Evaluate(exe_ctx, /*reg_ctx*/ nullptr, module_sp,
                                extractor, unit, lldb::eRegisterKindLLDB,
                                /*initial_value_ptr*/ nullptr,
                                /*object_address_ptr*/ nullptr);

  if (!result)
    llvm::consumeError(result.takeError());
}

class MockTarget : public Target {
public:
  MockTarget(Debugger &debugger, const ArchSpec &target_arch,
             const lldb::PlatformSP &platform_sp, llvm::ArrayRef<uint8_t> data)
      : Target(debugger, target_arch, platform_sp, true), m_data(data) {}

  size_t ReadMemory(const Address &addr, void *dst, size_t dst_len,
                    Status &error, bool force_live_memory = false,
                    lldb::addr_t *load_addr_ptr = nullptr) override {
    std::memcpy(dst, m_data.data(), m_data.size());
    return m_data.size();
  }

private:
  llvm::ArrayRef<uint8_t> m_data;
};

extern "C" int LLVMFuzzerTestOneInput(uint8_t *data, size_t size) {
  // We're going to use the first half of the input data as the DWARF expression
  // and the second half as memory.
  const size_t partition = size / 2;
  llvm::ArrayRef expression_data(data, partition);
  llvm::ArrayRef memory_data(data + partition, size - partition);

  // Create a mock target for reading memory.
  ArchSpec arch("i386-pc-linux");
  Platform::SetHostPlatform(
      platform_linux::PlatformLinux::CreateInstance(true, &arch));
  lldb::DebuggerSP debugger_sp = Debugger::CreateInstance();
  lldb::PlatformSP platform_sp;
  auto target_sp = std::make_shared<MockTarget>(*debugger_sp, arch, platform_sp,
                                                memory_data);
  ExecutionContext exe_ctx(static_cast<lldb::TargetSP>(target_sp), false);

  Evaluate(expression_data);
  return 0;
}
