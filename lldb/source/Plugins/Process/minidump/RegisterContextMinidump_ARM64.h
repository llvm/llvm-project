//===-- RegisterContextMinidump_ARM64.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_MINIDUMP_REGISTERCONTEXTMINIDUMP_ARM64_H
#define LLDB_SOURCE_PLUGINS_PROCESS_MINIDUMP_REGISTERCONTEXTMINIDUMP_ARM64_H

#include "MinidumpTypes.h"

#include "lldb/Target/ThreadRegisterContext.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitmaskEnum.h"

// C includes
// C++ includes

namespace lldb_private {

namespace minidump {

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

RegisterLayout CreateMinidumpArm64RegisterLayout();

class RegisterContextMinidump_ARM64
    : public lldb_private::ThreadRegisterContext {
public:
  RegisterContextMinidump_ARM64(lldb_private::Thread &thread,
                                const DataExtractor &data);

  ~RegisterContextMinidump_ARM64() override = default;

  void InvalidateAllRegisters() override {
    // Do nothing... registers are always valid...
  }

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override;

  llvm::Error
  WriteRegister(lldb_private::RegisterID id,
                const lldb_private::RegisterValue &reg_value) override;

  // Reference: see breakpad/crashpad source
  struct Context {
    uint64_t context_flags;
    uint64_t x[32];
    uint64_t pc;
    uint32_t cpsr;
    uint32_t fpsr;
    uint32_t fpcr;
    uint8_t v[32 * 16]; // 32 128-bit floating point registers
  };

  enum class Flags : uint32_t {
    ARM64_Flag = 0x80000000,
    Integer = ARM64_Flag | 0x00000002,
    FloatingPoint = ARM64_Flag | 0x00000004,
    LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue = */ FloatingPoint)
  };

protected:
  std::vector<uint8_t> m_context;
};

} // end namespace minidump
} // end namespace lldb_private
#endif // LLDB_SOURCE_PLUGINS_PROCESS_MINIDUMP_REGISTERCONTEXTMINIDUMP_ARM64_H
