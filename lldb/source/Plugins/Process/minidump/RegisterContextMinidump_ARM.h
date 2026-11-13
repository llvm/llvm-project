//===-- RegisterContextMinidump_ARM.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_MINIDUMP_REGISTERCONTEXTMINIDUMP_ARM_H
#define LLDB_SOURCE_PLUGINS_PROCESS_MINIDUMP_REGISTERCONTEXTMINIDUMP_ARM_H

#include "MinidumpTypes.h"

#include "lldb/Target/ThreadRegisterContext.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitmaskEnum.h"

// C includes
// C++ includes

namespace lldb_private {

namespace minidump {

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

RegisterLayout CreateMinidumpArmRegisterLayout(bool apple);

class RegisterContextMinidump_ARM : public lldb_private::ThreadRegisterContext {
public:
  RegisterContextMinidump_ARM(lldb_private::Thread &thread,
                              const DataExtractor &data, bool apple);

  ~RegisterContextMinidump_ARM() override = default;

  void InvalidateAllRegisters() override {
    // Do nothing... registers are always valid...
  }

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override;

  llvm::Error
  WriteRegister(lldb_private::RegisterID id,
                const lldb_private::RegisterValue &reg_value) override;

  // Reference: see breakpad/crashpad source
  struct QRegValue {
    uint64_t lo;
    uint64_t hi;
  };

  struct Context {
    uint32_t context_flags;
    uint32_t r[16];
    uint32_t cpsr;
    uint64_t fpscr;
    union {
      uint64_t d[32];
      uint32_t s[32];
      QRegValue q[16];
    };
    uint32_t extra[8];
  };

protected:
  enum class Flags : uint32_t {
    ARM_Flag = 0x40000000,
    Integer = ARM_Flag | 0x00000002,
    FloatingPoint = ARM_Flag | 0x00000004,
    LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue = */ FloatingPoint)
  };
  std::vector<uint8_t> m_context;
};

} // end namespace minidump
} // end namespace lldb_private
#endif // LLDB_SOURCE_PLUGINS_PROCESS_MINIDUMP_REGISTERCONTEXTMINIDUMP_ARM_H
