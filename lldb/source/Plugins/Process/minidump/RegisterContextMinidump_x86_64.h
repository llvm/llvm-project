//===-- RegisterContextMinidump_x86_64.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextMinidump_h_
#define liblldb_RegisterContextMinidump_h_

// Project includes
#include "MinidumpTypes.h"

// Other libraries and framework includes
#include "Plugins/Process/Utility/RegisterInfoInterface.h"
#include "Plugins/Process/Utility/lldb-x86-register-enums.h"

#include "lldb/Target/RegisterContext.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitmaskEnum.h"

// C includes
// C++ includes

namespace lldb_private {

namespace minidump {

// The content of the Minidump register context is as follows:
// (for reference see breakpad's source or WinNT.h)
// Register parameter home addresses: (p1_home .. p6_home)
// - uint64_t p1_home
// - uint64_t p2_home
// - uint64_t p3_home
// - uint64_t p4_home
// - uint64_t p5_home
// - uint64_t p6_home
//
// - uint32_t context_flags - field that determines the layout of the structure
//     and which parts of it are populated
// - uint32_t mx_csr
//
// - uint16_t cs - included if MinidumpContext_x86_64_Flags::Control
//
// - uint16_t ds - included if MinidumpContext_x86_64_Flags::Segments
// - uint16_t es - included if MinidumpContext_x86_64_Flags::Segments
// - uint16_t fs - included if MinidumpContext_x86_64_Flags::Segments
// - uint16_t gs - included if MinidumpContext_x86_64_Flags::Segments
//
// - uint16_t ss     - included if MinidumpContext_x86_64_Flags::Control
// - uint32_t rflags - included if MinidumpContext_x86_64_Flags::Control
//
// Debug registers: (included if MinidumpContext_x86_64_Flags::DebugRegisters)
// - uint64_t dr0
// - uint64_t dr1
// - uint64_t dr2
// - uint64_t dr3
// - uint64_t dr6
// - uint64_t dr7
//
// The next 4 registers are included if MinidumpContext_x86_64_Flags::Integer
// - uint64_t rax
// - uint64_t rcx
// - uint64_t rdx
// - uint64_t rbx
//
// - uint64_t rsp - included if MinidumpContext_x86_64_Flags::Control
//
// The next 11 registers are included if MinidumpContext_x86_64_Flags::Integer
// - uint64_t rbp
// - uint64_t rsi
// - uint64_t rdi
// - uint64_t r8
// - uint64_t r9
// - uint64_t r10
// - uint64_t r11
// - uint64_t r12
// - uint64_t r13
// - uint64_t r14
// - uint64_t r15
//
// - uint64_t rip - included if MinidumpContext_x86_64_Flags::Control
//
// TODO: add floating point registers here

// This function receives an ArrayRef pointing to the bytes of the Minidump
// register context and returns a DataBuffer that's ordered by the offsets
// specified in the RegisterInfoInterface argument
// This way we can reuse the already existing register contexts
lldb::DataBufferSP
ConvertMinidumpContextToRegIface(llvm::ArrayRef<uint8_t> source_data,
                                 RegisterInfoInterface *target_reg_interface);

// For context_flags. These values indicate the type of
// context stored in the structure.  The high 24 bits identify the CPU, the
// low 8 bits identify the type of context saved.
LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

enum class MinidumpContext_x86_64_Flags : uint32_t {
  x86_64_Flag = 0x00100000,
  Control = x86_64_Flag | 0x00000001,
  Integer = x86_64_Flag | 0x00000002,
  Segments = x86_64_Flag | 0x00000004,
  FloatingPoint = x86_64_Flag | 0x00000008,
  DebugRegisters = x86_64_Flag | 0x00000010,
  XState = x86_64_Flag | 0x00000040,

  Full = Control | Integer | FloatingPoint,
  All = Full | Segments | DebugRegisters,

  LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue = */ All)
};

} // end namespace minidump
} // end namespace lldb_private
#endif // liblldb_RegisterContextMinidump_h_
