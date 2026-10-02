//===--- AMDGPUObjLinkingInfo.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Enums shared between the AMDGPU backend (LLVM) and the ELF linker (LLD)
/// for the `.amdgpu.info` object-linking metadata section.
///
/// Binary layout of each entry: [kind: u8] [len: u8] [payload: <len> bytes].
/// Unknown kinds are forward-compatible: a consumer skips them by reading len.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_AMDGPUOBJECTLINKINGINFO_H
#define LLVM_SUPPORT_AMDGPUOBJECTLINKINGINFO_H

#include "llvm/ADT/BitmaskEnum.h"

#include <cstdint>

namespace llvm {
namespace AMDGPU {

/// Entry kind values for the `.amdgpu.info` section.
///
/// Entries that appear between an INFO_FUNC and the next INFO_FUNC (or end of
/// section) belong to the function scope opened by that INFO_FUNC.
enum class InfoKind : uint8_t {
  /// Opens a new function scope.  Payload is an 8-byte symbol reference
  /// (relocated) identifying the function.  All subsequent entries until the
  /// next INFO_FUNC belong to this function.
  INFO_FUNC = 1,
  /// Bitfield of FuncInfoFlags properties for the function.  [u32]
  INFO_FLAGS = 2,
  /// Number of SGPRs explicitly used by the function.  [u32]
  INFO_NUM_SGPR = 3,
  /// Number of architectural VGPRs used by the function.  [u32]
  INFO_NUM_VGPR = 4,
  /// Number of accumulator VGPRs (AGPRs) used by the function.  [u32]
  INFO_NUM_AGPR = 5,
  /// Private (scratch) memory size in bytes required by the function.  [u32]
  INFO_PRIVATE_SEGMENT_SIZE = 6,
  /// Dependency edge: the function uses the resource identified by the
  /// 8-byte relocated symbol (e.g. an LDS variable or named barrier).
  INFO_USE = 7,
  /// Direct call edge: the function calls the callee identified by the
  /// 8-byte relocated symbol.
  INFO_CALL = 8,
  /// Indirect call edge: the function contains an indirect call whose
  /// callee is expected to match the type-ID string at the given
  /// `.amdgpu.strtab` offset.  [u32]
  INFO_INDIRECT_CALL = 9,
  /// Function type ID: tags an address-taken function with a type-ID
  /// string (at the given `.amdgpu.strtab` offset) so the linker can match
  /// it against INFO_INDIRECT_CALL entries.  [u32]
  INFO_TYPEID = 10,
};

/// Per-function flags packed into INFO_FLAGS entries.
enum class FuncInfoFlags : uint32_t {
  FUNC_USES_VCC = 1U << 0,
  FUNC_USES_FLAT_SCRATCH = 1U << 1,
  FUNC_HAS_DYN_STACK = 1U << 2,
  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/FUNC_HAS_DYN_STACK),
};

} // namespace AMDGPU
} // namespace llvm

#endif // LLVM_SUPPORT_AMDGPUOBJECTLINKINGINFO_H
