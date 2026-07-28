//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_OBJECTFILE_WASM_WASMADDRESS_H
#define LLDB_SOURCE_PLUGINS_OBJECTFILE_WASM_WASMADDRESS_H

#include "lldb/lldb-types.h"
#include <cstdint>

namespace lldb_private {
namespace wasm {

/// Each WebAssembly module has separate address spaces for Code and Memory. A
/// WebAssembly module also has a Data section which, when the module is loaded,
/// gets mapped into a region in the module Memory.
///
/// Globals are not addressable: they live in an index space of their own. The
/// synthetic Global space stands in for one, holding the index where an address
/// would hold an offset, so that a global can be named and read.
///
/// The tag is two bits wide and these are the only spaces there are, so the
/// remaining value means an address that belongs to nothing.
enum WasmAddressType : uint8_t {
  Memory = 0x00,
  Object = 0x01,
  Global = 0x02,
  Invalid = 0x03,
};

/// The top two bits of a 64-bit address hold the space it belongs to.
static constexpr uint32_t kWasmAddressTypeShift = 62;
static constexpr uint64_t kWasmAddressTypeMask = uint64_t(0b11)
                                                 << kWasmAddressTypeShift;

/// For the purpose of debugging, we can represent all these separated 32-bit
/// address spaces with a single virtual 64-bit address space. The
/// wasm_addr_t provides this encoding using bitfields.
struct wasm_addr_t {
  uint64_t offset : 32;
  uint64_t module_id : 30;
  uint64_t type : 2;

  wasm_addr_t(lldb::addr_t addr)
      : offset(addr & 0x00000000ffffffff),
        module_id((addr & 0x00ffffff00000000) >> 32), type(addr >> 62) {}

  wasm_addr_t(WasmAddressType type, uint32_t module_id, uint32_t offset)
      : offset(offset), module_id(module_id), type(type) {}

  WasmAddressType GetType() const { return static_cast<WasmAddressType>(type); }
  uint32_t GetModuleID() const { return module_id; }
  uint32_t GetOffset() const { return offset; }

  operator lldb::addr_t() { return *(uint64_t *)this; }
};

static_assert(sizeof(wasm_addr_t) == 8, "");

} // namespace wasm
} // namespace lldb_private

#endif
