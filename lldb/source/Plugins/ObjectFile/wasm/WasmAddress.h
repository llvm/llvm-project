//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_OBJECTFILE_WASM_WASMADDRESS_H
#define LLDB_SOURCE_PLUGINS_OBJECTFILE_WASM_WASMADDRESS_H

#include "lldb/lldb-defines.h"
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

/// Widths of the fields a 64-bit address is made of, from the low bits up. The
/// bitfields below are declared with the same constants, so a field and the
/// mask that extracts it cannot come to disagree.
static constexpr uint32_t kWasmOffsetBits = 32;
static constexpr uint32_t kWasmModuleIDBits = 30;
static constexpr uint32_t kWasmAddressTypeBits = 2;

static_assert(kWasmOffsetBits + kWasmModuleIDBits + kWasmAddressTypeBits == 64,
              "a Wasm address has to account for all 64 bits");

static constexpr uint32_t kWasmModuleIDShift = kWasmOffsetBits;
static constexpr uint32_t kWasmAddressTypeShift =
    kWasmModuleIDShift + kWasmModuleIDBits;

static constexpr uint64_t MakeFieldMask(uint32_t bits, uint32_t shift) {
  return ((uint64_t(1) << bits) - 1) << shift;
}

static constexpr uint64_t kWasmOffsetMask = MakeFieldMask(kWasmOffsetBits, 0);
static constexpr uint64_t kWasmModuleIDMask =
    MakeFieldMask(kWasmModuleIDBits, kWasmModuleIDShift);
static constexpr uint64_t kWasmAddressTypeMask =
    MakeFieldMask(kWasmAddressTypeBits, kWasmAddressTypeShift);

/// A value that names no module. Every value the id field can hold names a
/// module, zero included, so the sentinel has to come from outside that range.
static constexpr uint32_t kWasmInvalidModuleID = UINT32_MAX;

static_assert(kWasmInvalidModuleID > (kWasmModuleIDMask >> kWasmModuleIDShift),
              "the sentinel has to fall outside the range of a module id");

/// For the purpose of debugging, we can represent all these separated 32-bit
/// address spaces with a single virtual 64-bit address space. The
/// wasm_addr_t provides this encoding using bitfields.
struct wasm_addr_t {
  uint64_t offset : kWasmOffsetBits;
  uint64_t module_id : kWasmModuleIDBits;
  uint64_t type : kWasmAddressTypeBits;

  wasm_addr_t(lldb::addr_t addr)
      : offset(addr & kWasmOffsetMask),
        module_id((addr & kWasmModuleIDMask) >> kWasmModuleIDShift),
        type(addr >> kWasmAddressTypeShift) {}

  wasm_addr_t(WasmAddressType type, uint32_t module_id, uint32_t offset)
      : offset(offset), module_id(module_id), type(type) {}

  WasmAddressType GetType() const { return static_cast<WasmAddressType>(type); }
  uint32_t GetModuleID() const { return module_id; }
  uint32_t GetOffset() const { return offset; }

  operator lldb::addr_t() { return *(uint64_t *)this; }
};

static_assert(sizeof(wasm_addr_t) == 8, "");

/// The module an address belongs to, or kWasmInvalidModuleID for an invalid
/// address.
inline uint32_t GetWasmModuleID(lldb::addr_t addr) {
  if (addr == LLDB_INVALID_ADDRESS)
    return kWasmInvalidModuleID;
  return wasm_addr_t(addr).GetModuleID();
}

} // namespace wasm
} // namespace lldb_private

#endif
