//===----------------------------- unwind-pe.h ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
// Pointer-Encoding decoder. Derived from:
//   - libcxxabi/src/Unwind/dwarf2.h
//   - libcxxabi/src/Unwind/AddressSpace.h
//
//===----------------------------------------------------------------------===//

#ifndef UNWIND_PE_H
#define UNWIND_PE_H

#include <assert.h>
#include <stdint.h>
#include <string.h>

// FSF exception handling Pointer-Encoding constants
// Used in CFI augmentation by GCC
enum {
  DW_EH_PE_ptr       = 0x00,
  DW_EH_PE_uleb128   = 0x01,
  DW_EH_PE_udata2    = 0x02,
  DW_EH_PE_udata4    = 0x03,
  DW_EH_PE_udata8    = 0x04,
  DW_EH_PE_signed    = 0x08,
  DW_EH_PE_sleb128   = 0x09,
  DW_EH_PE_sdata2    = 0x0A,
  DW_EH_PE_sdata4    = 0x0B,
  DW_EH_PE_sdata8    = 0x0C,
  DW_EH_PE_absptr    = 0x00,
  DW_EH_PE_pcrel     = 0x10,
  DW_EH_PE_textrel   = 0x20,
  DW_EH_PE_datarel   = 0x30,
  DW_EH_PE_funcrel   = 0x40,
  DW_EH_PE_aligned   = 0x50,
  DW_EH_PE_indirect  = 0x80,
  DW_EH_PE_omit      = 0xFF
};

/// Read a ULEB128 into a 64-bit word.
static uint64_t unw_getULEB128(uintptr_t *addr) {
  const uint8_t *p = (uint8_t *)*addr;
  uint64_t result = 0;
  int bit = 0;
  do {
    uint64_t b;

    b = *p & 0x7f;

    if (bit >= 64 || b << bit >> bit != b) {
      assert(!"malformed uleb128 expression");
    } else {
      result |= b << bit;
      bit += 7;
    }
  } while (*p++ >= 0x80);
  *addr = (uintptr_t) p;
  return result;
}

/// Read a SLEB128 into a 64-bit word.
static int64_t unw_getSLEB128(uintptr_t *addr) {
  const uint8_t *p = (uint8_t *)addr;
  int64_t result = 0;
  int bit = 0;
  uint8_t byte;
  do {
    byte = *p++;
    result |= ((byte & 0x7f) << bit);
    bit += 7;
  } while (byte & 0x80);
  // sign extend negative numbers
  if ((byte & 0x40) != 0)
    result |= (-1LL) << bit;
  *addr = (uintptr_t) p;
  return result;
}

static uint16_t unw_get16(uintptr_t addr) {
  uint16_t val;
  memcpy(&val, (void *)addr, sizeof(val));
  return val;
}

static uint32_t unw_get32(uintptr_t addr) {
  uint32_t val;
  memcpy(&val, (void *)addr, sizeof(val));
  return val;
}

static uint64_t unw_get64(uintptr_t addr) {
  uint64_t val;
  memcpy(&val, (void *)addr, sizeof(val));
  return val;
}

static uintptr_t unw_getP(uintptr_t addr) {
  if (sizeof(uintptr_t) == 8)
    return unw_get64(addr);
  else
    return unw_get32(addr);
}

static const unsigned char *read_uleb128(const unsigned char *p,
                                         _uleb128_t *ret) {
  uintptr_t addr = (uintptr_t)p;
  *ret = unw_getULEB128(&addr);
  return (unsigned char *)addr;
}

static const unsigned char *read_encoded_value(struct _Unwind_Context *ctx,
                                               unsigned char encoding,
                                               const unsigned char *p,
                                               _Unwind_Ptr *ret) {
  uintptr_t addr = (uintptr_t)p;
  uintptr_t startAddr = addr;
  uintptr_t result;

  (void)ctx;

  // first get value
  switch (encoding & 0x0F) {
  case DW_EH_PE_ptr:
    result = unw_getP(addr);
    p += sizeof(uintptr_t);
    break;
  case DW_EH_PE_uleb128:
    result = (uintptr_t)unw_getULEB128(&addr);
    p = (const unsigned char *)addr;
    break;
  case DW_EH_PE_udata2:
    result = unw_get16(addr);
    p += 2;
    break;
  case DW_EH_PE_udata4:
    result = unw_get32(addr);
    p += 4;
    break;
  case DW_EH_PE_udata8:
    result = (uintptr_t)unw_get64(addr);
    p += 8;
    break;
  case DW_EH_PE_sleb128:
    result = (uintptr_t)unw_getSLEB128(&addr);
    p = (const unsigned char *)addr;
    break;
  case DW_EH_PE_sdata2:
    // Sign extend from signed 16-bit value.
    result = (uintptr_t)(int16_t)unw_get16(addr);
    p += 2;
    break;
  case DW_EH_PE_sdata4:
    // Sign extend from signed 32-bit value.
    result = (uintptr_t)(int32_t)unw_get32(addr);
    p += 4;
    break;
  case DW_EH_PE_sdata8:
    result = (uintptr_t)unw_get64(addr);
    p += 8;
    break;
  default:
    assert(!"unknown pointer encoding");
  }

  // then add relative offset
  switch (encoding & 0x70) {
  case DW_EH_PE_absptr:
    // do nothing
    break;
  case DW_EH_PE_pcrel:
    result += startAddr;
    break;
  case DW_EH_PE_textrel:
    assert(!"DW_EH_PE_textrel pointer encoding not supported");
    break;
  case DW_EH_PE_datarel:
    assert(!"DW_EH_PE_datarel pointer encoding not supported");
    break;
  case DW_EH_PE_funcrel:
    assert(!"DW_EH_PE_funcrel pointer encoding not supported");
    break;
  case DW_EH_PE_aligned:
    assert(!"DW_EH_PE_aligned pointer encoding not supported");
    break;
  default:
    assert(!"unknown pointer encoding");
    break;
  }

  if (encoding & DW_EH_PE_indirect)
    result = unw_getP(result);

  *ret = result;
  return p;
}

#endif  // UNWIND_PE_H
