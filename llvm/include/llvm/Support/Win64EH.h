//===-- llvm/Support/Win64EH.h ---Win64 EH Constants-------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains constants and structures used for implementing
// exception handling on Win64 platforms. For more information, see
// http://msdn.microsoft.com/en-us/library/1eyas8tf.aspx
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_WIN64EH_H
#define LLVM_SUPPORT_WIN64EH_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace Win64EH {

/// UnwindOpcodes - Enumeration whose values specify a single operation in
/// the prolog of a function.
enum UnwindOpcodes {
  // The following set of unwind opcodes is for x86_64.  They are documented at
  // https://docs.microsoft.com/en-us/cpp/build/exception-handling-x64.
  // Some generic values from this set are used for other architectures too.
  UOP_PushNonVol = 0,
  UOP_AllocLarge,
  UOP_AllocSmall,
  UOP_SetFPReg,
  UOP_SaveNonVol,
  UOP_SaveNonVolBig,
  UOP_Epilog,
  UOP_SpareCode,
  UOP_SaveXMM128,
  UOP_SaveXMM128Big,
  UOP_PushMachFrame,
  // The following set of unwind opcodes is for ARM64.  They are documented at
  // https://docs.microsoft.com/en-us/cpp/build/arm64-exception-handling
  UOP_AllocMedium,
  UOP_SaveR19R20X,
  UOP_SaveFPLRX,
  UOP_SaveFPLR,
  UOP_SaveReg,
  UOP_SaveRegX,
  UOP_SaveRegP,
  UOP_SaveRegPX,
  UOP_SaveLRPair,
  UOP_SaveFReg,
  UOP_SaveFRegX,
  UOP_SaveFRegP,
  UOP_SaveFRegPX,
  UOP_SetFP,
  UOP_AddFP,
  UOP_Nop,
  UOP_End,
  UOP_SaveNext,
  UOP_TrapFrame,
  UOP_Context,
  UOP_ECContext,
  UOP_ClearUnwoundToCall,
  UOP_PACSignLR,
  UOP_SaveAnyRegI,
  UOP_SaveAnyRegIP,
  UOP_SaveAnyRegD,
  UOP_SaveAnyRegDP,
  UOP_SaveAnyRegQ,
  UOP_SaveAnyRegQP,
  UOP_SaveAnyRegIX,
  UOP_SaveAnyRegIPX,
  UOP_SaveAnyRegDX,
  UOP_SaveAnyRegDPX,
  UOP_SaveAnyRegQX,
  UOP_SaveAnyRegQPX,
  UOP_AllocZ,
  UOP_SaveZReg,
  UOP_SavePReg,

  // The following set of unwind opcodes is for ARM.  They are documented at
  // https://docs.microsoft.com/en-us/cpp/build/arm-exception-handling

  // Stack allocations use UOP_AllocSmall, UOP_AllocLarge from above, plus
  // the following. AllocSmall, AllocLarge and AllocHuge represent a 16 bit
  // instruction, while the WideAlloc* opcodes represent a 32 bit instruction.
  // Small can represent a stack offset of 0x7f*4 (252) bytes, Medium can
  // represent up to 0x3ff*4 (4092) bytes, Large up to 0xffff*4 (262140) bytes,
  // and Huge up to 0xffffff*4 (67108860) bytes.
  UOP_AllocHuge,
  UOP_WideAllocMedium,
  UOP_WideAllocLarge,
  UOP_WideAllocHuge,

  UOP_WideSaveRegMask,
  UOP_SaveSP,
  UOP_SaveRegsR4R7LR,
  UOP_WideSaveRegsR4R11LR,
  UOP_SaveFRegD8D15,
  UOP_SaveRegMask,
  UOP_SaveLR,
  UOP_SaveFRegD0D15,
  UOP_SaveFRegD16D31,
  // Using UOP_Nop from above
  UOP_WideNop,
  // Using UOP_End from above
  UOP_EndNop,
  UOP_WideEndNop,
  // A custom unspecified opcode, consisting of one or more bytes. This
  // allows producing opcodes in the implementation defined/reserved range.
  UOP_Custom,

  // V3-only x86_64 opcodes. They are documented at
  // https://learn.microsoft.com/en-us/cpp/build/x64-unwind-information-v3
  UOP_Push2, // PUSH2 — two registers in one instruction
};

/// UnwindCode - This union describes a single operation in a function prolog,
/// or part thereof.
union UnwindCode {
  struct {
    uint8_t CodeOffset;
    uint8_t UnwindOpAndOpInfo;
  } u;
  support::ulittle16_t FrameOffset;

  uint8_t getUnwindOp() const {
    return u.UnwindOpAndOpInfo & 0x0F;
  }
  uint8_t getOpInfo() const {
    return (u.UnwindOpAndOpInfo >> 4) & 0x0F;
  }
  /// Gets the offset for an UOP_Epilog unwind code.
  uint32_t getEpilogOffset() const {
    assert(getUnwindOp() == UOP_Epilog);
    return (getOpInfo() << 8) | static_cast<uint32_t>(u.CodeOffset);
  }
};

enum {
  /// UNW_ExceptionHandler - Specifies that this function has an exception
  /// handler.
  UNW_ExceptionHandler = 0x01,
  /// UNW_TerminateHandler - Specifies that this function has a termination
  /// handler.
  UNW_TerminateHandler = 0x02,
  /// UNW_ChainInfo - Specifies that this UnwindInfo structure is chained to
  /// another one.
  UNW_ChainInfo = 0x04,
  /// UNW_FlagLarge - V3 only. When set, the header is 5 bytes (an extra
  /// UNWIND_INFO_LARGE_V3 byte follows), SizeOfProlog extends to 16 bits,
  /// and prolog IP offset entries are 16-bit.
  UNW_FlagLarge = 0x08
};

/// RuntimeFunction - An entry in the table of functions with unwind info.
struct RuntimeFunction {
  support::ulittle32_t StartAddress;
  support::ulittle32_t EndAddress;
  support::ulittle32_t UnwindInfoOffset;
};

/// UnwindInfo - An entry in the exception table.
struct UnwindInfo {
  uint8_t VersionAndFlags;
  uint8_t PrologSize;
  uint8_t NumCodes;
  uint8_t FrameRegisterAndOffset;
  UnwindCode UnwindCodes[1];

  uint8_t getVersion() const {
    return VersionAndFlags & 0x07;
  }
  uint8_t getFlags() const {
    return (VersionAndFlags >> 3) & 0x1f;
  }
  uint8_t getFrameRegister() const {
    return FrameRegisterAndOffset & 0x0f;
  }
  uint8_t getFrameOffset() const {
    return (FrameRegisterAndOffset >> 4) & 0x0f;
  }

  // The data after unwindCodes depends on flags.
  // If UNW_ExceptionHandler or UNW_TerminateHandler is set then follows
  // the address of the language-specific exception handler.
  // If UNW_ChainInfo is set then follows a RuntimeFunction which defines
  // the chained unwind info.
  // For more information please see MSDN at:
  // http://msdn.microsoft.com/en-us/library/ddssxxy8.aspx

  /// Return pointer to language specific data part of UnwindInfo.
  void *getLanguageSpecificData() {
    return reinterpret_cast<void *>(&UnwindCodes[(NumCodes+1) & ~1]);
  }

  /// Return pointer to language specific data part of UnwindInfo.
  const void *getLanguageSpecificData() const {
    return reinterpret_cast<const void *>(&UnwindCodes[(NumCodes + 1) & ~1]);
  }

  /// Return image-relative offset of language-specific exception handler.
  uint32_t getLanguageSpecificHandlerOffset() const {
    return *reinterpret_cast<const support::ulittle32_t *>(
               getLanguageSpecificData());
  }

  /// Set image-relative offset of language-specific exception handler.
  void setLanguageSpecificHandlerOffset(uint32_t offset) {
    *reinterpret_cast<support::ulittle32_t *>(getLanguageSpecificData()) =
        offset;
  }

  /// Return pointer to exception-specific data.
  void *getExceptionData() {
    return reinterpret_cast<void *>(reinterpret_cast<uint32_t *>(
                                                  getLanguageSpecificData())+1);
  }

  /// Return pointer to chained unwind info.
  RuntimeFunction *getChainedFunctionEntry() {
    return reinterpret_cast<RuntimeFunction *>(getLanguageSpecificData());
  }

  /// Return pointer to chained unwind info.
  const RuntimeFunction *getChainedFunctionEntry() const {
    return reinterpret_cast<const RuntimeFunction *>(getLanguageSpecificData());
  }
};

//===----------------------------------------------------------------------===//
// V3 Unwind Information
//===----------------------------------------------------------------------===//

/// V3 Winding Operation Descriptor opcodes.
enum WODOpcode : uint8_t {
  WOD_SET_FPREG = 0,            // 8-bit opcode, 2 bytes
  WOD_ALLOC_HUGE = 1,           // 8-bit opcode, 5 bytes
  WOD_ALLOC_LARGE = 2,          // 8-bit opcode, 3 bytes
  WOD_PUSH_CANONICAL_FRAME = 3, // 8-bit opcode, 2 bytes
  WOD_PUSH = 4,                 // 3-bit opcode, 1 byte
  WOD_SAVE_NONVOL_FAR = 5,      // 3-bit opcode, 5 bytes
  WOD_SAVE_NONVOL = 6,          // 3-bit opcode, 3 bytes
  WOD_PUSH_CONSECUTIVE_2 = 7,   // 3-bit opcode, 1 byte
  WOD_ALLOC_SMALL = 8,          // 4-bit opcode, 1 byte
  WOD_SAVE_XMM128_FAR = 9,      // 4-bit opcode, 5 bytes
  WOD_SAVE_XMM128 = 10,         // 4-bit opcode, 3 bytes
  WOD_PUSH2 = 32,               // 6-bit opcode, 2 bytes
};

/// V3 EPILOG_INFO flags.
enum EpilogInfoFlagsV3 : uint8_t {
  EPILOG_PARENT_FRAGMENT_TRANSFER = 0x01,
  /// When set, the extended descriptor uses EPILOG_INFO_LARGE_EX_V3 (16-bit
  /// IpOffsetOfLastInstruction) and the IP offset array uses 16-bit entries.
  EPILOG_INFO_LARGE = 0x02,
};

/// Decoded V3 Winding Operation Descriptor.
struct DecodedWOD {
  WODOpcode Opcode;
  uint8_t Register;  // For applicable ops (5-bit for int, 4-bit for XMM)
  uint8_t Register2; // For WOD_PUSH2
  // TODO: Define a named enum for WOD_PUSH_CANONICAL_FRAME Type values once
  // the Windows x64 Unwind V3 spec is finalized. The set of valid values is
  // defined by the OS (see the Windows SDK headers) but is not yet stable.
  uint8_t Type;          // For WOD_PUSH_CANONICAL_FRAME
  uint8_t ByteSize;      // How many bytes this WOD consumed (max 5)
  uint32_t Size;         // For alloc ops: final computed size
  uint32_t Displacement; // For save ops: final computed displacement
};

/// Decoded V3 epilog descriptor.
struct DecodedEpilogV3 {
  uint8_t Flags;
  uint8_t NumberOfOps;
  uint16_t IpOffsetOfLastInstruction;
  uint16_t FirstOp;
  int32_t EpilogOffset; // Resolved absolute offset (accumulated from deltas).
  SmallVector<uint16_t, 8> IpOffsets;

  /// Whether the EPILOG_INFO_LARGE flag is set.
  bool isLarge() const { return Flags & EPILOG_INFO_LARGE; }
};

/// Decoded V3 UNWIND_INFO.
struct DecodedUnwindInfoV3 {
  uint8_t Version;
  uint8_t Flags;
  uint8_t PayloadWords;
  uint8_t NumberOfOps;
  uint8_t NumberOfEpilogs;
  uint16_t SizeOfProlog;
  /// Total bytes consumed by header + payload (used to locate handler/chain).
  uint16_t PayloadSize;
  SmallVector<uint16_t, 8> PrologIpOffsets;
  SmallVector<DecodedEpilogV3, 4> Epilogs;
  ArrayRef<uint8_t> WODPool;

  /// Whether the UNW_FlagLarge flag is set.
  bool isLarge() const { return Flags & UNW_FlagLarge; }
};

/// Return the register name for a 5-bit AMD64 integer register number.
/// Covers 0-15 (RAX-R15) and 16-31 (R16-R31 for APX).
LLVM_ABI StringRef getRegisterNameV3(unsigned Reg);

/// Decode one WOD from the pool at the given byte offset.
/// Returns an error on malformed data.
LLVM_ABI Expected<DecodedWOD> decodeWOD(ArrayRef<uint8_t> Pool,
                                        unsigned Offset);

/// Parse a V3 UNWIND_INFO from raw bytes.
/// Returns an error on malformed data.
LLVM_ABI Expected<DecodedUnwindInfoV3>
decodeUnwindInfoV3(ArrayRef<uint8_t> Data);

} // End of namespace Win64EH
} // End of namespace llvm

#endif
