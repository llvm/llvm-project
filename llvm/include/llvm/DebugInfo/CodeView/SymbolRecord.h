//===- SymbolRecord.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_SYMBOLRECORD_H
#define LLVM_DEBUGINFO_CODEVIEW_SYMBOLRECORD_H

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/Support/Endian.h"

namespace llvm {
namespace codeview {

using llvm::support::ulittle16_t;
using llvm::support::ulittle32_t;

/// Distinguishes individual records in the Symbols subsection of a .debug$S
/// section. Equivalent to SYM_ENUM_e in cvinfo.h.
enum SymbolRecordKind : uint16_t {
#define SYMBOL_TYPE(ename, value) ename = value,
#include "CVSymbolTypes.def"
};

/// Data preceding all symbol records.
struct SymRecord {
  ulittle16_t RecordLength; // Record length, starting from the next field
  ulittle16_t RecordKind;   // Record kind (SymbolRecordKind)
  // Symbol data follows.

  SymbolRecordKind getKind() const {
    return SymbolRecordKind(uint16_t(RecordKind));
  }
};

/// Corresponds to the CV_PROCFLAGS bitfield.
enum ProcFlags : uint8_t {
  HasFP = 1 << 0,
  HasIRET = 1 << 1,
  HasFRET = 1 << 2,
  IsNoReturn = 1 << 3,
  IsUnreachable = 1 << 4,
  HasCustomCallingConv = 1 << 5,
  IsNoInline = 1 << 6,
  HasOptimizedDebugInfo = 1 << 7,
};

// S_GPROC32, S_LPROC32, S_GPROC32_ID, S_LPROC32_ID, S_LPROC32_DPC or
// S_LPROC32_DPC_ID
struct ProcSym {
  ulittle32_t PtrParent;
  ulittle32_t PtrEnd;
  ulittle32_t PtrNext;
  ulittle32_t CodeSize;
  ulittle32_t DbgStart;
  ulittle32_t DbgEnd;
  TypeIndex FunctionType;
  ulittle32_t CodeOffset;
  ulittle16_t Segment;
  uint8_t Flags; // CV_PROCFLAGS
  // Name: The null-terminated name follows.
};

enum BinaryAnnotationsOpCode : uint32_t {
  Invalid,
  CodeOffset,
  ChangeCodeOffsetBase,
  ChangeCodeOffset,
  ChangeCodeLength,
  ChangeFile,
  ChangeLineOffset,
  ChangeLineEndDelta,
  ChangeRangeKind,
  ChangeColumnStart,
  ChangeColumnEndDelta,
  ChangeCodeOffsetAndLineOffset,
  ChangeCodeLengthAndCodeOffset,
  ChangeColumnEnd,
};

// S_INLINESITE
struct InlineSiteSym {
  ulittle32_t PtrParent;
  ulittle32_t PtrEnd;
  TypeIndex Inlinee;
  // BinaryAnnotations
};

// S_LOCAL
struct LocalSym {
  TypeIndex Type;
  ulittle16_t Flags;
  enum : uint16_t {
    IsParameter = 1 << 0,
    IsAddressTaken = 1 << 1,
    IsCompilerGenerated = 1 << 2,
    IsAggregate = 1 << 3,
    IsAggregated = 1 << 4,
    IsAliased = 1 << 5,
    IsAlias = 1 << 6,
    IsReturnValue = 1 << 7,
    IsOptimizedOut = 1 << 8,
    IsEnregisteredGlobal = 1 << 9,
    IsEnregisteredStatic = 1 << 10,
  };
  // Name: The null-terminated name follows.
};

// S_BLOCK32
struct BlockSym {
  ulittle32_t PtrParent;
  ulittle32_t PtrEnd;
  ulittle32_t CodeSize;
  ulittle32_t CodeOffset;
  ulittle16_t Segment;
  // Name: The null-terminated name follows.
};

// S_LABEL32
struct LabelSym {
  ulittle32_t CodeOffset;
  ulittle16_t Segment;
  uint8_t Flags; // CV_PROCFLAGS
  // Name: The null-terminated name follows.
};

// S_OBJNAME
struct ObjNameSym {
  ulittle32_t Signature;
  // Name: The null-terminated name follows.
};

// S_COMPILE3
struct CompileSym3 {
  ulittle32_t flags;
  uint8_t getLanguage() const { return flags & 0xff; }
  enum Flags : uint32_t {
    EC = 1 << 8,
    NoDbgInfo = 1 << 9,
    LTCG = 1 << 10,
    NoDataAlign = 1 << 11,
    ManagedPresent = 1 << 12,
    SecurityChecks = 1 << 13,
    HotPatch = 1 << 14,
    CVTCIL = 1 << 15,
    MSILModule = 1 << 16,
    Sdl = 1 << 17,
    PGO = 1 << 18,
    Exp = 1 << 19,
  };
  ulittle16_t Machine; // CPUType
  ulittle16_t VersionFrontendMajor;
  ulittle16_t VersionFrontendMinor;
  ulittle16_t VersionFrontendBuild;
  ulittle16_t VersionFrontendQFE;
  ulittle16_t VersionBackendMajor;
  ulittle16_t VersionBackendMinor;
  ulittle16_t VersionBackendBuild;
  ulittle16_t VersionBackendQFE;
  // VersionString: The null-terminated version string follows.
};

// S_FRAMEPROC
struct FrameProcSym {
  ulittle32_t TotalFrameBytes;
  ulittle32_t PaddingFrameBytes;
  ulittle32_t OffsetToPadding;
  ulittle32_t BytesOfCalleeSavedRegisters;
  ulittle32_t OffsetOfExceptionHandler;
  ulittle16_t SectionIdOfExceptionHandler;
  ulittle32_t Flags;
};

// S_CALLSITEINFO
struct CallSiteInfoSym {
  ulittle32_t CodeOffset;
  ulittle16_t Segment;
  ulittle16_t Reserved;
  TypeIndex Type;
};

// S_HEAPALLOCSITE
struct HeapAllocationSiteSym {
  ulittle32_t CodeOffset;
  ulittle16_t Segment;
  ulittle16_t CallInstructionSize;
  TypeIndex Type;
};

// S_FRAMECOOKIE
struct FrameCookieSym {
  ulittle32_t CodeOffset;
  ulittle16_t Register;
  ulittle16_t CookieKind;

  enum : uint16_t {
    Copy,
    XorStackPointer,
    XorFramePointer,
    XorR13,
  };
};

// S_UDT, S_COBOLUDT
struct UDTSym {
  TypeIndex Type; // Type of the UDT
  // Name: The null-terminated name follows.
};

// S_BUILDINFO
struct BuildInfoSym {
  ulittle32_t BuildId;
};

// S_BPREL32
struct BPRelativeSym {
  ulittle32_t Offset; // Offset from the base pointer register
  TypeIndex Type;     // Type of the variable
  // Name: The null-terminated name follows.
};

// S_REGREL32
struct RegRelativeSym {
  ulittle32_t Offset;   // Offset from the register
  TypeIndex Type;       // Type of the variable
  ulittle16_t Register; // Register to which the variable is relative
  // Name: The null-terminated name follows.
};

// S_CONSTANT, S_MANCONSTANT
struct ConstantSym {
  TypeIndex Type;
  // Value: The value of the constant.
  // Name: The null-terminated name follows.
};

// S_LDATA32, S_GDATA32, S_LMANDATA, S_GMANDATA
struct DataSym {
  TypeIndex Type;
  ulittle32_t DataOffset;
  ulittle16_t Segment;
  // Name: The null-terminated name follows.
};

// S_LTHREAD32, S_GTHREAD32
struct ThreadLocalDataSym {
  TypeIndex Type;
  ulittle32_t DataOffset;
  ulittle16_t Segment;
  // Name: The null-terminated name follows.
};

} // namespace codeview
} // namespace llvm

#endif
