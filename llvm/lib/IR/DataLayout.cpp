//===- DataLayout.cpp - Data size & alignment routines ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines layout properties related to datatype size/offset/alignment
// information.
//
// This structure should be created once, filled in if the defaults are not
// correct and then passed around by const&.  None of the members functions
// require modification to the object.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DataLayout.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemAlloc.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/TargetParser/ARMTargetParser.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <utility>

using namespace llvm;

//===----------------------------------------------------------------------===//
// Support for StructLayout
//===----------------------------------------------------------------------===//

StructLayout::StructLayout(StructType *ST, const DataLayout &DL)
    : StructSize(TypeSize::getFixed(0)) {
  assert(!ST->isOpaque() && "Cannot get layout of opaque structs");
  IsPadded = false;
  NumElements = ST->getNumElements();

  // Loop over each of the elements, placing them in memory.
  for (unsigned i = 0, e = NumElements; i != e; ++i) {
    Type *Ty = ST->getElementType(i);
    if (i == 0 && Ty->isScalableTy())
      StructSize = TypeSize::getScalable(0);

    const Align TyAlign = ST->isPacked() ? Align(1) : DL.getABITypeAlign(Ty);

    // Add padding if necessary to align the data element properly.
    // Currently the only structure with scalable size will be the homogeneous
    // scalable vector types. Homogeneous scalable vector types have members of
    // the same data type so no alignment issue will happen. The condition here
    // assumes so and needs to be adjusted if this assumption changes (e.g. we
    // support structures with arbitrary scalable data type, or structure that
    // contains both fixed size and scalable size data type members).
    if (!StructSize.isScalable() && !isAligned(TyAlign, StructSize)) {
      IsPadded = true;
      StructSize = TypeSize::getFixed(alignTo(StructSize, TyAlign));
    }

    // Keep track of maximum alignment constraint.
    StructAlignment = std::max(TyAlign, StructAlignment);

    getMemberOffsets()[i] = StructSize;
    // Consume space for this data item
    StructSize += DL.getTypeAllocSize(Ty);
  }

  // Add padding to the end of the struct so that it could be put in an array
  // and all array elements would be aligned correctly.
  if (!StructSize.isScalable() && !isAligned(StructAlignment, StructSize)) {
    IsPadded = true;
    StructSize = TypeSize::getFixed(alignTo(StructSize, StructAlignment));
  }
}

/// getElementContainingOffset - Given a valid offset into the structure,
/// return the structure index that contains it.
unsigned StructLayout::getElementContainingOffset(uint64_t FixedOffset) const {
  assert(!StructSize.isScalable() &&
         "Cannot get element at offset for structure containing scalable "
         "vector types");
  TypeSize Offset = TypeSize::getFixed(FixedOffset);
  ArrayRef<TypeSize> MemberOffsets = getMemberOffsets();

  const auto *SI = llvm::upper_bound(MemberOffsets, Offset,
                                     [](TypeSize LHS, TypeSize RHS) -> bool {
                                       return TypeSize::isKnownLT(LHS, RHS);
                                     });
  assert(SI != MemberOffsets.begin() && "Offset not in structure type!");
  --SI;
  assert(TypeSize::isKnownLE(*SI, Offset) && "upper_bound didn't work");
  assert(
      (SI == MemberOffsets.begin() || TypeSize::isKnownLE(*(SI - 1), Offset)) &&
      (SI + 1 == MemberOffsets.end() ||
       TypeSize::isKnownGT(*(SI + 1), Offset)) &&
      "Upper bound didn't work!");

  // Multiple fields can have the same offset if any of them are zero sized.
  // For example, in { i32, [0 x i32], i32 }, searching for offset 4 will stop
  // at the i32 element, because it is the last element at that offset.  This is
  // the right one to return, because anything after it will have a higher
  // offset, implying that this element is non-empty.
  return SI - MemberOffsets.begin();
}

namespace {

class StructLayoutMap {
  using LayoutInfoTy = DenseMap<StructType *, StructLayout *>;
  LayoutInfoTy LayoutInfo;

public:
  ~StructLayoutMap() {
    // Remove any layouts.
    for (const auto &I : LayoutInfo) {
      StructLayout *Value = I.second;
      Value->~StructLayout();
      free(Value);
    }
  }

  StructLayout *&operator[](StructType *STy) { return LayoutInfo[STy]; }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
//                       DataLayout Class Implementation
//===----------------------------------------------------------------------===//

bool DataLayout::PrimitiveSpec::operator==(const PrimitiveSpec &Other) const {
  return BitWidth == Other.BitWidth && ABIAlign == Other.ABIAlign &&
         PrefAlign == Other.PrefAlign;
}

bool DataLayout::PointerSpec::operator==(const PointerSpec &Other) const {
  return AddrSpace == Other.AddrSpace && BitWidth == Other.BitWidth &&
         ABIAlign == Other.ABIAlign && PrefAlign == Other.PrefAlign &&
         IndexBitWidth == Other.IndexBitWidth &&
         IsNonIntegral == Other.IsNonIntegral;
}

namespace {
/// Predicate to sort primitive specs by bit width.
struct LessPrimitiveBitWidth {
  bool operator()(const DataLayout::PrimitiveSpec &LHS,
                  unsigned RHSBitWidth) const {
    return LHS.BitWidth < RHSBitWidth;
  }
};

/// Predicate to sort pointer specs by address space number.
struct LessPointerAddrSpace {
  bool operator()(const DataLayout::PointerSpec &LHS,
                  unsigned RHSAddrSpace) const {
    return LHS.AddrSpace < RHSAddrSpace;
  }
};
} // namespace

static std::string computeARMDataLayout(const Triple &TT, StringRef ABIName) {
  auto ABI = ARM::computeTargetABI(TT, ABIName);
  std::string Ret;

  if (TT.isLittleEndian())
    // Little endian.
    Ret += "e";
  else
    // Big endian.
    Ret += "E";

  Ret += DataLayout::getManglingComponent(TT);

  // Pointers are 32 bits and aligned to 32 bits.
  Ret += "-p:32:32";

  // Function pointers are aligned to 8 bits (because the LSB stores the
  // ARM/Thumb state).
  Ret += "-Fi8";

  // ABIs other than APCS have 64 bit integers with natural alignment.
  if (ABI != ARM::ARM_ABI_APCS)
    Ret += "-i64:64";

  // We have 64 bits floats. The APCS ABI requires them to be aligned to 32
  // bits, others to 64 bits. We always try to align to 64 bits.
  if (ABI == ARM::ARM_ABI_APCS)
    Ret += "-f64:32:64";

  // We have 128 and 64 bit vectors. The APCS ABI aligns them to 32 bits, others
  // to 64. We always ty to give them natural alignment.
  if (ABI == ARM::ARM_ABI_APCS)
    Ret += "-v64:32:64-v128:32:128";
  else if (ABI != ARM::ARM_ABI_AAPCS16)
    Ret += "-v128:64:128";

  // Try to align aggregates to 32 bits (the default is 64 bits, which has no
  // particular hardware support on 32-bit ARM).
  Ret += "-a:0:32";

  // Integer registers are 32 bits.
  Ret += "-n32";

  // The stack is 64 bit aligned on AAPCS and 32 bit aligned everywhere else.
  if (ABI == ARM::ARM_ABI_AAPCS16)
    Ret += "-S128";
  else if (ABI == ARM::ARM_ABI_AAPCS)
    Ret += "-S64";
  else
    Ret += "-S32";

  return Ret;
}

// Helper function to build a DataLayout string
static std::string computeAArch64DataLayout(const Triple &TT) {
  if (TT.isOSBinFormatMachO()) {
    if (TT.getArch() == Triple::aarch64_32)
      return "e-m:o-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-"
             "n32:64-S128-Fn32";
    return "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-"
           "Fn32";
  }
  if (TT.isOSBinFormatCOFF())
    return "e-m:w-p270:32:32-p271:32:32-p272:64:64-p:64:64-i32:32-i64:64-i128:"
           "128-n32:64-S128-Fn32";
  std::string Endian = TT.isLittleEndian() ? "e" : "E";
  std::string Ptr32 = TT.getEnvironment() == Triple::GNUILP32 ? "-p:32:32" : "";
  return Endian + "-m:e" + Ptr32 +
         "-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-"
         "n32:64-S128-Fn32";
}

// DataLayout: little or big endian
static std::string computeBPFDataLayout(const Triple &TT) {
  if (TT.getArch() == Triple::bpfeb)
    return "E-m:e-p:64:64-i64:64-i128:128-n32:64-S128";
  else
    return "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128";
}

static std::string computeCSKYDataLayout(const Triple &TT) {
  std::string Ret;

  // Only support little endian for now.
  // TODO: Add support for big endian.
  Ret += "e";

  // CSKY is always 32-bit target with the CSKYv2 ABI as prefer now.
  // It's a 4-byte aligned stack with ELF mangling only.
  Ret += "-m:e-S32-p:32:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:32"
         "-v128:32:32-a:0:32-Fi32-n32";

  return Ret;
}

static std::string computeLoongArchDataLayout(const Triple &TT) {
  if (TT.isArch64Bit())
    return "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128";
  assert(TT.isArch32Bit() && "only LA32 and LA64 are currently supported");
  return "e-m:e-p:32:32-i64:64-n32-S128";
}

static std::string computeM68kDataLayout(const Triple &TT) {
  std::string Ret = "";
  // M68k is Big Endian
  Ret += "E";

  // FIXME how to wire it with the used object format?
  Ret += "-m:e";

  // M68k pointers are always 32 bit wide even for 16-bit CPUs.
  // The ABI only specifies 16-bit alignment.
  // On at least the 68020+ with a 32-bit bus, there is a performance benefit
  // to having 32-bit alignment.
  Ret += "-p:32:16:32";

  // Bytes do not require special alignment, words are word aligned and
  // long words are word aligned at minimum.
  Ret += "-i8:8:8-i16:16:16-i32:16:32";

  // FIXME no floats at the moment

  // The registers can hold 8, 16, 32 bits
  Ret += "-n8:16:32";

  Ret += "-a:0:16-S16";

  return Ret;
}

namespace {
enum class MipsABI {
  Unknown,
  O32,
  N32,
  N64
};
}

// FIXME: This duplicates MipsABIInfo::computeTargetABI, but duplicating this is
// preferable to violating layering rules. Ideally that information should live
// in LLVM TargetParser, but for now we just duplicate some ABI name string
// logic for simplicity.
static MipsABI getMipsABI(const Triple &TT, StringRef ABIName) {
  if (ABIName.starts_with("o32"))
    return MipsABI::O32;
  if (ABIName.starts_with("n32"))
    return MipsABI::N32;
  if (ABIName.starts_with("n64"))
    return MipsABI::N64;
  if (TT.isABIN32())
    return MipsABI::N32;
  assert(ABIName.empty() && "Unknown ABI option for MIPS");

  if (TT.isMIPS64())
    return MipsABI::N64;
  return MipsABI::O32;
}

static std::string computeMipsDataLayout(const Triple &TT, StringRef ABIName) {
  std::string Ret;
  MipsABI ABI = getMipsABI(TT, ABIName);

  // There are both little and big endian mips.
  if (TT.isLittleEndian())
    Ret += "e";
  else
    Ret += "E";

  if (ABI == MipsABI::O32)
    Ret += "-m:m";
  else
    Ret += "-m:e";

  // Pointers are 32 bit on some ABIs.
  if (ABI != MipsABI::N64)
    Ret += "-p:32:32";

  // 8 and 16 bit integers only need to have natural alignment, but try to
  // align them to 32 bits. 64 bit integers have natural alignment.
  Ret += "-i8:8:32-i16:16:32-i64:64";

  // 32 bit registers are always available and the stack is at least 64 bit
  // aligned. On N64 64 bit registers are also available and the stack is
  // 128 bit aligned.
  if (ABI == MipsABI::N64 || ABI == MipsABI::N32)
    Ret += "-i128:128-n32:64-S128";
  else
    Ret += "-n32-S64";

  return Ret;
}

static std::string computePowerDataLayout(const Triple &T) {
  bool is64Bit = T.getArch() == Triple::ppc64 || T.getArch() == Triple::ppc64le;
  std::string Ret;

  // Most PPC* platforms are big endian, PPC(64)LE is little endian.
  if (T.isLittleEndian())
    Ret = "e";
  else
    Ret = "E";

  Ret += DataLayout::getManglingComponent(T);

  // PPC32 has 32 bit pointers. The PS3 (OS Lv2) is a PPC64 machine with 32 bit
  // pointers.
  if (!is64Bit || T.getOS() == Triple::Lv2)
    Ret += "-p:32:32";

  // If the target ABI uses function descriptors, then the alignment of function
  // pointers depends on the alignment used to emit the descriptor. Otherwise,
  // function pointers are aligned to 32 bits because the instructions must be.
  if ((T.getArch() == Triple::ppc64 && !T.isPPC64ELFv2ABI())) {
    Ret += "-Fi64";
  } else if (T.isOSAIX()) {
    Ret += is64Bit ? "-Fi64" : "-Fi32";
  } else {
    Ret += "-Fn32";
  }

  // Note, the alignment values for f64 and i64 on ppc64 in Darwin
  // documentation are wrong; these are correct (i.e. "what gcc does").
  Ret += "-i64:64";

  // PPC64 has 32 and 64 bit registers, PPC32 has only 32 bit ones.
  if (is64Bit)
    Ret += "-i128:128-n32:64";
  else
    Ret += "-n32";

  // Specify the vector alignment explicitly. For v256i1 and v512i1, the
  // calculated alignment would be 256*alignment(i1) and 512*alignment(i1),
  // which is 256 and 512 bytes - way over aligned.
  if (is64Bit && (T.isOSAIX() || T.isOSLinux()))
    Ret += "-S128-v256:256:256-v512:512:512";

  return Ret;
}

static std::string computeAMDDataLayout(const Triple &TT) {
  if (TT.getArch() == Triple::r600) {
    // 32-bit pointers.
    return "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
           "-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1";
  }

  // 32-bit private, local, and region pointers. 64-bit global, constant and
  // flat. 160-bit non-integral fat buffer pointers that include a 128-bit
  // buffer descriptor and a 32-bit offset, which are indexed by 32-bit values
  // (address space 7), and 128-bit non-integral buffer resourcees (address
  // space 8) which cannot be non-trivilally accessed by LLVM memory operations
  // like getelementptr.
  return "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
         "-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-"
         "v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-"
         "v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9";
}

static std::string computeRISCVDataLayout(const Triple &TT, StringRef ABIName) {
  std::string Ret;

  if (TT.isLittleEndian())
    Ret += "e";
  else
    Ret += "E";

  Ret += "-m:e";

  // Pointer and integer sizes.
  if (TT.isArch64Bit()) {
    Ret += "-p:64:64-i64:64-i128:128";
    Ret += "-n32:64";
  } else {
    assert(TT.isArch32Bit() && "only RV32 and RV64 are currently supported");
    Ret += "-p:32:32-i64:64";
    Ret += "-n32";
  }

  // Stack alignment based on ABI.
  StringRef ABI = ABIName;
  if (ABI == "ilp32e")
    Ret += "-S32";
  else if (ABI == "lp64e")
    Ret += "-S64";
  else
    Ret += "-S128";

  return Ret;
}

static std::string computeSparcDataLayout(const Triple &T) {
  const bool is64Bit = T.isSPARC64();

  // Sparc is typically big endian, but some are little.
  std::string Ret = T.getArch() == Triple::sparcel ? "e" : "E";
  Ret += "-m:e";

  // Some ABIs have 32bit pointers.
  if (!is64Bit)
    Ret += "-p:32:32";

  // Alignments for 64 bit integers.
  Ret += "-i64:64";

  // Alignments for 128 bit integers.
  // This is not specified in the ABI document but is the de facto standard.
  Ret += "-i128:128";

  // On SparcV9 128 floats are aligned to 128 bits, on others only to 64.
  // On SparcV9 registers can hold 64 or 32 bits, on others only 32.
  if (is64Bit)
    Ret += "-n32:64";
  else
    Ret += "-f128:64-n32";

  if (is64Bit)
    Ret += "-S128";
  else
    Ret += "-S64";

  return Ret;
}

static std::string computeSystemZDataLayout(const Triple &TT) {
  std::string Ret;

  // Big endian.
  Ret += "E";

  // Data mangling.
  Ret += DataLayout::getManglingComponent(TT);

  // Special features for z/OS.
  if (TT.isOSzOS()) {
    if (TT.isArch64Bit()) {
      // Custom address space for ptr32.
      Ret += "-p1:32:32";
    }
  }

  // Make sure that global data has at least 16 bits of alignment by
  // default, so that we can refer to it using LARL.  We don't have any
  // special requirements for stack variables though.
  Ret += "-i1:8:16-i8:8:16";

  // 64-bit integers are naturally aligned.
  Ret += "-i64:64";

  // 128-bit floats are aligned only to 64 bits.
  Ret += "-f128:64";

  // The DataLayout string always holds a vector alignment of 64 bits, see
  // comment in clang/lib/Basic/Targets/SystemZ.h.
  Ret += "-v128:64";

  // We prefer 16 bits of aligned for all globals; see above.
  Ret += "-a:8:16";

  // Integer registers are 32 or 64 bits.
  Ret += "-n32:64";

  return Ret;
}

static std::string computeX86DataLayout(const Triple &TT) {
  // X86 is little endian
  std::string Ret = "e";

  Ret += DataLayout::getManglingComponent(TT);
  // X86 and x32 have 32 bit pointers.
  if (!TT.isArch64Bit() || TT.isX32())
    Ret += "-p:32:32";

  // Address spaces for 32 bit signed, 32 bit unsigned, and 64 bit pointers.
  Ret += "-p270:32:32-p271:32:32-p272:64:64";

  // Some ABIs align 64 bit integers and doubles to 64 bits, others to 32.
  // 128 bit integers are not specified in the 32-bit ABIs but are used
  // internally for lowering f128, so we match the alignment to that.
  if (TT.isArch64Bit() || TT.isOSWindows())
    Ret += "-i64:64-i128:128";
  else if (TT.isOSIAMCU())
    Ret += "-i64:32-f64:32";
  else
    Ret += "-i128:128-f64:32:64";

  // Some ABIs align long double to 128 bits, others to 32.
  if (TT.isOSIAMCU())
    ; // No f80
  else if (TT.isArch64Bit() || TT.isOSDarwin() || TT.isWindowsMSVCEnvironment())
    Ret += "-f80:128";
  else
    Ret += "-f80:32";

  if (TT.isOSIAMCU())
    Ret += "-f128:32";

  // The registers can hold 8, 16, 32 or, in x86-64, 64 bits.
  if (TT.isArch64Bit())
    Ret += "-n8:16:32:64";
  else
    Ret += "-n8:16:32";

  // The stack is aligned to 32 bits on some ABIs and 128 bits on others.
  if ((!TT.isArch64Bit() && TT.isOSWindows()) || TT.isOSIAMCU())
    Ret += "-a:0:32-S32";
  else
    Ret += "-S128";

  return Ret;
}


static cl::opt<bool> NVPTXUseShortPointers(
    "nvptx-short-ptr",
    cl::desc(
        "Use 32-bit pointers for accessing const/local/shared address spaces."),
    cl::init(false), cl::Hidden);

static std::string computeNVPTXDataLayout(const Triple &T) {
  std::string Ret = "e";

  // Tensor Memory (addrspace:6) is always 32-bits.
  // Distributed Shared Memory (addrspace:7) follows shared memory
  // (addrspace:3).
  if (!T.isArch64Bit())
    Ret += "-p:32:32-p6:32:32-p7:32:32";
  else if (NVPTXUseShortPointers)
    Ret += "-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32";
  else
    Ret += "-p6:32:32";

  Ret += "-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64";

  return Ret;
}

static std::string computeSPIRVDataLayout(const Triple &TT) {
  const auto Arch = TT.getArch();
  // TODO: this probably needs to be revisited:
  // Logical SPIR-V has no pointer size, so any fixed pointer size would be
  // wrong. The choice to default to 32 or 64 is just motivated by another
  // memory model used for graphics: PhysicalStorageBuffer64. But it shouldn't
  // mean anything.
  if (Arch == Triple::spirv32)
    return "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-"
           "v256:256-v512:512-v1024:1024-n8:16:32:64-G1";
  if (Arch == Triple::spirv)
    return "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-"
           "v512:512-v1024:1024-n8:16:32:64-G10";
  if (TT.getVendor() == Triple::VendorType::AMD &&
      TT.getOS() == Triple::OSType::AMDHSA)
    return "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-"
           "v512:512-v1024:1024-n32:64-S32-G1-P4-A0";
  return "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-"
         "v512:512-v1024:1024-n8:16:32:64-G1";
}

static std::string computeLanaiDataLayout() {
  // Data layout (keep in sync with clang/lib/Basic/Targets.cpp)
  return "E"        // Big endian
         "-m:e"     // ELF name manging
         "-p:32:32" // 32-bit pointers, 32 bit aligned
         "-i64:64"  // 64 bit integers, 64 bit aligned
         "-a:0:32"  // 32 bit alignment of objects of aggregate type
         "-n32"     // 32 bit native integer width
         "-S64";    // 64 bit natural stack alignment
}

static std::string computeWebAssemblyDataLayout(const Triple &TT) {
  return TT.isArch64Bit()
             ? (TT.isOSEmscripten() ? "e-m:e-p:64:64-p10:8:8-p20:8:8-i64:64-"
                                      "i128:128-f128:64-n32:64-S128-ni:1:10:20"
                                    : "e-m:e-p:64:64-p10:8:8-p20:8:8-i64:64-"
                                      "i128:128-n32:64-S128-ni:1:10:20")
             : (TT.isOSEmscripten() ? "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-"
                                      "i128:128-f128:64-n32:64-S128-ni:1:10:20"
                                    : "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-"
                                      "i128:128-n32:64-S128-ni:1:10:20");
}

static std::string computeVEDataLayout(const Triple &T) {
  // Aurora VE is little endian
  std::string Ret = "e";

  // Use ELF mangling
  Ret += "-m:e";

  // Alignments for 64 bit integers.
  Ret += "-i64:64";

  // VE supports 32 bit and 64 bits integer on registers
  Ret += "-n32:64";

  // Stack alignment is 128 bits
  Ret += "-S128";

  // Vector alignments are 64 bits
  // Need to define all of them.  Otherwise, each alignment becomes
  // the size of each data by default.
  Ret += "-v64:64:64"; // for v2f32
  Ret += "-v128:64:64";
  Ret += "-v256:64:64";
  Ret += "-v512:64:64";
  Ret += "-v1024:64:64";
  Ret += "-v2048:64:64";
  Ret += "-v4096:64:64";
  Ret += "-v8192:64:64";
  Ret += "-v16384:64:64"; // for v256f64

  return Ret;
}

// static
std::string DataLayout::computeStringForTriple(const Triple &T,
                                               StringRef ABIName) {
  switch (T.getArch()) {
  case Triple::arm:
  case Triple::armeb:
  case Triple::thumb:
  case Triple::thumbeb:
    return computeARMDataLayout(T, ABIName);
  case Triple::aarch64:
  case Triple::aarch64_be:
  case Triple::aarch64_32:
    return computeAArch64DataLayout(T);
  case Triple::arc:
    return "e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-"
           "f32:32:32-i64:32-f64:32-a:0:32-n32";
  case Triple::avr:
    return "e-P1-p:16:8-i8:8-i16:8-i32:8-i64:8-f32:8-f64:8-n8:16-a:8";
  case Triple::bpfel:
  case Triple::bpfeb:
    return computeBPFDataLayout(T);
  case Triple::csky:
    return computeCSKYDataLayout(T);
  case Triple::dxil:
    return "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-"
           "f32:32-f64:64-n8:16:32:64";
  case Triple::hexagon:
    return "e-m:e-p:32:32:32-a:0-n16:32-"
           "i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-"
           "v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048";
  case Triple::loongarch32:
  case Triple::loongarch64:
    return computeLoongArchDataLayout(T);
  case Triple::m68k:
    return computeM68kDataLayout(T);
  case Triple::mips:
  case Triple::mipsel:
  case Triple::mips64:
  case Triple::mips64el:
    return computeMipsDataLayout(T, ABIName);
  case Triple::msp430:
    return "e-m:e-p:16:16-i32:16-i64:16-f32:16-f64:16-a:8-n8:16-S16";
  case Triple::ppc:
  case Triple::ppcle:
  case Triple::ppc64:
  case Triple::ppc64le:
    return computePowerDataLayout(T);
  case Triple::r600:
  case Triple::amdgcn:
    return computeAMDDataLayout(T);
  case Triple::riscv32:
  case Triple::riscv64:
  case Triple::riscv32be:
  case Triple::riscv64be:
    return computeRISCVDataLayout(T, ABIName);
  case Triple::sparc:
  case Triple::sparcv9:
  case Triple::sparcel:
    return computeSparcDataLayout(T);
  case Triple::systemz:
    return computeSystemZDataLayout(T);
  case Triple::tce:
  case Triple::tcele:
  case Triple::x86:
  case Triple::x86_64:
    return computeX86DataLayout(T);
  case Triple::xcore:
  case Triple::xtensa:
    return "e-m:e-p:32:32-i8:8:32-i16:16:32-i64:64-n32";
  case Triple::nvptx:
  case Triple::nvptx64:
    return computeNVPTXDataLayout(T);
  case Triple::spir:
  case Triple::spir64:
  case Triple::spirv:
  case Triple::spirv32:
  case Triple::spirv64:
    return computeSPIRVDataLayout(T);
  case Triple::lanai:
    return computeLanaiDataLayout();
  case Triple::wasm32:
  case Triple::wasm64:
    return computeWebAssemblyDataLayout(T);
  case Triple::ve:
    return computeVEDataLayout(T);

  case Triple::amdil:
  case Triple::amdil64:
  case Triple::hsail:
  case Triple::hsail64:
  case Triple::kalimba:
  case Triple::shave:
  case Triple::renderscript32:
  case Triple::renderscript64:
    // These are all virtual ISAs with no LLVM backend, and therefore no fixed
    // LLVM data layout.
    return "";

  case Triple::UnknownArch:
    return "";
  }
  return "";
}

const char *DataLayout::getManglingComponent(const Triple &T) {
  if (T.isOSBinFormatGOFF())
    return "-m:l";
  if (T.isOSBinFormatMachO())
    return "-m:o";
  if ((T.isOSWindows() || T.isUEFI()) && T.isOSBinFormatCOFF())
    return T.getArch() == Triple::x86 ? "-m:x" : "-m:w";
  if (T.isOSBinFormatXCOFF())
    return "-m:a";
  return "-m:e";
}

// Default primitive type specifications.
// NOTE: These arrays must be sorted by type bit width.
constexpr DataLayout::PrimitiveSpec DefaultIntSpecs[] = {
    {8, Align::Constant<1>(), Align::Constant<1>()},  // i8:8:8
    {16, Align::Constant<2>(), Align::Constant<2>()}, // i16:16:16
    {32, Align::Constant<4>(), Align::Constant<4>()}, // i32:32:32
    {64, Align::Constant<4>(), Align::Constant<8>()}, // i64:32:64
};
constexpr DataLayout::PrimitiveSpec DefaultFloatSpecs[] = {
    {16, Align::Constant<2>(), Align::Constant<2>()},    // f16:16:16
    {32, Align::Constant<4>(), Align::Constant<4>()},    // f32:32:32
    {64, Align::Constant<8>(), Align::Constant<8>()},    // f64:64:64
    {128, Align::Constant<16>(), Align::Constant<16>()}, // f128:128:128
};
constexpr DataLayout::PrimitiveSpec DefaultVectorSpecs[] = {
    {64, Align::Constant<8>(), Align::Constant<8>()},    // v64:64:64
    {128, Align::Constant<16>(), Align::Constant<16>()}, // v128:128:128
};

// Default pointer type specifications.
constexpr DataLayout::PointerSpec DefaultPointerSpecs[] = {
    // p0:64:64:64:64
    {0, 64, Align::Constant<8>(), Align::Constant<8>(), 64, false},
};

DataLayout::DataLayout()
    : IntSpecs(ArrayRef(DefaultIntSpecs)),
      FloatSpecs(ArrayRef(DefaultFloatSpecs)),
      VectorSpecs(ArrayRef(DefaultVectorSpecs)),
      PointerSpecs(ArrayRef(DefaultPointerSpecs)) {}

DataLayout::DataLayout(StringRef LayoutString) : DataLayout() {
  if (Error Err = parseLayoutString(LayoutString))
    report_fatal_error(std::move(Err));
}

DataLayout &DataLayout::operator=(const DataLayout &Other) {
  delete static_cast<StructLayoutMap *>(LayoutMap);
  LayoutMap = nullptr;
  StringRepresentation = Other.StringRepresentation;
  BigEndian = Other.BigEndian;
  AllocaAddrSpace = Other.AllocaAddrSpace;
  ProgramAddrSpace = Other.ProgramAddrSpace;
  DefaultGlobalsAddrSpace = Other.DefaultGlobalsAddrSpace;
  StackNaturalAlign = Other.StackNaturalAlign;
  FunctionPtrAlign = Other.FunctionPtrAlign;
  TheFunctionPtrAlignType = Other.TheFunctionPtrAlignType;
  ManglingMode = Other.ManglingMode;
  LegalIntWidths = Other.LegalIntWidths;
  IntSpecs = Other.IntSpecs;
  FloatSpecs = Other.FloatSpecs;
  VectorSpecs = Other.VectorSpecs;
  PointerSpecs = Other.PointerSpecs;
  StructABIAlignment = Other.StructABIAlignment;
  StructPrefAlignment = Other.StructPrefAlignment;
  return *this;
}

bool DataLayout::operator==(const DataLayout &Other) const {
  // NOTE: StringRepresentation might differ, it is not canonicalized.
  return BigEndian == Other.BigEndian &&
         AllocaAddrSpace == Other.AllocaAddrSpace &&
         ProgramAddrSpace == Other.ProgramAddrSpace &&
         DefaultGlobalsAddrSpace == Other.DefaultGlobalsAddrSpace &&
         StackNaturalAlign == Other.StackNaturalAlign &&
         FunctionPtrAlign == Other.FunctionPtrAlign &&
         TheFunctionPtrAlignType == Other.TheFunctionPtrAlignType &&
         ManglingMode == Other.ManglingMode &&
         LegalIntWidths == Other.LegalIntWidths && IntSpecs == Other.IntSpecs &&
         FloatSpecs == Other.FloatSpecs && VectorSpecs == Other.VectorSpecs &&
         PointerSpecs == Other.PointerSpecs &&
         StructABIAlignment == Other.StructABIAlignment &&
         StructPrefAlignment == Other.StructPrefAlignment;
}

Expected<DataLayout> DataLayout::parse(StringRef LayoutString) {
  DataLayout Layout;
  if (Error Err = Layout.parseLayoutString(LayoutString))
    return std::move(Err);
  return Layout;
}

static Error createSpecFormatError(Twine Format) {
  return createStringError("malformed specification, must be of the form \"" +
                           Format + "\"");
}

/// Attempts to parse an address space component of a specification.
static Error parseAddrSpace(StringRef Str, unsigned &AddrSpace) {
  if (Str.empty())
    return createStringError("address space component cannot be empty");

  if (!to_integer(Str, AddrSpace, 10) || !isUInt<24>(AddrSpace))
    return createStringError("address space must be a 24-bit integer");

  return Error::success();
}

/// Attempts to parse a size component of a specification.
static Error parseSize(StringRef Str, unsigned &BitWidth,
                       StringRef Name = "size") {
  if (Str.empty())
    return createStringError(Name + " component cannot be empty");

  if (!to_integer(Str, BitWidth, 10) || BitWidth == 0 || !isUInt<24>(BitWidth))
    return createStringError(Name + " must be a non-zero 24-bit integer");

  return Error::success();
}

/// Attempts to parse an alignment component of a specification.
///
/// On success, returns the value converted to byte amount in \p Alignment.
/// If the value is zero and \p AllowZero is true, \p Alignment is set to one.
///
/// Return an error in a number of cases:
/// - \p Str is empty or contains characters other than decimal digits;
/// - the value is zero and \p AllowZero is false;
/// - the value is too large;
/// - the value is not a multiple of the byte width;
/// - the value converted to byte amount is not not a power of two.
static Error parseAlignment(StringRef Str, Align &Alignment, StringRef Name,
                            bool AllowZero = false) {
  if (Str.empty())
    return createStringError(Name + " alignment component cannot be empty");

  unsigned Value;
  if (!to_integer(Str, Value, 10) || !isUInt<16>(Value))
    return createStringError(Name + " alignment must be a 16-bit integer");

  if (Value == 0) {
    if (!AllowZero)
      return createStringError(Name + " alignment must be non-zero");
    Alignment = Align(1);
    return Error::success();
  }

  constexpr unsigned ByteWidth = 8;
  if (Value % ByteWidth || !isPowerOf2_32(Value / ByteWidth))
    return createStringError(
        Name + " alignment must be a power of two times the byte width");

  Alignment = Align(Value / ByteWidth);
  return Error::success();
}

Error DataLayout::parsePrimitiveSpec(StringRef Spec) {
  // [ifv]<size>:<abi>[:<pref>]
  SmallVector<StringRef, 3> Components;
  char Specifier = Spec.front();
  assert(Specifier == 'i' || Specifier == 'f' || Specifier == 'v');
  Spec.drop_front().split(Components, ':');

  if (Components.size() < 2 || Components.size() > 3)
    return createSpecFormatError(Twine(Specifier) + "<size>:<abi>[:<pref>]");

  // Size. Required, cannot be zero.
  unsigned BitWidth;
  if (Error Err = parseSize(Components[0], BitWidth))
    return Err;

  // ABI alignment.
  Align ABIAlign;
  if (Error Err = parseAlignment(Components[1], ABIAlign, "ABI"))
    return Err;

  if (Specifier == 'i' && BitWidth == 8 && ABIAlign != 1)
    return createStringError("i8 must be 8-bit aligned");

  // Preferred alignment. Optional, defaults to the ABI alignment.
  Align PrefAlign = ABIAlign;
  if (Components.size() > 2)
    if (Error Err = parseAlignment(Components[2], PrefAlign, "preferred"))
      return Err;

  if (PrefAlign < ABIAlign)
    return createStringError(
        "preferred alignment cannot be less than the ABI alignment");

  setPrimitiveSpec(Specifier, BitWidth, ABIAlign, PrefAlign);
  return Error::success();
}

Error DataLayout::parseAggregateSpec(StringRef Spec) {
  // a<size>:<abi>[:<pref>]
  SmallVector<StringRef, 3> Components;
  assert(Spec.front() == 'a');
  Spec.drop_front().split(Components, ':');

  if (Components.size() < 2 || Components.size() > 3)
    return createSpecFormatError("a:<abi>[:<pref>]");

  // According to LangRef, <size> component must be absent altogether.
  // For backward compatibility, allow it to be specified, but require
  // it to be zero.
  if (!Components[0].empty()) {
    unsigned BitWidth;
    if (!to_integer(Components[0], BitWidth, 10) || BitWidth != 0)
      return createStringError("size must be zero");
  }

  // ABI alignment. Required. Can be zero, meaning use one byte alignment.
  Align ABIAlign;
  if (Error Err =
          parseAlignment(Components[1], ABIAlign, "ABI", /*AllowZero=*/true))
    return Err;

  // Preferred alignment. Optional, defaults to the ABI alignment.
  Align PrefAlign = ABIAlign;
  if (Components.size() > 2)
    if (Error Err = parseAlignment(Components[2], PrefAlign, "preferred"))
      return Err;

  if (PrefAlign < ABIAlign)
    return createStringError(
        "preferred alignment cannot be less than the ABI alignment");

  StructABIAlignment = ABIAlign;
  StructPrefAlignment = PrefAlign;
  return Error::success();
}

Error DataLayout::parsePointerSpec(StringRef Spec) {
  // p[<n>]:<size>:<abi>[:<pref>[:<idx>]]
  SmallVector<StringRef, 5> Components;
  assert(Spec.front() == 'p');
  Spec.drop_front().split(Components, ':');

  if (Components.size() < 3 || Components.size() > 5)
    return createSpecFormatError("p[<n>]:<size>:<abi>[:<pref>[:<idx>]]");

  // Address space. Optional, defaults to 0.
  unsigned AddrSpace = 0;
  if (!Components[0].empty())
    if (Error Err = parseAddrSpace(Components[0], AddrSpace))
      return Err;

  // Size. Required, cannot be zero.
  unsigned BitWidth;
  if (Error Err = parseSize(Components[1], BitWidth, "pointer size"))
    return Err;

  // ABI alignment. Required, cannot be zero.
  Align ABIAlign;
  if (Error Err = parseAlignment(Components[2], ABIAlign, "ABI"))
    return Err;

  // Preferred alignment. Optional, defaults to the ABI alignment.
  // Cannot be zero.
  Align PrefAlign = ABIAlign;
  if (Components.size() > 3)
    if (Error Err = parseAlignment(Components[3], PrefAlign, "preferred"))
      return Err;

  if (PrefAlign < ABIAlign)
    return createStringError(
        "preferred alignment cannot be less than the ABI alignment");

  // Index size. Optional, defaults to pointer size. Cannot be zero.
  unsigned IndexBitWidth = BitWidth;
  if (Components.size() > 4)
    if (Error Err = parseSize(Components[4], IndexBitWidth, "index size"))
      return Err;

  if (IndexBitWidth > BitWidth)
    return createStringError(
        "index size cannot be larger than the pointer size");

  setPointerSpec(AddrSpace, BitWidth, ABIAlign, PrefAlign, IndexBitWidth,
                 false);
  return Error::success();
}

Error DataLayout::parseSpecification(
    StringRef Spec, SmallVectorImpl<unsigned> &NonIntegralAddressSpaces) {
  // The "ni" specifier is the only two-character specifier. Handle it first.
  if (Spec.starts_with("ni")) {
    // ni:<address space>[:<address space>]...
    StringRef Rest = Spec.drop_front(2);

    // Drop the first ':', then split the rest of the string the usual way.
    if (!Rest.consume_front(":"))
      return createSpecFormatError("ni:<address space>[:<address space>]...");

    for (StringRef Str : split(Rest, ':')) {
      unsigned AddrSpace;
      if (Error Err = parseAddrSpace(Str, AddrSpace))
        return Err;
      if (AddrSpace == 0)
        return createStringError("address space 0 cannot be non-integral");
      NonIntegralAddressSpaces.push_back(AddrSpace);
    }
    return Error::success();
  }

  // The rest of the specifiers are single-character.
  assert(!Spec.empty() && "Empty specification is handled by the caller");
  char Specifier = Spec.front();

  if (Specifier == 'i' || Specifier == 'f' || Specifier == 'v')
    return parsePrimitiveSpec(Spec);

  if (Specifier == 'a')
    return parseAggregateSpec(Spec);

  if (Specifier == 'p')
    return parsePointerSpec(Spec);

  StringRef Rest = Spec.drop_front();
  switch (Specifier) {
  case 's':
    // Deprecated, but ignoring here to preserve loading older textual llvm
    // ASM file
    break;
  case 'e':
  case 'E':
    if (!Rest.empty())
      return createStringError(
          "malformed specification, must be just 'e' or 'E'");
    BigEndian = Specifier == 'E';
    break;
  case 'n': // Native integer types.
    // n<size>[:<size>]...
    for (StringRef Str : split(Rest, ':')) {
      unsigned BitWidth;
      if (Error Err = parseSize(Str, BitWidth))
        return Err;
      LegalIntWidths.push_back(BitWidth);
    }
    break;
  case 'S': { // Stack natural alignment.
    // S<size>
    if (Rest.empty())
      return createSpecFormatError("S<size>");
    Align Alignment;
    if (Error Err = parseAlignment(Rest, Alignment, "stack natural"))
      return Err;
    StackNaturalAlign = Alignment;
    break;
  }
  case 'F': {
    // F<type><abi>
    if (Rest.empty())
      return createSpecFormatError("F<type><abi>");
    char Type = Rest.front();
    Rest = Rest.drop_front();
    switch (Type) {
    case 'i':
      TheFunctionPtrAlignType = FunctionPtrAlignType::Independent;
      break;
    case 'n':
      TheFunctionPtrAlignType = FunctionPtrAlignType::MultipleOfFunctionAlign;
      break;
    default:
      return createStringError("unknown function pointer alignment type '" +
                               Twine(Type) + "'");
    }
    Align Alignment;
    if (Error Err = parseAlignment(Rest, Alignment, "ABI"))
      return Err;
    FunctionPtrAlign = Alignment;
    break;
  }
  case 'P': { // Function address space.
    if (Rest.empty())
      return createSpecFormatError("P<address space>");
    if (Error Err = parseAddrSpace(Rest, ProgramAddrSpace))
      return Err;
    break;
  }
  case 'A': { // Default stack/alloca address space.
    if (Rest.empty())
      return createSpecFormatError("A<address space>");
    if (Error Err = parseAddrSpace(Rest, AllocaAddrSpace))
      return Err;
    break;
  }
  case 'G': { // Default address space for global variables.
    if (Rest.empty())
      return createSpecFormatError("G<address space>");
    if (Error Err = parseAddrSpace(Rest, DefaultGlobalsAddrSpace))
      return Err;
    break;
  }
  case 'm':
    if (!Rest.consume_front(":") || Rest.empty())
      return createSpecFormatError("m:<mangling>");
    if (Rest.size() > 1)
      return createStringError("unknown mangling mode");
    switch (Rest[0]) {
    default:
      return createStringError("unknown mangling mode");
    case 'e':
      ManglingMode = MM_ELF;
      break;
    case 'l':
      ManglingMode = MM_GOFF;
      break;
    case 'o':
      ManglingMode = MM_MachO;
      break;
    case 'm':
      ManglingMode = MM_Mips;
      break;
    case 'w':
      ManglingMode = MM_WinCOFF;
      break;
    case 'x':
      ManglingMode = MM_WinCOFFX86;
      break;
    case 'a':
      ManglingMode = MM_XCOFF;
      break;
    }
    break;
  default:
    return createStringError("unknown specifier '" + Twine(Specifier) + "'");
  }

  return Error::success();
}

Error DataLayout::parseLayoutString(StringRef LayoutString) {
  StringRepresentation = std::string(LayoutString);

  if (LayoutString.empty())
    return Error::success();

  // Split the data layout string into specifications separated by '-' and
  // parse each specification individually, updating internal data structures.
  SmallVector<unsigned, 8> NonIntegralAddressSpaces;
  for (StringRef Spec : split(LayoutString, '-')) {
    if (Spec.empty())
      return createStringError("empty specification is not allowed");
    if (Error Err = parseSpecification(Spec, NonIntegralAddressSpaces))
      return Err;
  }
  // Mark all address spaces that were qualified as non-integral now. This has
  // to be done later since the non-integral property is not part of the data
  // layout pointer specification.
  for (unsigned AS : NonIntegralAddressSpaces) {
    // If there is no special spec for a given AS, getPointerSpec(AS) returns
    // the spec for AS0, and we then update that to mark it non-integral.
    const PointerSpec &PS = getPointerSpec(AS);
    setPointerSpec(AS, PS.BitWidth, PS.ABIAlign, PS.PrefAlign, PS.IndexBitWidth,
                   true);
  }

  return Error::success();
}

void DataLayout::setPrimitiveSpec(char Specifier, uint32_t BitWidth,
                                  Align ABIAlign, Align PrefAlign) {
  SmallVectorImpl<PrimitiveSpec> *Specs;
  switch (Specifier) {
  default:
    llvm_unreachable("Unexpected specifier");
  case 'i':
    Specs = &IntSpecs;
    break;
  case 'f':
    Specs = &FloatSpecs;
    break;
  case 'v':
    Specs = &VectorSpecs;
    break;
  }

  auto I = lower_bound(*Specs, BitWidth, LessPrimitiveBitWidth());
  if (I != Specs->end() && I->BitWidth == BitWidth) {
    // Update the abi, preferred alignments.
    I->ABIAlign = ABIAlign;
    I->PrefAlign = PrefAlign;
  } else {
    // Insert before I to keep the vector sorted.
    Specs->insert(I, PrimitiveSpec{BitWidth, ABIAlign, PrefAlign});
  }
}

const DataLayout::PointerSpec &
DataLayout::getPointerSpec(uint32_t AddrSpace) const {
  if (AddrSpace != 0) {
    auto I = lower_bound(PointerSpecs, AddrSpace, LessPointerAddrSpace());
    if (I != PointerSpecs.end() && I->AddrSpace == AddrSpace)
      return *I;
  }

  assert(PointerSpecs[0].AddrSpace == 0);
  return PointerSpecs[0];
}

void DataLayout::setPointerSpec(uint32_t AddrSpace, uint32_t BitWidth,
                                Align ABIAlign, Align PrefAlign,
                                uint32_t IndexBitWidth, bool IsNonIntegral) {
  auto I = lower_bound(PointerSpecs, AddrSpace, LessPointerAddrSpace());
  if (I == PointerSpecs.end() || I->AddrSpace != AddrSpace) {
    PointerSpecs.insert(I, PointerSpec{AddrSpace, BitWidth, ABIAlign, PrefAlign,
                                       IndexBitWidth, IsNonIntegral});
  } else {
    I->BitWidth = BitWidth;
    I->ABIAlign = ABIAlign;
    I->PrefAlign = PrefAlign;
    I->IndexBitWidth = IndexBitWidth;
    I->IsNonIntegral = IsNonIntegral;
  }
}

Align DataLayout::getIntegerAlignment(uint32_t BitWidth,
                                      bool abi_or_pref) const {
  auto I = IntSpecs.begin();
  for (; I != IntSpecs.end(); ++I) {
    if (I->BitWidth >= BitWidth)
      break;
  }

  // If we don't have an exact match, use alignment of next larger integer
  // type. If there is none, use alignment of largest integer type by going
  // back one element.
  if (I == IntSpecs.end())
    --I;
  return abi_or_pref ? I->ABIAlign : I->PrefAlign;
}

DataLayout::~DataLayout() { delete static_cast<StructLayoutMap *>(LayoutMap); }

const StructLayout *DataLayout::getStructLayout(StructType *Ty) const {
  if (!LayoutMap)
    LayoutMap = new StructLayoutMap();

  StructLayoutMap *STM = static_cast<StructLayoutMap*>(LayoutMap);
  StructLayout *&SL = (*STM)[Ty];
  if (SL) return SL;

  // Otherwise, create the struct layout.  Because it is variable length, we
  // malloc it, then use placement new.
  StructLayout *L = (StructLayout *)safe_malloc(
      StructLayout::totalSizeToAlloc<TypeSize>(Ty->getNumElements()));

  // Set SL before calling StructLayout's ctor.  The ctor could cause other
  // entries to be added to TheMap, invalidating our reference.
  SL = L;

  new (L) StructLayout(Ty, *this);

  return L;
}

Align DataLayout::getPointerABIAlignment(unsigned AS) const {
  return getPointerSpec(AS).ABIAlign;
}

Align DataLayout::getPointerPrefAlignment(unsigned AS) const {
  return getPointerSpec(AS).PrefAlign;
}

unsigned DataLayout::getPointerSize(unsigned AS) const {
  return divideCeil(getPointerSpec(AS).BitWidth, 8);
}

unsigned DataLayout::getPointerTypeSizeInBits(Type *Ty) const {
  assert(Ty->isPtrOrPtrVectorTy() &&
         "This should only be called with a pointer or pointer vector type");
  Ty = Ty->getScalarType();
  return getPointerSizeInBits(cast<PointerType>(Ty)->getAddressSpace());
}

unsigned DataLayout::getIndexSize(unsigned AS) const {
  return divideCeil(getPointerSpec(AS).IndexBitWidth, 8);
}

unsigned DataLayout::getIndexTypeSizeInBits(Type *Ty) const {
  assert(Ty->isPtrOrPtrVectorTy() &&
         "This should only be called with a pointer or pointer vector type");
  Ty = Ty->getScalarType();
  return getIndexSizeInBits(cast<PointerType>(Ty)->getAddressSpace());
}

/*!
  \param abi_or_pref Flag that determines which alignment is returned. true
  returns the ABI alignment, false returns the preferred alignment.
  \param Ty The underlying type for which alignment is determined.

  Get the ABI (\a abi_or_pref == true) or preferred alignment (\a abi_or_pref
  == false) for the requested type \a Ty.
 */
Align DataLayout::getAlignment(Type *Ty, bool abi_or_pref) const {
  assert(Ty->isSized() && "Cannot getTypeInfo() on a type that is unsized!");
  switch (Ty->getTypeID()) {
  // Early escape for the non-numeric types.
  case Type::LabelTyID:
    return abi_or_pref ? getPointerABIAlignment(0) : getPointerPrefAlignment(0);
  case Type::PointerTyID: {
    unsigned AS = cast<PointerType>(Ty)->getAddressSpace();
    return abi_or_pref ? getPointerABIAlignment(AS)
                       : getPointerPrefAlignment(AS);
    }
  case Type::ArrayTyID:
    return getAlignment(cast<ArrayType>(Ty)->getElementType(), abi_or_pref);

  case Type::StructTyID: {
    // Packed structure types always have an ABI alignment of one.
    if (cast<StructType>(Ty)->isPacked() && abi_or_pref)
      return Align(1);

    // Get the layout annotation... which is lazily created on demand.
    const StructLayout *Layout = getStructLayout(cast<StructType>(Ty));
    const Align Align = abi_or_pref ? StructABIAlignment : StructPrefAlignment;
    return std::max(Align, Layout->getAlignment());
  }
  case Type::IntegerTyID:
    return getIntegerAlignment(Ty->getIntegerBitWidth(), abi_or_pref);
  case Type::HalfTyID:
  case Type::BFloatTyID:
  case Type::FloatTyID:
  case Type::DoubleTyID:
  // PPC_FP128TyID and FP128TyID have different data contents, but the
  // same size and alignment, so they look the same here.
  case Type::PPC_FP128TyID:
  case Type::FP128TyID:
  case Type::X86_FP80TyID: {
    unsigned BitWidth = getTypeSizeInBits(Ty).getFixedValue();
    auto I = lower_bound(FloatSpecs, BitWidth, LessPrimitiveBitWidth());
    if (I != FloatSpecs.end() && I->BitWidth == BitWidth)
      return abi_or_pref ? I->ABIAlign : I->PrefAlign;

    // If we still couldn't find a reasonable default alignment, fall back
    // to a simple heuristic that the alignment is the first power of two
    // greater-or-equal to the store size of the type.  This is a reasonable
    // approximation of reality, and if the user wanted something less
    // less conservative, they should have specified it explicitly in the data
    // layout.
    return Align(PowerOf2Ceil(BitWidth / 8));
  }
  case Type::FixedVectorTyID:
  case Type::ScalableVectorTyID: {
    unsigned BitWidth = getTypeSizeInBits(Ty).getKnownMinValue();
    auto I = lower_bound(VectorSpecs, BitWidth, LessPrimitiveBitWidth());
    if (I != VectorSpecs.end() && I->BitWidth == BitWidth)
      return abi_or_pref ? I->ABIAlign : I->PrefAlign;

    // By default, use natural alignment for vector types. This is consistent
    // with what clang and llvm-gcc do.
    //
    // We're only calculating a natural alignment, so it doesn't have to be
    // based on the full size for scalable vectors. Using the minimum element
    // count should be enough here.
    return Align(PowerOf2Ceil(getTypeStoreSize(Ty).getKnownMinValue()));
  }
  case Type::X86_AMXTyID:
    return Align(64);
  case Type::TargetExtTyID: {
    Type *LayoutTy = cast<TargetExtType>(Ty)->getLayoutType();
    return getAlignment(LayoutTy, abi_or_pref);
  }
  default:
    llvm_unreachable("Bad type for getAlignment!!!");
  }
}

TypeSize DataLayout::getTypeAllocSize(Type *Ty) const {
  switch (Ty->getTypeID()) {
  case Type::ArrayTyID: {
    // The alignment of the array is the alignment of the element, so there
    // is no need for further adjustment.
    auto *ATy = cast<ArrayType>(Ty);
    return ATy->getNumElements() * getTypeAllocSize(ATy->getElementType());
  }
  case Type::StructTyID: {
    const StructLayout *Layout = getStructLayout(cast<StructType>(Ty));
    TypeSize Size = Layout->getSizeInBytes();

    if (cast<StructType>(Ty)->isPacked())
      return Size;

    Align A = std::max(StructABIAlignment, Layout->getAlignment());
    return alignTo(Size, A.value());
  }
  case Type::IntegerTyID: {
    unsigned BitWidth = Ty->getIntegerBitWidth();
    TypeSize Size = TypeSize::getFixed(divideCeil(BitWidth, 8));
    Align A = getIntegerAlignment(BitWidth, /*ABI=*/true);
    return alignTo(Size, A.value());
  }
  case Type::PointerTyID: {
    unsigned AS = Ty->getPointerAddressSpace();
    TypeSize Size = TypeSize::getFixed(getPointerSize(AS));
    return alignTo(Size, getPointerABIAlignment(AS).value());
  }
  case Type::TargetExtTyID: {
    Type *LayoutTy = cast<TargetExtType>(Ty)->getLayoutType();
    return getTypeAllocSize(LayoutTy);
  }
  default:
    return alignTo(getTypeStoreSize(Ty), getABITypeAlign(Ty).value());
  }
}

Align DataLayout::getABITypeAlign(Type *Ty) const {
  return getAlignment(Ty, true);
}

Align DataLayout::getPrefTypeAlign(Type *Ty) const {
  return getAlignment(Ty, false);
}

IntegerType *DataLayout::getIntPtrType(LLVMContext &C,
                                       unsigned AddressSpace) const {
  return IntegerType::get(C, getPointerSizeInBits(AddressSpace));
}

Type *DataLayout::getIntPtrType(Type *Ty) const {
  assert(Ty->isPtrOrPtrVectorTy() &&
         "Expected a pointer or pointer vector type.");
  unsigned NumBits = getPointerTypeSizeInBits(Ty);
  IntegerType *IntTy = IntegerType::get(Ty->getContext(), NumBits);
  if (VectorType *VecTy = dyn_cast<VectorType>(Ty))
    return VectorType::get(IntTy, VecTy);
  return IntTy;
}

Type *DataLayout::getSmallestLegalIntType(LLVMContext &C, unsigned Width) const {
  for (unsigned LegalIntWidth : LegalIntWidths)
    if (Width <= LegalIntWidth)
      return Type::getIntNTy(C, LegalIntWidth);
  return nullptr;
}

unsigned DataLayout::getLargestLegalIntTypeSizeInBits() const {
  auto Max = llvm::max_element(LegalIntWidths);
  return Max != LegalIntWidths.end() ? *Max : 0;
}

IntegerType *DataLayout::getIndexType(LLVMContext &C,
                                      unsigned AddressSpace) const {
  return IntegerType::get(C, getIndexSizeInBits(AddressSpace));
}

Type *DataLayout::getIndexType(Type *Ty) const {
  assert(Ty->isPtrOrPtrVectorTy() &&
         "Expected a pointer or pointer vector type.");
  unsigned NumBits = getIndexTypeSizeInBits(Ty);
  IntegerType *IntTy = IntegerType::get(Ty->getContext(), NumBits);
  if (VectorType *VecTy = dyn_cast<VectorType>(Ty))
    return VectorType::get(IntTy, VecTy);
  return IntTy;
}

int64_t DataLayout::getIndexedOffsetInType(Type *ElemTy,
                                           ArrayRef<Value *> Indices) const {
  int64_t Result = 0;

  generic_gep_type_iterator<Value* const*>
    GTI = gep_type_begin(ElemTy, Indices),
    GTE = gep_type_end(ElemTy, Indices);
  for (; GTI != GTE; ++GTI) {
    Value *Idx = GTI.getOperand();
    if (StructType *STy = GTI.getStructTypeOrNull()) {
      assert(Idx->getType()->isIntegerTy(32) && "Illegal struct idx");
      unsigned FieldNo = cast<ConstantInt>(Idx)->getZExtValue();

      // Get structure layout information...
      const StructLayout *Layout = getStructLayout(STy);

      // Add in the offset, as calculated by the structure layout info...
      Result += Layout->getElementOffset(FieldNo);
    } else {
      if (int64_t ArrayIdx = cast<ConstantInt>(Idx)->getSExtValue())
        Result += ArrayIdx * GTI.getSequentialElementStride(*this);
    }
  }

  return Result;
}

static APInt getElementIndex(TypeSize ElemSize, APInt &Offset) {
  // Skip over scalable or zero size elements. Also skip element sizes larger
  // than the positive index space, because the arithmetic below may not be
  // correct in that case.
  unsigned BitWidth = Offset.getBitWidth();
  if (ElemSize.isScalable() || ElemSize == 0 ||
      !isUIntN(BitWidth - 1, ElemSize)) {
    return APInt::getZero(BitWidth);
  }

  uint64_t FixedElemSize = ElemSize.getFixedValue();
  APInt Index = Offset.sdiv(FixedElemSize);
  Offset -= Index * FixedElemSize;
  if (Offset.isNegative()) {
    // Prefer a positive remaining offset to allow struct indexing.
    --Index;
    Offset += FixedElemSize;
    assert(Offset.isNonNegative() && "Remaining offset shouldn't be negative");
  }
  return Index;
}

std::optional<APInt> DataLayout::getGEPIndexForOffset(Type *&ElemTy,
                                                      APInt &Offset) const {
  if (auto *ArrTy = dyn_cast<ArrayType>(ElemTy)) {
    ElemTy = ArrTy->getElementType();
    return getElementIndex(getTypeAllocSize(ElemTy), Offset);
  }

  if (isa<VectorType>(ElemTy)) {
    // Vector GEPs are partially broken (e.g. for overaligned element types),
    // and may be forbidden in the future, so avoid generating GEPs into
    // vectors. See https://discourse.llvm.org/t/67497
    return std::nullopt;
  }

  if (auto *STy = dyn_cast<StructType>(ElemTy)) {
    const StructLayout *SL = getStructLayout(STy);
    uint64_t IntOffset = Offset.getZExtValue();
    if (IntOffset >= SL->getSizeInBytes())
      return std::nullopt;

    unsigned Index = SL->getElementContainingOffset(IntOffset);
    Offset -= SL->getElementOffset(Index);
    ElemTy = STy->getElementType(Index);
    return APInt(32, Index);
  }

  // Non-aggregate type.
  return std::nullopt;
}

SmallVector<APInt> DataLayout::getGEPIndicesForOffset(Type *&ElemTy,
                                                      APInt &Offset) const {
  assert(ElemTy->isSized() && "Element type must be sized");
  SmallVector<APInt> Indices;
  Indices.push_back(getElementIndex(getTypeAllocSize(ElemTy), Offset));
  while (Offset != 0) {
    std::optional<APInt> Index = getGEPIndexForOffset(ElemTy, Offset);
    if (!Index)
      break;
    Indices.push_back(*Index);
  }

  return Indices;
}

/// getPreferredAlign - Return the preferred alignment of the specified global.
/// This includes an explicitly requested alignment (if the global has one).
Align DataLayout::getPreferredAlign(const GlobalVariable *GV) const {
  MaybeAlign GVAlignment = GV->getAlign();
  // If a section is specified, always precisely honor explicit alignment,
  // so we don't insert padding into a section we don't control.
  if (GVAlignment && GV->hasSection())
    return *GVAlignment;

  // If no explicit alignment is specified, compute the alignment based on
  // the IR type. If an alignment is specified, increase it to match the ABI
  // alignment of the IR type.
  //
  // FIXME: Not sure it makes sense to use the alignment of the type if
  // there's already an explicit alignment specification.
  Type *ElemType = GV->getValueType();
  Align Alignment = getPrefTypeAlign(ElemType);
  if (GVAlignment) {
    if (*GVAlignment >= Alignment)
      Alignment = *GVAlignment;
    else
      Alignment = std::max(*GVAlignment, getABITypeAlign(ElemType));
  }

  // If no explicit alignment is specified, and the global is large, increase
  // the alignment to 16.
  // FIXME: Why 16, specifically?
  if (GV->hasInitializer() && !GVAlignment) {
    if (Alignment < Align(16)) {
      // If the global is not external, see if it is large.  If so, give it a
      // larger alignment.
      if (getTypeSizeInBits(ElemType) > 128)
        Alignment = Align(16); // 16-byte alignment.
    }
  }
  return Alignment;
}
