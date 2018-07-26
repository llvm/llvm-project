//===-- TargetParser - Parser for target features ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a target parser to recognise hardware features such as
// FPU/CPU/ARCH names as well as specific support such as HDIV, etc.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_TARGETPARSER_H
#define LLVM_SUPPORT_TARGETPARSER_H

// FIXME: vector is used because that's what clang uses for subtarget feature
// lists, but SmallVector would probably be better
#include "llvm/ADT/Triple.h"
#include <vector>

namespace llvm {
class StringRef;

// Target specific information into their own namespaces. These should be
// generated from TableGen because the information is already there, and there
// is where new information about targets will be added.
// FIXME: To TableGen this we need to make some table generated files available
// even if the back-end is not compiled with LLVM, plus we need to create a new
// back-end to TableGen to create these clean tables.
namespace ARM {

// FPU Version
enum class FPUVersion {
  NONE,
  VFPV2,
  VFPV3,
  VFPV3_FP16,
  VFPV4,
  VFPV5
};

// An FPU name restricts the FPU in one of three ways:
enum class FPURestriction {
  None = 0, ///< No restriction
  D16,      ///< Only 16 D registers
  SP_D16    ///< Only single-precision instructions, with 16 D registers
};

// An FPU name implies one of three levels of Neon support:
enum class NeonSupportLevel {
  None = 0, ///< No Neon
  Neon,     ///< Neon
  Crypto    ///< Neon with Crypto
};

// FPU names.
enum FPUKind {
#define ARM_FPU(NAME, KIND, VERSION, NEON_SUPPORT, RESTRICTION) KIND,
#include "ARMTargetParser.def"
  FK_LAST
};

// Arch names.
enum class ArchKind {
#define ARM_ARCH(NAME, ID, CPU_ATTR, SUB_ARCH, ARCH_ATTR, ARCH_FPU, ARCH_BASE_EXT) ID,
#include "ARMTargetParser.def"
};

// Arch extension modifiers for CPUs.
enum ArchExtKind : unsigned {
  AEK_INVALID =     0,
  AEK_NONE =        1,
  AEK_CRC =         1 << 1,
  AEK_CRYPTO =      1 << 2,
  AEK_FP =          1 << 3,
  AEK_HWDIVTHUMB =  1 << 4,
  AEK_HWDIVARM =    1 << 5,
  AEK_MP =          1 << 6,
  AEK_SIMD =        1 << 7,
  AEK_SEC =         1 << 8,
  AEK_VIRT =        1 << 9,
  AEK_DSP =         1 << 10,
  AEK_FP16 =        1 << 11,
  AEK_RAS =         1 << 12,
  AEK_SVE =         1 << 13,
  AEK_DOTPROD =     1 << 14,
  AEK_SHA2    =     1 << 15,
  AEK_AES     =     1 << 16,
  // Unsupported extensions.
  AEK_OS = 0x8000000,
  AEK_IWMMXT = 0x10000000,
  AEK_IWMMXT2 = 0x20000000,
  AEK_MAVERICK = 0x40000000,
  AEK_XSCALE = 0x80000000,
};

// ISA kinds.
enum class ISAKind { INVALID = 0, ARM, THUMB, AARCH64 };

// Endianness
// FIXME: BE8 vs. BE32?
enum class EndianKind { INVALID = 0, LITTLE, BIG };

// v6/v7/v8 Profile
enum class ProfileKind { INVALID = 0, A, R, M };

StringRef getCanonicalArchName(StringRef Arch);

// Information by ID
StringRef getFPUName(unsigned FPUKind);
FPUVersion getFPUVersion(unsigned FPUKind);
NeonSupportLevel getFPUNeonSupportLevel(unsigned FPUKind);
FPURestriction getFPURestriction(unsigned FPUKind);

// FIXME: These should be moved to TargetTuple once it exists
bool getFPUFeatures(unsigned FPUKind, std::vector<StringRef> &Features);
bool getHWDivFeatures(unsigned HWDivKind, std::vector<StringRef> &Features);
bool getExtensionFeatures(unsigned Extensions,
                          std::vector<StringRef> &Features);

StringRef getArchName(ArchKind AK);
unsigned getArchAttr(ArchKind AK);
StringRef getCPUAttr(ArchKind AK);
StringRef getSubArch(ArchKind AK);
StringRef getArchExtName(unsigned ArchExtKind);
StringRef getArchExtFeature(StringRef ArchExt);
StringRef getHWDivName(unsigned HWDivKind);

// Information by Name
unsigned  getDefaultFPU(StringRef CPU, ArchKind AK);
unsigned  getDefaultExtensions(StringRef CPU, ArchKind AK);
StringRef getDefaultCPU(StringRef Arch);

// Parser
unsigned parseHWDiv(StringRef HWDiv);
unsigned parseFPU(StringRef FPU);
ArchKind parseArch(StringRef Arch);
unsigned parseArchExt(StringRef ArchExt);
ArchKind parseCPUArch(StringRef CPU);
void fillValidCPUArchList(SmallVectorImpl<StringRef> &Values);
ISAKind parseArchISA(StringRef Arch);
EndianKind parseArchEndian(StringRef Arch);
ProfileKind parseArchProfile(StringRef Arch);
unsigned parseArchVersion(StringRef Arch);

StringRef computeDefaultTargetABI(const Triple &TT, StringRef CPU);

} // namespace ARM

// FIXME:This should be made into class design,to avoid dupplication.
namespace AArch64 {

// Arch names.
enum class ArchKind {
#define AARCH64_ARCH(NAME, ID, CPU_ATTR, SUB_ARCH, ARCH_ATTR, ARCH_FPU, ARCH_BASE_EXT) ID,
#include "AArch64TargetParser.def"
};

// Arch extension modifiers for CPUs.
enum ArchExtKind : unsigned {
  AEK_INVALID =     0,
  AEK_NONE =        1,
  AEK_CRC =         1 << 1,
  AEK_CRYPTO =      1 << 2,
  AEK_FP =          1 << 3,
  AEK_SIMD =        1 << 4,
  AEK_FP16 =        1 << 5,
  AEK_PROFILE =     1 << 6,
  AEK_RAS =         1 << 7,
  AEK_LSE =         1 << 8,
  AEK_SVE =         1 << 9,
  AEK_DOTPROD =     1 << 10,
  AEK_RCPC =        1 << 11,
  AEK_RDM =         1 << 12,
  AEK_SM4 =         1 << 13,
  AEK_SHA3 =        1 << 14,
  AEK_SHA2 =        1 << 15,
  AEK_AES =         1 << 16,
};

StringRef getCanonicalArchName(StringRef Arch);

// Information by ID
StringRef getFPUName(unsigned FPUKind);
ARM::FPUVersion getFPUVersion(unsigned FPUKind);
ARM::NeonSupportLevel getFPUNeonSupportLevel(unsigned FPUKind);
ARM::FPURestriction getFPURestriction(unsigned FPUKind);

// FIXME: These should be moved to TargetTuple once it exists
bool getFPUFeatures(unsigned FPUKind, std::vector<StringRef> &Features);
bool getExtensionFeatures(unsigned Extensions,
                                   std::vector<StringRef> &Features);
bool getArchFeatures(ArchKind AK, std::vector<StringRef> &Features);

StringRef getArchName(ArchKind AK);
unsigned getArchAttr(ArchKind AK);
StringRef getCPUAttr(ArchKind AK);
StringRef getSubArch(ArchKind AK);
StringRef getArchExtName(unsigned ArchExtKind);
StringRef getArchExtFeature(StringRef ArchExt);
unsigned checkArchVersion(StringRef Arch);

// Information by Name
unsigned  getDefaultFPU(StringRef CPU, ArchKind AK);
unsigned  getDefaultExtensions(StringRef CPU, ArchKind AK);
StringRef getDefaultCPU(StringRef Arch);
AArch64::ArchKind getCPUArchKind(StringRef CPU);

// Parser
unsigned parseFPU(StringRef FPU);
AArch64::ArchKind parseArch(StringRef Arch);
ArchExtKind parseArchExt(StringRef ArchExt);
ArchKind parseCPUArch(StringRef CPU);
void fillValidCPUArchList(SmallVectorImpl<StringRef> &Values);
ARM::ISAKind parseArchISA(StringRef Arch);
ARM::EndianKind parseArchEndian(StringRef Arch);
ARM::ProfileKind parseArchProfile(StringRef Arch);
unsigned parseArchVersion(StringRef Arch);

bool isX18ReservedByDefault(const Triple &TT);

} // namespace AArch64

namespace X86 {

// This should be kept in sync with libcc/compiler-rt as its included by clang
// as a proxy for what's in libgcc/compiler-rt.
enum ProcessorVendors : unsigned {
  VENDOR_DUMMY,
#define X86_VENDOR(ENUM, STRING) \
  ENUM,
#include "llvm/Support/X86TargetParser.def"
  VENDOR_OTHER
};

// This should be kept in sync with libcc/compiler-rt as its included by clang
// as a proxy for what's in libgcc/compiler-rt.
enum ProcessorTypes : unsigned {
  CPU_TYPE_DUMMY,
#define X86_CPU_TYPE(ARCHNAME, ENUM) \
  ENUM,
#include "llvm/Support/X86TargetParser.def"
  CPU_TYPE_MAX
};

// This should be kept in sync with libcc/compiler-rt as its included by clang
// as a proxy for what's in libgcc/compiler-rt.
enum ProcessorSubtypes : unsigned {
  CPU_SUBTYPE_DUMMY,
#define X86_CPU_SUBTYPE(ARCHNAME, ENUM) \
  ENUM,
#include "llvm/Support/X86TargetParser.def"
  CPU_SUBTYPE_MAX
};

// This should be kept in sync with libcc/compiler-rt as it should be used
// by clang as a proxy for what's in libgcc/compiler-rt.
enum ProcessorFeatures {
#define X86_FEATURE(VAL, ENUM) \
  ENUM = VAL,
#include "llvm/Support/X86TargetParser.def"

};

} // namespace X86

} // namespace llvm

#endif
