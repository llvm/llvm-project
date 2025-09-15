//===--- TargetDataLayout.cpp - Map Triple to LLVM data layout string -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TargetParser/ARMTargetParser.h"
#include "llvm/TargetParser/Triple.h"
#include <cstring>
using namespace llvm;

static StringRef getManglingComponent(const Triple &T) {
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

static std::string computeARMDataLayout(const Triple &TT, StringRef ABIName) {
  auto ABI = ARM::computeTargetABI(TT, ABIName);
  std::string Ret;

  if (TT.isLittleEndian())
    // Little endian.
    Ret += "e";
  else
    // Big endian.
    Ret += "E";

  Ret += getManglingComponent(TT);

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
  // CSKY is always 32-bit target with the CSKYv2 ABI as prefer now.
  // It's a 4-byte aligned stack with ELF mangling only.
  // Only support little endian for now.
  // TODO: Add support for big endian.
  return "e-m:e-S32-p:32:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:32"
         "-v128:32:32-a:0:32-Fi32-n32";
}

static std::string computeLoongArchDataLayout(const Triple &TT) {
  if (TT.isLoongArch64())
    return "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128";
  assert(TT.isLoongArch32() && "only LA32 and LA64 are currently supported");
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
enum class MipsABI { Unknown, O32, N32, N64 };
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
  bool is64Bit = T.isPPC64();
  std::string Ret;

  // Most PPC* platforms are big endian, PPC(64)LE is little endian.
  if (T.isLittleEndian())
    Ret = "e";
  else
    Ret = "E";

  Ret += getManglingComponent(T);

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
  if (TT.isRISCV64()) {
    Ret += "-p:64:64-i64:64-i128:128";
    Ret += "-n32:64";
  } else {
    assert(TT.isRISCV32() && "only RV32 and RV64 are currently supported");
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
  const bool Is64Bit = T.isSPARC64();

  // Sparc is typically big endian, but some are little.
  std::string Ret = T.getArch() == Triple::sparcel ? "e" : "E";
  Ret += "-m:e";

  // Some ABIs have 32bit pointers.
  if (!Is64Bit)
    Ret += "-p:32:32";

  // Alignments for 64 bit integers.
  Ret += "-i64:64";

  // Alignments for 128 bit integers.
  // This is not specified in the ABI document but is the de facto standard.
  Ret += "-i128:128";

  // On SparcV9 128 floats are aligned to 128 bits, on others only to 64.
  // On SparcV9 registers can hold 64 or 32 bits, on others only 32.
  if (Is64Bit)
    Ret += "-n32:64";
  else
    Ret += "-f128:64-n32";

  if (Is64Bit)
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
  Ret += getManglingComponent(TT);

  // Special features for z/OS.
  if (TT.isOSzOS()) {
    // Custom address space for ptr32.
    Ret += "-p1:32:32";
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
  bool Is64Bit = TT.getArch() == Triple::x86_64;

  // X86 is little endian
  std::string Ret = "e";

  Ret += getManglingComponent(TT);
  // X86 and x32 have 32 bit pointers.
  if (!Is64Bit || TT.isX32())
    Ret += "-p:32:32";

  // Address spaces for 32 bit signed, 32 bit unsigned, and 64 bit pointers.
  Ret += "-p270:32:32-p271:32:32-p272:64:64";

  // Some ABIs align 64 bit integers and doubles to 64 bits, others to 32.
  // 128 bit integers are not specified in the 32-bit ABIs but are used
  // internally for lowering f128, so we match the alignment to that.
  if (Is64Bit || TT.isOSWindows())
    Ret += "-i64:64-i128:128";
  else if (TT.isOSIAMCU())
    Ret += "-i64:32-f64:32";
  else
    Ret += "-i128:128-f64:32:64";

  // Some ABIs align long double to 128 bits, others to 32.
  if (TT.isOSIAMCU())
    ; // No f80
  else if (Is64Bit || TT.isOSDarwin() || TT.isWindowsMSVCEnvironment())
    Ret += "-f80:128";
  else
    Ret += "-f80:32";

  if (TT.isOSIAMCU())
    Ret += "-f128:32";

  // The registers can hold 8, 16, 32 or, in x86-64, 64 bits.
  if (Is64Bit)
    Ret += "-n8:16:32:64";
  else
    Ret += "-n8:16:32";

  // The stack is aligned to 32 bits on some ABIs and 128 bits on others.
  if ((!Is64Bit && TT.isOSWindows()) || TT.isOSIAMCU())
    Ret += "-a:0:32-S32";
  else
    Ret += "-S128";

  return Ret;
}

static std::string computeNVPTXDataLayout(const Triple &T, StringRef ABIName) {
  bool Is64Bit = T.getArch() == Triple::nvptx64;
  std::string Ret = "e";

  // Tensor Memory (addrspace:6) is always 32-bits.
  // Distributed Shared Memory (addrspace:7) follows shared memory
  // (addrspace:3).
  if (!Is64Bit)
    Ret += "-p:32:32-p6:32:32-p7:32:32";
  else if (ABIName == "shortptr")
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
  if (TT.getVendor() == Triple::VendorType::Intel)
    return "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-"
           "v512:512-v1024:1024-n8:16:32:64-G1-P9-A0";
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
  return TT.getArch() == Triple::wasm64
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

std::string Triple::computeDataLayout(StringRef ABIName) const {
  switch (getArch()) {
  case Triple::arm:
  case Triple::armeb:
  case Triple::thumb:
  case Triple::thumbeb:
    return computeARMDataLayout(*this, ABIName);
  case Triple::aarch64:
  case Triple::aarch64_be:
  case Triple::aarch64_32:
    return computeAArch64DataLayout(*this);
  case Triple::arc:
    return "e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-"
           "f32:32:32-i64:32-f64:32-a:0:32-n32";
  case Triple::avr:
    return "e-P1-p:16:8-i8:8-i16:8-i32:8-i64:8-f32:8-f64:8-n8:16-a:8";
  case Triple::bpfel:
  case Triple::bpfeb:
    return computeBPFDataLayout(*this);
  case Triple::csky:
    return computeCSKYDataLayout(*this);
  case Triple::dxil:
    return "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-"
           "f32:32-f64:64-n8:16:32:64";
  case Triple::hexagon:
    return "e-m:e-p:32:32:32-a:0-n16:32-"
           "i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-"
           "v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048";
  case Triple::loongarch32:
  case Triple::loongarch64:
    return computeLoongArchDataLayout(*this);
  case Triple::m68k:
    return computeM68kDataLayout(*this);
  case Triple::mips:
  case Triple::mipsel:
  case Triple::mips64:
  case Triple::mips64el:
    return computeMipsDataLayout(*this, ABIName);
  case Triple::msp430:
    return "e-m:e-p:16:16-i32:16-i64:16-f32:16-f64:16-a:8-n8:16-S16";
  case Triple::ppc:
  case Triple::ppcle:
  case Triple::ppc64:
  case Triple::ppc64le:
    return computePowerDataLayout(*this);
  case Triple::r600:
  case Triple::amdgcn:
    return computeAMDDataLayout(*this);
  case Triple::riscv32:
  case Triple::riscv64:
  case Triple::riscv32be:
  case Triple::riscv64be:
    return computeRISCVDataLayout(*this, ABIName);
  case Triple::sparc:
  case Triple::sparcv9:
  case Triple::sparcel:
    return computeSparcDataLayout(*this);
  case Triple::systemz:
    return computeSystemZDataLayout(*this);
  case Triple::tce:
  case Triple::tcele:
  case Triple::x86:
  case Triple::x86_64:
    return computeX86DataLayout(*this);
  case Triple::xcore:
  case Triple::xtensa:
    return "e-m:e-p:32:32-i8:8:32-i16:16:32-i64:64-n32";
  case Triple::nvptx:
  case Triple::nvptx64:
    return computeNVPTXDataLayout(*this, ABIName);
  case Triple::spir:
  case Triple::spir64:
  case Triple::spirv:
  case Triple::spirv32:
  case Triple::spirv64:
    return computeSPIRVDataLayout(*this);
  case Triple::lanai:
    return computeLanaiDataLayout();
  case Triple::wasm32:
  case Triple::wasm64:
    return computeWebAssemblyDataLayout(*this);
  case Triple::ve:
    return computeVEDataLayout(*this);

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
  llvm_unreachable("Invalid arch");
}
