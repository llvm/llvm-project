//===--- CodeGenOptions.cpp - Shared codegen option handling --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/Driver/CodeGenOptions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/SystemLibraries.h"
#include "llvm/ProfileData/InstrProfCorrelator.h"
#include "llvm/TargetParser/ARMTargetParser.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {
extern llvm::cl::opt<llvm::InstrProfCorrelator::ProfCorrelatorKind>
    ProfileCorrelate;
} // namespace llvm

namespace llvm::driver {

/// Is the triple {arm,armeb,thumb,thumbeb}-none-none-{eabi,eabihf} ?
static bool useFramePointerForTargetByDefault(const llvm::Triple &Triple,
                                              const FramePointerOptions &Opts) {
  if (Opts.InstrumentationRequiresFramePointer)
    return true;

  if (Triple.isAndroid())
    return true;

  switch (Triple.getArch()) {
  case llvm::Triple::xcore:
  case llvm::Triple::wasm32:
  case llvm::Triple::wasm64:
  case llvm::Triple::msp430:
    // XCore never wants frame pointers, regardless of OS.
    // WebAssembly never wants frame pointers.
    return false;
  case llvm::Triple::ppc:
  case llvm::Triple::ppcle:
  case llvm::Triple::ppc64:
  case llvm::Triple::ppc64le:
  case llvm::Triple::riscv32:
  case llvm::Triple::riscv64:
  case llvm::Triple::riscv32be:
  case llvm::Triple::riscv64be:
  case llvm::Triple::sparc:
  case llvm::Triple::sparcel:
  case llvm::Triple::sparcv9:
  case llvm::Triple::amdgpu:
  case llvm::Triple::r600:
  case llvm::Triple::csky:
  case llvm::Triple::loongarch32:
  case llvm::Triple::loongarch64:
  case llvm::Triple::m68k:
  case llvm::Triple::mips64:
  case llvm::Triple::mips64el:
  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
    return !Opts.Optimized;
  default:
    break;
  }

  if (Triple.isOSFuchsia() || Triple.isOSNetBSD()) {
    return !Opts.Optimized;
  }

  if (Triple.isOSLinux() || Triple.isOSHurd()) {
    switch (Triple.getArch()) {
    // Don't use a frame pointer on linux if optimizing for certain targets.
    case llvm::Triple::arm:
    case llvm::Triple::armeb:
    case llvm::Triple::thumb:
    case llvm::Triple::thumbeb:
    case llvm::Triple::systemz:
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
      return !Opts.Optimized;
    default:
      return true;
    }
  }

  if (Triple.isOSWindows()) {
    switch (Triple.getArch()) {
    case llvm::Triple::x86:
      return !Opts.Optimized;
    case llvm::Triple::x86_64:
      return Triple.isOSBinFormatMachO();
    case llvm::Triple::arm:
    case llvm::Triple::thumb:
      // Windows on ARM builds with FPO disabled to aid fast stack walking
      return true;
    default:
      // All other supported Windows ISAs use xdata unwind information, so frame
      // pointers are not generally useful.
      return false;
    }
  }

  if (llvm::ARM::isARMEABIBareMetal(Triple))
    return false;

  return true;
}

static bool useLeafFramePointerForTargetByDefault(const llvm::Triple &Triple) {
  if (Triple.isAArch64() || Triple.isPS() || Triple.isVE() ||
      (Triple.isAndroid() && !Triple.isARM()))
    return false;

  if ((Triple.isARM() || Triple.isThumb()) && Triple.isOSBinFormatMachO())
    return false;

  return true;
}

static bool mustUseNonLeafFramePointerForTarget(const llvm::Triple &Triple) {
  switch (Triple.getArch()) {
  default:
    return false;
  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    // ARM Darwin targets require a frame pointer to be always present to aid
    // offline debugging via backtraces.
    return Triple.isOSDarwin();
  }
}

// True if a target-specific option requires the frame chain to be preserved,
// even if new frame records are not created.
static bool mustMaintainValidFrameChain(const FramePointerOptions &Opts,
                                        const llvm::Triple &Triple) {
  switch (Triple.getArch()) {
  default:
    return false;
  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb:
    // For 32-bit Arm, the -mframe-chain=aapcs and -mframe-chain=aapcs+leaf
    // options require the frame pointer register to be reserved (or point to a
    // new AAPCS-compilant frame record), even with -fno-omit-frame-pointer.
    return Opts.MaintainValidFrameChain;

  case llvm::Triple::aarch64:
    // Arm64 Windows requires that the frame chain is valid, as there is no
    // way to indicate during a stack walk that a frame has used the frame
    // pointer as a general purpose register.
    return Triple.isOSWindows();
  }
}

// True if a target-specific option causes -fno-omit-frame-pointer to also
// cause frame records to be created in leaf functions.
static bool framePointerImpliesLeafFramePointer(const FramePointerOptions &Opts,
                                                const llvm::Triple &Triple) {
  if (Triple.isARM() || Triple.isThumb()) {
    // For 32-bit Arm, the -mframe-chain=aapcs+leaf option causes the
    // -fno-omit-frame-pointer optiion to imply -mno-omit-leaf-frame-pointer,
    // but does not by itself imply either option.
    return Opts.FramePointerImpliesLeaf;
  }
  return false;
}

llvm::FramePointerKind getFramePointerKind(const llvm::Triple &Triple,
                                           const FramePointerOptions &Opts) {
  // There are four things to consider here:
  // * Should a frame record be created for non-leaf functions?
  // * Should a frame record be created for leaf functions?
  // * Is the frame pointer register reserved in non-leaf functions?
  //   i.e. must it always point to either a new, valid frame record or be
  //   un-modified?
  // * Is the frame pointer register reserved in leaf functions?
  //
  //  Not all combinations of these are valid:
  // * It's not useful to have leaf frame records without non-leaf ones.
  // * It's not useful to have frame records without reserving the frame
  //   pointer.
  //
  // | Frame Setup     | Reg Reserved    |
  // |-----------------|-----------------|
  // | Non-leaf | Leaf | Non-Leaf | Leaf |
  // |----------|------|----------|------|
  // | N        | N    | N        | N    | FramePointerKind::None
  // | N        | N    | N        | Y    | Invalid
  // | N        | N    | Y        | N    | Invalid
  // | N        | N    | Y        | Y    | FramePointerKind::Reserved
  // | N        | Y    | N        | N    | Invalid
  // | N        | Y    | N        | Y    | Invalid
  // | N        | Y    | Y        | N    | Invalid
  // | N        | Y    | Y        | Y    | Invalid
  // | Y        | N    | N        | N    | Invalid
  // | Y        | N    | N        | Y    | Invalid
  // | Y        | N    | Y        | N    | FramePointerKind::NonLeafNoReserve
  // | Y        | N    | Y        | Y    | FramePointerKind::NonLeaf
  // | Y        | Y    | N        | N    | Invalid
  // | Y        | Y    | N        | Y    | Invalid
  // | Y        | Y    | Y        | N    | Invalid
  // | Y        | Y    | Y        | Y    | FramePointerKind::All
  //
  // The FramePointerKind::Reserved case is currently only reachable for Arm,
  // which has the -mframe-chain= option which can (in combination with
  // -fno-omit-frame-pointer) specify that the frame chain must be valid,
  // without requiring new frame records to be created.

  bool DefaultFP = useFramePointerForTargetByDefault(Triple, Opts);
  bool EnableFP = mustUseNonLeafFramePointerForTarget(Triple) ||
                  Opts.EnableFramePointer.value_or(DefaultFP);

  bool DefaultLeafFP =
      useLeafFramePointerForTargetByDefault(Triple) ||
      (EnableFP && framePointerImpliesLeafFramePointer(Opts, Triple));
  bool EnableLeafFP = Opts.EnableLeafFramePointer.value_or(DefaultLeafFP);

  bool FPRegReserved = Opts.ReserveFramePointerRegister.value_or(
      mustMaintainValidFrameChain(Opts, Triple));

  if (EnableFP) {
    if (EnableLeafFP)
      return llvm::FramePointerKind::All;

    if (FPRegReserved)
      return llvm::FramePointerKind::NonLeaf;

    return llvm::FramePointerKind::NonLeafNoReserve;
  }
  if (FPRegReserved)
    return llvm::FramePointerKind::Reserved;
  return llvm::FramePointerKind::None;
}

llvm::VectorLibrary
convertDriverVectorLibraryToVectorLibrary(llvm::driver::VectorLibrary VecLib) {
  switch (VecLib) {
  case llvm::driver::VectorLibrary::NoLibrary:
    return llvm::VectorLibrary::NoLibrary;
  case llvm::driver::VectorLibrary::Accelerate:
    return llvm::VectorLibrary::Accelerate;
  case llvm::driver::VectorLibrary::Darwin_libsystem_m:
    return llvm::VectorLibrary::DarwinLibSystemM;
  case llvm::driver::VectorLibrary::LIBMVEC:
    return llvm::VectorLibrary::LIBMVEC;
  case llvm::driver::VectorLibrary::MASSV:
    return llvm::VectorLibrary::MASSV;
  case llvm::driver::VectorLibrary::SVML:
    return llvm::VectorLibrary::SVML;
  case llvm::driver::VectorLibrary::SLEEF:
    return llvm::VectorLibrary::SLEEFGNUABI;
  case llvm::driver::VectorLibrary::ArmPL:
    return llvm::VectorLibrary::ArmPL;
  case llvm::driver::VectorLibrary::AMDLIBM:
    return llvm::VectorLibrary::AMDLIBM;
  }
  llvm_unreachable("Unexpected driver::VectorLibrary");
}

TargetLibraryInfoImpl *createTLII(const llvm::Triple &TargetTriple,
                                  driver::VectorLibrary Veclib) {
  return new TargetLibraryInfoImpl(
      TargetTriple, convertDriverVectorLibraryToVectorLibrary(Veclib));
}

std::string getDefaultProfileGenName() {
  return llvm::ProfileCorrelate != InstrProfCorrelator::NONE
             ? "default_%m.proflite"
             : "default_%m.profraw";
}
} // namespace llvm::driver
