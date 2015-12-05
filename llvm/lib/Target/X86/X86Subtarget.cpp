//===-- X86Subtarget.cpp - X86 Subtarget Information ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the X86 specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "X86Subtarget.h"
#include "X86InstrInfo.h"
#include "X86TargetMachine.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#if defined(_MSC_VER)
#include <intrin.h>
#endif

using namespace llvm;

#define DEBUG_TYPE "subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "X86GenSubtargetInfo.inc"

// Temporary option to control early if-conversion for x86 while adding machine
// models.
static cl::opt<bool>
X86EarlyIfConv("x86-early-ifcvt", cl::Hidden,
               cl::desc("Enable early if-conversion on X86"));


/// Classify a blockaddress reference for the current subtarget according to how
/// we should reference it in a non-pcrel context.
unsigned char X86Subtarget::ClassifyBlockAddressReference() const {
  if (isPICStyleGOT())    // 32-bit ELF targets.
    return X86II::MO_GOTOFF;

  if (isPICStyleStubPIC())   // Darwin/32 in PIC mode.
    return X86II::MO_PIC_BASE_OFFSET;

  // Direct static reference to label.
  return X86II::MO_NO_FLAG;
}

/// Classify a global variable reference for the current subtarget according to
/// how we should reference it in a non-pcrel context.
unsigned char X86Subtarget::
ClassifyGlobalReference(const GlobalValue *GV, const TargetMachine &TM) const {
  // DLLImport only exists on windows, it is implemented as a load from a
  // DLLIMPORT stub.
  if (GV->hasDLLImportStorageClass())
    return X86II::MO_DLLIMPORT;

  bool isDef = GV->isStrongDefinitionForLinker();

  // X86-64 in PIC mode.
  if (isPICStyleRIPRel()) {
    // Large model never uses stubs.
    if (TM.getCodeModel() == CodeModel::Large)
      return X86II::MO_NO_FLAG;

    if (isTargetDarwin()) {
      // If symbol visibility is hidden, the extra load is not needed if
      // target is x86-64 or the symbol is definitely defined in the current
      // translation unit.
      if (GV->hasDefaultVisibility() && !isDef)
        return X86II::MO_GOTPCREL;
    } else if (!isTargetWin64()) {
      assert(isTargetELF() && "Unknown rip-relative target");

      // Extra load is needed for all externally visible.
      if (!GV->hasLocalLinkage() && GV->hasDefaultVisibility())
        return X86II::MO_GOTPCREL;
    }

    return X86II::MO_NO_FLAG;
  }

  if (isPICStyleGOT()) {   // 32-bit ELF targets.
    // Extra load is needed for all externally visible.
    if (GV->hasLocalLinkage() || GV->hasHiddenVisibility())
      return X86II::MO_GOTOFF;
    return X86II::MO_GOT;
  }

  if (isPICStyleStubPIC()) {  // Darwin/32 in PIC mode.
    // Determine whether we have a stub reference and/or whether the reference
    // is relative to the PIC base or not.

    // If this is a strong reference to a definition, it is definitely not
    // through a stub.
    if (isDef)
      return X86II::MO_PIC_BASE_OFFSET;

    // Unless we have a symbol with hidden visibility, we have to go through a
    // normal $non_lazy_ptr stub because this symbol might be resolved late.
    if (!GV->hasHiddenVisibility())  // Non-hidden $non_lazy_ptr reference.
      return X86II::MO_DARWIN_NONLAZY_PIC_BASE;

    // If symbol visibility is hidden, we have a stub for common symbol
    // references and external declarations.
    if (GV->isDeclarationForLinker() || GV->hasCommonLinkage()) {
      // Hidden $non_lazy_ptr reference.
      return X86II::MO_DARWIN_HIDDEN_NONLAZY_PIC_BASE;
    }

    // Otherwise, no stub.
    return X86II::MO_PIC_BASE_OFFSET;
  }

  if (isPICStyleStubNoDynamic()) {  // Darwin/32 in -mdynamic-no-pic mode.
    // Determine whether we have a stub reference.

    // If this is a strong reference to a definition, it is definitely not
    // through a stub.
    if (isDef)
      return X86II::MO_NO_FLAG;

    // Unless we have a symbol with hidden visibility, we have to go through a
    // normal $non_lazy_ptr stub because this symbol might be resolved late.
    if (!GV->hasHiddenVisibility())  // Non-hidden $non_lazy_ptr reference.
      return X86II::MO_DARWIN_NONLAZY;

    // Otherwise, no stub.
    return X86II::MO_NO_FLAG;
  }

  // Direct static reference to global.
  return X86II::MO_NO_FLAG;
}


/// This function returns the name of a function which has an interface like
/// the non-standard bzero function, if such a function exists on the
/// current subtarget and it is considered preferable over memset with zero
/// passed as the second argument. Otherwise it returns null.
const char *X86Subtarget::getBZeroEntry() const {
  // Darwin 10 has a __bzero entry point for this purpose.
  if (getTargetTriple().isMacOSX() &&
      !getTargetTriple().isMacOSXVersionLT(10, 6))
    return "__bzero";

  return nullptr;
}

bool X86Subtarget::hasSinCos() const {
  return getTargetTriple().isMacOSX() &&
    !getTargetTriple().isMacOSXVersionLT(10, 9) &&
    is64Bit();
}

/// Return true if the subtarget allows calls to immediate address.
bool X86Subtarget::IsLegalToCallImmediateAddr(const TargetMachine &TM) const {
  // FIXME: I386 PE/COFF supports PC relative calls using IMAGE_REL_I386_REL32
  // but WinCOFFObjectWriter::RecordRelocation cannot emit them.  Once it does,
  // the following check for Win32 should be removed.
  if (In64BitMode || isTargetWin32())
    return false;
  return isTargetELF() || TM.getRelocationModel() == Reloc::Static;
}

void X86Subtarget::initSubtargetFeatures(StringRef CPU, StringRef FS) {
  std::string CPUName = CPU;
  if (CPUName.empty())
    CPUName = "generic";

  // Make sure 64-bit features are available in 64-bit mode. (But make sure
  // SSE2 can be turned off explicitly.)
  std::string FullFS = FS;
  if (In64BitMode) {
    if (!FullFS.empty())
      FullFS = "+64bit,+sse2," + FullFS;
    else
      FullFS = "+64bit,+sse2";
  }

  // LAHF/SAHF are always supported in non-64-bit mode.
  if (!In64BitMode) {
    if (!FullFS.empty())
      FullFS = "+sahf," + FullFS;
    else
      FullFS = "+sahf";
  }


  // Parse features string and set the CPU.
  ParseSubtargetFeatures(CPUName, FullFS);

  // All CPUs that implement SSE4.2 or SSE4A support unaligned accesses of
  // 16-bytes and under that are reasonably fast. These features were
  // introduced with Intel's Nehalem/Silvermont and AMD's Family10h
  // micro-architectures respectively.
  if (hasSSE42() || hasSSE4A())
    IsUAMem16Slow = false;
  
  InstrItins = getInstrItineraryForCPU(CPUName);

  // It's important to keep the MCSubtargetInfo feature bits in sync with
  // target data structure which is shared with MC code emitter, etc.
  if (In64BitMode)
    ToggleFeature(X86::Mode64Bit);
  else if (In32BitMode)
    ToggleFeature(X86::Mode32Bit);
  else if (In16BitMode)
    ToggleFeature(X86::Mode16Bit);
  else
    llvm_unreachable("Not 16-bit, 32-bit or 64-bit mode!");

  DEBUG(dbgs() << "Subtarget features: SSELevel " << X86SSELevel
               << ", 3DNowLevel " << X863DNowLevel
               << ", 64bit " << HasX86_64 << "\n");
  assert((!In64BitMode || HasX86_64) &&
         "64-bit code requested on a subtarget that doesn't support it!");

  // Stack alignment is 16 bytes on Darwin, Linux and Solaris (both
  // 32 and 64 bit) and for all 64-bit targets.
  if (StackAlignOverride)
    stackAlignment = StackAlignOverride;
  else if (isTargetDarwin() || isTargetLinux() || isTargetSolaris() ||
           In64BitMode)
    stackAlignment = 16;
}

void X86Subtarget::initializeEnvironment() {
  X86SSELevel = NoSSE;
  X863DNowLevel = NoThreeDNow;
  HasCMov = false;
  HasX86_64 = false;
  HasPOPCNT = false;
  HasSSE4A = false;
  HasAES = false;
  HasFXSR = false;
  HasXSAVE = false;
  HasXSAVEOPT = false;
  HasXSAVEC = false;
  HasXSAVES = false;
  HasPCLMUL = false;
  HasFMA = false;
  HasFMA4 = false;
  HasXOP = false;
  HasTBM = false;
  HasMOVBE = false;
  HasRDRAND = false;
  HasF16C = false;
  HasFSGSBase = false;
  HasLZCNT = false;
  HasBMI = false;
  HasBMI2 = false;
  HasRTM = false;
  HasHLE = false;
  HasERI = false;
  HasCDI = false;
  HasPFI = false;
  HasDQI = false;
  HasBWI = false;
  HasVLX = false;
  HasADX = false;
  HasSHA = false;
  HasPRFCHW = false;
  HasRDSEED = false;
  HasLAHFSAHF = false;
  HasMPX = false;
  IsBTMemSlow = false;
  IsSHLDSlow = false;
  IsUAMem16Slow = false;
  IsUAMem32Slow = false;
  HasSSEUnalignedMem = false;
  HasCmpxchg16b = false;
  UseLeaForSP = false;
  HasSlowDivide32 = false;
  HasSlowDivide64 = false;
  PadShortFunctions = false;
  CallRegIndirect = false;
  LEAUsesAG = false;
  SlowLEA = false;
  SlowIncDec = false;
  stackAlignment = 4;
  // FIXME: this is a known good value for Yonah. How about others?
  MaxInlineSizeThreshold = 128;
  UseSoftFloat = false;
}

X86Subtarget &X86Subtarget::initializeSubtargetDependencies(StringRef CPU,
                                                            StringRef FS) {
  initializeEnvironment();
  initSubtargetFeatures(CPU, FS);
  return *this;
}

X86Subtarget::X86Subtarget(const Triple &TT, const std::string &CPU,
                           const std::string &FS, const X86TargetMachine &TM,
                           unsigned StackAlignOverride)
    : X86GenSubtargetInfo(TT, CPU, FS), X86ProcFamily(Others),
      PICStyle(PICStyles::None), TargetTriple(TT),
      StackAlignOverride(StackAlignOverride),
      In64BitMode(TargetTriple.getArch() == Triple::x86_64),
      In32BitMode(TargetTriple.getArch() == Triple::x86 &&
                  TargetTriple.getEnvironment() != Triple::CODE16),
      In16BitMode(TargetTriple.getArch() == Triple::x86 &&
                  TargetTriple.getEnvironment() == Triple::CODE16),
      TSInfo(), InstrInfo(initializeSubtargetDependencies(CPU, FS)),
      TLInfo(TM, *this), FrameLowering(*this, getStackAlignment()) {
  // Determine the PICStyle based on the target selected.
  if (TM.getRelocationModel() == Reloc::Static) {
    // Unless we're in PIC or DynamicNoPIC mode, set the PIC style to None.
    setPICStyle(PICStyles::None);
  } else if (is64Bit()) {
    // PIC in 64 bit mode is always rip-rel.
    setPICStyle(PICStyles::RIPRel);
  } else if (isTargetCOFF()) {
    setPICStyle(PICStyles::None);
  } else if (isTargetDarwin()) {
    if (TM.getRelocationModel() == Reloc::PIC_)
      setPICStyle(PICStyles::StubPIC);
    else {
      assert(TM.getRelocationModel() == Reloc::DynamicNoPIC);
      setPICStyle(PICStyles::StubDynamicNoPIC);
    }
  } else if (isTargetELF()) {
    setPICStyle(PICStyles::GOT);
  }
}

bool X86Subtarget::enableEarlyIfConversion() const {
  return hasCMov() && X86EarlyIfConv;
}

