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

#define DEBUG_TYPE "subtarget"
#include "X86Subtarget.h"
#include "X86InstrInfo.h"
#include "llvm/GlobalValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Host.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "X86GenSubtargetInfo.inc"

using namespace llvm;

#if defined(_MSC_VER)
#include <intrin.h>
#endif

/// ClassifyBlockAddressReference - Classify a blockaddress reference for the
/// current subtarget according to how we should reference it in a non-pcrel
/// context.
unsigned char X86Subtarget::
ClassifyBlockAddressReference() const {
  if (isPICStyleGOT())    // 32-bit ELF targets.
    return X86II::MO_GOTOFF;

  if (isPICStyleStubPIC())   // Darwin/32 in PIC mode.
    return X86II::MO_PIC_BASE_OFFSET;

  // Direct static reference to label.
  return X86II::MO_NO_FLAG;
}

/// ClassifyGlobalReference - Classify a global variable reference for the
/// current subtarget according to how we should reference it in a non-pcrel
/// context.
unsigned char X86Subtarget::
ClassifyGlobalReference(const GlobalValue *GV, const TargetMachine &TM) const {
  // DLLImport only exists on windows, it is implemented as a load from a
  // DLLIMPORT stub.
  if (GV->hasDLLImportLinkage())
    return X86II::MO_DLLIMPORT;

  // Determine whether this is a reference to a definition or a declaration.
  // Materializable GVs (in JIT lazy compilation mode) do not require an extra
  // load from stub.
  bool isDecl = GV->hasAvailableExternallyLinkage();
  if (GV->isDeclaration() && !GV->isMaterializable())
    isDecl = true;

  // X86-64 in PIC mode.
  if (isPICStyleRIPRel()) {
    // Large model never uses stubs.
    if (TM.getCodeModel() == CodeModel::Large)
      return X86II::MO_NO_FLAG;

    if (isTargetDarwin()) {
      // If symbol visibility is hidden, the extra load is not needed if
      // target is x86-64 or the symbol is definitely defined in the current
      // translation unit.
      if (GV->hasDefaultVisibility() &&
          (isDecl || GV->isWeakForLinker()))
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
    if (!isDecl && !GV->isWeakForLinker())
      return X86II::MO_PIC_BASE_OFFSET;

    // Unless we have a symbol with hidden visibility, we have to go through a
    // normal $non_lazy_ptr stub because this symbol might be resolved late.
    if (!GV->hasHiddenVisibility())  // Non-hidden $non_lazy_ptr reference.
      return X86II::MO_DARWIN_NONLAZY_PIC_BASE;

    // If symbol visibility is hidden, we have a stub for common symbol
    // references and external declarations.
    if (isDecl || GV->hasCommonLinkage()) {
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
    if (!isDecl && !GV->isWeakForLinker())
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


/// getBZeroEntry - This function returns the name of a function which has an
/// interface like the non-standard bzero function, if such a function exists on
/// the current subtarget and it is considered prefereable over memset with zero
/// passed as the second argument. Otherwise it returns null.
const char *X86Subtarget::getBZeroEntry() const {
  // Darwin 10 has a __bzero entry point for this purpose.
  if (getTargetTriple().isMacOSX() &&
      !getTargetTriple().isMacOSXVersionLT(10, 6))
    return "__bzero";

  return 0;
}

/// IsLegalToCallImmediateAddr - Return true if the subtarget allows calls
/// to immediate address.
bool X86Subtarget::IsLegalToCallImmediateAddr(const TargetMachine &TM) const {
  if (In64BitMode)
    return false;
  return isTargetELF() || TM.getRelocationModel() == Reloc::Static;
}

void X86Subtarget::AutoDetectSubtargetFeatures() {
  unsigned EAX = 0, EBX = 0, ECX = 0, EDX = 0;
  unsigned MaxLevel;
  union {
    unsigned u[3];
    char     c[12];
  } text;

  if (X86_MC::GetCpuIDAndInfo(0, &MaxLevel, text.u+0, text.u+2, text.u+1) ||
      MaxLevel < 1)
    return;

  X86_MC::GetCpuIDAndInfo(0x1, &EAX, &EBX, &ECX, &EDX);

  if ((EDX >> 15) & 1) { HasCMov = true;      ToggleFeature(X86::FeatureCMOV); }
  if ((EDX >> 23) & 1) { X86SSELevel = MMX;   ToggleFeature(X86::FeatureMMX);  }
  if ((EDX >> 25) & 1) { X86SSELevel = SSE1;  ToggleFeature(X86::FeatureSSE1); }
  if ((EDX >> 26) & 1) { X86SSELevel = SSE2;  ToggleFeature(X86::FeatureSSE2); }
  if (ECX & 0x1)       { X86SSELevel = SSE3;  ToggleFeature(X86::FeatureSSE3); }
  if ((ECX >> 9)  & 1) { X86SSELevel = SSSE3; ToggleFeature(X86::FeatureSSSE3);}
  if ((ECX >> 19) & 1) { X86SSELevel = SSE41; ToggleFeature(X86::FeatureSSE41);}
  if ((ECX >> 20) & 1) { X86SSELevel = SSE42; ToggleFeature(X86::FeatureSSE42);}
  if ((ECX >> 28) & 1) { X86SSELevel = AVX;   ToggleFeature(X86::FeatureAVX); }

  bool IsIntel = memcmp(text.c, "GenuineIntel", 12) == 0;
  bool IsAMD   = !IsIntel && memcmp(text.c, "AuthenticAMD", 12) == 0;

  if ((ECX >> 1) & 0x1) {
    HasPCLMUL = true;
    ToggleFeature(X86::FeaturePCLMUL);
  }
  if ((ECX >> 12) & 0x1) {
    HasFMA = true;
    ToggleFeature(X86::FeatureFMA);
  }
  if (IsIntel && ((ECX >> 22) & 0x1)) {
    HasMOVBE = true;
    ToggleFeature(X86::FeatureMOVBE);
  }
  if ((ECX >> 23) & 0x1) {
    HasPOPCNT = true;
    ToggleFeature(X86::FeaturePOPCNT);
  }
  if ((ECX >> 25) & 0x1) {
    HasAES = true;
    ToggleFeature(X86::FeatureAES);
  }
  if ((ECX >> 29) & 0x1) {
    HasF16C = true;
    ToggleFeature(X86::FeatureF16C);
  }
  if (IsIntel && ((ECX >> 30) & 0x1)) {
    HasRDRAND = true;
    ToggleFeature(X86::FeatureRDRAND);
  }

  if ((ECX >> 13) & 0x1) {
    HasCmpxchg16b = true;
    ToggleFeature(X86::FeatureCMPXCHG16B);
  }

  if (IsIntel || IsAMD) {
    // Determine if bit test memory instructions are slow.
    unsigned Family = 0;
    unsigned Model  = 0;
    X86_MC::DetectFamilyModel(EAX, Family, Model);
    if (IsAMD || (Family == 6 && Model >= 13)) {
      IsBTMemSlow = true;
      ToggleFeature(X86::FeatureSlowBTMem);
    }

    // If it's Nehalem, unaligned memory access is fast.
    // Include Westmere and Sandy Bridge as well.
    // FIXME: add later processors.
    if (IsIntel && ((Family == 6 && Model == 26) ||
        (Family == 6 && Model == 44) ||
        (Family == 6 && Model == 42))) {
      IsUAMemFast = true;
      ToggleFeature(X86::FeatureFastUAMem);
    }

    // Set processor type. Currently only Atom is detected.
    if (Family == 6 &&
        (Model == 28 || Model == 38 || Model == 39
         || Model == 53 || Model == 54)) {
      X86ProcFamily = IntelAtom;

      UseLeaForSP = true;
      ToggleFeature(X86::FeatureLeaForSP);
    }

    unsigned MaxExtLevel;
    X86_MC::GetCpuIDAndInfo(0x80000000, &MaxExtLevel, &EBX, &ECX, &EDX);

    if (MaxExtLevel >= 0x80000001) {
      X86_MC::GetCpuIDAndInfo(0x80000001, &EAX, &EBX, &ECX, &EDX);
      if ((EDX >> 29) & 0x1) {
        HasX86_64 = true;
        ToggleFeature(X86::Feature64Bit);
      }
      if ((ECX >> 5) & 0x1) {
        HasLZCNT = true;
        ToggleFeature(X86::FeatureLZCNT);
      }
      if (IsAMD) {
        if ((ECX >> 6) & 0x1) {
          HasSSE4A = true;
          ToggleFeature(X86::FeatureSSE4A);
        }
        if ((ECX >> 11) & 0x1) {
          HasXOP = true;
          ToggleFeature(X86::FeatureXOP);
        }
        if ((ECX >> 16) & 0x1) {
          HasFMA4 = true;
          ToggleFeature(X86::FeatureFMA4);
        }
      }
    }
  }

  if (MaxLevel >= 7) {
    if (!X86_MC::GetCpuIDAndInfoEx(0x7, 0x0, &EAX, &EBX, &ECX, &EDX)) {
      if (IsIntel && (EBX & 0x1)) {
        HasFSGSBase = true;
        ToggleFeature(X86::FeatureFSGSBase);
      }
      if ((EBX >> 3) & 0x1) {
        HasBMI = true;
        ToggleFeature(X86::FeatureBMI);
      }
      if (IsIntel && ((EBX >> 5) & 0x1)) {
        X86SSELevel = AVX2;
        ToggleFeature(X86::FeatureAVX2);
      }
      if (IsIntel && ((EBX >> 8) & 0x1)) {
        HasBMI2 = true;
        ToggleFeature(X86::FeatureBMI2);
      }
      if (IsIntel && ((EBX >> 11) & 0x1)) {
        HasRTM = true;
        ToggleFeature(X86::FeatureRTM);
      }
    }
  }
}

X86Subtarget::X86Subtarget(const std::string &TT, const std::string &CPU,
                           const std::string &FS,
                           unsigned StackAlignOverride, bool is64Bit)
  : X86GenSubtargetInfo(TT, CPU, FS)
  , X86ProcFamily(Others)
  , PICStyle(PICStyles::None)
  , X86SSELevel(NoMMXSSE)
  , X863DNowLevel(NoThreeDNow)
  , HasCMov(false)
  , HasX86_64(false)
  , HasPOPCNT(false)
  , HasSSE4A(false)
  , HasAES(false)
  , HasPCLMUL(false)
  , HasFMA(false)
  , HasFMA4(false)
  , HasXOP(false)
  , HasMOVBE(false)
  , HasRDRAND(false)
  , HasF16C(false)
  , HasFSGSBase(false)
  , HasLZCNT(false)
  , HasBMI(false)
  , HasBMI2(false)
  , HasRTM(false)
  , IsBTMemSlow(false)
  , IsUAMemFast(false)
  , HasVectorUAMem(false)
  , HasCmpxchg16b(false)
  , UseLeaForSP(false)
  , HasSlowDivide(false)
  , PostRAScheduler(false)
  , stackAlignment(4)
  // FIXME: this is a known good value for Yonah. How about others?
  , MaxInlineSizeThreshold(128)
  , TargetTriple(TT)
  , In64BitMode(is64Bit) {
  // Determine default and user specified characteristics
  std::string CPUName = CPU;
  if (!FS.empty() || !CPU.empty()) {
    if (CPUName.empty()) {
#if defined(i386) || defined(__i386__) || defined(__x86__) || defined(_M_IX86)\
    || defined(__x86_64__) || defined(_M_AMD64) || defined (_M_X64)
      CPUName = sys::getHostCPUName();
#else
      CPUName = "generic";
#endif
    }

    // Make sure 64-bit features are available in 64-bit mode. (But make sure
    // SSE2 can be turned off explicitly.)
    std::string FullFS = FS;
    if (In64BitMode) {
      if (!FullFS.empty())
        FullFS = "+64bit,+sse2," + FullFS;
      else
        FullFS = "+64bit,+sse2";
    }

    // If feature string is not empty, parse features string.
    ParseSubtargetFeatures(CPUName, FullFS);
  } else {
    if (CPUName.empty()) {
#if defined (__x86_64__) || defined(__i386__)
      CPUName = sys::getHostCPUName();
#else
      CPUName = "generic";
#endif
    }
    // Otherwise, use CPUID to auto-detect feature set.
    AutoDetectSubtargetFeatures();

    // Make sure 64-bit features are available in 64-bit mode.
    if (In64BitMode) {
      HasX86_64 = true; ToggleFeature(X86::Feature64Bit);
      HasCMov = true;   ToggleFeature(X86::FeatureCMOV);

      if (X86SSELevel < SSE2) {
        X86SSELevel = SSE2;
        ToggleFeature(X86::FeatureSSE1);
        ToggleFeature(X86::FeatureSSE2);
      }
    }
  }

  // CPUName may have been set by the CPU detection code. Make sure the
  // new MCSchedModel is used.
  InitMCProcessorInfo(CPUName, FS);

  if (X86ProcFamily == IntelAtom)
    PostRAScheduler = true;

  InstrItins = getInstrItineraryForCPU(CPUName);

  // It's important to keep the MCSubtargetInfo feature bits in sync with
  // target data structure which is shared with MC code emitter, etc.
  if (In64BitMode)
    ToggleFeature(X86::Mode64Bit);

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

bool X86Subtarget::enablePostRAScheduler(
           CodeGenOpt::Level OptLevel,
           TargetSubtargetInfo::AntiDepBreakMode& Mode,
           RegClassVector& CriticalPathRCs) const {
  Mode = TargetSubtargetInfo::ANTIDEP_CRITICAL;
  CriticalPathRCs.clear();
  return PostRAScheduler && OptLevel >= CodeGenOpt::Default;
}
