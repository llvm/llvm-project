//===-- X86Subtarget.cpp - X86 Subtarget Information ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the X86 specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "subtarget"
#include "X86Subtarget.h"
#include "X86InstrInfo.h"
#include "X86GenSubtarget.inc"
#include "llvm/GlobalValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

#if defined(_MSC_VER)
#include <intrin.h>
#endif

/// ClassifyGlobalReference - Classify a global variable reference for the
/// current subtarget according to how we should reference it in a non-pcrel
/// context.
unsigned char X86Subtarget::
ClassifyGlobalReference(const GlobalValue *GV, const TargetMachine &TM) const {
  // DLLImport only exists on windows, it is implemented as a load from a
  // DLLIMPORT stub.
  if (GV->hasDLLImportLinkage())
    return X86II::MO_DLLIMPORT;

  // GV with ghost linkage (in JIT lazy compilation mode) do not require an
  // extra load from stub.
  bool isDecl = GV->isDeclaration() && !GV->hasNotBeenReadFromBitcode();

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
    } else {
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
    
    // If symbol visibility is hidden, we have a stub for common symbol
    // references and external declarations.
    if (isDecl || GV->hasCommonLinkage()) {
      // Hidden $non_lazy_ptr reference.
      return X86II::MO_DARWIN_HIDDEN_NONLAZY;
    }
    
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
  if (getDarwinVers() >= 10)
    return "__bzero";

  return 0;
}

/// IsLegalToCallImmediateAddr - Return true if the subtarget allows calls
/// to immediate address.
bool X86Subtarget::IsLegalToCallImmediateAddr(const TargetMachine &TM) const {
  if (Is64Bit)
    return false;
  return isTargetELF() || TM.getRelocationModel() == Reloc::Static;
}

/// getSpecialAddressLatency - For targets where it is beneficial to
/// backschedule instructions that compute addresses, return a value
/// indicating the number of scheduling cycles of backscheduling that
/// should be attempted.
unsigned X86Subtarget::getSpecialAddressLatency() const {
  // For x86 out-of-order targets, back-schedule address computations so
  // that loads and stores aren't blocked.
  // This value was chosen arbitrarily.
  return 200;
}

/// GetCpuIDAndInfo - Execute the specified cpuid and return the 4 values in the
/// specified arguments.  If we can't run cpuid on the host, return true.
bool X86::GetCpuIDAndInfo(unsigned value, unsigned *rEAX, unsigned *rEBX,
                          unsigned *rECX, unsigned *rEDX) {
#if defined(__x86_64__) || defined(_M_AMD64) || defined (_M_X64)
  #if defined(__GNUC__)
    // gcc doesn't know cpuid would clobber ebx/rbx. Preseve it manually.
    asm ("movq\t%%rbx, %%rsi\n\t"
         "cpuid\n\t"
         "xchgq\t%%rbx, %%rsi\n\t"
         : "=a" (*rEAX),
           "=S" (*rEBX),
           "=c" (*rECX),
           "=d" (*rEDX)
         :  "a" (value));
    return false;
  #elif defined(_MSC_VER)
    int registers[4];
    __cpuid(registers, value);
    *rEAX = registers[0];
    *rEBX = registers[1];
    *rECX = registers[2];
    *rEDX = registers[3];
    return false;
  #endif
#elif defined(i386) || defined(__i386__) || defined(__x86__) || defined(_M_IX86)
  #if defined(__GNUC__)
    asm ("movl\t%%ebx, %%esi\n\t"
         "cpuid\n\t"
         "xchgl\t%%ebx, %%esi\n\t"
         : "=a" (*rEAX),
           "=S" (*rEBX),
           "=c" (*rECX),
           "=d" (*rEDX)
         :  "a" (value));
    return false;
  #elif defined(_MSC_VER)
    __asm {
      mov   eax,value
      cpuid
      mov   esi,rEAX
      mov   dword ptr [esi],eax
      mov   esi,rEBX
      mov   dword ptr [esi],ebx
      mov   esi,rECX
      mov   dword ptr [esi],ecx
      mov   esi,rEDX
      mov   dword ptr [esi],edx
    }
    return false;
  #endif
#endif
  return true;
}

static void DetectFamilyModel(unsigned EAX, unsigned &Family, unsigned &Model) {
  Family = (EAX >> 8) & 0xf; // Bits 8 - 11
  Model  = (EAX >> 4) & 0xf; // Bits 4 - 7
  if (Family == 6 || Family == 0xf) {
    if (Family == 0xf)
      // Examine extended family ID if family ID is F.
      Family += (EAX >> 20) & 0xff;    // Bits 20 - 27
    // Examine extended model ID if family ID is 6 or F.
    Model += ((EAX >> 16) & 0xf) << 4; // Bits 16 - 19
  }
}

void X86Subtarget::AutoDetectSubtargetFeatures() {
  unsigned EAX = 0, EBX = 0, ECX = 0, EDX = 0;
  union {
    unsigned u[3];
    char     c[12];
  } text;
  
  if (X86::GetCpuIDAndInfo(0, &EAX, text.u+0, text.u+2, text.u+1))
    return;

  X86::GetCpuIDAndInfo(0x1, &EAX, &EBX, &ECX, &EDX);
  
  if ((EDX >> 23) & 0x1) X86SSELevel = MMX;
  if ((EDX >> 25) & 0x1) X86SSELevel = SSE1;
  if ((EDX >> 26) & 0x1) X86SSELevel = SSE2;
  if (ECX & 0x1)         X86SSELevel = SSE3;
  if ((ECX >> 9)  & 0x1) X86SSELevel = SSSE3;
  if ((ECX >> 19) & 0x1) X86SSELevel = SSE41;
  if ((ECX >> 20) & 0x1) X86SSELevel = SSE42;

  bool IsIntel = memcmp(text.c, "GenuineIntel", 12) == 0;
  bool IsAMD   = !IsIntel && memcmp(text.c, "AuthenticAMD", 12) == 0;

  HasFMA3 = IsIntel && ((ECX >> 12) & 0x1);
  HasAVX = ((ECX >> 28) & 0x1);

  if (IsIntel || IsAMD) {
    // Determine if bit test memory instructions are slow.
    unsigned Family = 0;
    unsigned Model  = 0;
    DetectFamilyModel(EAX, Family, Model);
    IsBTMemSlow = IsAMD || (Family == 6 && Model >= 13);

    X86::GetCpuIDAndInfo(0x80000001, &EAX, &EBX, &ECX, &EDX);
    HasX86_64 = (EDX >> 29) & 0x1;
    HasSSE4A = IsAMD && ((ECX >> 6) & 0x1);
    HasFMA4 = IsAMD && ((ECX >> 16) & 0x1);
  }
}

static const char *GetCurrentX86CPU() {
  unsigned EAX = 0, EBX = 0, ECX = 0, EDX = 0;
  if (X86::GetCpuIDAndInfo(0x1, &EAX, &EBX, &ECX, &EDX))
    return "generic";
  unsigned Family = 0;
  unsigned Model  = 0;
  DetectFamilyModel(EAX, Family, Model);

  X86::GetCpuIDAndInfo(0x80000001, &EAX, &EBX, &ECX, &EDX);
  bool Em64T = (EDX >> 29) & 0x1;
  bool HasSSE3 = (ECX & 0x1);

  union {
    unsigned u[3];
    char     c[12];
  } text;

  X86::GetCpuIDAndInfo(0, &EAX, text.u+0, text.u+2, text.u+1);
  if (memcmp(text.c, "GenuineIntel", 12) == 0) {
    switch (Family) {
      case 3:
        return "i386";
      case 4:
        return "i486";
      case 5:
        switch (Model) {
        case 4:  return "pentium-mmx";
        default: return "pentium";
        }
      case 6:
        switch (Model) {
        case 1:  return "pentiumpro";
        case 3:
        case 5:
        case 6:  return "pentium2";
        case 7:
        case 8:
        case 10:
        case 11: return "pentium3";
        case 9:
        case 13: return "pentium-m";
        case 14: return "yonah";
        case 15:
        case 22: // Celeron M 540
          return "core2";
        case 23: // 45nm: Penryn , Wolfdale, Yorkfield (XE)
          return "penryn";
        default: return "i686";
        }
      case 15: {
        switch (Model) {
        case 3:  
        case 4:
        case 6: // same as 4, but 65nm
          return (Em64T) ? "nocona" : "prescott";
        case 26:
          return "corei7";
        case 28:
          return "atom";
        default:
          return (Em64T) ? "x86-64" : "pentium4";
        }
      }
        
    default:
      return "generic";
    }
  } else if (memcmp(text.c, "AuthenticAMD", 12) == 0) {
    // FIXME: this poorly matches the generated SubtargetFeatureKV table.  There
    // appears to be no way to generate the wide variety of AMD-specific targets
    // from the information returned from CPUID.
    switch (Family) {
      case 4:
        return "i486";
      case 5:
        switch (Model) {
        case 6:
        case 7:  return "k6";
        case 8:  return "k6-2";
        case 9:
        case 13: return "k6-3";
        default: return "pentium";
        }
      case 6:
        switch (Model) {
        case 4:  return "athlon-tbird";
        case 6:
        case 7:
        case 8:  return "athlon-mp";
        case 10: return "athlon-xp";
        default: return "athlon";
        }
      case 15:
        if (HasSSE3) {
          return "k8-sse3";
        } else {
          switch (Model) {
          case 1:  return "opteron";
          case 5:  return "athlon-fx"; // also opteron
          default: return "athlon64";
          }
        }
      case 16:
        return "amdfam10";
    default:
      return "generic";
    }
  } else {
    return "generic";
  }
}

X86Subtarget::X86Subtarget(const std::string &TT, const std::string &FS, 
                           bool is64Bit)
  : PICStyle(PICStyles::None)
  , X86SSELevel(NoMMXSSE)
  , X863DNowLevel(NoThreeDNow)
  , HasX86_64(false)
  , HasSSE4A(false)
  , HasAVX(false)
  , HasFMA3(false)
  , HasFMA4(false)
  , IsBTMemSlow(false)
  , DarwinVers(0)
  , IsLinux(false)
  , stackAlignment(8)
  // FIXME: this is a known good value for Yonah. How about others?
  , MaxInlineSizeThreshold(128)
  , Is64Bit(is64Bit)
  , TargetType(isELF) { // Default to ELF unless otherwise specified.

  // default to hard float ABI
  if (FloatABIType == FloatABI::Default)
    FloatABIType = FloatABI::Hard;
    
  // Determine default and user specified characteristics
  if (!FS.empty()) {
    // If feature string is not empty, parse features string.
    std::string CPU = GetCurrentX86CPU();
    ParseSubtargetFeatures(FS, CPU);
    // All X86-64 CPUs also have SSE2, however user might request no SSE via 
    // -mattr, so don't force SSELevel here.
  } else {
    // Otherwise, use CPUID to auto-detect feature set.
    AutoDetectSubtargetFeatures();
    // Make sure SSE2 is enabled; it is available on all X86-64 CPUs.
    if (Is64Bit && X86SSELevel < SSE2)
      X86SSELevel = SSE2;
  }

  // If requesting codegen for X86-64, make sure that 64-bit features
  // are enabled.
  if (Is64Bit)
    HasX86_64 = true;

  DEBUG(errs() << "Subtarget features: SSELevel " << X86SSELevel
               << ", 3DNowLevel " << X863DNowLevel
               << ", 64bit " << HasX86_64 << "\n");
  assert((!Is64Bit || HasX86_64) &&
         "64-bit code requested on a subtarget that doesn't support it!");

  // Set the boolean corresponding to the current target triple, or the default
  // if one cannot be determined, to true.
  if (TT.length() > 5) {
    size_t Pos;
    if ((Pos = TT.find("-darwin")) != std::string::npos) {
      TargetType = isDarwin;
      
      // Compute the darwin version number.
      if (isdigit(TT[Pos+7]))
        DarwinVers = atoi(&TT[Pos+7]);
      else
        DarwinVers = 8;  // Minimum supported darwin is Tiger.
    } else if (TT.find("linux") != std::string::npos) {
      // Linux doesn't imply ELF, but we don't currently support anything else.
      TargetType = isELF;
      IsLinux = true;
    } else if (TT.find("cygwin") != std::string::npos) {
      TargetType = isCygwin;
    } else if (TT.find("mingw") != std::string::npos) {
      TargetType = isMingw;
    } else if (TT.find("win32") != std::string::npos) {
      TargetType = isWindows;
    } else if (TT.find("windows") != std::string::npos) {
      TargetType = isWindows;
    } else if (TT.find("-cl") != std::string::npos) {
      TargetType = isDarwin;
      DarwinVers = 9;
    }
  }

  // Stack alignment is 16 bytes on Darwin (both 32 and 64 bit) and for all 64
  // bit targets.
  if (TargetType == isDarwin || Is64Bit)
    stackAlignment = 16;

  if (StackAlignment)
    stackAlignment = StackAlignment;
}
