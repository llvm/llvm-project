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
#include "X86GenSubtarget.inc"
#include "llvm/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

static cl::opt<X86Subtarget::AsmWriterFlavorTy>
AsmWriterFlavor("x86-asm-syntax", cl::init(X86Subtarget::Unset),
  cl::desc("Choose style of code to emit from X86 backend:"),
  cl::values(
    clEnumValN(X86Subtarget::ATT,   "att",   "Emit AT&T-style assembly"),
    clEnumValN(X86Subtarget::Intel, "intel", "Emit Intel-style assembly"),
    clEnumValEnd));


/// True if accessing the GV requires an extra load. For Windows, dllimported
/// symbols are indirect, loading the value at address GV rather then the
/// value of GV itself. This means that the GlobalAddress must be in the base
/// or index register of the address, not the GV offset field.
bool X86Subtarget::GVRequiresExtraLoad(const GlobalValue* GV,
                                       const TargetMachine& TM,
                                       bool isDirectCall) const
{
  // FIXME: PIC
  if (TM.getRelocationModel() != Reloc::Static &&
      TM.getCodeModel() != CodeModel::Large) {
    if (isTargetDarwin()) {
      if (isDirectCall)
        return false;
      bool isDecl = GV->isDeclaration() && !GV->hasNotBeenReadFromBitcode();
      if (GV->hasHiddenVisibility() &&
          (Is64Bit || (!isDecl && !GV->hasCommonLinkage())))
        // If symbol visibility is hidden, the extra load is not needed if
        // target is x86-64 or the symbol is definitely defined in the current
        // translation unit.
        return false;
      return !isDirectCall && (isDecl || GV->mayBeOverridden());
    } else if (isTargetELF()) {
      // Extra load is needed for all externally visible.
      if (isDirectCall)
        return false;
      if (GV->hasLocalLinkage() || GV->hasHiddenVisibility())
        return false;
      return true;
    } else if (isTargetCygMing() || isTargetWindows()) {
      return (GV->hasDLLImportLinkage());
    }
  }
  return false;
}

/// True if accessing the GV requires a register.  This is a superset of the
/// cases where GVRequiresExtraLoad is true.  Some variations of PIC require
/// a register, but not an extra load.
bool X86Subtarget::GVRequiresRegister(const GlobalValue *GV,
                                       const TargetMachine& TM,
                                       bool isDirectCall) const
{
  if (GVRequiresExtraLoad(GV, TM, isDirectCall))
    return true;
  // Code below here need only consider cases where GVRequiresExtraLoad
  // returns false.
  if (TM.getRelocationModel() == Reloc::PIC_)
    return !isDirectCall && 
      (GV->hasLocalLinkage() || GV->hasExternalLinkage());
  return false;
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
#if defined(__x86_64__)
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
  if (IsIntel || IsAMD) {
    // Determine if bit test memory instructions are slow.
    unsigned Family = 0;
    unsigned Model  = 0;
    DetectFamilyModel(EAX, Family, Model);
    IsBTMemSlow = IsAMD || (Family == 6 && Model >= 13);

    X86::GetCpuIDAndInfo(0x80000001, &EAX, &EBX, &ECX, &EDX);
    HasX86_64 = (EDX >> 29) & 0x1;
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
        switch (Model) {
        case 1:  return "opteron";
        case 5:  return "athlon-fx"; // also opteron
        default: return "athlon64";
        }
    default:
      return "generic";
    }
  } else {
    return "generic";
  }
}

X86Subtarget::X86Subtarget(const Module &M, const std::string &FS, bool is64Bit)
  : AsmFlavor(AsmWriterFlavor)
  , PICStyle(PICStyles::None)
  , X86SSELevel(NoMMXSSE)
  , X863DNowLevel(NoThreeDNow)
  , HasX86_64(false)
  , IsBTMemSlow(false)
  , DarwinVers(0)
  , IsLinux(false)
  , stackAlignment(8)
  // FIXME: this is a known good value for Yonah. How about others?
  , MaxInlineSizeThreshold(128)
  , Is64Bit(is64Bit)
  , TargetType(isELF) { // Default to ELF unless otherwise specified.
    
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
    // If requesting codegen for X86-64, make sure that 64-bit features
    // are enabled.
    if (Is64Bit)
      HasX86_64 = true;
    // Make sure SSE2 is enabled; it is available on all X86-64 CPUs.
    if (Is64Bit && X86SSELevel < SSE2)
      X86SSELevel = SSE2;
  }

  DOUT << "Subtarget features: SSELevel " << X86SSELevel
       << ", 3DNowLevel " << X863DNowLevel
       << ", 64bit " << HasX86_64 << "\n";
  assert((!Is64Bit || HasX86_64) &&
         "64-bit code requested on a subtarget that doesn't support it!");

  // Set the boolean corresponding to the current target triple, or the default
  // if one cannot be determined, to true.
  const std::string& TT = M.getTargetTriple();
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
    }
  } else if (TT.empty()) {
#if defined(__CYGWIN__)
    TargetType = isCygwin;
#elif defined(__MINGW32__) || defined(__MINGW64__)
    TargetType = isMingw;
#elif defined(__APPLE__)
    TargetType = isDarwin;
#if __APPLE_CC__ > 5400
    DarwinVers = 9;  // GCC 5400+ is Leopard.
#else
    DarwinVers = 8;  // Minimum supported darwin is Tiger.
#endif
    
#elif defined(_WIN32) || defined(_WIN64)
    TargetType = isWindows;
#elif defined(__linux__)
    // Linux doesn't imply ELF, but we don't currently support anything else.
    TargetType = isELF;
    IsLinux = true;
#endif
  }

  // If the asm syntax hasn't been overridden on the command line, use whatever
  // the target wants.
  if (AsmFlavor == X86Subtarget::Unset) {
    AsmFlavor = (TargetType == isWindows)
      ? X86Subtarget::Intel : X86Subtarget::ATT;
  }

  // Stack alignment is 16 bytes on Darwin (both 32 and 64 bit) and for all 64
  // bit targets.
  if (TargetType == isDarwin || Is64Bit)
    stackAlignment = 16;

  if (StackAlignment)
    stackAlignment = StackAlignment;
}
