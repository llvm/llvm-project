//===-- Host.cpp - Implement OS Host Concept --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This header file implements the operating system Host concept.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/DataStream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Config/config.h"
#include <string.h>

// Include the platform-specific parts of this class.
#ifdef LLVM_ON_UNIX
#include "Unix/Host.inc"
#endif
#ifdef LLVM_ON_WIN32
#include "Windows/Host.inc"
#endif
#ifdef _MSC_VER
#include <intrin.h>
#endif
#if defined(__APPLE__) && (defined(__ppc__) || defined(__powerpc__))
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/host_info.h>
#include <mach/machine.h>
#endif

//===----------------------------------------------------------------------===//
//
//  Implementations of the CPU detection routines
//
//===----------------------------------------------------------------------===//

using namespace llvm;

#if defined(i386) || defined(__i386__) || defined(__x86__) || defined(_M_IX86)\
 || defined(__x86_64__) || defined(_M_AMD64) || defined (_M_X64)

/// GetX86CpuIDAndInfo - Execute the specified cpuid and return the 4 values in the
/// specified arguments.  If we can't run cpuid on the host, return true.
static bool GetX86CpuIDAndInfo(unsigned value, unsigned *rEAX,
                            unsigned *rEBX, unsigned *rECX, unsigned *rEDX) {
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
  #else
    return true;
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
// pedantic #else returns to appease -Wunreachable-code (so we don't generate
// postprocessed code that looks like "return true; return false;")
  #else
    return true;
  #endif
#else
  return true;
#endif
}

static void DetectX86FamilyModel(unsigned EAX, unsigned &Family,
                                 unsigned &Model) {
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

std::string sys::getHostCPUName() {
  unsigned EAX = 0, EBX = 0, ECX = 0, EDX = 0;
  if (GetX86CpuIDAndInfo(0x1, &EAX, &EBX, &ECX, &EDX))
    return "generic";
  unsigned Family = 0;
  unsigned Model  = 0;
  DetectX86FamilyModel(EAX, Family, Model);

  bool HasSSE3 = (ECX & 0x1);
  GetX86CpuIDAndInfo(0x80000001, &EAX, &EBX, &ECX, &EDX);
  bool Em64T = (EDX >> 29) & 0x1;

  union {
    unsigned u[3];
    char     c[12];
  } text;

  GetX86CpuIDAndInfo(0, &EAX, text.u+0, text.u+2, text.u+1);
  if (memcmp(text.c, "GenuineIntel", 12) == 0) {
    switch (Family) {
    case 3:
      return "i386";
    case 4:
      switch (Model) {
      case 0: // Intel486 DX processors
      case 1: // Intel486 DX processors
      case 2: // Intel486 SX processors
      case 3: // Intel487 processors, IntelDX2 OverDrive processors,
              // IntelDX2 processors
      case 4: // Intel486 SL processor
      case 5: // IntelSX2 processors
      case 7: // Write-Back Enhanced IntelDX2 processors
      case 8: // IntelDX4 OverDrive processors, IntelDX4 processors
      default: return "i486";
      }
    case 5:
      switch (Model) {
      case  1: // Pentium OverDrive processor for Pentium processor (60, 66),
               // Pentium processors (60, 66)
      case  2: // Pentium OverDrive processor for Pentium processor (75, 90,
               // 100, 120, 133), Pentium processors (75, 90, 100, 120, 133,
               // 150, 166, 200)
      case  3: // Pentium OverDrive processors for Intel486 processor-based
               // systems
        return "pentium";

      case  4: // Pentium OverDrive processor with MMX technology for Pentium
               // processor (75, 90, 100, 120, 133), Pentium processor with
               // MMX technology (166, 200)
        return "pentium-mmx";

      default: return "pentium";
      }
    case 6:
      switch (Model) {
      case  1: // Pentium Pro processor
        return "pentiumpro";

      case  3: // Intel Pentium II OverDrive processor, Pentium II processor,
               // model 03
      case  5: // Pentium II processor, model 05, Pentium II Xeon processor,
               // model 05, and Intel Celeron processor, model 05
      case  6: // Celeron processor, model 06
        return "pentium2";

      case  7: // Pentium III processor, model 07, and Pentium III Xeon
               // processor, model 07
      case  8: // Pentium III processor, model 08, Pentium III Xeon processor,
               // model 08, and Celeron processor, model 08
      case 10: // Pentium III Xeon processor, model 0Ah
      case 11: // Pentium III processor, model 0Bh
        return "pentium3";

      case  9: // Intel Pentium M processor, Intel Celeron M processor model 09.
      case 13: // Intel Pentium M processor, Intel Celeron M processor, model
               // 0Dh. All processors are manufactured using the 90 nm process.
        return "pentium-m";

      case 14: // Intel Core Duo processor, Intel Core Solo processor, model
               // 0Eh. All processors are manufactured using the 65 nm process.
        return "yonah";

      case 15: // Intel Core 2 Duo processor, Intel Core 2 Duo mobile
               // processor, Intel Core 2 Quad processor, Intel Core 2 Quad
               // mobile processor, Intel Core 2 Extreme processor, Intel
               // Pentium Dual-Core processor, Intel Xeon processor, model
               // 0Fh. All processors are manufactured using the 65 nm process.
      case 22: // Intel Celeron processor model 16h. All processors are
               // manufactured using the 65 nm process
        return "core2";

      case 21: // Intel EP80579 Integrated Processor and Intel EP80579
               // Integrated Processor with Intel QuickAssist Technology
        return "i686"; // FIXME: ???

      case 23: // Intel Core 2 Extreme processor, Intel Xeon processor, model
               // 17h. All processors are manufactured using the 45 nm process.
               //
               // 45nm: Penryn , Wolfdale, Yorkfield (XE)
        return "penryn";

      case 26: // Intel Core i7 processor and Intel Xeon processor. All
               // processors are manufactured using the 45 nm process.
      case 29: // Intel Xeon processor MP. All processors are manufactured using
               // the 45 nm process.
      case 30: // Intel(R) Core(TM) i7 CPU         870  @ 2.93GHz.
               // As found in a Summer 2010 model iMac.
      case 37: // Intel Core i7, laptop version.
      case 44: // Intel Core i7 processor and Intel Xeon processor. All
               // processors are manufactured using the 32 nm process.
      case 46: // Nehalem EX
      case 47: // Westmere EX
        return "corei7";

      // SandyBridge:
      case 42: // Intel Core i7 processor. All processors are manufactured
               // using the 32 nm process.
      case 45:
        return "corei7-avx";

      // Ivy Bridge:
      case 58:
        return "core-avx-i";

      case 28: // Most 45 nm Intel Atom processors
      case 38: // 45 nm Atom Lincroft
      case 39: // 32 nm Atom Medfield
      case 53: // 32 nm Atom Midview
      case 54: // 32 nm Atom Midview
        return "atom";

      default: return (Em64T) ? "x86-64" : "i686";
      }
    case 15: {
      switch (Model) {
      case  0: // Pentium 4 processor, Intel Xeon processor. All processors are
               // model 00h and manufactured using the 0.18 micron process.
      case  1: // Pentium 4 processor, Intel Xeon processor, Intel Xeon
               // processor MP, and Intel Celeron processor. All processors are
               // model 01h and manufactured using the 0.18 micron process.
      case  2: // Pentium 4 processor, Mobile Intel Pentium 4 processor - M,
               // Intel Xeon processor, Intel Xeon processor MP, Intel Celeron
               // processor, and Mobile Intel Celeron processor. All processors
               // are model 02h and manufactured using the 0.13 micron process.
        return (Em64T) ? "x86-64" : "pentium4";

      case  3: // Pentium 4 processor, Intel Xeon processor, Intel Celeron D
               // processor. All processors are model 03h and manufactured using
               // the 90 nm process.
      case  4: // Pentium 4 processor, Pentium 4 processor Extreme Edition,
               // Pentium D processor, Intel Xeon processor, Intel Xeon
               // processor MP, Intel Celeron D processor. All processors are
               // model 04h and manufactured using the 90 nm process.
      case  6: // Pentium 4 processor, Pentium D processor, Pentium processor
               // Extreme Edition, Intel Xeon processor, Intel Xeon processor
               // MP, Intel Celeron D processor. All processors are model 06h
               // and manufactured using the 65 nm process.
        return (Em64T) ? "nocona" : "prescott";

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
        case 10: return "geode";
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
        if (HasSSE3)
          return "k8-sse3";
        switch (Model) {
        case 1:  return "opteron";
        case 5:  return "athlon-fx"; // also opteron
        default: return "athlon64";
        }
      case 16:
        return "amdfam10";
      case 20:
        return "btver1";
      case 21:
        return "bdver1";
    default:
      return "generic";
    }
  }
  return "generic";
}
#elif defined(__APPLE__) && (defined(__ppc__) || defined(__powerpc__))
std::string sys::getHostCPUName() {
  host_basic_info_data_t hostInfo;
  mach_msg_type_number_t infoCount;

  infoCount = HOST_BASIC_INFO_COUNT;
  host_info(mach_host_self(), HOST_BASIC_INFO, (host_info_t)&hostInfo, 
            &infoCount);
            
  if (hostInfo.cpu_type != CPU_TYPE_POWERPC) return "generic";

  switch(hostInfo.cpu_subtype) {
  case CPU_SUBTYPE_POWERPC_601:   return "601";
  case CPU_SUBTYPE_POWERPC_602:   return "602";
  case CPU_SUBTYPE_POWERPC_603:   return "603";
  case CPU_SUBTYPE_POWERPC_603e:  return "603e";
  case CPU_SUBTYPE_POWERPC_603ev: return "603ev";
  case CPU_SUBTYPE_POWERPC_604:   return "604";
  case CPU_SUBTYPE_POWERPC_604e:  return "604e";
  case CPU_SUBTYPE_POWERPC_620:   return "620";
  case CPU_SUBTYPE_POWERPC_750:   return "750";
  case CPU_SUBTYPE_POWERPC_7400:  return "7400";
  case CPU_SUBTYPE_POWERPC_7450:  return "7450";
  case CPU_SUBTYPE_POWERPC_970:   return "970";
  default: ;
  }
  
  return "generic";
}
#elif defined(__linux__) && (defined(__ppc__) || defined(__powerpc__))
std::string sys::getHostCPUName() {
  // Access to the Processor Version Register (PVR) on PowerPC is privileged,
  // and so we must use an operating-system interface to determine the current
  // processor type. On Linux, this is exposed through the /proc/cpuinfo file.
  const char *generic = "generic";

  // Note: We cannot mmap /proc/cpuinfo here and then process the resulting
  // memory buffer because the 'file' has 0 size (it can be read from only
  // as a stream).

  std::string Err;
  DataStreamer *DS = getDataFileStreamer("/proc/cpuinfo", &Err);
  if (!DS) {
    DEBUG(dbgs() << "Unable to open /proc/cpuinfo: " << Err << "\n");
    return generic;
  }

  // The cpu line is second (after the 'processor: 0' line), so if this
  // buffer is too small then something has changed (or is wrong).
  char buffer[1024];
  size_t CPUInfoSize = DS->GetBytes((unsigned char*) buffer, sizeof(buffer));
  delete DS;

  const char *CPUInfoStart = buffer;
  const char *CPUInfoEnd = buffer + CPUInfoSize;

  const char *CIP = CPUInfoStart;

  const char *CPUStart = 0;
  size_t CPULen = 0;

  // We need to find the first line which starts with cpu, spaces, and a colon.
  // After the colon, there may be some additional spaces and then the cpu type.
  while (CIP < CPUInfoEnd && CPUStart == 0) {
    if (CIP < CPUInfoEnd && *CIP == '\n')
      ++CIP;

    if (CIP < CPUInfoEnd && *CIP == 'c') {
      ++CIP;
      if (CIP < CPUInfoEnd && *CIP == 'p') {
        ++CIP;
        if (CIP < CPUInfoEnd && *CIP == 'u') {
          ++CIP;
          while (CIP < CPUInfoEnd && (*CIP == ' ' || *CIP == '\t'))
            ++CIP;
  
          if (CIP < CPUInfoEnd && *CIP == ':') {
            ++CIP;
            while (CIP < CPUInfoEnd && (*CIP == ' ' || *CIP == '\t'))
              ++CIP;
  
            if (CIP < CPUInfoEnd) {
              CPUStart = CIP;
              while (CIP < CPUInfoEnd && (*CIP != ' ' && *CIP != '\t' &&
                                          *CIP != ',' && *CIP != '\n'))
                ++CIP;
              CPULen = CIP - CPUStart;
            }
          }
        }
      }
    }

    if (CPUStart == 0)
      while (CIP < CPUInfoEnd && *CIP != '\n')
        ++CIP;
  }

  if (CPUStart == 0)
    return generic;

  return StringSwitch<const char *>(StringRef(CPUStart, CPULen))
    .Case("604e", "604e")
    .Case("604", "604")
    .Case("7400", "7400")
    .Case("7410", "7400")
    .Case("7447", "7400")
    .Case("7455", "7450")
    .Case("G4", "g4")
    .Case("POWER4", "970")
    .Case("PPC970FX", "970")
    .Case("PPC970MP", "970")
    .Case("G5", "g5")
    .Case("POWER5", "g5")
    .Case("A2", "a2")
    .Case("POWER6", "pwr6")
    .Case("POWER7", "pwr7")
    .Default(generic);
}
#elif defined(__linux__) && defined(__arm__)
std::string sys::getHostCPUName() {
  // The cpuid register on arm is not accessible from user space. On Linux,
  // it is exposed through the /proc/cpuinfo file.
  // Note: We cannot mmap /proc/cpuinfo here and then process the resulting
  // memory buffer because the 'file' has 0 size (it can be read from only
  // as a stream).

  std::string Err;
  DataStreamer *DS = getDataFileStreamer("/proc/cpuinfo", &Err);
  if (!DS) {
    DEBUG(dbgs() << "Unable to open /proc/cpuinfo: " << Err << "\n");
    return "generic";
  }

  // Read 1024 bytes from /proc/cpuinfo, which should contain the CPU part line
  // in all cases.
  char buffer[1024];
  size_t CPUInfoSize = DS->GetBytes((unsigned char*) buffer, sizeof(buffer));
  delete DS;

  StringRef Str(buffer, CPUInfoSize);

  SmallVector<StringRef, 32> Lines;
  Str.split(Lines, "\n");

  // Look for the CPU implementer line.
  StringRef Implementer;
  for (unsigned I = 0, E = Lines.size(); I != E; ++I)
    if (Lines[I].startswith("CPU implementer"))
      Implementer = Lines[I].substr(15).ltrim("\t :");

  if (Implementer == "0x41") // ARM Ltd.
    // Look for the CPU part line.
    for (unsigned I = 0, E = Lines.size(); I != E; ++I)
      if (Lines[I].startswith("CPU part"))
        // The CPU part is a 3 digit hexadecimal number with a 0x prefix. The
        // values correspond to the "Part number" in the CP15/c0 register. The
        // contents are specified in the various processor manuals.
        return StringSwitch<const char *>(Lines[I].substr(8).ltrim("\t :"))
          .Case("0x926", "arm926ej-s")
          .Case("0xb02", "mpcore")
          .Case("0xb36", "arm1136j-s")
          .Case("0xb56", "arm1156t2-s")
          .Case("0xb76", "arm1176jz-s")
          .Case("0xc08", "cortex-a8")
          .Case("0xc09", "cortex-a9")
          .Case("0xc0f", "cortex-a15")
          .Case("0xc20", "cortex-m0")
          .Case("0xc23", "cortex-m3")
          .Case("0xc24", "cortex-m4")
          .Default("generic");

  return "generic";
}
#else
std::string sys::getHostCPUName() {
  return "generic";
}
#endif

bool sys::getHostCPUFeatures(StringMap<bool> &Features){
  return false;
}
