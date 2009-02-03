//===-- llvm/Target/TargetMachOWriterInfo.h - MachO Writer Info--*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the TargetMachOWriterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETMACHOWRITERINFO_H
#define LLVM_TARGET_TARGETMACHOWRITERINFO_H

#include "llvm/CodeGen/MachineRelocation.h"

namespace llvm {

  class MachineBasicBlock;
  class OutputBuffer;

  //===--------------------------------------------------------------------===//
  //                        TargetMachOWriterInfo
  //===--------------------------------------------------------------------===//

  class TargetMachOWriterInfo {
    uint32_t CPUType;                 // CPU specifier
    uint32_t CPUSubType;              // Machine specifier
  public:
    // The various CPU_TYPE_* constants are already defined by at least one
    // system header file and create compilation errors if not respected.
#if !defined(CPU_TYPE_I386)
#define CPU_TYPE_I386       7
#endif
#if !defined(CPU_TYPE_X86_64)
#define CPU_TYPE_X86_64     (CPU_TYPE_I386 | 0x1000000)
#endif
#if !defined(CPU_TYPE_ARM)
#define CPU_TYPE_ARM        12
#endif
#if !defined(CPU_TYPE_SPARC)
#define CPU_TYPE_SPARC      14
#endif
#if !defined(CPU_TYPE_POWERPC)
#define CPU_TYPE_POWERPC    18
#endif
#if !defined(CPU_TYPE_POWERPC64)
#define CPU_TYPE_POWERPC64  (CPU_TYPE_POWERPC | 0x1000000)
#endif

    // Constants for the cputype field
    // see <mach/machine.h>
    enum {
      HDR_CPU_TYPE_I386      = CPU_TYPE_I386,
      HDR_CPU_TYPE_X86_64    = CPU_TYPE_X86_64,
      HDR_CPU_TYPE_ARM       = CPU_TYPE_ARM,
      HDR_CPU_TYPE_SPARC     = CPU_TYPE_SPARC,
      HDR_CPU_TYPE_POWERPC   = CPU_TYPE_POWERPC,
      HDR_CPU_TYPE_POWERPC64 = CPU_TYPE_POWERPC64
    };
      
#if !defined(CPU_SUBTYPE_I386_ALL)
#define CPU_SUBTYPE_I386_ALL    3
#endif
#if !defined(CPU_SUBTYPE_X86_64_ALL)
#define CPU_SUBTYPE_X86_64_ALL  3
#endif
#if !defined(CPU_SUBTYPE_ARM_ALL)
#define CPU_SUBTYPE_ARM_ALL     0
#endif
#if !defined(CPU_SUBTYPE_SPARC_ALL)
#define CPU_SUBTYPE_SPARC_ALL   0
#endif
#if !defined(CPU_SUBTYPE_POWERPC_ALL)
#define CPU_SUBTYPE_POWERPC_ALL 0
#endif

    // Constants for the cpusubtype field
    // see <mach/machine.h>
    enum {
      HDR_CPU_SUBTYPE_I386_ALL    = CPU_SUBTYPE_I386_ALL,
      HDR_CPU_SUBTYPE_X86_64_ALL  = CPU_SUBTYPE_X86_64_ALL,
      HDR_CPU_SUBTYPE_ARM_ALL     = CPU_SUBTYPE_ARM_ALL,
      HDR_CPU_SUBTYPE_SPARC_ALL   = CPU_SUBTYPE_SPARC_ALL,
      HDR_CPU_SUBTYPE_POWERPC_ALL = CPU_SUBTYPE_POWERPC_ALL
    };

    TargetMachOWriterInfo(uint32_t cputype, uint32_t cpusubtype)
      : CPUType(cputype), CPUSubType(cpusubtype) {}
    virtual ~TargetMachOWriterInfo();

    virtual MachineRelocation GetJTRelocation(unsigned Offset,
                                              MachineBasicBlock *MBB) const;

    virtual unsigned GetTargetRelocation(MachineRelocation &MR,
                                         unsigned FromIdx,
                                         unsigned ToAddr,
                                         unsigned ToIdx,
                                         OutputBuffer &RelocOut,
                                         OutputBuffer &SecOut,
                                         bool Scattered,
                                         bool Extern) const { return 0; }

    uint32_t getCPUType() const { return CPUType; }
    uint32_t getCPUSubType() const { return CPUSubType; }
  };

} // end llvm namespace

#endif // LLVM_TARGET_TARGETMACHOWRITERINFO_H
