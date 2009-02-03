//===-- PPCMachOWriterInfo.cpp - Mach-O Writer Info for the PowerPC -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Mach-O writer information for the PowerPC backend.
//
//===----------------------------------------------------------------------===//

#include "PPCMachOWriterInfo.h"
#include "PPCRelocations.h"
#include "PPCTargetMachine.h"
#include "llvm/CodeGen/MachORelocation.h"
#include "llvm/Support/OutputBuffer.h"
#include <cstdio>
using namespace llvm;

PPCMachOWriterInfo::PPCMachOWriterInfo(const PPCTargetMachine &TM)
  : TargetMachOWriterInfo(TM.getTargetData()->getPointerSizeInBits() == 64 ?
                          HDR_CPU_TYPE_POWERPC64 :
                          HDR_CPU_TYPE_POWERPC,
                          HDR_CPU_SUBTYPE_POWERPC_ALL) {}
PPCMachOWriterInfo::~PPCMachOWriterInfo() {}

/// GetTargetRelocation - For the MachineRelocation MR, convert it to one or
/// more PowerPC MachORelocation(s), add the new relocations to the
/// MachOSection, and rewrite the instruction at the section offset if required
/// by that relocation type.
unsigned PPCMachOWriterInfo::GetTargetRelocation(MachineRelocation &MR,
                                                 unsigned FromIdx,
                                                 unsigned ToAddr,
                                                 unsigned ToIdx,
                                                 OutputBuffer &RelocOut,
                                                 OutputBuffer &SecOut,
                                                 bool Scattered,
                                                 bool isExtern) const {
  unsigned NumRelocs = 0;
  uint64_t Addr = 0;

  // Get the address of whatever it is we're relocating, if possible.
  if (!isExtern)
    Addr = (uintptr_t)MR.getResultPointer() + ToAddr;

  switch ((PPC::RelocationType)MR.getRelocationType()) {
  default: assert(0 && "Unknown PPC relocation type!");
  case PPC::reloc_absolute_low_ix:
    assert(0 && "Unhandled PPC relocation type!");
    break;
  case PPC::reloc_vanilla:
    {
      // FIXME: need to handle 64 bit vanilla relocs
      MachORelocation VANILLA(MR.getMachineCodeOffset(), ToIdx,
                              false, 2, isExtern,
                              PPC_RELOC_VANILLA,
                              Scattered, (intptr_t)MR.getResultPointer());
      ++NumRelocs;

      if (Scattered) {
        RelocOut.outword(VANILLA.getPackedFields());
        RelocOut.outword(VANILLA.getAddress());
      } else {
        RelocOut.outword(VANILLA.getAddress());
        RelocOut.outword(VANILLA.getPackedFields());
      }
      
      intptr_t SymbolOffset;

      if (Scattered)
        SymbolOffset = Addr + MR.getConstantVal();
      else
        SymbolOffset = Addr;

      printf("vanilla fixup: sec_%x[%x] = %x\n", FromIdx,
             unsigned(MR.getMachineCodeOffset()),
             unsigned(SymbolOffset));
      SecOut.fixword(SymbolOffset, MR.getMachineCodeOffset());
    }
    break;
  case PPC::reloc_pcrel_bx:
    {
      // FIXME: Presumably someday we will need to branch to other, non-extern
      // functions too.  Need to figure out some way to distinguish between
      // target is BB and target is function.
      if (isExtern) {
        MachORelocation BR24(MR.getMachineCodeOffset(), ToIdx, true, 2, 
                             isExtern, PPC_RELOC_BR24, Scattered, 
                             (intptr_t)MR.getMachineCodeOffset());
        RelocOut.outword(BR24.getAddress());
        RelocOut.outword(BR24.getPackedFields());
        ++NumRelocs;
      }

      Addr -= MR.getMachineCodeOffset();
      Addr >>= 2;
      Addr &= 0xFFFFFF;
      Addr <<= 2;
      Addr |= (SecOut[MR.getMachineCodeOffset()] << 24);
      Addr |= (SecOut[MR.getMachineCodeOffset()+3] & 0x3);
      SecOut.fixword(Addr, MR.getMachineCodeOffset());
      break;
    }
  case PPC::reloc_pcrel_bcx:
    {
      Addr -= MR.getMachineCodeOffset();
      Addr &= 0xFFFC;

      SecOut.fixhalf(Addr, MR.getMachineCodeOffset() + 2);
      break;
    }
  case PPC::reloc_absolute_high:
    {
      MachORelocation HA16(MR.getMachineCodeOffset(), ToIdx, false, 2,
                           isExtern, PPC_RELOC_HA16);
      MachORelocation PAIR(Addr & 0xFFFF, 0xFFFFFF, false, 2, isExtern,
                           PPC_RELOC_PAIR);
      NumRelocs = 2;

      RelocOut.outword(HA16.getRawAddress());
      RelocOut.outword(HA16.getPackedFields());
      RelocOut.outword(PAIR.getRawAddress());
      RelocOut.outword(PAIR.getPackedFields());

      Addr += 0x8000;

      SecOut.fixhalf(Addr >> 16, MR.getMachineCodeOffset() + 2);
      break;
    }
  case PPC::reloc_absolute_low:
    {
      MachORelocation LO16(MR.getMachineCodeOffset(), ToIdx, false, 2,
                           isExtern, PPC_RELOC_LO16);
      MachORelocation PAIR(Addr >> 16, 0xFFFFFF, false, 2, isExtern,
                           PPC_RELOC_PAIR);
      NumRelocs = 2;

      RelocOut.outword(LO16.getRawAddress());
      RelocOut.outword(LO16.getPackedFields());
      RelocOut.outword(PAIR.getRawAddress());
      RelocOut.outword(PAIR.getPackedFields());

      SecOut.fixhalf(Addr, MR.getMachineCodeOffset() + 2);
      break;
    }
  }

  return NumRelocs;
}
