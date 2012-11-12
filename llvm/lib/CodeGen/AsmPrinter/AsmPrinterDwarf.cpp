//===-- AsmPrinterDwarf.cpp - AsmPrinter Dwarf Support --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Dwarf emissions parts of AsmPrinter.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/MachineLocation.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// Dwarf Emission Helper Routines
//===----------------------------------------------------------------------===//

/// EmitSLEB128 - emit the specified signed leb128 value.
void AsmPrinter::EmitSLEB128(int Value, const char *Desc) const {
  if (isVerbose() && Desc)
    OutStreamer.AddComment(Desc);

  OutStreamer.EmitSLEB128IntValue(Value);
}

/// EmitULEB128 - emit the specified signed leb128 value.
void AsmPrinter::EmitULEB128(unsigned Value, const char *Desc,
                             unsigned PadTo) const {
  if (isVerbose() && Desc)
    OutStreamer.AddComment(Desc);

  OutStreamer.EmitULEB128IntValue(Value, 0/*addrspace*/, PadTo);
}

/// EmitCFAByte - Emit a .byte 42 directive for a DW_CFA_xxx value.
void AsmPrinter::EmitCFAByte(unsigned Val) const {
  if (isVerbose()) {
    if (Val >= dwarf::DW_CFA_offset && Val < dwarf::DW_CFA_offset+64)
      OutStreamer.AddComment("DW_CFA_offset + Reg (" + 
                             Twine(Val-dwarf::DW_CFA_offset) + ")");
    else
      OutStreamer.AddComment(dwarf::CallFrameString(Val));
  }
  OutStreamer.EmitIntValue(Val, 1, 0/*addrspace*/);
}

static const char *DecodeDWARFEncoding(unsigned Encoding) {
  switch (Encoding) {
  case dwarf::DW_EH_PE_absptr: return "absptr";
  case dwarf::DW_EH_PE_omit:   return "omit";
  case dwarf::DW_EH_PE_pcrel:  return "pcrel";
  case dwarf::DW_EH_PE_udata4: return "udata4";
  case dwarf::DW_EH_PE_udata8: return "udata8";
  case dwarf::DW_EH_PE_sdata4: return "sdata4";
  case dwarf::DW_EH_PE_sdata8: return "sdata8";
  case dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_udata4: return "pcrel udata4";
  case dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata4: return "pcrel sdata4";
  case dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_udata8: return "pcrel udata8";
  case dwarf::DW_EH_PE_pcrel | dwarf::DW_EH_PE_sdata8: return "pcrel sdata8";
  case dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |dwarf::DW_EH_PE_udata4:
    return "indirect pcrel udata4";
  case dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |dwarf::DW_EH_PE_sdata4:
    return "indirect pcrel sdata4";
  case dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |dwarf::DW_EH_PE_udata8:
    return "indirect pcrel udata8";
  case dwarf::DW_EH_PE_indirect | dwarf::DW_EH_PE_pcrel |dwarf::DW_EH_PE_sdata8:
    return "indirect pcrel sdata8";
  }
  
  return "<unknown encoding>";
}


/// EmitEncodingByte - Emit a .byte 42 directive that corresponds to an
/// encoding.  If verbose assembly output is enabled, we output comments
/// describing the encoding.  Desc is an optional string saying what the
/// encoding is specifying (e.g. "LSDA").
void AsmPrinter::EmitEncodingByte(unsigned Val, const char *Desc) const {
  if (isVerbose()) {
    if (Desc != 0)
      OutStreamer.AddComment(Twine(Desc)+" Encoding = " +
                             Twine(DecodeDWARFEncoding(Val)));
    else
      OutStreamer.AddComment(Twine("Encoding = ") +
                             DecodeDWARFEncoding(Val));
  }
  
  OutStreamer.EmitIntValue(Val, 1, 0/*addrspace*/);
}

/// GetSizeOfEncodedValue - Return the size of the encoding in bytes.
unsigned AsmPrinter::GetSizeOfEncodedValue(unsigned Encoding) const {
  if (Encoding == dwarf::DW_EH_PE_omit)
    return 0;
  
  switch (Encoding & 0x07) {
  default: llvm_unreachable("Invalid encoded value.");
  case dwarf::DW_EH_PE_absptr: return TM.getDataLayout()->getPointerSize();
  case dwarf::DW_EH_PE_udata2: return 2;
  case dwarf::DW_EH_PE_udata4: return 4;
  case dwarf::DW_EH_PE_udata8: return 8;
  }
}

void AsmPrinter::EmitReference(const MCSymbol *Sym, unsigned Encoding) const {
  const TargetLoweringObjectFile &TLOF = getObjFileLowering();
  
  const MCExpr *Exp =
    TLOF.getExprForDwarfReference(Sym, Encoding, OutStreamer);
  OutStreamer.EmitAbsValue(Exp, GetSizeOfEncodedValue(Encoding));
}

void AsmPrinter::EmitReference(const GlobalValue *GV, unsigned Encoding)const{
  const TargetLoweringObjectFile &TLOF = getObjFileLowering();
  
  const MCExpr *Exp =
    TLOF.getExprForDwarfGlobalReference(GV, Mang, MMI, Encoding, OutStreamer);
  OutStreamer.EmitValue(Exp, GetSizeOfEncodedValue(Encoding), /*addrspace*/0);
}

/// EmitSectionOffset - Emit the 4-byte offset of Label from the start of its
/// section.  This can be done with a special directive if the target supports
/// it (e.g. cygwin) or by emitting it as an offset from a label at the start
/// of the section.
///
/// SectionLabel is a temporary label emitted at the start of the section that
/// Label lives in.
void AsmPrinter::EmitSectionOffset(const MCSymbol *Label,
                                   const MCSymbol *SectionLabel) const {
  // On COFF targets, we have to emit the special .secrel32 directive.
  if (MAI->getDwarfSectionOffsetDirective()) {
    OutStreamer.EmitCOFFSecRel32(Label);
    return;
  }
  
  // Get the section that we're referring to, based on SectionLabel.
  const MCSection &Section = SectionLabel->getSection();
  
  // If Label has already been emitted, verify that it is in the same section as
  // section label for sanity.
  assert((!Label->isInSection() || &Label->getSection() == &Section) &&
         "Section offset using wrong section base for label");
  
  // If the section in question will end up with an address of 0 anyway, we can
  // just emit an absolute reference to save a relocation.
  if (Section.isBaseAddressKnownZero()) {
    OutStreamer.EmitSymbolValue(Label, 4, 0/*AddrSpace*/);
    return;
  }
  
  // Otherwise, emit it as a label difference from the start of the section.
  EmitLabelDifference(Label, SectionLabel, 4);
}

//===----------------------------------------------------------------------===//
// Dwarf Lowering Routines
//===----------------------------------------------------------------------===//

/// EmitCFIFrameMove - Emit a frame instruction.
void AsmPrinter::EmitCFIFrameMove(const MachineMove &Move) const {
  const TargetRegisterInfo *RI = TM.getRegisterInfo();

  const MachineLocation &Dst = Move.getDestination();
  const MachineLocation &Src = Move.getSource();

  // If advancing cfa.
  if (Dst.isReg() && Dst.getReg() == MachineLocation::VirtualFP) {
    if (Src.getReg() == MachineLocation::VirtualFP) {
      OutStreamer.EmitCFIDefCfaOffset(-Src.getOffset());
    } else {
      // Reg + Offset
      OutStreamer.EmitCFIDefCfa(RI->getDwarfRegNum(Src.getReg(), true),
                                Src.getOffset());
    }
  } else if (Src.isReg() && Src.getReg() == MachineLocation::VirtualFP) {
    assert(Dst.isReg() && "Machine move not supported yet.");
    OutStreamer.EmitCFIDefCfaRegister(RI->getDwarfRegNum(Dst.getReg(), true));
  } else {
    assert(!Dst.isReg() && "Machine move not supported yet.");
    OutStreamer.EmitCFIOffset(RI->getDwarfRegNum(Src.getReg(), true),
                              Dst.getOffset());
  }
}
