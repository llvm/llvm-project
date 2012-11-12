//===-- ARMAsmPrinter.cpp - Print machine code to an ARM .s file ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format ARM assembly language.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "ARMAsmPrinter.h"
#include "ARM.h"
#include "ARMBuildAttrs.h"
#include "ARMConstantPoolValue.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMTargetMachine.h"
#include "ARMTargetObjectFile.h"
#include "InstPrinter/ARMInstPrinter.h"
#include "MCTargetDesc/ARMAddressingModes.h"
#include "MCTargetDesc/ARMMCExpr.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Constants.h"
#include "llvm/DebugInfo.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Target/Mangler.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
using namespace llvm;

namespace {

  // Per section and per symbol attributes are not supported.
  // To implement them we would need the ability to delay this emission
  // until the assembly file is fully parsed/generated as only then do we
  // know the symbol and section numbers.
  class AttributeEmitter {
  public:
    virtual void MaybeSwitchVendor(StringRef Vendor) = 0;
    virtual void EmitAttribute(unsigned Attribute, unsigned Value) = 0;
    virtual void EmitTextAttribute(unsigned Attribute, StringRef String) = 0;
    virtual void Finish() = 0;
    virtual ~AttributeEmitter() {}
  };

  class AsmAttributeEmitter : public AttributeEmitter {
    MCStreamer &Streamer;

  public:
    AsmAttributeEmitter(MCStreamer &Streamer_) : Streamer(Streamer_) {}
    void MaybeSwitchVendor(StringRef Vendor) { }

    void EmitAttribute(unsigned Attribute, unsigned Value) {
      Streamer.EmitRawText("\t.eabi_attribute " +
                           Twine(Attribute) + ", " + Twine(Value));
    }

    void EmitTextAttribute(unsigned Attribute, StringRef String) {
      switch (Attribute) {
      default: llvm_unreachable("Unsupported Text attribute in ASM Mode");
      case ARMBuildAttrs::CPU_name:
        Streamer.EmitRawText(StringRef("\t.cpu ") + String.lower());
        break;
      /* GAS requires .fpu to be emitted regardless of EABI attribute */
      case ARMBuildAttrs::Advanced_SIMD_arch:
      case ARMBuildAttrs::VFP_arch:
        Streamer.EmitRawText(StringRef("\t.fpu ") + String.lower());
        break;
      }
    }
    void Finish() { }
  };

  class ObjectAttributeEmitter : public AttributeEmitter {
    // This structure holds all attributes, accounting for
    // their string/numeric value, so we can later emmit them
    // in declaration order, keeping all in the same vector
    struct AttributeItemType {
      enum {
        HiddenAttribute = 0,
        NumericAttribute,
        TextAttribute
      } Type;
      unsigned Tag;
      unsigned IntValue;
      StringRef StringValue;
    } AttributeItem;

    MCObjectStreamer &Streamer;
    StringRef CurrentVendor;
    SmallVector<AttributeItemType, 64> Contents;

    // Account for the ULEB/String size of each item,
    // not just the number of items
    size_t ContentsSize;
    // FIXME: this should be in a more generic place, but
    // getULEBSize() is in MCAsmInfo and will be moved to MCDwarf
    size_t getULEBSize(int Value) {
      size_t Size = 0;
      do {
        Value >>= 7;
        Size += sizeof(int8_t); // Is this really necessary?
      } while (Value);
      return Size;
    }

  public:
    ObjectAttributeEmitter(MCObjectStreamer &Streamer_) :
      Streamer(Streamer_), CurrentVendor(""), ContentsSize(0) { }

    void MaybeSwitchVendor(StringRef Vendor) {
      assert(!Vendor.empty() && "Vendor cannot be empty.");

      if (CurrentVendor.empty())
        CurrentVendor = Vendor;
      else if (CurrentVendor == Vendor)
        return;
      else
        Finish();

      CurrentVendor = Vendor;

      assert(Contents.size() == 0);
    }

    void EmitAttribute(unsigned Attribute, unsigned Value) {
      AttributeItemType attr = {
        AttributeItemType::NumericAttribute,
        Attribute,
        Value,
        StringRef("")
      };
      ContentsSize += getULEBSize(Attribute);
      ContentsSize += getULEBSize(Value);
      Contents.push_back(attr);
    }

    void EmitTextAttribute(unsigned Attribute, StringRef String) {
      AttributeItemType attr = {
        AttributeItemType::TextAttribute,
        Attribute,
        0,
        String
      };
      ContentsSize += getULEBSize(Attribute);
      // String + \0
      ContentsSize += String.size()+1;

      Contents.push_back(attr);
    }

    void Finish() {
      // Vendor size + Vendor name + '\0'
      const size_t VendorHeaderSize = 4 + CurrentVendor.size() + 1;

      // Tag + Tag Size
      const size_t TagHeaderSize = 1 + 4;

      Streamer.EmitIntValue(VendorHeaderSize + TagHeaderSize + ContentsSize, 4);
      Streamer.EmitBytes(CurrentVendor, 0);
      Streamer.EmitIntValue(0, 1); // '\0'

      Streamer.EmitIntValue(ARMBuildAttrs::File, 1);
      Streamer.EmitIntValue(TagHeaderSize + ContentsSize, 4);

      // Size should have been accounted for already, now
      // emit each field as its type (ULEB or String)
      for (unsigned int i=0; i<Contents.size(); ++i) {
        AttributeItemType item = Contents[i];
        Streamer.EmitULEB128IntValue(item.Tag, 0);
        switch (item.Type) {
        default: llvm_unreachable("Invalid attribute type");
        case AttributeItemType::NumericAttribute:
          Streamer.EmitULEB128IntValue(item.IntValue, 0);
          break;
        case AttributeItemType::TextAttribute:
          Streamer.EmitBytes(item.StringValue.upper(), 0);
          Streamer.EmitIntValue(0, 1); // '\0'
          break;
        }
      }

      Contents.clear();
    }
  };

} // end of anonymous namespace

MachineLocation ARMAsmPrinter::
getDebugValueLocation(const MachineInstr *MI) const {
  MachineLocation Location;
  assert(MI->getNumOperands() == 4 && "Invalid no. of machine operands!");
  // Frame address.  Currently handles register +- offset only.
  if (MI->getOperand(0).isReg() && MI->getOperand(1).isImm())
    Location.set(MI->getOperand(0).getReg(), MI->getOperand(1).getImm());
  else {
    DEBUG(dbgs() << "DBG_VALUE instruction ignored! " << *MI << "\n");
  }
  return Location;
}

/// EmitDwarfRegOp - Emit dwarf register operation.
void ARMAsmPrinter::EmitDwarfRegOp(const MachineLocation &MLoc) const {
  const TargetRegisterInfo *RI = TM.getRegisterInfo();
  if (RI->getDwarfRegNum(MLoc.getReg(), false) != -1)
    AsmPrinter::EmitDwarfRegOp(MLoc);
  else {
    unsigned Reg = MLoc.getReg();
    if (Reg >= ARM::S0 && Reg <= ARM::S31) {
      assert(ARM::S0 + 31 == ARM::S31 && "Unexpected ARM S register numbering");
      // S registers are described as bit-pieces of a register
      // S[2x] = DW_OP_regx(256 + (x>>1)) DW_OP_bit_piece(32, 0)
      // S[2x+1] = DW_OP_regx(256 + (x>>1)) DW_OP_bit_piece(32, 32)

      unsigned SReg = Reg - ARM::S0;
      bool odd = SReg & 0x1;
      unsigned Rx = 256 + (SReg >> 1);

      OutStreamer.AddComment("DW_OP_regx for S register");
      EmitInt8(dwarf::DW_OP_regx);

      OutStreamer.AddComment(Twine(SReg));
      EmitULEB128(Rx);

      if (odd) {
        OutStreamer.AddComment("DW_OP_bit_piece 32 32");
        EmitInt8(dwarf::DW_OP_bit_piece);
        EmitULEB128(32);
        EmitULEB128(32);
      } else {
        OutStreamer.AddComment("DW_OP_bit_piece 32 0");
        EmitInt8(dwarf::DW_OP_bit_piece);
        EmitULEB128(32);
        EmitULEB128(0);
      }
    } else if (Reg >= ARM::Q0 && Reg <= ARM::Q15) {
      assert(ARM::Q0 + 15 == ARM::Q15 && "Unexpected ARM Q register numbering");
      // Q registers Q0-Q15 are described by composing two D registers together.
      // Qx = DW_OP_regx(256+2x) DW_OP_piece(8) DW_OP_regx(256+2x+1)
      // DW_OP_piece(8)

      unsigned QReg = Reg - ARM::Q0;
      unsigned D1 = 256 + 2 * QReg;
      unsigned D2 = D1 + 1;

      OutStreamer.AddComment("DW_OP_regx for Q register: D1");
      EmitInt8(dwarf::DW_OP_regx);
      EmitULEB128(D1);
      OutStreamer.AddComment("DW_OP_piece 8");
      EmitInt8(dwarf::DW_OP_piece);
      EmitULEB128(8);

      OutStreamer.AddComment("DW_OP_regx for Q register: D2");
      EmitInt8(dwarf::DW_OP_regx);
      EmitULEB128(D2);
      OutStreamer.AddComment("DW_OP_piece 8");
      EmitInt8(dwarf::DW_OP_piece);
      EmitULEB128(8);
    }
  }
}

void ARMAsmPrinter::EmitFunctionBodyEnd() {
  // Make sure to terminate any constant pools that were at the end
  // of the function.
  if (!InConstantPool)
    return;
  InConstantPool = false;
  OutStreamer.EmitDataRegion(MCDR_DataRegionEnd);
}

void ARMAsmPrinter::EmitFunctionEntryLabel() {
  if (AFI->isThumbFunction()) {
    OutStreamer.EmitAssemblerFlag(MCAF_Code16);
    OutStreamer.EmitThumbFunc(CurrentFnSym);
  }

  OutStreamer.EmitLabel(CurrentFnSym);
}

void ARMAsmPrinter::EmitXXStructor(const Constant *CV) {
  uint64_t Size = TM.getDataLayout()->getTypeAllocSize(CV->getType());
  assert(Size && "C++ constructor pointer had zero size!");

  const GlobalValue *GV = dyn_cast<GlobalValue>(CV->stripPointerCasts());
  assert(GV && "C++ constructor pointer was not a GlobalValue!");

  const MCExpr *E = MCSymbolRefExpr::Create(Mang->getSymbol(GV),
                                            (Subtarget->isTargetDarwin()
                                             ? MCSymbolRefExpr::VK_None
                                             : MCSymbolRefExpr::VK_ARM_TARGET1),
                                            OutContext);
  
  OutStreamer.EmitValue(E, Size);
}

/// runOnMachineFunction - This uses the EmitInstruction()
/// method to print assembly for each instruction.
///
bool ARMAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  AFI = MF.getInfo<ARMFunctionInfo>();
  MCP = MF.getConstantPool();

  return AsmPrinter::runOnMachineFunction(MF);
}

void ARMAsmPrinter::printOperand(const MachineInstr *MI, int OpNum,
                                 raw_ostream &O, const char *Modifier) {
  const MachineOperand &MO = MI->getOperand(OpNum);
  unsigned TF = MO.getTargetFlags();

  switch (MO.getType()) {
  default: llvm_unreachable("<unknown operand type>");
  case MachineOperand::MO_Register: {
    unsigned Reg = MO.getReg();
    assert(TargetRegisterInfo::isPhysicalRegister(Reg));
    assert(!MO.getSubReg() && "Subregs should be eliminated!");
    O << ARMInstPrinter::getRegisterName(Reg);
    break;
  }
  case MachineOperand::MO_Immediate: {
    int64_t Imm = MO.getImm();
    O << '#';
    if ((Modifier && strcmp(Modifier, "lo16") == 0) ||
        (TF == ARMII::MO_LO16))
      O << ":lower16:";
    else if ((Modifier && strcmp(Modifier, "hi16") == 0) ||
             (TF == ARMII::MO_HI16))
      O << ":upper16:";
    O << Imm;
    break;
  }
  case MachineOperand::MO_MachineBasicBlock:
    O << *MO.getMBB()->getSymbol();
    return;
  case MachineOperand::MO_GlobalAddress: {
    const GlobalValue *GV = MO.getGlobal();
    if ((Modifier && strcmp(Modifier, "lo16") == 0) ||
        (TF & ARMII::MO_LO16))
      O << ":lower16:";
    else if ((Modifier && strcmp(Modifier, "hi16") == 0) ||
             (TF & ARMII::MO_HI16))
      O << ":upper16:";
    O << *Mang->getSymbol(GV);

    printOffset(MO.getOffset(), O);
    if (TF == ARMII::MO_PLT)
      O << "(PLT)";
    break;
  }
  case MachineOperand::MO_ExternalSymbol: {
    O << *GetExternalSymbolSymbol(MO.getSymbolName());
    if (TF == ARMII::MO_PLT)
      O << "(PLT)";
    break;
  }
  case MachineOperand::MO_ConstantPoolIndex:
    O << *GetCPISymbol(MO.getIndex());
    break;
  case MachineOperand::MO_JumpTableIndex:
    O << *GetJTISymbol(MO.getIndex());
    break;
  }
}

//===--------------------------------------------------------------------===//

MCSymbol *ARMAsmPrinter::
GetARMJTIPICJumpTableLabel2(unsigned uid, unsigned uid2) const {
  SmallString<60> Name;
  raw_svector_ostream(Name) << MAI->getPrivateGlobalPrefix() << "JTI"
    << getFunctionNumber() << '_' << uid << '_' << uid2;
  return OutContext.GetOrCreateSymbol(Name.str());
}


MCSymbol *ARMAsmPrinter::GetARMSJLJEHLabel(void) const {
  SmallString<60> Name;
  raw_svector_ostream(Name) << MAI->getPrivateGlobalPrefix() << "SJLJEH"
    << getFunctionNumber();
  return OutContext.GetOrCreateSymbol(Name.str());
}

bool ARMAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNum,
                                    unsigned AsmVariant, const char *ExtraCode,
                                    raw_ostream &O) {
  // Does this asm operand have a single letter operand modifier?
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0) return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default:
      // See if this is a generic print operand
      return AsmPrinter::PrintAsmOperand(MI, OpNum, AsmVariant, ExtraCode, O);
    case 'a': // Print as a memory address.
      if (MI->getOperand(OpNum).isReg()) {
        O << "["
          << ARMInstPrinter::getRegisterName(MI->getOperand(OpNum).getReg())
          << "]";
        return false;
      }
      // Fallthrough
    case 'c': // Don't print "#" before an immediate operand.
      if (!MI->getOperand(OpNum).isImm())
        return true;
      O << MI->getOperand(OpNum).getImm();
      return false;
    case 'P': // Print a VFP double precision register.
    case 'q': // Print a NEON quad precision register.
      printOperand(MI, OpNum, O);
      return false;
    case 'y': // Print a VFP single precision register as indexed double.
      if (MI->getOperand(OpNum).isReg()) {
        unsigned Reg = MI->getOperand(OpNum).getReg();
        const TargetRegisterInfo *TRI = MF->getTarget().getRegisterInfo();
        // Find the 'd' register that has this 's' register as a sub-register,
        // and determine the lane number.
        for (MCSuperRegIterator SR(Reg, TRI); SR.isValid(); ++SR) {
          if (!ARM::DPRRegClass.contains(*SR))
            continue;
          bool Lane0 = TRI->getSubReg(*SR, ARM::ssub_0) == Reg;
          O << ARMInstPrinter::getRegisterName(*SR) << (Lane0 ? "[0]" : "[1]");
          return false;
        }
      }
      return true;
    case 'B': // Bitwise inverse of integer or symbol without a preceding #.
      if (!MI->getOperand(OpNum).isImm())
        return true;
      O << ~(MI->getOperand(OpNum).getImm());
      return false;
    case 'L': // The low 16 bits of an immediate constant.
      if (!MI->getOperand(OpNum).isImm())
        return true;
      O << (MI->getOperand(OpNum).getImm() & 0xffff);
      return false;
    case 'M': { // A register range suitable for LDM/STM.
      if (!MI->getOperand(OpNum).isReg())
        return true;
      const MachineOperand &MO = MI->getOperand(OpNum);
      unsigned RegBegin = MO.getReg();
      // This takes advantage of the 2 operand-ness of ldm/stm and that we've
      // already got the operands in registers that are operands to the
      // inline asm statement.

      O << "{" << ARMInstPrinter::getRegisterName(RegBegin);

      // FIXME: The register allocator not only may not have given us the
      // registers in sequence, but may not be in ascending registers. This
      // will require changes in the register allocator that'll need to be
      // propagated down here if the operands change.
      unsigned RegOps = OpNum + 1;
      while (MI->getOperand(RegOps).isReg()) {
        O << ", "
          << ARMInstPrinter::getRegisterName(MI->getOperand(RegOps).getReg());
        RegOps++;
      }

      O << "}";

      return false;
    }
    case 'R': // The most significant register of a pair.
    case 'Q': { // The least significant register of a pair.
      if (OpNum == 0)
        return true;
      const MachineOperand &FlagsOP = MI->getOperand(OpNum - 1);
      if (!FlagsOP.isImm())
        return true;
      unsigned Flags = FlagsOP.getImm();
      unsigned NumVals = InlineAsm::getNumOperandRegisters(Flags);
      if (NumVals != 2)
        return true;
      unsigned RegOp = ExtraCode[0] == 'Q' ? OpNum : OpNum + 1;
      if (RegOp >= MI->getNumOperands())
        return true;
      const MachineOperand &MO = MI->getOperand(RegOp);
      if (!MO.isReg())
        return true;
      unsigned Reg = MO.getReg();
      O << ARMInstPrinter::getRegisterName(Reg);
      return false;
    }

    case 'e': // The low doubleword register of a NEON quad register.
    case 'f': { // The high doubleword register of a NEON quad register.
      if (!MI->getOperand(OpNum).isReg())
        return true;
      unsigned Reg = MI->getOperand(OpNum).getReg();
      if (!ARM::QPRRegClass.contains(Reg))
        return true;
      const TargetRegisterInfo *TRI = MF->getTarget().getRegisterInfo();
      unsigned SubReg = TRI->getSubReg(Reg, ExtraCode[0] == 'e' ?
                                       ARM::dsub_0 : ARM::dsub_1);
      O << ARMInstPrinter::getRegisterName(SubReg);
      return false;
    }

    // This modifier is not yet supported.
    case 'h': // A range of VFP/NEON registers suitable for VLD1/VST1.
      return true;
    case 'H': { // The highest-numbered register of a pair.
      const MachineOperand &MO = MI->getOperand(OpNum);
      if (!MO.isReg())
        return true;
      const TargetRegisterClass &RC = ARM::GPRRegClass;
      const MachineFunction &MF = *MI->getParent()->getParent();
      const TargetRegisterInfo *TRI = MF.getTarget().getRegisterInfo();

      unsigned RegIdx = TRI->getEncodingValue(MO.getReg());
      RegIdx |= 1; //The odd register is also the higher-numbered one of a pair.

      unsigned Reg = RC.getRegister(RegIdx);
      O << ARMInstPrinter::getRegisterName(Reg);
      return false;
    }
    }
  }

  printOperand(MI, OpNum, O);
  return false;
}

bool ARMAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                          unsigned OpNum, unsigned AsmVariant,
                                          const char *ExtraCode,
                                          raw_ostream &O) {
  // Does this asm operand have a single letter operand modifier?
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0) return true; // Unknown modifier.

    switch (ExtraCode[0]) {
      case 'A': // A memory operand for a VLD1/VST1 instruction.
      default: return true;  // Unknown modifier.
      case 'm': // The base register of a memory operand.
        if (!MI->getOperand(OpNum).isReg())
          return true;
        O << ARMInstPrinter::getRegisterName(MI->getOperand(OpNum).getReg());
        return false;
    }
  }

  const MachineOperand &MO = MI->getOperand(OpNum);
  assert(MO.isReg() && "unexpected inline asm memory operand");
  O << "[" << ARMInstPrinter::getRegisterName(MO.getReg()) << "]";
  return false;
}

void ARMAsmPrinter::EmitStartOfAsmFile(Module &M) {
  if (Subtarget->isTargetDarwin()) {
    Reloc::Model RelocM = TM.getRelocationModel();
    if (RelocM == Reloc::PIC_ || RelocM == Reloc::DynamicNoPIC) {
      // Declare all the text sections up front (before the DWARF sections
      // emitted by AsmPrinter::doInitialization) so the assembler will keep
      // them together at the beginning of the object file.  This helps
      // avoid out-of-range branches that are due a fundamental limitation of
      // the way symbol offsets are encoded with the current Darwin ARM
      // relocations.
      const TargetLoweringObjectFileMachO &TLOFMacho =
        static_cast<const TargetLoweringObjectFileMachO &>(
          getObjFileLowering());

      // Collect the set of sections our functions will go into.
      SetVector<const MCSection *, SmallVector<const MCSection *, 8>,
        SmallPtrSet<const MCSection *, 8> > TextSections;
      // Default text section comes first.
      TextSections.insert(TLOFMacho.getTextSection());
      // Now any user defined text sections from function attributes.
      for (Module::iterator F = M.begin(), e = M.end(); F != e; ++F)
        if (!F->isDeclaration() && !F->hasAvailableExternallyLinkage())
          TextSections.insert(TLOFMacho.SectionForGlobal(F, Mang, TM));
      // Now the coalescable sections.
      TextSections.insert(TLOFMacho.getTextCoalSection());
      TextSections.insert(TLOFMacho.getConstTextCoalSection());

      // Emit the sections in the .s file header to fix the order.
      for (unsigned i = 0, e = TextSections.size(); i != e; ++i)
        OutStreamer.SwitchSection(TextSections[i]);

      if (RelocM == Reloc::DynamicNoPIC) {
        const MCSection *sect =
          OutContext.getMachOSection("__TEXT", "__symbol_stub4",
                                     MCSectionMachO::S_SYMBOL_STUBS,
                                     12, SectionKind::getText());
        OutStreamer.SwitchSection(sect);
      } else {
        const MCSection *sect =
          OutContext.getMachOSection("__TEXT", "__picsymbolstub4",
                                     MCSectionMachO::S_SYMBOL_STUBS,
                                     16, SectionKind::getText());
        OutStreamer.SwitchSection(sect);
      }
      const MCSection *StaticInitSect =
        OutContext.getMachOSection("__TEXT", "__StaticInit",
                                   MCSectionMachO::S_REGULAR |
                                   MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                                   SectionKind::getText());
      OutStreamer.SwitchSection(StaticInitSect);
    }
  }

  // Use unified assembler syntax.
  OutStreamer.EmitAssemblerFlag(MCAF_SyntaxUnified);

  // Emit ARM Build Attributes
  if (Subtarget->isTargetELF())
    emitAttributes();
}


void ARMAsmPrinter::EmitEndOfAsmFile(Module &M) {
  if (Subtarget->isTargetDarwin()) {
    // All darwin targets use mach-o.
    const TargetLoweringObjectFileMachO &TLOFMacho =
      static_cast<const TargetLoweringObjectFileMachO &>(getObjFileLowering());
    MachineModuleInfoMachO &MMIMacho =
      MMI->getObjFileInfo<MachineModuleInfoMachO>();

    // Output non-lazy-pointers for external and common global variables.
    MachineModuleInfoMachO::SymbolListTy Stubs = MMIMacho.GetGVStubList();

    if (!Stubs.empty()) {
      // Switch with ".non_lazy_symbol_pointer" directive.
      OutStreamer.SwitchSection(TLOFMacho.getNonLazySymbolPointerSection());
      EmitAlignment(2);
      for (unsigned i = 0, e = Stubs.size(); i != e; ++i) {
        // L_foo$stub:
        OutStreamer.EmitLabel(Stubs[i].first);
        //   .indirect_symbol _foo
        MachineModuleInfoImpl::StubValueTy &MCSym = Stubs[i].second;
        OutStreamer.EmitSymbolAttribute(MCSym.getPointer(),MCSA_IndirectSymbol);

        if (MCSym.getInt())
          // External to current translation unit.
          OutStreamer.EmitIntValue(0, 4/*size*/, 0/*addrspace*/);
        else
          // Internal to current translation unit.
          //
          // When we place the LSDA into the TEXT section, the type info
          // pointers need to be indirect and pc-rel. We accomplish this by
          // using NLPs; however, sometimes the types are local to the file.
          // We need to fill in the value for the NLP in those cases.
          OutStreamer.EmitValue(MCSymbolRefExpr::Create(MCSym.getPointer(),
                                                        OutContext),
                                4/*size*/, 0/*addrspace*/);
      }

      Stubs.clear();
      OutStreamer.AddBlankLine();
    }

    Stubs = MMIMacho.GetHiddenGVStubList();
    if (!Stubs.empty()) {
      OutStreamer.SwitchSection(getObjFileLowering().getDataSection());
      EmitAlignment(2);
      for (unsigned i = 0, e = Stubs.size(); i != e; ++i) {
        // L_foo$stub:
        OutStreamer.EmitLabel(Stubs[i].first);
        //   .long _foo
        OutStreamer.EmitValue(MCSymbolRefExpr::
                              Create(Stubs[i].second.getPointer(),
                                     OutContext),
                              4/*size*/, 0/*addrspace*/);
      }

      Stubs.clear();
      OutStreamer.AddBlankLine();
    }

    // Funny Darwin hack: This flag tells the linker that no global symbols
    // contain code that falls through to other global symbols (e.g. the obvious
    // implementation of multiple entry points).  If this doesn't occur, the
    // linker can safely perform dead code stripping.  Since LLVM never
    // generates code that does this, it is always safe to set.
    OutStreamer.EmitAssemblerFlag(MCAF_SubsectionsViaSymbols);
  }
}

//===----------------------------------------------------------------------===//
// Helper routines for EmitStartOfAsmFile() and EmitEndOfAsmFile()
// FIXME:
// The following seem like one-off assembler flags, but they actually need
// to appear in the .ARM.attributes section in ELF.
// Instead of subclassing the MCELFStreamer, we do the work here.

void ARMAsmPrinter::emitAttributes() {

  emitARMAttributeSection();

  /* GAS expect .fpu to be emitted, regardless of VFP build attribute */
  bool emitFPU = false;
  AttributeEmitter *AttrEmitter;
  if (OutStreamer.hasRawTextSupport()) {
    AttrEmitter = new AsmAttributeEmitter(OutStreamer);
    emitFPU = true;
  } else {
    MCObjectStreamer &O = static_cast<MCObjectStreamer&>(OutStreamer);
    AttrEmitter = new ObjectAttributeEmitter(O);
  }

  AttrEmitter->MaybeSwitchVendor("aeabi");

  std::string CPUString = Subtarget->getCPUString();

  if (CPUString == "cortex-a8" ||
      Subtarget->isCortexA8()) {
    AttrEmitter->EmitTextAttribute(ARMBuildAttrs::CPU_name, "cortex-a8");
    AttrEmitter->EmitAttribute(ARMBuildAttrs::CPU_arch, ARMBuildAttrs::v7);
    AttrEmitter->EmitAttribute(ARMBuildAttrs::CPU_arch_profile,
                               ARMBuildAttrs::ApplicationProfile);
    AttrEmitter->EmitAttribute(ARMBuildAttrs::ARM_ISA_use,
                               ARMBuildAttrs::Allowed);
    AttrEmitter->EmitAttribute(ARMBuildAttrs::THUMB_ISA_use,
                               ARMBuildAttrs::AllowThumb32);
    // Fixme: figure out when this is emitted.
    //AttrEmitter->EmitAttribute(ARMBuildAttrs::WMMX_arch,
    //                           ARMBuildAttrs::AllowWMMXv1);
    //

    /// ADD additional Else-cases here!
  } else if (CPUString == "xscale") {
    AttrEmitter->EmitAttribute(ARMBuildAttrs::CPU_arch, ARMBuildAttrs::v5TEJ);
    AttrEmitter->EmitAttribute(ARMBuildAttrs::ARM_ISA_use,
                               ARMBuildAttrs::Allowed);
    AttrEmitter->EmitAttribute(ARMBuildAttrs::THUMB_ISA_use,
                               ARMBuildAttrs::Allowed);
  } else if (CPUString == "generic") {
    // For a generic CPU, we assume a standard v7a architecture in Subtarget.
    AttrEmitter->EmitAttribute(ARMBuildAttrs::CPU_arch, ARMBuildAttrs::v7);
    AttrEmitter->EmitAttribute(ARMBuildAttrs::CPU_arch_profile,
                               ARMBuildAttrs::ApplicationProfile);
    AttrEmitter->EmitAttribute(ARMBuildAttrs::ARM_ISA_use,
                               ARMBuildAttrs::Allowed);
    AttrEmitter->EmitAttribute(ARMBuildAttrs::THUMB_ISA_use,
                               ARMBuildAttrs::AllowThumb32);
  } else if (Subtarget->hasV7Ops()) {
    AttrEmitter->EmitAttribute(ARMBuildAttrs::CPU_arch, ARMBuildAttrs::v7);
    AttrEmitter->EmitAttribute(ARMBuildAttrs::THUMB_ISA_use,
                               ARMBuildAttrs::AllowThumb32);
  } else if (Subtarget->hasV6T2Ops())
    AttrEmitter->EmitAttribute(ARMBuildAttrs::CPU_arch, ARMBuildAttrs::v6T2);
  else if (Subtarget->hasV6Ops())
    AttrEmitter->EmitAttribute(ARMBuildAttrs::CPU_arch, ARMBuildAttrs::v6);
  else if (Subtarget->hasV5TEOps())
    AttrEmitter->EmitAttribute(ARMBuildAttrs::CPU_arch, ARMBuildAttrs::v5TE);
  else if (Subtarget->hasV5TOps())
    AttrEmitter->EmitAttribute(ARMBuildAttrs::CPU_arch, ARMBuildAttrs::v5T);
  else if (Subtarget->hasV4TOps())
    AttrEmitter->EmitAttribute(ARMBuildAttrs::CPU_arch, ARMBuildAttrs::v4T);

  if (Subtarget->hasNEON() && emitFPU) {
    /* NEON is not exactly a VFP architecture, but GAS emit one of
     * neon/neon-vfpv4/vfpv3/vfpv2 for .fpu parameters */
    if (Subtarget->hasVFP4())
      AttrEmitter->EmitTextAttribute(ARMBuildAttrs::Advanced_SIMD_arch,
                                     "neon-vfpv4");
    else
      AttrEmitter->EmitTextAttribute(ARMBuildAttrs::Advanced_SIMD_arch, "neon");
    /* If emitted for NEON, omit from VFP below, since you can have both
     * NEON and VFP in build attributes but only one .fpu */
    emitFPU = false;
  }

  /* VFPv4 + .fpu */
  if (Subtarget->hasVFP4()) {
    AttrEmitter->EmitAttribute(ARMBuildAttrs::VFP_arch,
                               ARMBuildAttrs::AllowFPv4A);
    if (emitFPU)
      AttrEmitter->EmitTextAttribute(ARMBuildAttrs::VFP_arch, "vfpv4");

  /* VFPv3 + .fpu */
  } else if (Subtarget->hasVFP3()) {
    AttrEmitter->EmitAttribute(ARMBuildAttrs::VFP_arch,
                               ARMBuildAttrs::AllowFPv3A);
    if (emitFPU)
      AttrEmitter->EmitTextAttribute(ARMBuildAttrs::VFP_arch, "vfpv3");

  /* VFPv2 + .fpu */
  } else if (Subtarget->hasVFP2()) {
    AttrEmitter->EmitAttribute(ARMBuildAttrs::VFP_arch,
                               ARMBuildAttrs::AllowFPv2);
    if (emitFPU)
      AttrEmitter->EmitTextAttribute(ARMBuildAttrs::VFP_arch, "vfpv2");
  }

  /* TODO: ARMBuildAttrs::Allowed is not completely accurate,
   * since NEON can have 1 (allowed) or 2 (MAC operations) */
  if (Subtarget->hasNEON()) {
    AttrEmitter->EmitAttribute(ARMBuildAttrs::Advanced_SIMD_arch,
                               ARMBuildAttrs::Allowed);
  }

  // Signal various FP modes.
  if (!TM.Options.UnsafeFPMath) {
    AttrEmitter->EmitAttribute(ARMBuildAttrs::ABI_FP_denormal,
                               ARMBuildAttrs::Allowed);
    AttrEmitter->EmitAttribute(ARMBuildAttrs::ABI_FP_exceptions,
                               ARMBuildAttrs::Allowed);
  }

  if (TM.Options.NoInfsFPMath && TM.Options.NoNaNsFPMath)
    AttrEmitter->EmitAttribute(ARMBuildAttrs::ABI_FP_number_model,
                               ARMBuildAttrs::Allowed);
  else
    AttrEmitter->EmitAttribute(ARMBuildAttrs::ABI_FP_number_model,
                               ARMBuildAttrs::AllowIEE754);

  // FIXME: add more flags to ARMBuildAttrs.h
  // 8-bytes alignment stuff.
  AttrEmitter->EmitAttribute(ARMBuildAttrs::ABI_align8_needed, 1);
  AttrEmitter->EmitAttribute(ARMBuildAttrs::ABI_align8_preserved, 1);

  // Hard float.  Use both S and D registers and conform to AAPCS-VFP.
  if (Subtarget->isAAPCS_ABI() && TM.Options.FloatABIType == FloatABI::Hard) {
    AttrEmitter->EmitAttribute(ARMBuildAttrs::ABI_HardFP_use, 3);
    AttrEmitter->EmitAttribute(ARMBuildAttrs::ABI_VFP_args, 1);
  }
  // FIXME: Should we signal R9 usage?

  if (Subtarget->hasDivide())
    AttrEmitter->EmitAttribute(ARMBuildAttrs::DIV_use, 1);

  AttrEmitter->Finish();
  delete AttrEmitter;
}

void ARMAsmPrinter::emitARMAttributeSection() {
  // <format-version>
  // [ <section-length> "vendor-name"
  // [ <file-tag> <size> <attribute>*
  //   | <section-tag> <size> <section-number>* 0 <attribute>*
  //   | <symbol-tag> <size> <symbol-number>* 0 <attribute>*
  //   ]+
  // ]*

  if (OutStreamer.hasRawTextSupport())
    return;

  const ARMElfTargetObjectFile &TLOFELF =
    static_cast<const ARMElfTargetObjectFile &>
    (getObjFileLowering());

  OutStreamer.SwitchSection(TLOFELF.getAttributesSection());

  // Format version
  OutStreamer.EmitIntValue(0x41, 1);
}

//===----------------------------------------------------------------------===//

static MCSymbol *getPICLabel(const char *Prefix, unsigned FunctionNumber,
                             unsigned LabelId, MCContext &Ctx) {

  MCSymbol *Label = Ctx.GetOrCreateSymbol(Twine(Prefix)
                       + "PC" + Twine(FunctionNumber) + "_" + Twine(LabelId));
  return Label;
}

static MCSymbolRefExpr::VariantKind
getModifierVariantKind(ARMCP::ARMCPModifier Modifier) {
  switch (Modifier) {
  case ARMCP::no_modifier: return MCSymbolRefExpr::VK_None;
  case ARMCP::TLSGD:       return MCSymbolRefExpr::VK_ARM_TLSGD;
  case ARMCP::TPOFF:       return MCSymbolRefExpr::VK_ARM_TPOFF;
  case ARMCP::GOTTPOFF:    return MCSymbolRefExpr::VK_ARM_GOTTPOFF;
  case ARMCP::GOT:         return MCSymbolRefExpr::VK_ARM_GOT;
  case ARMCP::GOTOFF:      return MCSymbolRefExpr::VK_ARM_GOTOFF;
  }
  llvm_unreachable("Invalid ARMCPModifier!");
}

MCSymbol *ARMAsmPrinter::GetARMGVSymbol(const GlobalValue *GV) {
  bool isIndirect = Subtarget->isTargetDarwin() &&
    Subtarget->GVIsIndirectSymbol(GV, TM.getRelocationModel());
  if (!isIndirect)
    return Mang->getSymbol(GV);

  // FIXME: Remove this when Darwin transition to @GOT like syntax.
  MCSymbol *MCSym = GetSymbolWithGlobalValueBase(GV, "$non_lazy_ptr");
  MachineModuleInfoMachO &MMIMachO =
    MMI->getObjFileInfo<MachineModuleInfoMachO>();
  MachineModuleInfoImpl::StubValueTy &StubSym =
    GV->hasHiddenVisibility() ? MMIMachO.getHiddenGVStubEntry(MCSym) :
    MMIMachO.getGVStubEntry(MCSym);
  if (StubSym.getPointer() == 0)
    StubSym = MachineModuleInfoImpl::
      StubValueTy(Mang->getSymbol(GV), !GV->hasInternalLinkage());
  return MCSym;
}

void ARMAsmPrinter::
EmitMachineConstantPoolValue(MachineConstantPoolValue *MCPV) {
  int Size = TM.getDataLayout()->getTypeAllocSize(MCPV->getType());

  ARMConstantPoolValue *ACPV = static_cast<ARMConstantPoolValue*>(MCPV);

  MCSymbol *MCSym;
  if (ACPV->isLSDA()) {
    SmallString<128> Str;
    raw_svector_ostream OS(Str);
    OS << MAI->getPrivateGlobalPrefix() << "_LSDA_" << getFunctionNumber();
    MCSym = OutContext.GetOrCreateSymbol(OS.str());
  } else if (ACPV->isBlockAddress()) {
    const BlockAddress *BA =
      cast<ARMConstantPoolConstant>(ACPV)->getBlockAddress();
    MCSym = GetBlockAddressSymbol(BA);
  } else if (ACPV->isGlobalValue()) {
    const GlobalValue *GV = cast<ARMConstantPoolConstant>(ACPV)->getGV();
    MCSym = GetARMGVSymbol(GV);
  } else if (ACPV->isMachineBasicBlock()) {
    const MachineBasicBlock *MBB = cast<ARMConstantPoolMBB>(ACPV)->getMBB();
    MCSym = MBB->getSymbol();
  } else {
    assert(ACPV->isExtSymbol() && "unrecognized constant pool value");
    const char *Sym = cast<ARMConstantPoolSymbol>(ACPV)->getSymbol();
    MCSym = GetExternalSymbolSymbol(Sym);
  }

  // Create an MCSymbol for the reference.
  const MCExpr *Expr =
    MCSymbolRefExpr::Create(MCSym, getModifierVariantKind(ACPV->getModifier()),
                            OutContext);

  if (ACPV->getPCAdjustment()) {
    MCSymbol *PCLabel = getPICLabel(MAI->getPrivateGlobalPrefix(),
                                    getFunctionNumber(),
                                    ACPV->getLabelId(),
                                    OutContext);
    const MCExpr *PCRelExpr = MCSymbolRefExpr::Create(PCLabel, OutContext);
    PCRelExpr =
      MCBinaryExpr::CreateAdd(PCRelExpr,
                              MCConstantExpr::Create(ACPV->getPCAdjustment(),
                                                     OutContext),
                              OutContext);
    if (ACPV->mustAddCurrentAddress()) {
      // We want "(<expr> - .)", but MC doesn't have a concept of the '.'
      // label, so just emit a local label end reference that instead.
      MCSymbol *DotSym = OutContext.CreateTempSymbol();
      OutStreamer.EmitLabel(DotSym);
      const MCExpr *DotExpr = MCSymbolRefExpr::Create(DotSym, OutContext);
      PCRelExpr = MCBinaryExpr::CreateSub(PCRelExpr, DotExpr, OutContext);
    }
    Expr = MCBinaryExpr::CreateSub(Expr, PCRelExpr, OutContext);
  }
  OutStreamer.EmitValue(Expr, Size);
}

void ARMAsmPrinter::EmitJumpTable(const MachineInstr *MI) {
  unsigned Opcode = MI->getOpcode();
  int OpNum = 1;
  if (Opcode == ARM::BR_JTadd)
    OpNum = 2;
  else if (Opcode == ARM::BR_JTm)
    OpNum = 3;

  const MachineOperand &MO1 = MI->getOperand(OpNum);
  const MachineOperand &MO2 = MI->getOperand(OpNum+1); // Unique Id
  unsigned JTI = MO1.getIndex();

  // Emit a label for the jump table.
  MCSymbol *JTISymbol = GetARMJTIPICJumpTableLabel2(JTI, MO2.getImm());
  OutStreamer.EmitLabel(JTISymbol);

  // Mark the jump table as data-in-code.
  OutStreamer.EmitDataRegion(MCDR_DataRegionJT32);

  // Emit each entry of the table.
  const MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  const std::vector<MachineBasicBlock*> &JTBBs = JT[JTI].MBBs;

  for (unsigned i = 0, e = JTBBs.size(); i != e; ++i) {
    MachineBasicBlock *MBB = JTBBs[i];
    // Construct an MCExpr for the entry. We want a value of the form:
    // (BasicBlockAddr - TableBeginAddr)
    //
    // For example, a table with entries jumping to basic blocks BB0 and BB1
    // would look like:
    // LJTI_0_0:
    //    .word (LBB0 - LJTI_0_0)
    //    .word (LBB1 - LJTI_0_0)
    const MCExpr *Expr = MCSymbolRefExpr::Create(MBB->getSymbol(), OutContext);

    if (TM.getRelocationModel() == Reloc::PIC_)
      Expr = MCBinaryExpr::CreateSub(Expr, MCSymbolRefExpr::Create(JTISymbol,
                                                                   OutContext),
                                     OutContext);
    // If we're generating a table of Thumb addresses in static relocation
    // model, we need to add one to keep interworking correctly.
    else if (AFI->isThumbFunction())
      Expr = MCBinaryExpr::CreateAdd(Expr, MCConstantExpr::Create(1,OutContext),
                                     OutContext);
    OutStreamer.EmitValue(Expr, 4);
  }
  // Mark the end of jump table data-in-code region.
  OutStreamer.EmitDataRegion(MCDR_DataRegionEnd);
}

void ARMAsmPrinter::EmitJump2Table(const MachineInstr *MI) {
  unsigned Opcode = MI->getOpcode();
  int OpNum = (Opcode == ARM::t2BR_JT) ? 2 : 1;
  const MachineOperand &MO1 = MI->getOperand(OpNum);
  const MachineOperand &MO2 = MI->getOperand(OpNum+1); // Unique Id
  unsigned JTI = MO1.getIndex();

  MCSymbol *JTISymbol = GetARMJTIPICJumpTableLabel2(JTI, MO2.getImm());
  OutStreamer.EmitLabel(JTISymbol);

  // Emit each entry of the table.
  const MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  const std::vector<MachineBasicBlock*> &JTBBs = JT[JTI].MBBs;
  unsigned OffsetWidth = 4;
  if (MI->getOpcode() == ARM::t2TBB_JT) {
    OffsetWidth = 1;
    // Mark the jump table as data-in-code.
    OutStreamer.EmitDataRegion(MCDR_DataRegionJT8);
  } else if (MI->getOpcode() == ARM::t2TBH_JT) {
    OffsetWidth = 2;
    // Mark the jump table as data-in-code.
    OutStreamer.EmitDataRegion(MCDR_DataRegionJT16);
  }

  for (unsigned i = 0, e = JTBBs.size(); i != e; ++i) {
    MachineBasicBlock *MBB = JTBBs[i];
    const MCExpr *MBBSymbolExpr = MCSymbolRefExpr::Create(MBB->getSymbol(),
                                                      OutContext);
    // If this isn't a TBB or TBH, the entries are direct branch instructions.
    if (OffsetWidth == 4) {
      MCInst BrInst;
      BrInst.setOpcode(ARM::t2B);
      BrInst.addOperand(MCOperand::CreateExpr(MBBSymbolExpr));
      BrInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      BrInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(BrInst);
      continue;
    }
    // Otherwise it's an offset from the dispatch instruction. Construct an
    // MCExpr for the entry. We want a value of the form:
    // (BasicBlockAddr - TableBeginAddr) / 2
    //
    // For example, a TBB table with entries jumping to basic blocks BB0 and BB1
    // would look like:
    // LJTI_0_0:
    //    .byte (LBB0 - LJTI_0_0) / 2
    //    .byte (LBB1 - LJTI_0_0) / 2
    const MCExpr *Expr =
      MCBinaryExpr::CreateSub(MBBSymbolExpr,
                              MCSymbolRefExpr::Create(JTISymbol, OutContext),
                              OutContext);
    Expr = MCBinaryExpr::CreateDiv(Expr, MCConstantExpr::Create(2, OutContext),
                                   OutContext);
    OutStreamer.EmitValue(Expr, OffsetWidth);
  }
  // Mark the end of jump table data-in-code region. 32-bit offsets use
  // actual branch instructions here, so we don't mark those as a data-region
  // at all.
  if (OffsetWidth != 4)
    OutStreamer.EmitDataRegion(MCDR_DataRegionEnd);
}

void ARMAsmPrinter::PrintDebugValueComment(const MachineInstr *MI,
                                           raw_ostream &OS) {
  unsigned NOps = MI->getNumOperands();
  assert(NOps==4);
  OS << '\t' << MAI->getCommentString() << "DEBUG_VALUE: ";
  // cast away const; DIetc do not take const operands for some reason.
  DIVariable V(const_cast<MDNode *>(MI->getOperand(NOps-1).getMetadata()));
  OS << V.getName();
  OS << " <- ";
  // Frame address.  Currently handles register +- offset only.
  assert(MI->getOperand(0).isReg() && MI->getOperand(1).isImm());
  OS << '['; printOperand(MI, 0, OS); OS << '+'; printOperand(MI, 1, OS);
  OS << ']';
  OS << "+";
  printOperand(MI, NOps-2, OS);
}

static void populateADROperands(MCInst &Inst, unsigned Dest,
                                const MCSymbol *Label,
                                unsigned pred, unsigned ccreg,
                                MCContext &Ctx) {
  const MCExpr *SymbolExpr = MCSymbolRefExpr::Create(Label, Ctx);
  Inst.addOperand(MCOperand::CreateReg(Dest));
  Inst.addOperand(MCOperand::CreateExpr(SymbolExpr));
  // Add predicate operands.
  Inst.addOperand(MCOperand::CreateImm(pred));
  Inst.addOperand(MCOperand::CreateReg(ccreg));
}

void ARMAsmPrinter::EmitUnwindingInstruction(const MachineInstr *MI) {
  assert(MI->getFlag(MachineInstr::FrameSetup) &&
      "Only instruction which are involved into frame setup code are allowed");

  const MachineFunction &MF = *MI->getParent()->getParent();
  const TargetRegisterInfo *RegInfo = MF.getTarget().getRegisterInfo();
  const ARMFunctionInfo &AFI = *MF.getInfo<ARMFunctionInfo>();

  unsigned FramePtr = RegInfo->getFrameRegister(MF);
  unsigned Opc = MI->getOpcode();
  unsigned SrcReg, DstReg;

  if (Opc == ARM::tPUSH || Opc == ARM::tLDRpci) {
    // Two special cases:
    // 1) tPUSH does not have src/dst regs.
    // 2) for Thumb1 code we sometimes materialize the constant via constpool
    // load. Yes, this is pretty fragile, but for now I don't see better
    // way... :(
    SrcReg = DstReg = ARM::SP;
  } else {
    SrcReg = MI->getOperand(1).getReg();
    DstReg = MI->getOperand(0).getReg();
  }

  // Try to figure out the unwinding opcode out of src / dst regs.
  if (MI->mayStore()) {
    // Register saves.
    assert(DstReg == ARM::SP &&
           "Only stack pointer as a destination reg is supported");

    SmallVector<unsigned, 4> RegList;
    // Skip src & dst reg, and pred ops.
    unsigned StartOp = 2 + 2;
    // Use all the operands.
    unsigned NumOffset = 0;

    switch (Opc) {
    default:
      MI->dump();
      llvm_unreachable("Unsupported opcode for unwinding information");
    case ARM::tPUSH:
      // Special case here: no src & dst reg, but two extra imp ops.
      StartOp = 2; NumOffset = 2;
    case ARM::STMDB_UPD:
    case ARM::t2STMDB_UPD:
    case ARM::VSTMDDB_UPD:
      assert(SrcReg == ARM::SP &&
             "Only stack pointer as a source reg is supported");
      for (unsigned i = StartOp, NumOps = MI->getNumOperands() - NumOffset;
           i != NumOps; ++i) {
        const MachineOperand &MO = MI->getOperand(i);
        // Actually, there should never be any impdef stuff here. Skip it
        // temporary to workaround PR11902.
        if (MO.isImplicit())
          continue;
        RegList.push_back(MO.getReg());
      }
      break;
    case ARM::STR_PRE_IMM:
    case ARM::STR_PRE_REG:
    case ARM::t2STR_PRE:
      assert(MI->getOperand(2).getReg() == ARM::SP &&
             "Only stack pointer as a source reg is supported");
      RegList.push_back(SrcReg);
      break;
    }
    OutStreamer.EmitRegSave(RegList, Opc == ARM::VSTMDDB_UPD);
  } else {
    // Changes of stack / frame pointer.
    if (SrcReg == ARM::SP) {
      int64_t Offset = 0;
      switch (Opc) {
      default:
        MI->dump();
        llvm_unreachable("Unsupported opcode for unwinding information");
      case ARM::MOVr:
      case ARM::tMOVr:
        Offset = 0;
        break;
      case ARM::ADDri:
        Offset = -MI->getOperand(2).getImm();
        break;
      case ARM::SUBri:
      case ARM::t2SUBri:
        Offset = MI->getOperand(2).getImm();
        break;
      case ARM::tSUBspi:
        Offset = MI->getOperand(2).getImm()*4;
        break;
      case ARM::tADDspi:
      case ARM::tADDrSPi:
        Offset = -MI->getOperand(2).getImm()*4;
        break;
      case ARM::tLDRpci: {
        // Grab the constpool index and check, whether it corresponds to
        // original or cloned constpool entry.
        unsigned CPI = MI->getOperand(1).getIndex();
        const MachineConstantPool *MCP = MF.getConstantPool();
        if (CPI >= MCP->getConstants().size())
          CPI = AFI.getOriginalCPIdx(CPI);
        assert(CPI != -1U && "Invalid constpool index");

        // Derive the actual offset.
        const MachineConstantPoolEntry &CPE = MCP->getConstants()[CPI];
        assert(!CPE.isMachineConstantPoolEntry() && "Invalid constpool entry");
        // FIXME: Check for user, it should be "add" instruction!
        Offset = -cast<ConstantInt>(CPE.Val.ConstVal)->getSExtValue();
        break;
      }
      }

      if (DstReg == FramePtr && FramePtr != ARM::SP)
        // Set-up of the frame pointer. Positive values correspond to "add"
        // instruction.
        OutStreamer.EmitSetFP(FramePtr, ARM::SP, -Offset);
      else if (DstReg == ARM::SP) {
        // Change of SP by an offset. Positive values correspond to "sub"
        // instruction.
        OutStreamer.EmitPad(Offset);
      } else {
        MI->dump();
        llvm_unreachable("Unsupported opcode for unwinding information");
      }
    } else if (DstReg == ARM::SP) {
      // FIXME: .movsp goes here
      MI->dump();
      llvm_unreachable("Unsupported opcode for unwinding information");
    }
    else {
      MI->dump();
      llvm_unreachable("Unsupported opcode for unwinding information");
    }
  }
}

extern cl::opt<bool> EnableARMEHABI;

// Simple pseudo-instructions have their lowering (with expansion to real
// instructions) auto-generated.
#include "ARMGenMCPseudoLowering.inc"

void ARMAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  // If we just ended a constant pool, mark it as such.
  if (InConstantPool && MI->getOpcode() != ARM::CONSTPOOL_ENTRY) {
    OutStreamer.EmitDataRegion(MCDR_DataRegionEnd);
    InConstantPool = false;
  }

  // Emit unwinding stuff for frame-related instructions
  if (EnableARMEHABI && MI->getFlag(MachineInstr::FrameSetup))
    EmitUnwindingInstruction(MI);

  // Do any auto-generated pseudo lowerings.
  if (emitPseudoExpansionLowering(OutStreamer, MI))
    return;

  assert(!convertAddSubFlagsOpcode(MI->getOpcode()) &&
         "Pseudo flag setting opcode should be expanded early");

  // Check for manual lowerings.
  unsigned Opc = MI->getOpcode();
  switch (Opc) {
  case ARM::t2MOVi32imm: llvm_unreachable("Should be lowered by thumb2it pass");
  case ARM::DBG_VALUE: {
    if (isVerbose() && OutStreamer.hasRawTextSupport()) {
      SmallString<128> TmpStr;
      raw_svector_ostream OS(TmpStr);
      PrintDebugValueComment(MI, OS);
      OutStreamer.EmitRawText(StringRef(OS.str()));
    }
    return;
  }
  case ARM::LEApcrel:
  case ARM::tLEApcrel:
  case ARM::t2LEApcrel: {
    // FIXME: Need to also handle globals and externals
    MCInst TmpInst;
    TmpInst.setOpcode(MI->getOpcode() == ARM::t2LEApcrel ? ARM::t2ADR
                      : (MI->getOpcode() == ARM::tLEApcrel ? ARM::tADR
                         : ARM::ADR));
    populateADROperands(TmpInst, MI->getOperand(0).getReg(),
                        GetCPISymbol(MI->getOperand(1).getIndex()),
                        MI->getOperand(2).getImm(), MI->getOperand(3).getReg(),
                        OutContext);
    OutStreamer.EmitInstruction(TmpInst);
    return;
  }
  case ARM::LEApcrelJT:
  case ARM::tLEApcrelJT:
  case ARM::t2LEApcrelJT: {
    MCInst TmpInst;
    TmpInst.setOpcode(MI->getOpcode() == ARM::t2LEApcrelJT ? ARM::t2ADR
                      : (MI->getOpcode() == ARM::tLEApcrelJT ? ARM::tADR
                         : ARM::ADR));
    populateADROperands(TmpInst, MI->getOperand(0).getReg(),
                      GetARMJTIPICJumpTableLabel2(MI->getOperand(1).getIndex(),
                                                  MI->getOperand(2).getImm()),
                      MI->getOperand(3).getImm(), MI->getOperand(4).getReg(),
                      OutContext);
    OutStreamer.EmitInstruction(TmpInst);
    return;
  }
  // Darwin call instructions are just normal call instructions with different
  // clobber semantics (they clobber R9).
  case ARM::BX_CALL: {
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::MOVr);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::LR));
      TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
      // Add predicate operands.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      // Add 's' bit operand (always reg0 for this)
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::BX);
      TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
      OutStreamer.EmitInstruction(TmpInst);
    }
    return;
  }
  case ARM::tBX_CALL: {
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::tMOVr);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::LR));
      TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
      // Add predicate operands.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::tBX);
      TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
      // Add predicate operands.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    return;
  }
  case ARM::BMOVPCRX_CALL: {
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::MOVr);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::LR));
      TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
      // Add predicate operands.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      // Add 's' bit operand (always reg0 for this)
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::MOVr);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
      TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
      // Add predicate operands.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      // Add 's' bit operand (always reg0 for this)
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    return;
  }
  case ARM::BMOVPCB_CALL: {
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::MOVr);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::LR));
      TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
      // Add predicate operands.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      // Add 's' bit operand (always reg0 for this)
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::Bcc);
      const GlobalValue *GV = MI->getOperand(0).getGlobal();
      MCSymbol *GVSym = Mang->getSymbol(GV);
      const MCExpr *GVSymExpr = MCSymbolRefExpr::Create(GVSym, OutContext);
      TmpInst.addOperand(MCOperand::CreateExpr(GVSymExpr));
      // Add predicate operands.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    return;
  }
  case ARM::MOVi16_ga_pcrel:
  case ARM::t2MOVi16_ga_pcrel: {
    MCInst TmpInst;
    TmpInst.setOpcode(Opc == ARM::MOVi16_ga_pcrel? ARM::MOVi16 : ARM::t2MOVi16);
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));

    unsigned TF = MI->getOperand(1).getTargetFlags();
    bool isPIC = TF == ARMII::MO_LO16_NONLAZY_PIC;
    const GlobalValue *GV = MI->getOperand(1).getGlobal();
    MCSymbol *GVSym = GetARMGVSymbol(GV);
    const MCExpr *GVSymExpr = MCSymbolRefExpr::Create(GVSym, OutContext);
    if (isPIC) {
      MCSymbol *LabelSym = getPICLabel(MAI->getPrivateGlobalPrefix(),
                                       getFunctionNumber(),
                                       MI->getOperand(2).getImm(), OutContext);
      const MCExpr *LabelSymExpr= MCSymbolRefExpr::Create(LabelSym, OutContext);
      unsigned PCAdj = (Opc == ARM::MOVi16_ga_pcrel) ? 8 : 4;
      const MCExpr *PCRelExpr =
        ARMMCExpr::CreateLower16(MCBinaryExpr::CreateSub(GVSymExpr,
                                  MCBinaryExpr::CreateAdd(LabelSymExpr,
                                      MCConstantExpr::Create(PCAdj, OutContext),
                                          OutContext), OutContext), OutContext);
      TmpInst.addOperand(MCOperand::CreateExpr(PCRelExpr));
    } else {
      const MCExpr *RefExpr= ARMMCExpr::CreateLower16(GVSymExpr, OutContext);
      TmpInst.addOperand(MCOperand::CreateExpr(RefExpr));
    }

    // Add predicate operands.
    TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    TmpInst.addOperand(MCOperand::CreateReg(0));
    // Add 's' bit operand (always reg0 for this)
    TmpInst.addOperand(MCOperand::CreateReg(0));
    OutStreamer.EmitInstruction(TmpInst);
    return;
  }
  case ARM::MOVTi16_ga_pcrel:
  case ARM::t2MOVTi16_ga_pcrel: {
    MCInst TmpInst;
    TmpInst.setOpcode(Opc == ARM::MOVTi16_ga_pcrel
                      ? ARM::MOVTi16 : ARM::t2MOVTi16);
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(1).getReg()));

    unsigned TF = MI->getOperand(2).getTargetFlags();
    bool isPIC = TF == ARMII::MO_HI16_NONLAZY_PIC;
    const GlobalValue *GV = MI->getOperand(2).getGlobal();
    MCSymbol *GVSym = GetARMGVSymbol(GV);
    const MCExpr *GVSymExpr = MCSymbolRefExpr::Create(GVSym, OutContext);
    if (isPIC) {
      MCSymbol *LabelSym = getPICLabel(MAI->getPrivateGlobalPrefix(),
                                       getFunctionNumber(),
                                       MI->getOperand(3).getImm(), OutContext);
      const MCExpr *LabelSymExpr= MCSymbolRefExpr::Create(LabelSym, OutContext);
      unsigned PCAdj = (Opc == ARM::MOVTi16_ga_pcrel) ? 8 : 4;
      const MCExpr *PCRelExpr =
        ARMMCExpr::CreateUpper16(MCBinaryExpr::CreateSub(GVSymExpr,
                                   MCBinaryExpr::CreateAdd(LabelSymExpr,
                                      MCConstantExpr::Create(PCAdj, OutContext),
                                          OutContext), OutContext), OutContext);
      TmpInst.addOperand(MCOperand::CreateExpr(PCRelExpr));
    } else {
      const MCExpr *RefExpr= ARMMCExpr::CreateUpper16(GVSymExpr, OutContext);
      TmpInst.addOperand(MCOperand::CreateExpr(RefExpr));
    }
    // Add predicate operands.
    TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    TmpInst.addOperand(MCOperand::CreateReg(0));
    // Add 's' bit operand (always reg0 for this)
    TmpInst.addOperand(MCOperand::CreateReg(0));
    OutStreamer.EmitInstruction(TmpInst);
    return;
  }
  case ARM::tPICADD: {
    // This is a pseudo op for a label + instruction sequence, which looks like:
    // LPC0:
    //     add r0, pc
    // This adds the address of LPC0 to r0.

    // Emit the label.
    OutStreamer.EmitLabel(getPICLabel(MAI->getPrivateGlobalPrefix(),
                          getFunctionNumber(), MI->getOperand(2).getImm(),
                          OutContext));

    // Form and emit the add.
    MCInst AddInst;
    AddInst.setOpcode(ARM::tADDhirr);
    AddInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    AddInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    AddInst.addOperand(MCOperand::CreateReg(ARM::PC));
    // Add predicate operands.
    AddInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    AddInst.addOperand(MCOperand::CreateReg(0));
    OutStreamer.EmitInstruction(AddInst);
    return;
  }
  case ARM::PICADD: {
    // This is a pseudo op for a label + instruction sequence, which looks like:
    // LPC0:
    //     add r0, pc, r0
    // This adds the address of LPC0 to r0.

    // Emit the label.
    OutStreamer.EmitLabel(getPICLabel(MAI->getPrivateGlobalPrefix(),
                          getFunctionNumber(), MI->getOperand(2).getImm(),
                          OutContext));

    // Form and emit the add.
    MCInst AddInst;
    AddInst.setOpcode(ARM::ADDrr);
    AddInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    AddInst.addOperand(MCOperand::CreateReg(ARM::PC));
    AddInst.addOperand(MCOperand::CreateReg(MI->getOperand(1).getReg()));
    // Add predicate operands.
    AddInst.addOperand(MCOperand::CreateImm(MI->getOperand(3).getImm()));
    AddInst.addOperand(MCOperand::CreateReg(MI->getOperand(4).getReg()));
    // Add 's' bit operand (always reg0 for this)
    AddInst.addOperand(MCOperand::CreateReg(0));
    OutStreamer.EmitInstruction(AddInst);
    return;
  }
  case ARM::PICSTR:
  case ARM::PICSTRB:
  case ARM::PICSTRH:
  case ARM::PICLDR:
  case ARM::PICLDRB:
  case ARM::PICLDRH:
  case ARM::PICLDRSB:
  case ARM::PICLDRSH: {
    // This is a pseudo op for a label + instruction sequence, which looks like:
    // LPC0:
    //     OP r0, [pc, r0]
    // The LCP0 label is referenced by a constant pool entry in order to get
    // a PC-relative address at the ldr instruction.

    // Emit the label.
    OutStreamer.EmitLabel(getPICLabel(MAI->getPrivateGlobalPrefix(),
                          getFunctionNumber(), MI->getOperand(2).getImm(),
                          OutContext));

    // Form and emit the load
    unsigned Opcode;
    switch (MI->getOpcode()) {
    default:
      llvm_unreachable("Unexpected opcode!");
    case ARM::PICSTR:   Opcode = ARM::STRrs; break;
    case ARM::PICSTRB:  Opcode = ARM::STRBrs; break;
    case ARM::PICSTRH:  Opcode = ARM::STRH; break;
    case ARM::PICLDR:   Opcode = ARM::LDRrs; break;
    case ARM::PICLDRB:  Opcode = ARM::LDRBrs; break;
    case ARM::PICLDRH:  Opcode = ARM::LDRH; break;
    case ARM::PICLDRSB: Opcode = ARM::LDRSB; break;
    case ARM::PICLDRSH: Opcode = ARM::LDRSH; break;
    }
    MCInst LdStInst;
    LdStInst.setOpcode(Opcode);
    LdStInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    LdStInst.addOperand(MCOperand::CreateReg(ARM::PC));
    LdStInst.addOperand(MCOperand::CreateReg(MI->getOperand(1).getReg()));
    LdStInst.addOperand(MCOperand::CreateImm(0));
    // Add predicate operands.
    LdStInst.addOperand(MCOperand::CreateImm(MI->getOperand(3).getImm()));
    LdStInst.addOperand(MCOperand::CreateReg(MI->getOperand(4).getReg()));
    OutStreamer.EmitInstruction(LdStInst);

    return;
  }
  case ARM::CONSTPOOL_ENTRY: {
    /// CONSTPOOL_ENTRY - This instruction represents a floating constant pool
    /// in the function.  The first operand is the ID# for this instruction, the
    /// second is the index into the MachineConstantPool that this is, the third
    /// is the size in bytes of this constant pool entry.
    /// The required alignment is specified on the basic block holding this MI.
    unsigned LabelId = (unsigned)MI->getOperand(0).getImm();
    unsigned CPIdx   = (unsigned)MI->getOperand(1).getIndex();

    // If this is the first entry of the pool, mark it.
    if (!InConstantPool) {
      OutStreamer.EmitDataRegion(MCDR_DataRegion);
      InConstantPool = true;
    }

    OutStreamer.EmitLabel(GetCPISymbol(LabelId));

    const MachineConstantPoolEntry &MCPE = MCP->getConstants()[CPIdx];
    if (MCPE.isMachineConstantPoolEntry())
      EmitMachineConstantPoolValue(MCPE.Val.MachineCPVal);
    else
      EmitGlobalConstant(MCPE.Val.ConstVal);
    return;
  }
  case ARM::t2BR_JT: {
    // Lower and emit the instruction itself, then the jump table following it.
    MCInst TmpInst;
    TmpInst.setOpcode(ARM::tMOVr);
    TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    // Add predicate operands.
    TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    TmpInst.addOperand(MCOperand::CreateReg(0));
    OutStreamer.EmitInstruction(TmpInst);
    // Output the data for the jump table itself
    EmitJump2Table(MI);
    return;
  }
  case ARM::t2TBB_JT: {
    // Lower and emit the instruction itself, then the jump table following it.
    MCInst TmpInst;

    TmpInst.setOpcode(ARM::t2TBB);
    TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    // Add predicate operands.
    TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    TmpInst.addOperand(MCOperand::CreateReg(0));
    OutStreamer.EmitInstruction(TmpInst);
    // Output the data for the jump table itself
    EmitJump2Table(MI);
    // Make sure the next instruction is 2-byte aligned.
    EmitAlignment(1);
    return;
  }
  case ARM::t2TBH_JT: {
    // Lower and emit the instruction itself, then the jump table following it.
    MCInst TmpInst;

    TmpInst.setOpcode(ARM::t2TBH);
    TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    // Add predicate operands.
    TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    TmpInst.addOperand(MCOperand::CreateReg(0));
    OutStreamer.EmitInstruction(TmpInst);
    // Output the data for the jump table itself
    EmitJump2Table(MI);
    return;
  }
  case ARM::tBR_JTr:
  case ARM::BR_JTr: {
    // Lower and emit the instruction itself, then the jump table following it.
    // mov pc, target
    MCInst TmpInst;
    unsigned Opc = MI->getOpcode() == ARM::BR_JTr ?
      ARM::MOVr : ARM::tMOVr;
    TmpInst.setOpcode(Opc);
    TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    // Add predicate operands.
    TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    TmpInst.addOperand(MCOperand::CreateReg(0));
    // Add 's' bit operand (always reg0 for this)
    if (Opc == ARM::MOVr)
      TmpInst.addOperand(MCOperand::CreateReg(0));
    OutStreamer.EmitInstruction(TmpInst);

    // Make sure the Thumb jump table is 4-byte aligned.
    if (Opc == ARM::tMOVr)
      EmitAlignment(2);

    // Output the data for the jump table itself
    EmitJumpTable(MI);
    return;
  }
  case ARM::BR_JTm: {
    // Lower and emit the instruction itself, then the jump table following it.
    // ldr pc, target
    MCInst TmpInst;
    if (MI->getOperand(1).getReg() == 0) {
      // literal offset
      TmpInst.setOpcode(ARM::LDRi12);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
      TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
      TmpInst.addOperand(MCOperand::CreateImm(MI->getOperand(2).getImm()));
    } else {
      TmpInst.setOpcode(ARM::LDRrs);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
      TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
      TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(1).getReg()));
      TmpInst.addOperand(MCOperand::CreateImm(0));
    }
    // Add predicate operands.
    TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    TmpInst.addOperand(MCOperand::CreateReg(0));
    OutStreamer.EmitInstruction(TmpInst);

    // Output the data for the jump table itself
    EmitJumpTable(MI);
    return;
  }
  case ARM::BR_JTadd: {
    // Lower and emit the instruction itself, then the jump table following it.
    // add pc, target, idx
    MCInst TmpInst;
    TmpInst.setOpcode(ARM::ADDrr);
    TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(1).getReg()));
    // Add predicate operands.
    TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
    TmpInst.addOperand(MCOperand::CreateReg(0));
    // Add 's' bit operand (always reg0 for this)
    TmpInst.addOperand(MCOperand::CreateReg(0));
    OutStreamer.EmitInstruction(TmpInst);

    // Output the data for the jump table itself
    EmitJumpTable(MI);
    return;
  }
  case ARM::TRAP: {
    // Non-Darwin binutils don't yet support the "trap" mnemonic.
    // FIXME: Remove this special case when they do.
    if (!Subtarget->isTargetDarwin()) {
      //.long 0xe7ffdefe @ trap
      uint32_t Val = 0xe7ffdefeUL;
      OutStreamer.AddComment("trap");
      OutStreamer.EmitIntValue(Val, 4);
      return;
    }
    break;
  }
  case ARM::tTRAP: {
    // Non-Darwin binutils don't yet support the "trap" mnemonic.
    // FIXME: Remove this special case when they do.
    if (!Subtarget->isTargetDarwin()) {
      //.short 57086 @ trap
      uint16_t Val = 0xdefe;
      OutStreamer.AddComment("trap");
      OutStreamer.EmitIntValue(Val, 2);
      return;
    }
    break;
  }
  case ARM::t2Int_eh_sjlj_setjmp:
  case ARM::t2Int_eh_sjlj_setjmp_nofp:
  case ARM::tInt_eh_sjlj_setjmp: {
    // Two incoming args: GPR:$src, GPR:$val
    // mov $val, pc
    // adds $val, #7
    // str $val, [$src, #4]
    // movs r0, #0
    // b 1f
    // movs r0, #1
    // 1:
    unsigned SrcReg = MI->getOperand(0).getReg();
    unsigned ValReg = MI->getOperand(1).getReg();
    MCSymbol *Label = GetARMSJLJEHLabel();
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::tMOVr);
      TmpInst.addOperand(MCOperand::CreateReg(ValReg));
      TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.AddComment("eh_setjmp begin");
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::tADDi3);
      TmpInst.addOperand(MCOperand::CreateReg(ValReg));
      // 's' bit operand
      TmpInst.addOperand(MCOperand::CreateReg(ARM::CPSR));
      TmpInst.addOperand(MCOperand::CreateReg(ValReg));
      TmpInst.addOperand(MCOperand::CreateImm(7));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::tSTRi);
      TmpInst.addOperand(MCOperand::CreateReg(ValReg));
      TmpInst.addOperand(MCOperand::CreateReg(SrcReg));
      // The offset immediate is #4. The operand value is scaled by 4 for the
      // tSTR instruction.
      TmpInst.addOperand(MCOperand::CreateImm(1));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::tMOVi8);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::R0));
      TmpInst.addOperand(MCOperand::CreateReg(ARM::CPSR));
      TmpInst.addOperand(MCOperand::CreateImm(0));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      const MCExpr *SymbolExpr = MCSymbolRefExpr::Create(Label, OutContext);
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::tB);
      TmpInst.addOperand(MCOperand::CreateExpr(SymbolExpr));
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::tMOVi8);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::R0));
      TmpInst.addOperand(MCOperand::CreateReg(ARM::CPSR));
      TmpInst.addOperand(MCOperand::CreateImm(1));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.AddComment("eh_setjmp end");
      OutStreamer.EmitInstruction(TmpInst);
    }
    OutStreamer.EmitLabel(Label);
    return;
  }

  case ARM::Int_eh_sjlj_setjmp_nofp:
  case ARM::Int_eh_sjlj_setjmp: {
    // Two incoming args: GPR:$src, GPR:$val
    // add $val, pc, #8
    // str $val, [$src, #+4]
    // mov r0, #0
    // add pc, pc, #0
    // mov r0, #1
    unsigned SrcReg = MI->getOperand(0).getReg();
    unsigned ValReg = MI->getOperand(1).getReg();

    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::ADDri);
      TmpInst.addOperand(MCOperand::CreateReg(ValReg));
      TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
      TmpInst.addOperand(MCOperand::CreateImm(8));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      // 's' bit operand (always reg0 for this).
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.AddComment("eh_setjmp begin");
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::STRi12);
      TmpInst.addOperand(MCOperand::CreateReg(ValReg));
      TmpInst.addOperand(MCOperand::CreateReg(SrcReg));
      TmpInst.addOperand(MCOperand::CreateImm(4));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::MOVi);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::R0));
      TmpInst.addOperand(MCOperand::CreateImm(0));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      // 's' bit operand (always reg0 for this).
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::ADDri);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
      TmpInst.addOperand(MCOperand::CreateReg(ARM::PC));
      TmpInst.addOperand(MCOperand::CreateImm(0));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      // 's' bit operand (always reg0 for this).
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::MOVi);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::R0));
      TmpInst.addOperand(MCOperand::CreateImm(1));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      // 's' bit operand (always reg0 for this).
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.AddComment("eh_setjmp end");
      OutStreamer.EmitInstruction(TmpInst);
    }
    return;
  }
  case ARM::Int_eh_sjlj_longjmp: {
    // ldr sp, [$src, #8]
    // ldr $scratch, [$src, #4]
    // ldr r7, [$src]
    // bx $scratch
    unsigned SrcReg = MI->getOperand(0).getReg();
    unsigned ScratchReg = MI->getOperand(1).getReg();
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::LDRi12);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::SP));
      TmpInst.addOperand(MCOperand::CreateReg(SrcReg));
      TmpInst.addOperand(MCOperand::CreateImm(8));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::LDRi12);
      TmpInst.addOperand(MCOperand::CreateReg(ScratchReg));
      TmpInst.addOperand(MCOperand::CreateReg(SrcReg));
      TmpInst.addOperand(MCOperand::CreateImm(4));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::LDRi12);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::R7));
      TmpInst.addOperand(MCOperand::CreateReg(SrcReg));
      TmpInst.addOperand(MCOperand::CreateImm(0));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::BX);
      TmpInst.addOperand(MCOperand::CreateReg(ScratchReg));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    return;
  }
  case ARM::tInt_eh_sjlj_longjmp: {
    // ldr $scratch, [$src, #8]
    // mov sp, $scratch
    // ldr $scratch, [$src, #4]
    // ldr r7, [$src]
    // bx $scratch
    unsigned SrcReg = MI->getOperand(0).getReg();
    unsigned ScratchReg = MI->getOperand(1).getReg();
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::tLDRi);
      TmpInst.addOperand(MCOperand::CreateReg(ScratchReg));
      TmpInst.addOperand(MCOperand::CreateReg(SrcReg));
      // The offset immediate is #8. The operand value is scaled by 4 for the
      // tLDR instruction.
      TmpInst.addOperand(MCOperand::CreateImm(2));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::tMOVr);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::SP));
      TmpInst.addOperand(MCOperand::CreateReg(ScratchReg));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::tLDRi);
      TmpInst.addOperand(MCOperand::CreateReg(ScratchReg));
      TmpInst.addOperand(MCOperand::CreateReg(SrcReg));
      TmpInst.addOperand(MCOperand::CreateImm(1));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::tLDRi);
      TmpInst.addOperand(MCOperand::CreateReg(ARM::R7));
      TmpInst.addOperand(MCOperand::CreateReg(SrcReg));
      TmpInst.addOperand(MCOperand::CreateImm(0));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    {
      MCInst TmpInst;
      TmpInst.setOpcode(ARM::tBX);
      TmpInst.addOperand(MCOperand::CreateReg(ScratchReg));
      // Predicate.
      TmpInst.addOperand(MCOperand::CreateImm(ARMCC::AL));
      TmpInst.addOperand(MCOperand::CreateReg(0));
      OutStreamer.EmitInstruction(TmpInst);
    }
    return;
  }
  }

  MCInst TmpInst;
  LowerARMMachineInstrToMCInst(MI, TmpInst, *this);

  OutStreamer.EmitInstruction(TmpInst);
}

//===----------------------------------------------------------------------===//
// Target Registry Stuff
//===----------------------------------------------------------------------===//

// Force static initialization.
extern "C" void LLVMInitializeARMAsmPrinter() {
  RegisterAsmPrinter<ARMAsmPrinter> X(TheARMTarget);
  RegisterAsmPrinter<ARMAsmPrinter> Y(TheThumbTarget);
}
