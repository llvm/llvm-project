//===- lib/MC/ARMELFStreamer.cpp - ELF Object Output for ARM --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file assembles .s files and emits ARM ELF .o object files. Different
// from generic ELF streamer in emitting mapping symbols ($a, $t and $d) to
// delimit regions of data and code.
//
//===----------------------------------------------------------------------===//

#include "ARMRegisterInfo.h"
#include "ARMUnwindOpAsm.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ARMBuildAttributes.h"
#include "llvm/Support/ARMEHABI.h"
#include "llvm/Support/TargetParser.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace llvm;

static std::string GetAEABIUnwindPersonalityName(unsigned Index) {
  assert(Index < ARM::EHABI::NUM_PERSONALITY_INDEX &&
         "Invalid personality index");
  return (Twine("__aeabi_unwind_cpp_pr") + Twine(Index)).str();
}

namespace {

class ARMELFStreamer;

class ARMTargetAsmStreamer : public ARMTargetStreamer {
  formatted_raw_ostream &OS;
  MCInstPrinter &InstPrinter;
  bool IsVerboseAsm;

  void emitFnStart() override;
  void emitFnEnd() override;
  void emitCantUnwind() override;
  void emitPersonality(const MCSymbol *Personality) override;
  void emitPersonalityIndex(unsigned Index) override;
  void emitHandlerData() override;
  void emitSetFP(unsigned FpReg, unsigned SpReg, int64_t Offset = 0) override;
  void emitMovSP(unsigned Reg, int64_t Offset = 0) override;
  void emitPad(int64_t Offset) override;
  void emitRegSave(const SmallVectorImpl<unsigned> &RegList,
                   bool isVector) override;
  void emitUnwindRaw(int64_t Offset,
                     const SmallVectorImpl<uint8_t> &Opcodes) override;

  void switchVendor(StringRef Vendor) override;
  void emitAttribute(unsigned Attribute, unsigned Value) override;
  void emitTextAttribute(unsigned Attribute, StringRef String) override;
  void emitIntTextAttribute(unsigned Attribute, unsigned IntValue,
                            StringRef StrinValue) override;
  void emitArch(unsigned Arch) override;
  void emitArchExtension(unsigned ArchExt) override;
  void emitObjectArch(unsigned Arch) override;
  void emitFPU(unsigned FPU) override;
  void emitInst(uint32_t Inst, char Suffix = '\0') override;
  void finishAttributeSection() override;

  void AnnotateTLSDescriptorSequence(const MCSymbolRefExpr *SRE) override;
  void emitThumbSet(MCSymbol *Symbol, const MCExpr *Value) override;

public:
  ARMTargetAsmStreamer(MCStreamer &S, formatted_raw_ostream &OS,
                       MCInstPrinter &InstPrinter, bool VerboseAsm);
};

ARMTargetAsmStreamer::ARMTargetAsmStreamer(MCStreamer &S,
                                           formatted_raw_ostream &OS,
                                           MCInstPrinter &InstPrinter,
                                           bool VerboseAsm)
    : ARMTargetStreamer(S), OS(OS), InstPrinter(InstPrinter),
      IsVerboseAsm(VerboseAsm) {}
void ARMTargetAsmStreamer::emitFnStart() { OS << "\t.fnstart\n"; }
void ARMTargetAsmStreamer::emitFnEnd() { OS << "\t.fnend\n"; }
void ARMTargetAsmStreamer::emitCantUnwind() { OS << "\t.cantunwind\n"; }
void ARMTargetAsmStreamer::emitPersonality(const MCSymbol *Personality) {
  OS << "\t.personality " << Personality->getName() << '\n';
}
void ARMTargetAsmStreamer::emitPersonalityIndex(unsigned Index) {
  OS << "\t.personalityindex " << Index << '\n';
}
void ARMTargetAsmStreamer::emitHandlerData() { OS << "\t.handlerdata\n"; }
void ARMTargetAsmStreamer::emitSetFP(unsigned FpReg, unsigned SpReg,
                                     int64_t Offset) {
  OS << "\t.setfp\t";
  InstPrinter.printRegName(OS, FpReg);
  OS << ", ";
  InstPrinter.printRegName(OS, SpReg);
  if (Offset)
    OS << ", #" << Offset;
  OS << '\n';
}
void ARMTargetAsmStreamer::emitMovSP(unsigned Reg, int64_t Offset) {
  assert((Reg != ARM::SP && Reg != ARM::PC) &&
         "the operand of .movsp cannot be either sp or pc");

  OS << "\t.movsp\t";
  InstPrinter.printRegName(OS, Reg);
  if (Offset)
    OS << ", #" << Offset;
  OS << '\n';
}
void ARMTargetAsmStreamer::emitPad(int64_t Offset) {
  OS << "\t.pad\t#" << Offset << '\n';
}
void ARMTargetAsmStreamer::emitRegSave(const SmallVectorImpl<unsigned> &RegList,
                                       bool isVector) {
  assert(RegList.size() && "RegList should not be empty");
  if (isVector)
    OS << "\t.vsave\t{";
  else
    OS << "\t.save\t{";

  InstPrinter.printRegName(OS, RegList[0]);

  for (unsigned i = 1, e = RegList.size(); i != e; ++i) {
    OS << ", ";
    InstPrinter.printRegName(OS, RegList[i]);
  }

  OS << "}\n";
}
void ARMTargetAsmStreamer::switchVendor(StringRef Vendor) {
}
void ARMTargetAsmStreamer::emitAttribute(unsigned Attribute, unsigned Value) {
  OS << "\t.eabi_attribute\t" << Attribute << ", " << Twine(Value);
  if (IsVerboseAsm) {
    StringRef Name = ARMBuildAttrs::AttrTypeAsString(Attribute);
    if (!Name.empty())
      OS << "\t@ " << Name;
  }
  OS << "\n";
}
void ARMTargetAsmStreamer::emitTextAttribute(unsigned Attribute,
                                             StringRef String) {
  switch (Attribute) {
  case ARMBuildAttrs::CPU_name:
    OS << "\t.cpu\t" << String.lower();
    break;
  default:
    OS << "\t.eabi_attribute\t" << Attribute << ", \"" << String << "\"";
    if (IsVerboseAsm) {
      StringRef Name = ARMBuildAttrs::AttrTypeAsString(Attribute);
      if (!Name.empty())
        OS << "\t@ " << Name;
    }
    break;
  }
  OS << "\n";
}
void ARMTargetAsmStreamer::emitIntTextAttribute(unsigned Attribute,
                                                unsigned IntValue,
                                                StringRef StringValue) {
  switch (Attribute) {
  default: llvm_unreachable("unsupported multi-value attribute in asm mode");
  case ARMBuildAttrs::compatibility:
    OS << "\t.eabi_attribute\t" << Attribute << ", " << IntValue;
    if (!StringValue.empty())
      OS << ", \"" << StringValue << "\"";
    if (IsVerboseAsm)
      OS << "\t@ " << ARMBuildAttrs::AttrTypeAsString(Attribute);
    break;
  }
  OS << "\n";
}
void ARMTargetAsmStreamer::emitArch(unsigned Arch) {
  OS << "\t.arch\t" << ARM::getArchName(Arch) << "\n";
}
void ARMTargetAsmStreamer::emitArchExtension(unsigned ArchExt) {
  OS << "\t.arch_extension\t" << ARM::getArchExtName(ArchExt) << "\n";
}
void ARMTargetAsmStreamer::emitObjectArch(unsigned Arch) {
  OS << "\t.object_arch\t" << ARM::getArchName(Arch) << '\n';
}
void ARMTargetAsmStreamer::emitFPU(unsigned FPU) {
  OS << "\t.fpu\t" << ARM::getFPUName(FPU) << "\n";
}
void ARMTargetAsmStreamer::finishAttributeSection() {
}
void
ARMTargetAsmStreamer::AnnotateTLSDescriptorSequence(const MCSymbolRefExpr *S) {
  OS << "\t.tlsdescseq\t" << S->getSymbol().getName();
}

void ARMTargetAsmStreamer::emitThumbSet(MCSymbol *Symbol, const MCExpr *Value) {
  const MCAsmInfo *MAI = Streamer.getContext().getAsmInfo();

  OS << "\t.thumb_set\t";
  Symbol->print(OS, MAI);
  OS << ", ";
  Value->print(OS, MAI);
  OS << '\n';
}

void ARMTargetAsmStreamer::emitInst(uint32_t Inst, char Suffix) {
  OS << "\t.inst";
  if (Suffix)
    OS << "." << Suffix;
  OS << "\t0x" << Twine::utohexstr(Inst) << "\n";
}

void ARMTargetAsmStreamer::emitUnwindRaw(int64_t Offset,
                                      const SmallVectorImpl<uint8_t> &Opcodes) {
  OS << "\t.unwind_raw " << Offset;
  for (SmallVectorImpl<uint8_t>::const_iterator OCI = Opcodes.begin(),
                                                OCE = Opcodes.end();
       OCI != OCE; ++OCI)
    OS << ", 0x" << Twine::utohexstr(*OCI);
  OS << '\n';
}

class ARMTargetELFStreamer : public ARMTargetStreamer {
private:
  // This structure holds all attributes, accounting for
  // their string/numeric value, so we can later emmit them
  // in declaration order, keeping all in the same vector
  struct AttributeItem {
    enum {
      HiddenAttribute = 0,
      NumericAttribute,
      TextAttribute,
      NumericAndTextAttributes
    } Type;
    unsigned Tag;
    unsigned IntValue;
    StringRef StringValue;

    static bool LessTag(const AttributeItem &LHS, const AttributeItem &RHS) {
      // The conformance tag must be emitted first when serialised
      // into an object file. Specifically, the addenda to the ARM ABI
      // states that (2.3.7.4):
      //
      // "To simplify recognition by consumers in the common case of
      // claiming conformity for the whole file, this tag should be
      // emitted first in a file-scope sub-subsection of the first
      // public subsection of the attributes section."
      //
      // So it is special-cased in this comparison predicate when the
      // attributes are sorted in finishAttributeSection().
      return (RHS.Tag != ARMBuildAttrs::conformance) &&
             ((LHS.Tag == ARMBuildAttrs::conformance) || (LHS.Tag < RHS.Tag));
    }
  };

  StringRef CurrentVendor;
  unsigned FPU;
  unsigned Arch;
  unsigned EmittedArch;
  SmallVector<AttributeItem, 64> Contents;

  MCSection *AttributeSection;

  AttributeItem *getAttributeItem(unsigned Attribute) {
    for (size_t i = 0; i < Contents.size(); ++i)
      if (Contents[i].Tag == Attribute)
        return &Contents[i];
    return nullptr;
  }

  void setAttributeItem(unsigned Attribute, unsigned Value,
                        bool OverwriteExisting) {
    // Look for existing attribute item
    if (AttributeItem *Item = getAttributeItem(Attribute)) {
      if (!OverwriteExisting)
        return;
      Item->Type = AttributeItem::NumericAttribute;
      Item->IntValue = Value;
      return;
    }

    // Create new attribute item
    AttributeItem Item = {
      AttributeItem::NumericAttribute,
      Attribute,
      Value,
      StringRef("")
    };
    Contents.push_back(Item);
  }

  void setAttributeItem(unsigned Attribute, StringRef Value,
                        bool OverwriteExisting) {
    // Look for existing attribute item
    if (AttributeItem *Item = getAttributeItem(Attribute)) {
      if (!OverwriteExisting)
        return;
      Item->Type = AttributeItem::TextAttribute;
      Item->StringValue = Value;
      return;
    }

    // Create new attribute item
    AttributeItem Item = {
      AttributeItem::TextAttribute,
      Attribute,
      0,
      Value
    };
    Contents.push_back(Item);
  }

  void setAttributeItems(unsigned Attribute, unsigned IntValue,
                         StringRef StringValue, bool OverwriteExisting) {
    // Look for existing attribute item
    if (AttributeItem *Item = getAttributeItem(Attribute)) {
      if (!OverwriteExisting)
        return;
      Item->Type = AttributeItem::NumericAndTextAttributes;
      Item->IntValue = IntValue;
      Item->StringValue = StringValue;
      return;
    }

    // Create new attribute item
    AttributeItem Item = {
      AttributeItem::NumericAndTextAttributes,
      Attribute,
      IntValue,
      StringValue
    };
    Contents.push_back(Item);
  }

  void emitArchDefaultAttributes();
  void emitFPUDefaultAttributes();

  ARMELFStreamer &getStreamer();

  void emitFnStart() override;
  void emitFnEnd() override;
  void emitCantUnwind() override;
  void emitPersonality(const MCSymbol *Personality) override;
  void emitPersonalityIndex(unsigned Index) override;
  void emitHandlerData() override;
  void emitSetFP(unsigned FpReg, unsigned SpReg, int64_t Offset = 0) override;
  void emitMovSP(unsigned Reg, int64_t Offset = 0) override;
  void emitPad(int64_t Offset) override;
  void emitRegSave(const SmallVectorImpl<unsigned> &RegList,
                   bool isVector) override;
  void emitUnwindRaw(int64_t Offset,
                     const SmallVectorImpl<uint8_t> &Opcodes) override;

  void switchVendor(StringRef Vendor) override;
  void emitAttribute(unsigned Attribute, unsigned Value) override;
  void emitTextAttribute(unsigned Attribute, StringRef String) override;
  void emitIntTextAttribute(unsigned Attribute, unsigned IntValue,
                            StringRef StringValue) override;
  void emitArch(unsigned Arch) override;
  void emitObjectArch(unsigned Arch) override;
  void emitFPU(unsigned FPU) override;
  void emitInst(uint32_t Inst, char Suffix = '\0') override;
  void finishAttributeSection() override;
  void emitLabel(MCSymbol *Symbol) override;

  void AnnotateTLSDescriptorSequence(const MCSymbolRefExpr *SRE) override;
  void emitThumbSet(MCSymbol *Symbol, const MCExpr *Value) override;

  size_t calculateContentSize() const;

public:
  ARMTargetELFStreamer(MCStreamer &S)
    : ARMTargetStreamer(S), CurrentVendor("aeabi"), FPU(ARM::FK_INVALID),
      Arch(ARM::AK_INVALID), EmittedArch(ARM::AK_INVALID),
      AttributeSection(nullptr) {}
};

/// Extend the generic ELFStreamer class so that it can emit mapping symbols at
/// the appropriate points in the object files. These symbols are defined in the
/// ARM ELF ABI: infocenter.arm.com/help/topic/com.arm.../IHI0044D_aaelf.pdf.
///
/// In brief: $a, $t or $d should be emitted at the start of each contiguous
/// region of ARM code, Thumb code or data in a section. In practice, this
/// emission does not rely on explicit assembler directives but on inherent
/// properties of the directives doing the emission (e.g. ".byte" is data, "add
/// r0, r0, r0" an instruction).
///
/// As a result this system is orthogonal to the DataRegion infrastructure used
/// by MachO. Beware!
class ARMELFStreamer : public MCELFStreamer {
public:
  friend class ARMTargetELFStreamer;

  ARMELFStreamer(MCContext &Context, MCAsmBackend &TAB, raw_pwrite_stream &OS,
                 MCCodeEmitter *Emitter, bool IsThumb)
      : MCELFStreamer(Context, TAB, OS, Emitter), IsThumb(IsThumb),
        MappingSymbolCounter(0), LastEMS(EMS_None) {
    Reset();
  }

  ~ARMELFStreamer() {}

  void FinishImpl() override;

  // ARM exception handling directives
  void emitFnStart();
  void emitFnEnd();
  void emitCantUnwind();
  void emitPersonality(const MCSymbol *Per);
  void emitPersonalityIndex(unsigned index);
  void emitHandlerData();
  void emitSetFP(unsigned NewFpReg, unsigned NewSpReg, int64_t Offset = 0);
  void emitMovSP(unsigned Reg, int64_t Offset = 0);
  void emitPad(int64_t Offset);
  void emitRegSave(const SmallVectorImpl<unsigned> &RegList, bool isVector);
  void emitUnwindRaw(int64_t Offset, const SmallVectorImpl<uint8_t> &Opcodes);

  void ChangeSection(MCSection *Section, const MCExpr *Subsection) override {
    // We have to keep track of the mapping symbol state of any sections we
    // use. Each one should start off as EMS_None, which is provided as the
    // default constructor by DenseMap::lookup.
    LastMappingSymbols[getPreviousSection().first] = LastEMS;
    LastEMS = LastMappingSymbols.lookup(Section);

    MCELFStreamer::ChangeSection(Section, Subsection);
  }

  /// This function is the one used to emit instruction data into the ELF
  /// streamer. We override it to add the appropriate mapping symbol if
  /// necessary.
  void EmitInstruction(const MCInst& Inst,
                       const MCSubtargetInfo &STI) override {
    if (IsThumb)
      EmitThumbMappingSymbol();
    else
      EmitARMMappingSymbol();

    MCELFStreamer::EmitInstruction(Inst, STI);
  }

  void emitInst(uint32_t Inst, char Suffix) {
    unsigned Size;
    char Buffer[4];
    const bool LittleEndian = getContext().getAsmInfo()->isLittleEndian();

    switch (Suffix) {
    case '\0':
      Size = 4;

      assert(!IsThumb);
      EmitARMMappingSymbol();
      for (unsigned II = 0, IE = Size; II != IE; II++) {
        const unsigned I = LittleEndian ? (Size - II - 1) : II;
        Buffer[Size - II - 1] = uint8_t(Inst >> I * CHAR_BIT);
      }

      break;
    case 'n':
    case 'w':
      Size = (Suffix == 'n' ? 2 : 4);

      assert(IsThumb);
      EmitThumbMappingSymbol();
      for (unsigned II = 0, IE = Size; II != IE; II = II + 2) {
        const unsigned I0 = LittleEndian ? II + 0 : (Size - II - 1);
        const unsigned I1 = LittleEndian ? II + 1 : (Size - II - 2);
        Buffer[Size - II - 2] = uint8_t(Inst >> I0 * CHAR_BIT);
        Buffer[Size - II - 1] = uint8_t(Inst >> I1 * CHAR_BIT);
      }

      break;
    default:
      llvm_unreachable("Invalid Suffix");
    }

    MCELFStreamer::EmitBytes(StringRef(Buffer, Size));
  }

  /// This is one of the functions used to emit data into an ELF section, so the
  /// ARM streamer overrides it to add the appropriate mapping symbol ($d) if
  /// necessary.
  void EmitBytes(StringRef Data) override {
    EmitDataMappingSymbol();
    MCELFStreamer::EmitBytes(Data);
  }

  /// This is one of the functions used to emit data into an ELF section, so the
  /// ARM streamer overrides it to add the appropriate mapping symbol ($d) if
  /// necessary.
  void EmitValueImpl(const MCExpr *Value, unsigned Size, SMLoc Loc) override {
    if (const MCSymbolRefExpr *SRE = dyn_cast_or_null<MCSymbolRefExpr>(Value))
      if (SRE->getKind() == MCSymbolRefExpr::VK_ARM_SBREL && !(Size == 4))
        getContext().reportFatalError(Loc, "relocated expression must be 32-bit");

    EmitDataMappingSymbol();
    MCELFStreamer::EmitValueImpl(Value, Size);
  }

  void EmitAssemblerFlag(MCAssemblerFlag Flag) override {
    MCELFStreamer::EmitAssemblerFlag(Flag);

    switch (Flag) {
    case MCAF_SyntaxUnified:
      return; // no-op here.
    case MCAF_Code16:
      IsThumb = true;
      return; // Change to Thumb mode
    case MCAF_Code32:
      IsThumb = false;
      return; // Change to ARM mode
    case MCAF_Code64:
      return;
    case MCAF_SubsectionsViaSymbols:
      return;
    }
  }

private:
  enum ElfMappingSymbol {
    EMS_None,
    EMS_ARM,
    EMS_Thumb,
    EMS_Data
  };

  void EmitDataMappingSymbol() {
    if (LastEMS == EMS_Data) return;
    EmitMappingSymbol("$d");
    LastEMS = EMS_Data;
  }

  void EmitThumbMappingSymbol() {
    if (LastEMS == EMS_Thumb) return;
    EmitMappingSymbol("$t");
    LastEMS = EMS_Thumb;
  }

  void EmitARMMappingSymbol() {
    if (LastEMS == EMS_ARM) return;
    EmitMappingSymbol("$a");
    LastEMS = EMS_ARM;
  }

  void EmitMappingSymbol(StringRef Name) {
    auto *Symbol = cast<MCSymbolELF>(getContext().getOrCreateSymbol(
        Name + "." + Twine(MappingSymbolCounter++)));
    EmitLabel(Symbol);

    Symbol->setType(ELF::STT_NOTYPE);
    Symbol->setBinding(ELF::STB_LOCAL);
    Symbol->setExternal(false);
  }

  void EmitThumbFunc(MCSymbol *Func) override {
    getAssembler().setIsThumbFunc(Func);
    EmitSymbolAttribute(Func, MCSA_ELF_TypeFunction);
  }

  // Helper functions for ARM exception handling directives
  void Reset();

  void EmitPersonalityFixup(StringRef Name);
  void FlushPendingOffset();
  void FlushUnwindOpcodes(bool NoHandlerData);

  void SwitchToEHSection(const char *Prefix, unsigned Type, unsigned Flags,
                         SectionKind Kind, const MCSymbol &Fn);
  void SwitchToExTabSection(const MCSymbol &FnStart);
  void SwitchToExIdxSection(const MCSymbol &FnStart);

  void EmitFixup(const MCExpr *Expr, MCFixupKind Kind);

  bool IsThumb;
  int64_t MappingSymbolCounter;

  DenseMap<const MCSection *, ElfMappingSymbol> LastMappingSymbols;
  ElfMappingSymbol LastEMS;

  // ARM Exception Handling Frame Information
  MCSymbol *ExTab;
  MCSymbol *FnStart;
  const MCSymbol *Personality;
  unsigned PersonalityIndex;
  unsigned FPReg; // Frame pointer register
  int64_t FPOffset; // Offset: (final frame pointer) - (initial $sp)
  int64_t SPOffset; // Offset: (final $sp) - (initial $sp)
  int64_t PendingOffset; // Offset: (final $sp) - (emitted $sp)
  bool UsedFP;
  bool CantUnwind;
  SmallVector<uint8_t, 64> Opcodes;
  UnwindOpcodeAssembler UnwindOpAsm;
};
} // end anonymous namespace

ARMELFStreamer &ARMTargetELFStreamer::getStreamer() {
  return static_cast<ARMELFStreamer &>(Streamer);
}

void ARMTargetELFStreamer::emitFnStart() { getStreamer().emitFnStart(); }
void ARMTargetELFStreamer::emitFnEnd() { getStreamer().emitFnEnd(); }
void ARMTargetELFStreamer::emitCantUnwind() { getStreamer().emitCantUnwind(); }
void ARMTargetELFStreamer::emitPersonality(const MCSymbol *Personality) {
  getStreamer().emitPersonality(Personality);
}
void ARMTargetELFStreamer::emitPersonalityIndex(unsigned Index) {
  getStreamer().emitPersonalityIndex(Index);
}
void ARMTargetELFStreamer::emitHandlerData() {
  getStreamer().emitHandlerData();
}
void ARMTargetELFStreamer::emitSetFP(unsigned FpReg, unsigned SpReg,
                                     int64_t Offset) {
  getStreamer().emitSetFP(FpReg, SpReg, Offset);
}
void ARMTargetELFStreamer::emitMovSP(unsigned Reg, int64_t Offset) {
  getStreamer().emitMovSP(Reg, Offset);
}
void ARMTargetELFStreamer::emitPad(int64_t Offset) {
  getStreamer().emitPad(Offset);
}
void ARMTargetELFStreamer::emitRegSave(const SmallVectorImpl<unsigned> &RegList,
                                       bool isVector) {
  getStreamer().emitRegSave(RegList, isVector);
}
void ARMTargetELFStreamer::emitUnwindRaw(int64_t Offset,
                                      const SmallVectorImpl<uint8_t> &Opcodes) {
  getStreamer().emitUnwindRaw(Offset, Opcodes);
}
void ARMTargetELFStreamer::switchVendor(StringRef Vendor) {
  assert(!Vendor.empty() && "Vendor cannot be empty.");

  if (CurrentVendor == Vendor)
    return;

  if (!CurrentVendor.empty())
    finishAttributeSection();

  assert(Contents.empty() &&
         ".ARM.attributes should be flushed before changing vendor");
  CurrentVendor = Vendor;

}
void ARMTargetELFStreamer::emitAttribute(unsigned Attribute, unsigned Value) {
  setAttributeItem(Attribute, Value, /* OverwriteExisting= */ true);
}
void ARMTargetELFStreamer::emitTextAttribute(unsigned Attribute,
                                             StringRef Value) {
  setAttributeItem(Attribute, Value, /* OverwriteExisting= */ true);
}
void ARMTargetELFStreamer::emitIntTextAttribute(unsigned Attribute,
                                                unsigned IntValue,
                                                StringRef StringValue) {
  setAttributeItems(Attribute, IntValue, StringValue,
                    /* OverwriteExisting= */ true);
}
void ARMTargetELFStreamer::emitArch(unsigned Value) {
  Arch = Value;
}
void ARMTargetELFStreamer::emitObjectArch(unsigned Value) {
  EmittedArch = Value;
}
void ARMTargetELFStreamer::emitArchDefaultAttributes() {
  using namespace ARMBuildAttrs;

  setAttributeItem(CPU_name,
                   ARM::getCPUAttr(Arch),
                   false);

  if (EmittedArch == ARM::AK_INVALID)
    setAttributeItem(CPU_arch,
                     ARM::getArchAttr(Arch),
                     false);
  else
    setAttributeItem(CPU_arch,
                     ARM::getArchAttr(EmittedArch),
                     false);

  switch (Arch) {
  case ARM::AK_ARMV2:
  case ARM::AK_ARMV2A:
  case ARM::AK_ARMV3:
  case ARM::AK_ARMV3M:
  case ARM::AK_ARMV4:
  case ARM::AK_ARMV5:
    setAttributeItem(ARM_ISA_use, Allowed, false);
    break;

  case ARM::AK_ARMV4T:
  case ARM::AK_ARMV5T:
  case ARM::AK_ARMV5TE:
  case ARM::AK_ARMV6:
  case ARM::AK_ARMV6J:
    setAttributeItem(ARM_ISA_use, Allowed, false);
    setAttributeItem(THUMB_ISA_use, Allowed, false);
    break;

  case ARM::AK_ARMV6T2:
    setAttributeItem(ARM_ISA_use, Allowed, false);
    setAttributeItem(THUMB_ISA_use, AllowThumb32, false);
    break;

  case ARM::AK_ARMV6K:
  case ARM::AK_ARMV6Z:
  case ARM::AK_ARMV6ZK:
    setAttributeItem(ARM_ISA_use, Allowed, false);
    setAttributeItem(THUMB_ISA_use, Allowed, false);
    setAttributeItem(Virtualization_use, AllowTZ, false);
    break;

  case ARM::AK_ARMV6M:
    setAttributeItem(THUMB_ISA_use, Allowed, false);
    break;

  case ARM::AK_ARMV7:
    setAttributeItem(THUMB_ISA_use, AllowThumb32, false);
    break;

  case ARM::AK_ARMV7A:
    setAttributeItem(CPU_arch_profile, ApplicationProfile, false);
    setAttributeItem(ARM_ISA_use, Allowed, false);
    setAttributeItem(THUMB_ISA_use, AllowThumb32, false);
    break;

  case ARM::AK_ARMV7R:
    setAttributeItem(CPU_arch_profile, RealTimeProfile, false);
    setAttributeItem(ARM_ISA_use, Allowed, false);
    setAttributeItem(THUMB_ISA_use, AllowThumb32, false);
    break;

  case ARM::AK_ARMV7M:
    setAttributeItem(CPU_arch_profile, MicroControllerProfile, false);
    setAttributeItem(THUMB_ISA_use, AllowThumb32, false);
    break;

  case ARM::AK_ARMV8A:
  case ARM::AK_ARMV8_1A:
    setAttributeItem(CPU_arch_profile, ApplicationProfile, false);
    setAttributeItem(ARM_ISA_use, Allowed, false);
    setAttributeItem(THUMB_ISA_use, AllowThumb32, false);
    setAttributeItem(MPextension_use, Allowed, false);
    setAttributeItem(Virtualization_use, AllowTZVirtualization, false);
    break;

  case ARM::AK_IWMMXT:
    setAttributeItem(ARM_ISA_use, Allowed, false);
    setAttributeItem(THUMB_ISA_use, Allowed, false);
    setAttributeItem(WMMX_arch, AllowWMMXv1, false);
    break;

  case ARM::AK_IWMMXT2:
    setAttributeItem(ARM_ISA_use, Allowed, false);
    setAttributeItem(THUMB_ISA_use, Allowed, false);
    setAttributeItem(WMMX_arch, AllowWMMXv2, false);
    break;

  default:
    report_fatal_error("Unknown Arch: " + Twine(Arch));
    break;
  }
}
void ARMTargetELFStreamer::emitFPU(unsigned Value) {
  FPU = Value;
}
void ARMTargetELFStreamer::emitFPUDefaultAttributes() {
  switch (FPU) {
  case ARM::FK_VFP:
  case ARM::FK_VFPV2:
    setAttributeItem(ARMBuildAttrs::FP_arch,
                     ARMBuildAttrs::AllowFPv2,
                     /* OverwriteExisting= */ false);
    break;

  case ARM::FK_VFPV3:
    setAttributeItem(ARMBuildAttrs::FP_arch,
                     ARMBuildAttrs::AllowFPv3A,
                     /* OverwriteExisting= */ false);
    break;

  case ARM::FK_VFPV3_FP16:
    setAttributeItem(ARMBuildAttrs::FP_arch,
                     ARMBuildAttrs::AllowFPv3A,
                     /* OverwriteExisting= */ false);
    setAttributeItem(ARMBuildAttrs::FP_HP_extension,
                     ARMBuildAttrs::AllowHPFP,
                     /* OverwriteExisting= */ false);
    break;

  case ARM::FK_VFPV3_D16:
    setAttributeItem(ARMBuildAttrs::FP_arch,
                     ARMBuildAttrs::AllowFPv3B,
                     /* OverwriteExisting= */ false);
    break;

  case ARM::FK_VFPV3_D16_FP16:
    setAttributeItem(ARMBuildAttrs::FP_arch,
                     ARMBuildAttrs::AllowFPv3B,
                     /* OverwriteExisting= */ false);
    setAttributeItem(ARMBuildAttrs::FP_HP_extension,
                     ARMBuildAttrs::AllowHPFP,
                     /* OverwriteExisting= */ false);
    break;

  case ARM::FK_VFPV3XD:
    setAttributeItem(ARMBuildAttrs::FP_arch,
                     ARMBuildAttrs::AllowFPv3B,
                     /* OverwriteExisting= */ false);
    break;
  case ARM::FK_VFPV3XD_FP16:
    setAttributeItem(ARMBuildAttrs::FP_arch,
                     ARMBuildAttrs::AllowFPv3B,
                     /* OverwriteExisting= */ false);
    setAttributeItem(ARMBuildAttrs::FP_HP_extension,
                     ARMBuildAttrs::AllowHPFP,
                     /* OverwriteExisting= */ false);
    break;

  case ARM::FK_VFPV4:
    setAttributeItem(ARMBuildAttrs::FP_arch,
                     ARMBuildAttrs::AllowFPv4A,
                     /* OverwriteExisting= */ false);
    break;

  // ABI_HardFP_use is handled in ARMAsmPrinter, so _SP_D16 is treated the same
  // as _D16 here.
  case ARM::FK_FPV4_SP_D16:
  case ARM::FK_VFPV4_D16:
    setAttributeItem(ARMBuildAttrs::FP_arch,
                     ARMBuildAttrs::AllowFPv4B,
                     /* OverwriteExisting= */ false);
    break;

  case ARM::FK_FP_ARMV8:
    setAttributeItem(ARMBuildAttrs::FP_arch,
                     ARMBuildAttrs::AllowFPARMv8A,
                     /* OverwriteExisting= */ false);
    break;

  // FPV5_D16 is identical to FP_ARMV8 except for the number of D registers, so
  // uses the FP_ARMV8_D16 build attribute.
  case ARM::FK_FPV5_SP_D16:
  case ARM::FK_FPV5_D16:
    setAttributeItem(ARMBuildAttrs::FP_arch,
                     ARMBuildAttrs::AllowFPARMv8B,
                     /* OverwriteExisting= */ false);
    break;

  case ARM::FK_NEON:
    setAttributeItem(ARMBuildAttrs::FP_arch,
                     ARMBuildAttrs::AllowFPv3A,
                     /* OverwriteExisting= */ false);
    setAttributeItem(ARMBuildAttrs::Advanced_SIMD_arch,
                     ARMBuildAttrs::AllowNeon,
                     /* OverwriteExisting= */ false);
    break;

  case ARM::FK_NEON_FP16:
    setAttributeItem(ARMBuildAttrs::FP_arch,
                     ARMBuildAttrs::AllowFPv3A,
                     /* OverwriteExisting= */ false);
    setAttributeItem(ARMBuildAttrs::Advanced_SIMD_arch,
                     ARMBuildAttrs::AllowNeon,
                     /* OverwriteExisting= */ false);
    setAttributeItem(ARMBuildAttrs::FP_HP_extension,
                     ARMBuildAttrs::AllowHPFP,
                     /* OverwriteExisting= */ false);
    break;

  case ARM::FK_NEON_VFPV4:
    setAttributeItem(ARMBuildAttrs::FP_arch,
                     ARMBuildAttrs::AllowFPv4A,
                     /* OverwriteExisting= */ false);
    setAttributeItem(ARMBuildAttrs::Advanced_SIMD_arch,
                     ARMBuildAttrs::AllowNeon2,
                     /* OverwriteExisting= */ false);
    break;

  case ARM::FK_NEON_FP_ARMV8:
  case ARM::FK_CRYPTO_NEON_FP_ARMV8:
    setAttributeItem(ARMBuildAttrs::FP_arch,
                     ARMBuildAttrs::AllowFPARMv8A,
                     /* OverwriteExisting= */ false);
    // 'Advanced_SIMD_arch' must be emitted not here, but within
    // ARMAsmPrinter::emitAttributes(), depending on hasV8Ops() and hasV8_1a()
    break;

  case ARM::FK_SOFTVFP:
  case ARM::FK_NONE:
    break;

  default:
    report_fatal_error("Unknown FPU: " + Twine(FPU));
    break;
  }
}
size_t ARMTargetELFStreamer::calculateContentSize() const {
  size_t Result = 0;
  for (size_t i = 0; i < Contents.size(); ++i) {
    AttributeItem item = Contents[i];
    switch (item.Type) {
    case AttributeItem::HiddenAttribute:
      break;
    case AttributeItem::NumericAttribute:
      Result += getULEB128Size(item.Tag);
      Result += getULEB128Size(item.IntValue);
      break;
    case AttributeItem::TextAttribute:
      Result += getULEB128Size(item.Tag);
      Result += item.StringValue.size() + 1; // string + '\0'
      break;
    case AttributeItem::NumericAndTextAttributes:
      Result += getULEB128Size(item.Tag);
      Result += getULEB128Size(item.IntValue);
      Result += item.StringValue.size() + 1; // string + '\0';
      break;
    }
  }
  return Result;
}
void ARMTargetELFStreamer::finishAttributeSection() {
  // <format-version>
  // [ <section-length> "vendor-name"
  // [ <file-tag> <size> <attribute>*
  //   | <section-tag> <size> <section-number>* 0 <attribute>*
  //   | <symbol-tag> <size> <symbol-number>* 0 <attribute>*
  //   ]+
  // ]*

  if (FPU != ARM::FK_INVALID)
    emitFPUDefaultAttributes();

  if (Arch != ARM::AK_INVALID)
    emitArchDefaultAttributes();

  if (Contents.empty())
    return;

  std::sort(Contents.begin(), Contents.end(), AttributeItem::LessTag);

  ARMELFStreamer &Streamer = getStreamer();

  // Switch to .ARM.attributes section
  if (AttributeSection) {
    Streamer.SwitchSection(AttributeSection);
  } else {
    AttributeSection = Streamer.getContext().getELFSection(
        ".ARM.attributes", ELF::SHT_ARM_ATTRIBUTES, 0);
    Streamer.SwitchSection(AttributeSection);

    // Format version
    Streamer.EmitIntValue(0x41, 1);
  }

  // Vendor size + Vendor name + '\0'
  const size_t VendorHeaderSize = 4 + CurrentVendor.size() + 1;

  // Tag + Tag Size
  const size_t TagHeaderSize = 1 + 4;

  const size_t ContentsSize = calculateContentSize();

  Streamer.EmitIntValue(VendorHeaderSize + TagHeaderSize + ContentsSize, 4);
  Streamer.EmitBytes(CurrentVendor);
  Streamer.EmitIntValue(0, 1); // '\0'

  Streamer.EmitIntValue(ARMBuildAttrs::File, 1);
  Streamer.EmitIntValue(TagHeaderSize + ContentsSize, 4);

  // Size should have been accounted for already, now
  // emit each field as its type (ULEB or String)
  for (size_t i = 0; i < Contents.size(); ++i) {
    AttributeItem item = Contents[i];
    Streamer.EmitULEB128IntValue(item.Tag);
    switch (item.Type) {
    default: llvm_unreachable("Invalid attribute type");
    case AttributeItem::NumericAttribute:
      Streamer.EmitULEB128IntValue(item.IntValue);
      break;
    case AttributeItem::TextAttribute:
      Streamer.EmitBytes(item.StringValue);
      Streamer.EmitIntValue(0, 1); // '\0'
      break;
    case AttributeItem::NumericAndTextAttributes:
      Streamer.EmitULEB128IntValue(item.IntValue);
      Streamer.EmitBytes(item.StringValue);
      Streamer.EmitIntValue(0, 1); // '\0'
      break;
    }
  }

  Contents.clear();
  FPU = ARM::FK_INVALID;
}

void ARMTargetELFStreamer::emitLabel(MCSymbol *Symbol) {
  ARMELFStreamer &Streamer = getStreamer();
  if (!Streamer.IsThumb)
    return;

  Streamer.getAssembler().registerSymbol(*Symbol);
  unsigned Type = cast<MCSymbolELF>(Symbol)->getType();
  if (Type == ELF::STT_FUNC || Type == ELF::STT_GNU_IFUNC)
    Streamer.EmitThumbFunc(Symbol);
}

void
ARMTargetELFStreamer::AnnotateTLSDescriptorSequence(const MCSymbolRefExpr *S) {
  getStreamer().EmitFixup(S, FK_Data_4);
}

void ARMTargetELFStreamer::emitThumbSet(MCSymbol *Symbol, const MCExpr *Value) {
  if (const MCSymbolRefExpr *SRE = dyn_cast<MCSymbolRefExpr>(Value)) {
    const MCSymbol &Sym = SRE->getSymbol();
    if (!Sym.isDefined()) {
      getStreamer().EmitAssignment(Symbol, Value);
      return;
    }
  }

  getStreamer().EmitThumbFunc(Symbol);
  getStreamer().EmitAssignment(Symbol, Value);
}

void ARMTargetELFStreamer::emitInst(uint32_t Inst, char Suffix) {
  getStreamer().emitInst(Inst, Suffix);
}

void ARMELFStreamer::FinishImpl() {
  MCTargetStreamer &TS = *getTargetStreamer();
  ARMTargetStreamer &ATS = static_cast<ARMTargetStreamer &>(TS);
  ATS.finishAttributeSection();

  MCELFStreamer::FinishImpl();
}

inline void ARMELFStreamer::SwitchToEHSection(const char *Prefix,
                                              unsigned Type,
                                              unsigned Flags,
                                              SectionKind Kind,
                                              const MCSymbol &Fn) {
  const MCSectionELF &FnSection =
    static_cast<const MCSectionELF &>(Fn.getSection());

  // Create the name for new section
  StringRef FnSecName(FnSection.getSectionName());
  SmallString<128> EHSecName(Prefix);
  if (FnSecName != ".text") {
    EHSecName += FnSecName;
  }

  // Get .ARM.extab or .ARM.exidx section
  const MCSymbolELF *Group = FnSection.getGroup();
  if (Group)
    Flags |= ELF::SHF_GROUP;
  MCSectionELF *EHSection =
      getContext().getELFSection(EHSecName, Type, Flags, 0, Group,
                                 FnSection.getUniqueID(), nullptr, &FnSection);

  assert(EHSection && "Failed to get the required EH section");

  // Switch to .ARM.extab or .ARM.exidx section
  SwitchSection(EHSection);
  EmitCodeAlignment(4);
}

inline void ARMELFStreamer::SwitchToExTabSection(const MCSymbol &FnStart) {
  SwitchToEHSection(".ARM.extab",
                    ELF::SHT_PROGBITS,
                    ELF::SHF_ALLOC,
                    SectionKind::getDataRel(),
                    FnStart);
}

inline void ARMELFStreamer::SwitchToExIdxSection(const MCSymbol &FnStart) {
  SwitchToEHSection(".ARM.exidx",
                    ELF::SHT_ARM_EXIDX,
                    ELF::SHF_ALLOC | ELF::SHF_LINK_ORDER,
                    SectionKind::getDataRel(),
                    FnStart);
}
void ARMELFStreamer::EmitFixup(const MCExpr *Expr, MCFixupKind Kind) {
  MCDataFragment *Frag = getOrCreateDataFragment();
  Frag->getFixups().push_back(MCFixup::create(Frag->getContents().size(), Expr,
                                              Kind));
}

void ARMELFStreamer::Reset() {
  ExTab = nullptr;
  FnStart = nullptr;
  Personality = nullptr;
  PersonalityIndex = ARM::EHABI::NUM_PERSONALITY_INDEX;
  FPReg = ARM::SP;
  FPOffset = 0;
  SPOffset = 0;
  PendingOffset = 0;
  UsedFP = false;
  CantUnwind = false;

  Opcodes.clear();
  UnwindOpAsm.Reset();
}

void ARMELFStreamer::emitFnStart() {
  assert(FnStart == nullptr);
  FnStart = getContext().createTempSymbol();
  EmitLabel(FnStart);
}

void ARMELFStreamer::emitFnEnd() {
  assert(FnStart && ".fnstart must precedes .fnend");

  // Emit unwind opcodes if there is no .handlerdata directive
  if (!ExTab && !CantUnwind)
    FlushUnwindOpcodes(true);

  // Emit the exception index table entry
  SwitchToExIdxSection(*FnStart);

  if (PersonalityIndex < ARM::EHABI::NUM_PERSONALITY_INDEX)
    EmitPersonalityFixup(GetAEABIUnwindPersonalityName(PersonalityIndex));

  const MCSymbolRefExpr *FnStartRef =
    MCSymbolRefExpr::create(FnStart,
                            MCSymbolRefExpr::VK_ARM_PREL31,
                            getContext());

  EmitValue(FnStartRef, 4);

  if (CantUnwind) {
    EmitIntValue(ARM::EHABI::EXIDX_CANTUNWIND, 4);
  } else if (ExTab) {
    // Emit a reference to the unwind opcodes in the ".ARM.extab" section.
    const MCSymbolRefExpr *ExTabEntryRef =
      MCSymbolRefExpr::create(ExTab,
                              MCSymbolRefExpr::VK_ARM_PREL31,
                              getContext());
    EmitValue(ExTabEntryRef, 4);
  } else {
    // For the __aeabi_unwind_cpp_pr0, we have to emit the unwind opcodes in
    // the second word of exception index table entry.  The size of the unwind
    // opcodes should always be 4 bytes.
    assert(PersonalityIndex == ARM::EHABI::AEABI_UNWIND_CPP_PR0 &&
           "Compact model must use __aeabi_unwind_cpp_pr0 as personality");
    assert(Opcodes.size() == 4u &&
           "Unwind opcode size for __aeabi_unwind_cpp_pr0 must be equal to 4");
    uint64_t Intval = Opcodes[0] |
                      Opcodes[1] << 8 |
                      Opcodes[2] << 16 |
                      Opcodes[3] << 24;
    EmitIntValue(Intval, Opcodes.size());
  }

  // Switch to the section containing FnStart
  SwitchSection(&FnStart->getSection());

  // Clean exception handling frame information
  Reset();
}

void ARMELFStreamer::emitCantUnwind() { CantUnwind = true; }

// Add the R_ARM_NONE fixup at the same position
void ARMELFStreamer::EmitPersonalityFixup(StringRef Name) {
  const MCSymbol *PersonalitySym = getContext().getOrCreateSymbol(Name);

  const MCSymbolRefExpr *PersonalityRef = MCSymbolRefExpr::create(
      PersonalitySym, MCSymbolRefExpr::VK_ARM_NONE, getContext());

  visitUsedExpr(*PersonalityRef);
  MCDataFragment *DF = getOrCreateDataFragment();
  DF->getFixups().push_back(MCFixup::create(DF->getContents().size(),
                                            PersonalityRef,
                                            MCFixup::getKindForSize(4, false)));
}

void ARMELFStreamer::FlushPendingOffset() {
  if (PendingOffset != 0) {
    UnwindOpAsm.EmitSPOffset(-PendingOffset);
    PendingOffset = 0;
  }
}

void ARMELFStreamer::FlushUnwindOpcodes(bool NoHandlerData) {
  // Emit the unwind opcode to restore $sp.
  if (UsedFP) {
    const MCRegisterInfo *MRI = getContext().getRegisterInfo();
    int64_t LastRegSaveSPOffset = SPOffset - PendingOffset;
    UnwindOpAsm.EmitSPOffset(LastRegSaveSPOffset - FPOffset);
    UnwindOpAsm.EmitSetSP(MRI->getEncodingValue(FPReg));
  } else {
    FlushPendingOffset();
  }

  // Finalize the unwind opcode sequence
  UnwindOpAsm.Finalize(PersonalityIndex, Opcodes);

  // For compact model 0, we have to emit the unwind opcodes in the .ARM.exidx
  // section.  Thus, we don't have to create an entry in the .ARM.extab
  // section.
  if (NoHandlerData && PersonalityIndex == ARM::EHABI::AEABI_UNWIND_CPP_PR0)
    return;

  // Switch to .ARM.extab section.
  SwitchToExTabSection(*FnStart);

  // Create .ARM.extab label for offset in .ARM.exidx
  assert(!ExTab);
  ExTab = getContext().createTempSymbol();
  EmitLabel(ExTab);

  // Emit personality
  if (Personality) {
    const MCSymbolRefExpr *PersonalityRef =
      MCSymbolRefExpr::create(Personality,
                              MCSymbolRefExpr::VK_ARM_PREL31,
                              getContext());

    EmitValue(PersonalityRef, 4);
  }

  // Emit unwind opcodes
  assert((Opcodes.size() % 4) == 0 &&
         "Unwind opcode size for __aeabi_cpp_unwind_pr0 must be multiple of 4");
  for (unsigned I = 0; I != Opcodes.size(); I += 4) {
    uint64_t Intval = Opcodes[I] |
                      Opcodes[I + 1] << 8 |
                      Opcodes[I + 2] << 16 |
                      Opcodes[I + 3] << 24;
    EmitIntValue(Intval, 4);
  }

  // According to ARM EHABI section 9.2, if the __aeabi_unwind_cpp_pr1() or
  // __aeabi_unwind_cpp_pr2() is used, then the handler data must be emitted
  // after the unwind opcodes.  The handler data consists of several 32-bit
  // words, and should be terminated by zero.
  //
  // In case that the .handlerdata directive is not specified by the
  // programmer, we should emit zero to terminate the handler data.
  if (NoHandlerData && !Personality)
    EmitIntValue(0, 4);
}

void ARMELFStreamer::emitHandlerData() { FlushUnwindOpcodes(false); }

void ARMELFStreamer::emitPersonality(const MCSymbol *Per) {
  Personality = Per;
  UnwindOpAsm.setPersonality(Per);
}

void ARMELFStreamer::emitPersonalityIndex(unsigned Index) {
  assert(Index < ARM::EHABI::NUM_PERSONALITY_INDEX && "invalid index");
  PersonalityIndex = Index;
}

void ARMELFStreamer::emitSetFP(unsigned NewFPReg, unsigned NewSPReg,
                               int64_t Offset) {
  assert((NewSPReg == ARM::SP || NewSPReg == FPReg) &&
         "the operand of .setfp directive should be either $sp or $fp");

  UsedFP = true;
  FPReg = NewFPReg;

  if (NewSPReg == ARM::SP)
    FPOffset = SPOffset + Offset;
  else
    FPOffset += Offset;
}

void ARMELFStreamer::emitMovSP(unsigned Reg, int64_t Offset) {
  assert((Reg != ARM::SP && Reg != ARM::PC) &&
         "the operand of .movsp cannot be either sp or pc");
  assert(FPReg == ARM::SP && "current FP must be SP");

  FlushPendingOffset();

  FPReg = Reg;
  FPOffset = SPOffset + Offset;

  const MCRegisterInfo *MRI = getContext().getRegisterInfo();
  UnwindOpAsm.EmitSetSP(MRI->getEncodingValue(FPReg));
}

void ARMELFStreamer::emitPad(int64_t Offset) {
  // Track the change of the $sp offset
  SPOffset -= Offset;

  // To squash multiple .pad directives, we should delay the unwind opcode
  // until the .save, .vsave, .handlerdata, or .fnend directives.
  PendingOffset -= Offset;
}

void ARMELFStreamer::emitRegSave(const SmallVectorImpl<unsigned> &RegList,
                                 bool IsVector) {
  // Collect the registers in the register list
  unsigned Count = 0;
  uint32_t Mask = 0;
  const MCRegisterInfo *MRI = getContext().getRegisterInfo();
  for (size_t i = 0; i < RegList.size(); ++i) {
    unsigned Reg = MRI->getEncodingValue(RegList[i]);
    assert(Reg < (IsVector ? 32U : 16U) && "Register out of range");
    unsigned Bit = (1u << Reg);
    if ((Mask & Bit) == 0) {
      Mask |= Bit;
      ++Count;
    }
  }

  // Track the change the $sp offset: For the .save directive, the
  // corresponding push instruction will decrease the $sp by (4 * Count).
  // For the .vsave directive, the corresponding vpush instruction will
  // decrease $sp by (8 * Count).
  SPOffset -= Count * (IsVector ? 8 : 4);

  // Emit the opcode
  FlushPendingOffset();
  if (IsVector)
    UnwindOpAsm.EmitVFPRegSave(Mask);
  else
    UnwindOpAsm.EmitRegSave(Mask);
}

void ARMELFStreamer::emitUnwindRaw(int64_t Offset,
                                   const SmallVectorImpl<uint8_t> &Opcodes) {
  FlushPendingOffset();
  SPOffset = SPOffset - Offset;
  UnwindOpAsm.EmitRaw(Opcodes);
}

namespace llvm {

MCTargetStreamer *createARMTargetAsmStreamer(MCStreamer &S,
                                             formatted_raw_ostream &OS,
                                             MCInstPrinter *InstPrint,
                                             bool isVerboseAsm) {
  return new ARMTargetAsmStreamer(S, OS, *InstPrint, isVerboseAsm);
}

MCTargetStreamer *createARMNullTargetStreamer(MCStreamer &S) {
  return new ARMTargetStreamer(S);
}

MCTargetStreamer *createARMObjectTargetStreamer(MCStreamer &S,
                                                const MCSubtargetInfo &STI) {
  const Triple &TT = STI.getTargetTriple();
  if (TT.isOSBinFormatELF())
    return new ARMTargetELFStreamer(S);
  return new ARMTargetStreamer(S);
}

MCELFStreamer *createARMELFStreamer(MCContext &Context, MCAsmBackend &TAB,
                                    raw_pwrite_stream &OS,
                                    MCCodeEmitter *Emitter, bool RelaxAll,
                                    bool IsThumb) {
    ARMELFStreamer *S = new ARMELFStreamer(Context, TAB, OS, Emitter, IsThumb);
    // FIXME: This should eventually end up somewhere else where more
    // intelligent flag decisions can be made. For now we are just maintaining
    // the status quo for ARM and setting EF_ARM_EABI_VER5 as the default.
    S->getAssembler().setELFHeaderEFlags(ELF::EF_ARM_EABI_VER5);

    if (RelaxAll)
      S->getAssembler().setRelaxAll(true);
    return S;
  }

}


