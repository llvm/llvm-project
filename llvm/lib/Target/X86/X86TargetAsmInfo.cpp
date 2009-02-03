//===-- X86TargetAsmInfo.cpp - X86 asm properties ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the X86TargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "X86TargetAsmInfo.h"
#include "X86TargetMachine.h"
#include "X86Subtarget.h"
#include "llvm/DerivedTypes.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Dwarf.h"

using namespace llvm;
using namespace llvm::dwarf;

const char *const llvm::x86_asm_table[] = {
  "{si}", "S",
  "{di}", "D",
  "{ax}", "a",
  "{cx}", "c",
  "{memory}", "memory",
  "{flags}", "",
  "{dirflag}", "",
  "{fpsr}", "",
  "{cc}", "cc",
  0,0};

X86DarwinTargetAsmInfo::X86DarwinTargetAsmInfo(const X86TargetMachine &TM):
  X86TargetAsmInfo<DarwinTargetAsmInfo>(TM) {
  const X86Subtarget* Subtarget = &TM.getSubtarget<X86Subtarget>();
  bool is64Bit = Subtarget->is64Bit();

  AlignmentIsInBytes = false;
  TextAlignFillValue = 0x90;
  GlobalPrefix = "_";
  if (!is64Bit)
    Data64bitsDirective = 0;       // we can't emit a 64-bit unit
  ZeroDirective = "\t.space\t";  // ".space N" emits N zeros.
  PrivateGlobalPrefix = "L";     // Marker for constant pool idxs
  LessPrivateGlobalPrefix = "l";  // Marker for some ObjC metadata
  BSSSection = 0;                       // no BSS section.
  ZeroFillDirective = "\t.zerofill\t";  // Uses .zerofill
  if (TM.getRelocationModel() != Reloc::Static)
    ConstantPoolSection = "\t.const_data";
  else
    ConstantPoolSection = "\t.const\n";
  JumpTableDataSection = "\t.const\n";
  CStringSection = "\t.cstring";
  // FIXME: Why don't always use this section?
  if (is64Bit) {
    SixteenByteConstantSection = getUnnamedSection("\t.literal16\n",
                                                   SectionFlags::Mergeable);
  }
  LCOMMDirective = "\t.lcomm\t";
  SwitchToSectionDirective = "\t.section ";
  StringConstantPrefix = "\1LC";
  // Leopard and above support aligned common symbols.
  COMMDirectiveTakesAlignment = (Subtarget->getDarwinVers() >= 9);
  HasDotTypeDotSizeDirective = false;
  HasSingleParameterDotFile = false;
  NonLocalEHFrameLabel = true;
  if (TM.getRelocationModel() == Reloc::Static) {
    StaticCtorsSection = ".constructor";
    StaticDtorsSection = ".destructor";
  } else {
    StaticCtorsSection = ".mod_init_func";
    StaticDtorsSection = ".mod_term_func";
  }
  if (is64Bit) {
    PersonalityPrefix = "";
    PersonalitySuffix = "+4@GOTPCREL";
  } else {
    PersonalityPrefix = "L";
    PersonalitySuffix = "$non_lazy_ptr";
  }
  NeedsIndirectEncoding = true;
  InlineAsmStart = "## InlineAsm Start";
  InlineAsmEnd = "## InlineAsm End";
  CommentString = "##";
  SetDirective = "\t.set";
  PCSymbol = ".";
  UsedDirective = "\t.no_dead_strip\t";
  WeakDefDirective = "\t.weak_definition ";
  WeakRefDirective = "\t.weak_reference ";
  HiddenDirective = "\t.private_extern ";
  ProtectedDirective = "\t.globl\t";

  // In non-PIC modes, emit a special label before jump tables so that the
  // linker can perform more accurate dead code stripping.
  if (TM.getRelocationModel() != Reloc::PIC_) {
    // Emit a local label that is preserved until the linker runs.
    JumpTableSpecialLabelPrefix = "l";
  }

  SupportsDebugInformation = true;
  NeedsSet = true;
  DwarfAbbrevSection = ".section __DWARF,__debug_abbrev,regular,debug";
  DwarfInfoSection = ".section __DWARF,__debug_info,regular,debug";
  DwarfLineSection = ".section __DWARF,__debug_line,regular,debug";
  DwarfFrameSection = ".section __DWARF,__debug_frame,regular,debug";
  DwarfPubNamesSection = ".section __DWARF,__debug_pubnames,regular,debug";
  DwarfPubTypesSection = ".section __DWARF,__debug_pubtypes,regular,debug";
  DwarfStrSection = ".section __DWARF,__debug_str,regular,debug";
  DwarfLocSection = ".section __DWARF,__debug_loc,regular,debug";
  DwarfARangesSection = ".section __DWARF,__debug_aranges,regular,debug";
  DwarfRangesSection = ".section __DWARF,__debug_ranges,regular,debug";
  DwarfMacInfoSection = ".section __DWARF,__debug_macinfo,regular,debug";

  // Exceptions handling
  SupportsExceptionHandling = true;
  GlobalEHDirective = "\t.globl\t";
  SupportsWeakOmittedEHFrame = false;
  AbsoluteEHSectionOffsets = false;
  DwarfEHFrameSection =
  ".section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support";
  DwarfExceptionSection = ".section __DATA,__gcc_except_tab";
}

unsigned
X86DarwinTargetAsmInfo::PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                              bool Global) const {
  if (Reason == DwarfEncoding::Functions && Global)
    return (DW_EH_PE_pcrel | DW_EH_PE_indirect | DW_EH_PE_sdata4);
  else if (Reason == DwarfEncoding::CodeLabels || !Global)
    return DW_EH_PE_pcrel;
  else
    return DW_EH_PE_absptr;
}

const char *
X86DarwinTargetAsmInfo::getEHGlobalPrefix() const
{
  const X86Subtarget* Subtarget = &TM.getSubtarget<X86Subtarget>();
  if (Subtarget->getDarwinVers() > 9)
    return PrivateGlobalPrefix;
  else
    return "";
}

X86ELFTargetAsmInfo::X86ELFTargetAsmInfo(const X86TargetMachine &TM):
  X86TargetAsmInfo<ELFTargetAsmInfo>(TM) {

  CStringSection = ".rodata.str";
  PrivateGlobalPrefix = ".L";
  WeakRefDirective = "\t.weak\t";
  SetDirective = "\t.set\t";
  PCSymbol = ".";

  // Set up DWARF directives
  HasLEB128 = true;  // Target asm supports leb128 directives (little-endian)

  // Debug Information
  AbsoluteDebugSectionOffsets = true;
  SupportsDebugInformation = true;
  DwarfAbbrevSection =  "\t.section\t.debug_abbrev,\"\",@progbits";
  DwarfInfoSection =    "\t.section\t.debug_info,\"\",@progbits";
  DwarfLineSection =    "\t.section\t.debug_line,\"\",@progbits";
  DwarfFrameSection =   "\t.section\t.debug_frame,\"\",@progbits";
  DwarfPubNamesSection ="\t.section\t.debug_pubnames,\"\",@progbits";
  DwarfPubTypesSection ="\t.section\t.debug_pubtypes,\"\",@progbits";
  DwarfStrSection =     "\t.section\t.debug_str,\"\",@progbits";
  DwarfLocSection =     "\t.section\t.debug_loc,\"\",@progbits";
  DwarfARangesSection = "\t.section\t.debug_aranges,\"\",@progbits";
  DwarfRangesSection =  "\t.section\t.debug_ranges,\"\",@progbits";
  DwarfMacInfoSection = "\t.section\t.debug_macinfo,\"\",@progbits";

  // Exceptions handling
  SupportsExceptionHandling = true;
  AbsoluteEHSectionOffsets = false;
  DwarfEHFrameSection = "\t.section\t.eh_frame,\"aw\",@progbits";
  DwarfExceptionSection = "\t.section\t.gcc_except_table,\"a\",@progbits";

  // On Linux we must declare when we can use a non-executable stack.
  if (TM.getSubtarget<X86Subtarget>().isLinux())
    NonexecutableStackDirective = "\t.section\t.note.GNU-stack,\"\",@progbits";
}

unsigned
X86ELFTargetAsmInfo::PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                           bool Global) const {
  CodeModel::Model CM = TM.getCodeModel();
  bool is64Bit = TM.getSubtarget<X86Subtarget>().is64Bit();

  if (TM.getRelocationModel() == Reloc::PIC_) {
    unsigned Format = 0;

    if (!is64Bit)
      // 32 bit targets always encode pointers as 4 bytes
      Format = DW_EH_PE_sdata4;
    else {
      // 64 bit targets encode pointers in 4 bytes iff:
      // - code model is small OR
      // - code model is medium and we're emitting externally visible symbols
      //   or any code symbols
      if (CM == CodeModel::Small ||
          (CM == CodeModel::Medium && (Global ||
                                       Reason != DwarfEncoding::Data)))
        Format = DW_EH_PE_sdata4;
      else
        Format = DW_EH_PE_sdata8;
    }

    if (Global)
      Format |= DW_EH_PE_indirect;

    return (Format | DW_EH_PE_pcrel);
  } else {
    if (is64Bit &&
        (CM == CodeModel::Small ||
         (CM == CodeModel::Medium && Reason != DwarfEncoding::Data)))
      return DW_EH_PE_udata4;
    else
      return DW_EH_PE_absptr;
  }
}

X86COFFTargetAsmInfo::X86COFFTargetAsmInfo(const X86TargetMachine &TM):
  X86GenericTargetAsmInfo(TM) {

  GlobalPrefix = "_";
  LCOMMDirective = "\t.lcomm\t";
  COMMDirectiveTakesAlignment = false;
  HasDotTypeDotSizeDirective = false;
  HasSingleParameterDotFile = false;
  StaticCtorsSection = "\t.section .ctors,\"aw\"";
  StaticDtorsSection = "\t.section .dtors,\"aw\"";
  HiddenDirective = NULL;
  PrivateGlobalPrefix = "L";  // Prefix for private global symbols
  WeakRefDirective = "\t.weak\t";
  SetDirective = "\t.set\t";

  // Set up DWARF directives
  HasLEB128 = true;  // Target asm supports leb128 directives (little-endian)
  AbsoluteDebugSectionOffsets = true;
  AbsoluteEHSectionOffsets = false;
  SupportsDebugInformation = true;
  DwarfSectionOffsetDirective = "\t.secrel32\t";
  DwarfAbbrevSection =  "\t.section\t.debug_abbrev,\"dr\"";
  DwarfInfoSection =    "\t.section\t.debug_info,\"dr\"";
  DwarfLineSection =    "\t.section\t.debug_line,\"dr\"";
  DwarfFrameSection =   "\t.section\t.debug_frame,\"dr\"";
  DwarfPubNamesSection ="\t.section\t.debug_pubnames,\"dr\"";
  DwarfPubTypesSection ="\t.section\t.debug_pubtypes,\"dr\"";
  DwarfStrSection =     "\t.section\t.debug_str,\"dr\"";
  DwarfLocSection =     "\t.section\t.debug_loc,\"dr\"";
  DwarfARangesSection = "\t.section\t.debug_aranges,\"dr\"";
  DwarfRangesSection =  "\t.section\t.debug_ranges,\"dr\"";
  DwarfMacInfoSection = "\t.section\t.debug_macinfo,\"dr\"";
}

unsigned
X86COFFTargetAsmInfo::PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                            bool Global) const {
  CodeModel::Model CM = TM.getCodeModel();
  bool is64Bit = TM.getSubtarget<X86Subtarget>().is64Bit();

  if (TM.getRelocationModel() == Reloc::PIC_) {
    unsigned Format = 0;

    if (!is64Bit)
      // 32 bit targets always encode pointers as 4 bytes
      Format = DW_EH_PE_sdata4;
    else {
      // 64 bit targets encode pointers in 4 bytes iff:
      // - code model is small OR
      // - code model is medium and we're emitting externally visible symbols
      //   or any code symbols
      if (CM == CodeModel::Small ||
          (CM == CodeModel::Medium && (Global ||
                                       Reason != DwarfEncoding::Data)))
        Format = DW_EH_PE_sdata4;
      else
        Format = DW_EH_PE_sdata8;
    }

    if (Global)
      Format |= DW_EH_PE_indirect;

    return (Format | DW_EH_PE_pcrel);
  } else {
    if (is64Bit &&
        (CM == CodeModel::Small ||
         (CM == CodeModel::Medium && Reason != DwarfEncoding::Data)))
      return DW_EH_PE_udata4;
    else
      return DW_EH_PE_absptr;
  }
}

std::string
X86COFFTargetAsmInfo::UniqueSectionForGlobal(const GlobalValue* GV,
                                             SectionKind::Kind kind) const {
  switch (kind) {
   case SectionKind::Text:
    return ".text$linkonce" + GV->getName();
   case SectionKind::Data:
   case SectionKind::BSS:
   case SectionKind::ThreadData:
   case SectionKind::ThreadBSS:
    return ".data$linkonce" + GV->getName();
   case SectionKind::ROData:
   case SectionKind::RODataMergeConst:
   case SectionKind::RODataMergeStr:
    return ".rdata$linkonce" + GV->getName();
   default:
    assert(0 && "Unknown section kind");
  }
  return NULL;
}

std::string X86COFFTargetAsmInfo::printSectionFlags(unsigned flags) const {
  std::string Flags = ",\"";

  if (flags & SectionFlags::Code)
    Flags += 'x';
  if (flags & SectionFlags::Writeable)
    Flags += 'w';

  Flags += "\"";

  return Flags;
}

X86WinTargetAsmInfo::X86WinTargetAsmInfo(const X86TargetMachine &TM):
  X86GenericTargetAsmInfo(TM) {
  GlobalPrefix = "_";
  CommentString = ";";

  PrivateGlobalPrefix = "$";
  AlignDirective = "\talign\t";
  ZeroDirective = "\tdb\t";
  ZeroDirectiveSuffix = " dup(0)";
  AsciiDirective = "\tdb\t";
  AscizDirective = 0;
  Data8bitsDirective = "\tdb\t";
  Data16bitsDirective = "\tdw\t";
  Data32bitsDirective = "\tdd\t";
  Data64bitsDirective = "\tdq\t";
  HasDotTypeDotSizeDirective = false;
  HasSingleParameterDotFile = false;

  TextSection = getUnnamedSection("_text", SectionFlags::Code);
  DataSection = getUnnamedSection("_data", SectionFlags::Writeable);

  JumpTableDataSection = NULL;
  SwitchToSectionDirective = "";
  TextSectionStartSuffix = "\tsegment 'CODE'";
  DataSectionStartSuffix = "\tsegment 'DATA'";
  SectionEndDirectiveSuffix = "\tends\n";
}

template <class BaseTAI>
bool X86TargetAsmInfo<BaseTAI>::LowerToBSwap(CallInst *CI) const {
  // FIXME: this should verify that we are targetting a 486 or better.  If not,
  // we will turn this bswap into something that will be lowered to logical ops
  // instead of emitting the bswap asm.  For now, we don't support 486 or lower
  // so don't worry about this.

  // Verify this is a simple bswap.
  if (CI->getNumOperands() != 2 ||
      CI->getType() != CI->getOperand(1)->getType() ||
      !CI->getType()->isInteger())
    return false;

  const IntegerType *Ty = dyn_cast<IntegerType>(CI->getType());
  if (!Ty || Ty->getBitWidth() % 16 != 0)
    return false;

  // Okay, we can do this xform, do so now.
  const Type *Tys[] = { Ty };
  Module *M = CI->getParent()->getParent()->getParent();
  Constant *Int = Intrinsic::getDeclaration(M, Intrinsic::bswap, Tys, 1);

  Value *Op = CI->getOperand(1);
  Op = CallInst::Create(Int, Op, CI->getName(), CI);

  CI->replaceAllUsesWith(Op);
  CI->eraseFromParent();
  return true;
}

template <class BaseTAI>
bool X86TargetAsmInfo<BaseTAI>::ExpandInlineAsm(CallInst *CI) const {
  InlineAsm *IA = cast<InlineAsm>(CI->getCalledValue());
  std::vector<InlineAsm::ConstraintInfo> Constraints = IA->ParseConstraints();

  std::string AsmStr = IA->getAsmString();

  // TODO: should remove alternatives from the asmstring: "foo {a|b}" -> "foo a"
  std::vector<std::string> AsmPieces;
  SplitString(AsmStr, AsmPieces, "\n");  // ; as separator?

  switch (AsmPieces.size()) {
  default: return false;
  case 1:
    AsmStr = AsmPieces[0];
    AsmPieces.clear();
    SplitString(AsmStr, AsmPieces, " \t");  // Split with whitespace.

    // bswap $0
    if (AsmPieces.size() == 2 &&
        AsmPieces[0] == "bswap" && (AsmPieces[1] == "$0" ||
                                    AsmPieces[1] == "${0:q}")) {
      // No need to check constraints, nothing other than the equivalent of
      // "=r,0" would be valid here.
      return LowerToBSwap(CI);
    }
    // rorw $$8, ${0:w}  -->  llvm.bswap.i16
    if (CI->getType() == Type::Int16Ty &&
        AsmPieces.size() == 3 &&
        AsmPieces[0] == "rorw" &&
        AsmPieces[1] == "$$8," &&
        AsmPieces[2] == "${0:w}" &&
        IA->getConstraintString() == "=r,0,~{dirflag},~{fpsr},~{flags},~{cc}") {
      return LowerToBSwap(CI);
    }
    break;
  case 3:
    if (CI->getType() == Type::Int64Ty && Constraints.size() >= 2 &&
        Constraints[0].Codes.size() == 1 && Constraints[0].Codes[0] == "A" &&
        Constraints[1].Codes.size() == 1 && Constraints[1].Codes[0] == "0") {
      // bswap %eax / bswap %edx / xchgl %eax, %edx  -> llvm.bswap.i64
      std::vector<std::string> Words;
      SplitString(AsmPieces[0], Words, " \t");
      if (Words.size() == 2 && Words[0] == "bswap" && Words[1] == "%eax") {
        Words.clear();
        SplitString(AsmPieces[1], Words, " \t");
        if (Words.size() == 2 && Words[0] == "bswap" && Words[1] == "%edx") {
          Words.clear();
          SplitString(AsmPieces[2], Words, " \t,");
          if (Words.size() == 3 && Words[0] == "xchgl" && Words[1] == "%eax" &&
              Words[2] == "%edx") {
            return LowerToBSwap(CI);
          }
        }
      }
    }
    break;
  }
  return false;
}

// Instantiate default implementation.
TEMPLATE_INSTANTIATION(class X86TargetAsmInfo<TargetAsmInfo>);
