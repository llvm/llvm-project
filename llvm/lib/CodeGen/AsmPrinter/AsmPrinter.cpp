//===-- AsmPrinter.cpp - Common AsmPrinter code ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AsmPrinter class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/GCMetadataPrinter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/DwarfWriter.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include <cerrno>
using namespace llvm;

char AsmPrinter::ID = 0;
AsmPrinter::AsmPrinter(raw_ostream &o, TargetMachine &tm,
                       const TargetAsmInfo *T)
  : MachineFunctionPass(&ID), FunctionNumber(0), O(o),
    TM(tm), TAI(T), TRI(tm.getRegisterInfo()),
    IsInTextSection(false)
{}

AsmPrinter::~AsmPrinter() {
  for (gcp_iterator I = GCMetadataPrinters.begin(),
                    E = GCMetadataPrinters.end(); I != E; ++I)
    delete I->second;
}

/// SwitchToTextSection - Switch to the specified text section of the executable
/// if we are not already in it!
///
void AsmPrinter::SwitchToTextSection(const char *NewSection,
                                     const GlobalValue *GV) {
  std::string NS;
  if (GV && GV->hasSection())
    NS = TAI->getSwitchToSectionDirective() + GV->getSection();
  else
    NS = NewSection;
  
  // If we're already in this section, we're done.
  if (CurrentSection == NS) return;

  // Close the current section, if applicable.
  if (TAI->getSectionEndDirectiveSuffix() && !CurrentSection.empty())
    O << CurrentSection << TAI->getSectionEndDirectiveSuffix() << '\n';

  CurrentSection = NS;

  if (!CurrentSection.empty())
    O << CurrentSection << TAI->getTextSectionStartSuffix() << '\n';

  IsInTextSection = true;
}

/// SwitchToDataSection - Switch to the specified data section of the executable
/// if we are not already in it!
///
void AsmPrinter::SwitchToDataSection(const char *NewSection,
                                     const GlobalValue *GV) {
  std::string NS;
  if (GV && GV->hasSection())
    NS = TAI->getSwitchToSectionDirective() + GV->getSection();
  else
    NS = NewSection;
  
  // If we're already in this section, we're done.
  if (CurrentSection == NS) return;

  // Close the current section, if applicable.
  if (TAI->getSectionEndDirectiveSuffix() && !CurrentSection.empty())
    O << CurrentSection << TAI->getSectionEndDirectiveSuffix() << '\n';

  CurrentSection = NS;
  
  if (!CurrentSection.empty())
    O << CurrentSection << TAI->getDataSectionStartSuffix() << '\n';

  IsInTextSection = false;
}

/// SwitchToSection - Switch to the specified section of the executable if we
/// are not already in it!
void AsmPrinter::SwitchToSection(const Section* NS) {
  const std::string& NewSection = NS->getName();

  // If we're already in this section, we're done.
  if (CurrentSection == NewSection) return;

  // Close the current section, if applicable.
  if (TAI->getSectionEndDirectiveSuffix() && !CurrentSection.empty())
    O << CurrentSection << TAI->getSectionEndDirectiveSuffix() << '\n';

  // FIXME: Make CurrentSection a Section* in the future
  CurrentSection = NewSection;
  CurrentSection_ = NS;

  if (!CurrentSection.empty()) {
    // If section is named we need to switch into it via special '.section'
    // directive and also append funky flags. Otherwise - section name is just
    // some magic assembler directive.
    if (NS->isNamed())
      O << TAI->getSwitchToSectionDirective()
        << CurrentSection
        << TAI->getSectionFlags(NS->getFlags());
    else
      O << CurrentSection;
    O << TAI->getDataSectionStartSuffix() << '\n';
  }

  IsInTextSection = (NS->getFlags() & SectionFlags::Code);
}

void AsmPrinter::getAnalysisUsage(AnalysisUsage &AU) const {
  MachineFunctionPass::getAnalysisUsage(AU);
  AU.addRequired<GCModuleInfo>();
}

bool AsmPrinter::doInitialization(Module &M) {
  Mang = new Mangler(M, TAI->getGlobalPrefix(), TAI->getPrivateGlobalPrefix());
  
  GCModuleInfo *MI = getAnalysisIfAvailable<GCModuleInfo>();
  assert(MI && "AsmPrinter didn't require GCModuleInfo?");

  if (TAI->hasSingleParameterDotFile()) {
    /* Very minimal debug info. It is ignored if we emit actual
       debug info. If we don't, this at helps the user find where
       a function came from. */
    O << "\t.file\t\"" << M.getModuleIdentifier() << "\"\n";
  }

  for (GCModuleInfo::iterator I = MI->begin(), E = MI->end(); I != E; ++I)
    if (GCMetadataPrinter *MP = GetOrCreateGCPrinter(*I))
      MP->beginAssembly(O, *this, *TAI);
  
  if (!M.getModuleInlineAsm().empty())
    O << TAI->getCommentString() << " Start of file scope inline assembly\n"
      << M.getModuleInlineAsm()
      << '\n' << TAI->getCommentString()
      << " End of file scope inline assembly\n";

  SwitchToDataSection("");   // Reset back to no section.
  
  MachineModuleInfo *MMI = getAnalysisIfAvailable<MachineModuleInfo>();
  if (MMI) MMI->AnalyzeModule(M);
  DW = getAnalysisIfAvailable<DwarfWriter>();
  return false;
}

bool AsmPrinter::doFinalization(Module &M) {
  if (TAI->getWeakRefDirective()) {
    if (!ExtWeakSymbols.empty())
      SwitchToDataSection("");

    for (std::set<const GlobalValue*>::iterator i = ExtWeakSymbols.begin(),
         e = ExtWeakSymbols.end(); i != e; ++i) {
      const GlobalValue *GV = *i;
      std::string Name = Mang->getValueName(GV);
      O << TAI->getWeakRefDirective() << Name << '\n';
    }
  }

  if (TAI->getSetDirective()) {
    if (!M.alias_empty())
      SwitchToSection(TAI->getTextSection());

    O << '\n';
    for (Module::const_alias_iterator I = M.alias_begin(), E = M.alias_end();
         I!=E; ++I) {
      std::string Name = Mang->getValueName(I);
      std::string Target;

      const GlobalValue *GV = cast<GlobalValue>(I->getAliasedGlobal());
      Target = Mang->getValueName(GV);

      if (I->hasExternalLinkage() || !TAI->getWeakRefDirective())
        O << "\t.globl\t" << Name << '\n';
      else if (I->hasWeakLinkage())
        O << TAI->getWeakRefDirective() << Name << '\n';
      else if (!I->hasLocalLinkage())
        assert(0 && "Invalid alias linkage");

      printVisibility(Name, I->getVisibility());

      O << TAI->getSetDirective() << ' ' << Name << ", " << Target << '\n';

      // If the aliasee has external weak linkage it can be referenced only by
      // alias itself. In this case it can be not in ExtWeakSymbols list. Emit
      // weak reference in such case.
      if (GV->hasExternalWeakLinkage()) {
        if (TAI->getWeakRefDirective())
          O << TAI->getWeakRefDirective() << Target << '\n';
        else
          O << "\t.globl\t" << Target << '\n';
      }
    }
  }

  GCModuleInfo *MI = getAnalysisIfAvailable<GCModuleInfo>();
  assert(MI && "AsmPrinter didn't require GCModuleInfo?");
  for (GCModuleInfo::iterator I = MI->end(), E = MI->begin(); I != E; )
    if (GCMetadataPrinter *MP = GetOrCreateGCPrinter(*--I))
      MP->finishAssembly(O, *this, *TAI);

  // If we don't have any trampolines, then we don't require stack memory
  // to be executable. Some targets have a directive to declare this.
  Function* InitTrampolineIntrinsic = M.getFunction("llvm.init.trampoline");
  if (!InitTrampolineIntrinsic || InitTrampolineIntrinsic->use_empty())
    if (TAI->getNonexecutableStackDirective())
      O << TAI->getNonexecutableStackDirective() << '\n';

  delete Mang; Mang = 0;
  return false;
}

std::string AsmPrinter::getCurrentFunctionEHName(const MachineFunction *MF) {
  assert(MF && "No machine function?");
  std::string Name = MF->getFunction()->getName();
  if (Name.empty())
    Name = Mang->getValueName(MF->getFunction());
  return Mang->makeNameProper(TAI->getEHGlobalPrefix() +
                              Name + ".eh", TAI->getGlobalPrefix());
}

void AsmPrinter::SetupMachineFunction(MachineFunction &MF) {
  // What's my mangled name?
  CurrentFnName = Mang->getValueName(MF.getFunction());
  IncrementFunctionNumber();
}

/// EmitConstantPool - Print to the current output stream assembly
/// representations of the constants in the constant pool MCP. This is
/// used to print out constants which have been "spilled to memory" by
/// the code generator.
///
void AsmPrinter::EmitConstantPool(MachineConstantPool *MCP) {
  const std::vector<MachineConstantPoolEntry> &CP = MCP->getConstants();
  if (CP.empty()) return;

  // Calculate sections for constant pool entries. We collect entries to go into
  // the same section together to reduce amount of section switch statements.
  typedef
    std::multimap<const Section*,
                  std::pair<MachineConstantPoolEntry, unsigned> > CPMap;
  CPMap  CPs;
  SmallPtrSet<const Section*, 5> Sections;

  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    MachineConstantPoolEntry CPE = CP[i];
    const Section* S = TAI->SelectSectionForMachineConst(CPE.getType());
    CPs.insert(std::make_pair(S, std::make_pair(CPE, i)));
    Sections.insert(S);
  }

  // Now print stuff into the calculated sections.
  for (SmallPtrSet<const Section*, 5>::iterator IS = Sections.begin(),
         ES = Sections.end(); IS != ES; ++IS) {
    SwitchToSection(*IS);
    EmitAlignment(MCP->getConstantPoolAlignment());

    std::pair<CPMap::iterator, CPMap::iterator> II = CPs.equal_range(*IS);
    for (CPMap::iterator I = II.first, E = II.second; I != E; ++I) {
      CPMap::iterator J = next(I);
      MachineConstantPoolEntry Entry = I->second.first;
      unsigned index = I->second.second;

      O << TAI->getPrivateGlobalPrefix() << "CPI" << getFunctionNumber() << '_'
        << index << ":\t\t\t\t\t";
    // O << TAI->getCommentString() << ' ' << 
    //      WriteTypeSymbolic(O, CP[i].first.getType(), 0);
      O << '\n';
      if (Entry.isMachineConstantPoolEntry())
        EmitMachineConstantPoolValue(Entry.Val.MachineCPVal);
      else
        EmitGlobalConstant(Entry.Val.ConstVal);

      // Emit inter-object padding for alignment.
      if (J != E) {
        const Type *Ty = Entry.getType();
        unsigned EntSize = TM.getTargetData()->getTypePaddedSize(Ty);
        unsigned ValEnd = Entry.getOffset() + EntSize;
        EmitZeros(J->second.first.getOffset()-ValEnd);
      }
    }
  }
}

/// EmitJumpTableInfo - Print assembly representations of the jump tables used
/// by the current function to the current output stream.  
///
void AsmPrinter::EmitJumpTableInfo(MachineJumpTableInfo *MJTI,
                                   MachineFunction &MF) {
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  if (JT.empty()) return;

  bool IsPic = TM.getRelocationModel() == Reloc::PIC_;
  
  // Pick the directive to use to print the jump table entries, and switch to 
  // the appropriate section.
  TargetLowering *LoweringInfo = TM.getTargetLowering();

  const char* JumpTableDataSection = TAI->getJumpTableDataSection();
  const Function *F = MF.getFunction();
  unsigned SectionFlags = TAI->SectionFlagsForGlobal(F);
  if ((IsPic && !(LoweringInfo && LoweringInfo->usesGlobalOffsetTable())) ||
     !JumpTableDataSection ||
      SectionFlags & SectionFlags::Linkonce) {
    // In PIC mode, we need to emit the jump table to the same section as the
    // function body itself, otherwise the label differences won't make sense.
    // We should also do if the section name is NULL or function is declared in
    // discardable section.
    SwitchToSection(TAI->SectionForGlobal(F));
  } else {
    SwitchToDataSection(JumpTableDataSection);
  }
  
  EmitAlignment(Log2_32(MJTI->getAlignment()));
  
  for (unsigned i = 0, e = JT.size(); i != e; ++i) {
    const std::vector<MachineBasicBlock*> &JTBBs = JT[i].MBBs;
    
    // If this jump table was deleted, ignore it. 
    if (JTBBs.empty()) continue;

    // For PIC codegen, if possible we want to use the SetDirective to reduce
    // the number of relocations the assembler will generate for the jump table.
    // Set directives are all printed before the jump table itself.
    SmallPtrSet<MachineBasicBlock*, 16> EmittedSets;
    if (TAI->getSetDirective() && IsPic)
      for (unsigned ii = 0, ee = JTBBs.size(); ii != ee; ++ii)
        if (EmittedSets.insert(JTBBs[ii]))
          printPICJumpTableSetLabel(i, JTBBs[ii]);
    
    // On some targets (e.g. darwin) we want to emit two consequtive labels
    // before each jump table.  The first label is never referenced, but tells
    // the assembler and linker the extents of the jump table object.  The
    // second label is actually referenced by the code.
    if (const char *JTLabelPrefix = TAI->getJumpTableSpecialLabelPrefix())
      O << JTLabelPrefix << "JTI" << getFunctionNumber() << '_' << i << ":\n";
    
    O << TAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber() 
      << '_' << i << ":\n";
    
    for (unsigned ii = 0, ee = JTBBs.size(); ii != ee; ++ii) {
      printPICJumpTableEntry(MJTI, JTBBs[ii], i);
      O << '\n';
    }
  }
}

void AsmPrinter::printPICJumpTableEntry(const MachineJumpTableInfo *MJTI,
                                        const MachineBasicBlock *MBB,
                                        unsigned uid)  const {
  bool IsPic = TM.getRelocationModel() == Reloc::PIC_;
  
  // Use JumpTableDirective otherwise honor the entry size from the jump table
  // info.
  const char *JTEntryDirective = TAI->getJumpTableDirective();
  bool HadJTEntryDirective = JTEntryDirective != NULL;
  if (!HadJTEntryDirective) {
    JTEntryDirective = MJTI->getEntrySize() == 4 ?
      TAI->getData32bitsDirective() : TAI->getData64bitsDirective();
  }

  O << JTEntryDirective << ' ';

  // If we have emitted set directives for the jump table entries, print 
  // them rather than the entries themselves.  If we're emitting PIC, then
  // emit the table entries as differences between two text section labels.
  // If we're emitting non-PIC code, then emit the entries as direct
  // references to the target basic blocks.
  if (IsPic) {
    if (TAI->getSetDirective()) {
      O << TAI->getPrivateGlobalPrefix() << getFunctionNumber()
        << '_' << uid << "_set_" << MBB->getNumber();
    } else {
      printBasicBlockLabel(MBB, false, false, false);
      // If the arch uses custom Jump Table directives, don't calc relative to
      // JT
      if (!HadJTEntryDirective) 
        O << '-' << TAI->getPrivateGlobalPrefix() << "JTI"
          << getFunctionNumber() << '_' << uid;
    }
  } else {
    printBasicBlockLabel(MBB, false, false, false);
  }
}


/// EmitSpecialLLVMGlobal - Check to see if the specified global is a
/// special global used by LLVM.  If so, emit it and return true, otherwise
/// do nothing and return false.
bool AsmPrinter::EmitSpecialLLVMGlobal(const GlobalVariable *GV) {
  if (GV->getName() == "llvm.used") {
    if (TAI->getUsedDirective() != 0)    // No need to emit this at all.
      EmitLLVMUsedList(GV->getInitializer());
    return true;
  }

  // Ignore debug and non-emitted data.
  if (GV->getSection() == "llvm.metadata") return true;
  
  if (!GV->hasAppendingLinkage()) return false;

  assert(GV->hasInitializer() && "Not a special LLVM global!");
  
  const TargetData *TD = TM.getTargetData();
  unsigned Align = Log2_32(TD->getPointerPrefAlignment());
  if (GV->getName() == "llvm.global_ctors" && GV->use_empty()) {
    SwitchToDataSection(TAI->getStaticCtorsSection());
    EmitAlignment(Align, 0);
    EmitXXStructorList(GV->getInitializer());
    return true;
  } 
  
  if (GV->getName() == "llvm.global_dtors" && GV->use_empty()) {
    SwitchToDataSection(TAI->getStaticDtorsSection());
    EmitAlignment(Align, 0);
    EmitXXStructorList(GV->getInitializer());
    return true;
  }
  
  return false;
}

/// findGlobalValue - if CV is an expression equivalent to a single
/// global value, return that value.
const GlobalValue * AsmPrinter::findGlobalValue(const Constant *CV) {
  if (const GlobalValue *GV = dyn_cast<GlobalValue>(CV))
    return GV;
  else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV)) {
    const TargetData *TD = TM.getTargetData();
    unsigned Opcode = CE->getOpcode();    
    switch (Opcode) {
    case Instruction::GetElementPtr: {
      const Constant *ptrVal = CE->getOperand(0);
      SmallVector<Value*, 8> idxVec(CE->op_begin()+1, CE->op_end());
      if (TD->getIndexedOffset(ptrVal->getType(), &idxVec[0], idxVec.size()))
        return 0;
      return findGlobalValue(ptrVal);
    }
    case Instruction::BitCast:
      return findGlobalValue(CE->getOperand(0));
    default:
      return 0;
    }
  }
  return 0;
}

/// EmitLLVMUsedList - For targets that define a TAI::UsedDirective, mark each
/// global in the specified llvm.used list for which emitUsedDirectiveFor
/// is true, as being used with this directive.

void AsmPrinter::EmitLLVMUsedList(Constant *List) {
  const char *Directive = TAI->getUsedDirective();

  // Should be an array of 'sbyte*'.
  ConstantArray *InitList = dyn_cast<ConstantArray>(List);
  if (InitList == 0) return;
  
  for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i) {
    const GlobalValue *GV = findGlobalValue(InitList->getOperand(i));
    if (TAI->emitUsedDirectiveFor(GV, Mang)) {
      O << Directive;
      EmitConstantValueOnly(InitList->getOperand(i));
      O << '\n';
    }
  }
}

/// EmitXXStructorList - Emit the ctor or dtor list.  This just prints out the 
/// function pointers, ignoring the init priority.
void AsmPrinter::EmitXXStructorList(Constant *List) {
  // Should be an array of '{ int, void ()* }' structs.  The first value is the
  // init priority, which we ignore.
  if (!isa<ConstantArray>(List)) return;
  ConstantArray *InitList = cast<ConstantArray>(List);
  for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i)
    if (ConstantStruct *CS = dyn_cast<ConstantStruct>(InitList->getOperand(i))){
      if (CS->getNumOperands() != 2) return;  // Not array of 2-element structs.

      if (CS->getOperand(1)->isNullValue())
        return;  // Found a null terminator, exit printing.
      // Emit the function pointer.
      EmitGlobalConstant(CS->getOperand(1));
    }
}

/// getGlobalLinkName - Returns the asm/link name of of the specified
/// global variable.  Should be overridden by each target asm printer to
/// generate the appropriate value.
const std::string AsmPrinter::getGlobalLinkName(const GlobalVariable *GV) const{
  std::string LinkName;
  
  if (isa<Function>(GV)) {
    LinkName += TAI->getFunctionAddrPrefix();
    LinkName += Mang->getValueName(GV);
    LinkName += TAI->getFunctionAddrSuffix();
  } else {
    LinkName += TAI->getGlobalVarAddrPrefix();
    LinkName += Mang->getValueName(GV);
    LinkName += TAI->getGlobalVarAddrSuffix();
  }  
  
  return LinkName;
}

/// EmitExternalGlobal - Emit the external reference to a global variable.
/// Should be overridden if an indirect reference should be used.
void AsmPrinter::EmitExternalGlobal(const GlobalVariable *GV) {
  O << getGlobalLinkName(GV);
}



//===----------------------------------------------------------------------===//
/// LEB 128 number encoding.

/// PrintULEB128 - Print a series of hexidecimal values (separated by commas)
/// representing an unsigned leb128 value.
void AsmPrinter::PrintULEB128(unsigned Value) const {
  char Buffer[20];
  do {
    unsigned char Byte = static_cast<unsigned char>(Value & 0x7f);
    Value >>= 7;
    if (Value) Byte |= 0x80;
    O << "0x" << utohex_buffer(Byte, Buffer+20);
    if (Value) O << ", ";
  } while (Value);
}

/// PrintSLEB128 - Print a series of hexidecimal values (separated by commas)
/// representing a signed leb128 value.
void AsmPrinter::PrintSLEB128(int Value) const {
  int Sign = Value >> (8 * sizeof(Value) - 1);
  bool IsMore;
  char Buffer[20];

  do {
    unsigned char Byte = static_cast<unsigned char>(Value & 0x7f);
    Value >>= 7;
    IsMore = Value != Sign || ((Byte ^ Sign) & 0x40) != 0;
    if (IsMore) Byte |= 0x80;
    O << "0x" << utohex_buffer(Byte, Buffer+20);
    if (IsMore) O << ", ";
  } while (IsMore);
}

//===--------------------------------------------------------------------===//
// Emission and print routines
//

/// PrintHex - Print a value as a hexidecimal value.
///
void AsmPrinter::PrintHex(int Value) const { 
  char Buffer[20];
  O << "0x" << utohex_buffer(static_cast<unsigned>(Value), Buffer+20);
}

/// EOL - Print a newline character to asm stream.  If a comment is present
/// then it will be printed first.  Comments should not contain '\n'.
void AsmPrinter::EOL() const {
  O << '\n';
}

void AsmPrinter::EOL(const std::string &Comment) const {
  if (VerboseAsm && !Comment.empty()) {
    O << '\t'
      << TAI->getCommentString()
      << ' '
      << Comment;
  }
  O << '\n';
}

void AsmPrinter::EOL(const char* Comment) const {
  if (VerboseAsm && *Comment) {
    O << '\t'
      << TAI->getCommentString()
      << ' '
      << Comment;
  }
  O << '\n';
}

/// EmitULEB128Bytes - Emit an assembler byte data directive to compose an
/// unsigned leb128 value.
void AsmPrinter::EmitULEB128Bytes(unsigned Value) const {
  if (TAI->hasLEB128()) {
    O << "\t.uleb128\t"
      << Value;
  } else {
    O << TAI->getData8bitsDirective();
    PrintULEB128(Value);
  }
}

/// EmitSLEB128Bytes - print an assembler byte data directive to compose a
/// signed leb128 value.
void AsmPrinter::EmitSLEB128Bytes(int Value) const {
  if (TAI->hasLEB128()) {
    O << "\t.sleb128\t"
      << Value;
  } else {
    O << TAI->getData8bitsDirective();
    PrintSLEB128(Value);
  }
}

/// EmitInt8 - Emit a byte directive and value.
///
void AsmPrinter::EmitInt8(int Value) const {
  O << TAI->getData8bitsDirective();
  PrintHex(Value & 0xFF);
}

/// EmitInt16 - Emit a short directive and value.
///
void AsmPrinter::EmitInt16(int Value) const {
  O << TAI->getData16bitsDirective();
  PrintHex(Value & 0xFFFF);
}

/// EmitInt32 - Emit a long directive and value.
///
void AsmPrinter::EmitInt32(int Value) const {
  O << TAI->getData32bitsDirective();
  PrintHex(Value);
}

/// EmitInt64 - Emit a long long directive and value.
///
void AsmPrinter::EmitInt64(uint64_t Value) const {
  if (TAI->getData64bitsDirective()) {
    O << TAI->getData64bitsDirective();
    PrintHex(Value);
  } else {
    if (TM.getTargetData()->isBigEndian()) {
      EmitInt32(unsigned(Value >> 32)); O << '\n';
      EmitInt32(unsigned(Value));
    } else {
      EmitInt32(unsigned(Value)); O << '\n';
      EmitInt32(unsigned(Value >> 32));
    }
  }
}

/// toOctal - Convert the low order bits of X into an octal digit.
///
static inline char toOctal(int X) {
  return (X&7)+'0';
}

/// printStringChar - Print a char, escaped if necessary.
///
static void printStringChar(raw_ostream &O, char C) {
  if (C == '"') {
    O << "\\\"";
  } else if (C == '\\') {
    O << "\\\\";
  } else if (isprint((unsigned char)C)) {
    O << C;
  } else {
    switch(C) {
    case '\b': O << "\\b"; break;
    case '\f': O << "\\f"; break;
    case '\n': O << "\\n"; break;
    case '\r': O << "\\r"; break;
    case '\t': O << "\\t"; break;
    default:
      O << '\\';
      O << toOctal(C >> 6);
      O << toOctal(C >> 3);
      O << toOctal(C >> 0);
      break;
    }
  }
}

/// EmitString - Emit a string with quotes and a null terminator.
/// Special characters are emitted properly.
/// \literal (Eg. '\t') \endliteral
void AsmPrinter::EmitString(const std::string &String) const {
  const char* AscizDirective = TAI->getAscizDirective();
  if (AscizDirective)
    O << AscizDirective;
  else
    O << TAI->getAsciiDirective();
  O << '\"';
  for (unsigned i = 0, N = String.size(); i < N; ++i) {
    unsigned char C = String[i];
    printStringChar(O, C);
  }
  if (AscizDirective)
    O << '\"';
  else
    O << "\\0\"";
}


/// EmitFile - Emit a .file directive.
void AsmPrinter::EmitFile(unsigned Number, const std::string &Name) const {
  O << "\t.file\t" << Number << " \"";
  for (unsigned i = 0, N = Name.size(); i < N; ++i) {
    unsigned char C = Name[i];
    printStringChar(O, C);
  }
  O << '\"';
}


//===----------------------------------------------------------------------===//

// EmitAlignment - Emit an alignment directive to the specified power of
// two boundary.  For example, if you pass in 3 here, you will get an 8
// byte alignment.  If a global value is specified, and if that global has
// an explicit alignment requested, it will unconditionally override the
// alignment request.  However, if ForcedAlignBits is specified, this value
// has final say: the ultimate alignment will be the max of ForcedAlignBits
// and the alignment computed with NumBits and the global.
//
// The algorithm is:
//     Align = NumBits;
//     if (GV && GV->hasalignment) Align = GV->getalignment();
//     Align = std::max(Align, ForcedAlignBits);
//
void AsmPrinter::EmitAlignment(unsigned NumBits, const GlobalValue *GV,
                               unsigned ForcedAlignBits,
                               bool UseFillExpr) const {
  if (GV && GV->getAlignment())
    NumBits = Log2_32(GV->getAlignment());
  NumBits = std::max(NumBits, ForcedAlignBits);
  
  if (NumBits == 0) return;   // No need to emit alignment.
  if (TAI->getAlignmentIsInBytes()) NumBits = 1 << NumBits;
  O << TAI->getAlignDirective() << NumBits;

  unsigned FillValue = TAI->getTextAlignFillValue();
  UseFillExpr &= IsInTextSection && FillValue;
  if (UseFillExpr) {
    O << ',';
    PrintHex(FillValue);
  }
  O << '\n';
}

    
/// EmitZeros - Emit a block of zeros.
///
void AsmPrinter::EmitZeros(uint64_t NumZeros, unsigned AddrSpace) const {
  if (NumZeros) {
    if (TAI->getZeroDirective()) {
      O << TAI->getZeroDirective() << NumZeros;
      if (TAI->getZeroDirectiveSuffix())
        O << TAI->getZeroDirectiveSuffix();
      O << '\n';
    } else {
      for (; NumZeros; --NumZeros)
        O << TAI->getData8bitsDirective(AddrSpace) << "0\n";
    }
  }
}

// Print out the specified constant, without a storage class.  Only the
// constants valid in constant expressions can occur here.
void AsmPrinter::EmitConstantValueOnly(const Constant *CV) {
  if (CV->isNullValue() || isa<UndefValue>(CV))
    O << '0';
  else if (const ConstantInt *CI = dyn_cast<ConstantInt>(CV)) {
    O << CI->getZExtValue();
  } else if (const GlobalValue *GV = dyn_cast<GlobalValue>(CV)) {
    // This is a constant address for a global variable or function. Use the
    // name of the variable or function as the address value, possibly
    // decorating it with GlobalVarAddrPrefix/Suffix or
    // FunctionAddrPrefix/Suffix (these all default to "" )
    if (isa<Function>(GV)) {
      O << TAI->getFunctionAddrPrefix()
        << Mang->getValueName(GV)
        << TAI->getFunctionAddrSuffix();
    } else {
      O << TAI->getGlobalVarAddrPrefix()
        << Mang->getValueName(GV)
        << TAI->getGlobalVarAddrSuffix();
    }
  } else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CV)) {
    const TargetData *TD = TM.getTargetData();
    unsigned Opcode = CE->getOpcode();    
    switch (Opcode) {
    case Instruction::GetElementPtr: {
      // generate a symbolic expression for the byte address
      const Constant *ptrVal = CE->getOperand(0);
      SmallVector<Value*, 8> idxVec(CE->op_begin()+1, CE->op_end());
      if (int64_t Offset = TD->getIndexedOffset(ptrVal->getType(), &idxVec[0],
                                                idxVec.size())) {
        if (Offset)
          O << '(';
        EmitConstantValueOnly(ptrVal);
        if (Offset > 0)
          O << ") + " << Offset;
        else if (Offset < 0)
          O << ") - " << -Offset;
      } else {
        EmitConstantValueOnly(ptrVal);
      }
      break;
    }
    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
      assert(0 && "FIXME: Don't yet support this kind of constant cast expr");
      break;
    case Instruction::BitCast:
      return EmitConstantValueOnly(CE->getOperand(0));

    case Instruction::IntToPtr: {
      // Handle casts to pointers by changing them into casts to the appropriate
      // integer type.  This promotes constant folding and simplifies this code.
      Constant *Op = CE->getOperand(0);
      Op = ConstantExpr::getIntegerCast(Op, TD->getIntPtrType(), false/*ZExt*/);
      return EmitConstantValueOnly(Op);
    }
      
      
    case Instruction::PtrToInt: {
      // Support only foldable casts to/from pointers that can be eliminated by
      // changing the pointer to the appropriately sized integer type.
      Constant *Op = CE->getOperand(0);
      const Type *Ty = CE->getType();

      // We can emit the pointer value into this slot if the slot is an
      // integer slot greater or equal to the size of the pointer.
      if (TD->getTypePaddedSize(Ty) >= TD->getTypePaddedSize(Op->getType()))
        return EmitConstantValueOnly(Op);

      O << "((";
      EmitConstantValueOnly(Op);
      APInt ptrMask = APInt::getAllOnesValue(TD->getTypePaddedSizeInBits(Ty));
      
      SmallString<40> S;
      ptrMask.toStringUnsigned(S);
      O << ") & " << S.c_str() << ')';
      break;
    }
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
      O << '(';
      EmitConstantValueOnly(CE->getOperand(0));
      O << ')';
      switch (Opcode) {
      case Instruction::Add:
       O << " + ";
       break;
      case Instruction::Sub:
       O << " - ";
       break;
      case Instruction::And:
       O << " & ";
       break;
      case Instruction::Or:
       O << " | ";
       break;
      case Instruction::Xor:
       O << " ^ ";
       break;
      default:
       break;
      }
      O << '(';
      EmitConstantValueOnly(CE->getOperand(1));
      O << ')';
      break;
    default:
      assert(0 && "Unsupported operator!");
    }
  } else {
    assert(0 && "Unknown constant value!");
  }
}

/// printAsCString - Print the specified array as a C compatible string, only if
/// the predicate isString is true.
///
static void printAsCString(raw_ostream &O, const ConstantArray *CVA,
                           unsigned LastElt) {
  assert(CVA->isString() && "Array is not string compatible!");

  O << '\"';
  for (unsigned i = 0; i != LastElt; ++i) {
    unsigned char C =
        (unsigned char)cast<ConstantInt>(CVA->getOperand(i))->getZExtValue();
    printStringChar(O, C);
  }
  O << '\"';
}

/// EmitString - Emit a zero-byte-terminated string constant.
///
void AsmPrinter::EmitString(const ConstantArray *CVA) const {
  unsigned NumElts = CVA->getNumOperands();
  if (TAI->getAscizDirective() && NumElts && 
      cast<ConstantInt>(CVA->getOperand(NumElts-1))->getZExtValue() == 0) {
    O << TAI->getAscizDirective();
    printAsCString(O, CVA, NumElts-1);
  } else {
    O << TAI->getAsciiDirective();
    printAsCString(O, CVA, NumElts);
  }
  O << '\n';
}

void AsmPrinter::EmitGlobalConstantArray(const ConstantArray *CVA) {
  if (CVA->isString()) {
    EmitString(CVA);
  } else { // Not a string.  Print the values in successive locations
    for (unsigned i = 0, e = CVA->getNumOperands(); i != e; ++i)
      EmitGlobalConstant(CVA->getOperand(i));
  }
}

void AsmPrinter::EmitGlobalConstantVector(const ConstantVector *CP) {
  const VectorType *PTy = CP->getType();
  
  for (unsigned I = 0, E = PTy->getNumElements(); I < E; ++I)
    EmitGlobalConstant(CP->getOperand(I));
}

void AsmPrinter::EmitGlobalConstantStruct(const ConstantStruct *CVS,
                                          unsigned AddrSpace) {
  // Print the fields in successive locations. Pad to align if needed!
  const TargetData *TD = TM.getTargetData();
  unsigned Size = TD->getTypePaddedSize(CVS->getType());
  const StructLayout *cvsLayout = TD->getStructLayout(CVS->getType());
  uint64_t sizeSoFar = 0;
  for (unsigned i = 0, e = CVS->getNumOperands(); i != e; ++i) {
    const Constant* field = CVS->getOperand(i);

    // Check if padding is needed and insert one or more 0s.
    uint64_t fieldSize = TD->getTypePaddedSize(field->getType());
    uint64_t padSize = ((i == e-1 ? Size : cvsLayout->getElementOffset(i+1))
                        - cvsLayout->getElementOffset(i)) - fieldSize;
    sizeSoFar += fieldSize + padSize;

    // Now print the actual field value.
    EmitGlobalConstant(field, AddrSpace);

    // Insert padding - this may include padding to increase the size of the
    // current field up to the ABI size (if the struct is not packed) as well
    // as padding to ensure that the next field starts at the right offset.
    EmitZeros(padSize, AddrSpace);
  }
  assert(sizeSoFar == cvsLayout->getSizeInBytes() &&
         "Layout of constant struct may be incorrect!");
}

void AsmPrinter::EmitGlobalConstantFP(const ConstantFP *CFP, 
                                      unsigned AddrSpace) {
  // FP Constants are printed as integer constants to avoid losing
  // precision...
  const TargetData *TD = TM.getTargetData();
  if (CFP->getType() == Type::DoubleTy) {
    double Val = CFP->getValueAPF().convertToDouble();  // for comment only
    uint64_t i = CFP->getValueAPF().bitcastToAPInt().getZExtValue();
    if (TAI->getData64bitsDirective(AddrSpace))
      O << TAI->getData64bitsDirective(AddrSpace) << i << '\t'
        << TAI->getCommentString() << " double value: " << Val << '\n';
    else if (TD->isBigEndian()) {
      O << TAI->getData32bitsDirective(AddrSpace) << unsigned(i >> 32)
        << '\t' << TAI->getCommentString()
        << " double most significant word " << Val << '\n';
      O << TAI->getData32bitsDirective(AddrSpace) << unsigned(i)
        << '\t' << TAI->getCommentString()
        << " double least significant word " << Val << '\n';
    } else {
      O << TAI->getData32bitsDirective(AddrSpace) << unsigned(i)
        << '\t' << TAI->getCommentString()
        << " double least significant word " << Val << '\n';
      O << TAI->getData32bitsDirective(AddrSpace) << unsigned(i >> 32)
        << '\t' << TAI->getCommentString()
        << " double most significant word " << Val << '\n';
    }
    return;
  } else if (CFP->getType() == Type::FloatTy) {
    float Val = CFP->getValueAPF().convertToFloat();  // for comment only
    O << TAI->getData32bitsDirective(AddrSpace)
      << CFP->getValueAPF().bitcastToAPInt().getZExtValue()
      << '\t' << TAI->getCommentString() << " float " << Val << '\n';
    return;
  } else if (CFP->getType() == Type::X86_FP80Ty) {
    // all long double variants are printed as hex
    // api needed to prevent premature destruction
    APInt api = CFP->getValueAPF().bitcastToAPInt();
    const uint64_t *p = api.getRawData();
    // Convert to double so we can print the approximate val as a comment.
    APFloat DoubleVal = CFP->getValueAPF();
    bool ignored;
    DoubleVal.convert(APFloat::IEEEdouble, APFloat::rmNearestTiesToEven,
                      &ignored);
    if (TD->isBigEndian()) {
      O << TAI->getData16bitsDirective(AddrSpace) << uint16_t(p[0] >> 48)
        << '\t' << TAI->getCommentString()
        << " long double most significant halfword of ~"
        << DoubleVal.convertToDouble() << '\n';
      O << TAI->getData16bitsDirective(AddrSpace) << uint16_t(p[0] >> 32)
        << '\t' << TAI->getCommentString()
        << " long double next halfword\n";
      O << TAI->getData16bitsDirective(AddrSpace) << uint16_t(p[0] >> 16)
        << '\t' << TAI->getCommentString()
        << " long double next halfword\n";
      O << TAI->getData16bitsDirective(AddrSpace) << uint16_t(p[0])
        << '\t' << TAI->getCommentString()
        << " long double next halfword\n";
      O << TAI->getData16bitsDirective(AddrSpace) << uint16_t(p[1])
        << '\t' << TAI->getCommentString()
        << " long double least significant halfword\n";
     } else {
      O << TAI->getData16bitsDirective(AddrSpace) << uint16_t(p[1])
        << '\t' << TAI->getCommentString()
        << " long double least significant halfword of ~"
        << DoubleVal.convertToDouble() << '\n';
      O << TAI->getData16bitsDirective(AddrSpace) << uint16_t(p[0])
        << '\t' << TAI->getCommentString()
        << " long double next halfword\n";
      O << TAI->getData16bitsDirective(AddrSpace) << uint16_t(p[0] >> 16)
        << '\t' << TAI->getCommentString()
        << " long double next halfword\n";
      O << TAI->getData16bitsDirective(AddrSpace) << uint16_t(p[0] >> 32)
        << '\t' << TAI->getCommentString()
        << " long double next halfword\n";
      O << TAI->getData16bitsDirective(AddrSpace) << uint16_t(p[0] >> 48)
        << '\t' << TAI->getCommentString()
        << " long double most significant halfword\n";
    }
    EmitZeros(TD->getTypePaddedSize(Type::X86_FP80Ty) -
              TD->getTypeStoreSize(Type::X86_FP80Ty), AddrSpace);
    return;
  } else if (CFP->getType() == Type::PPC_FP128Ty) {
    // all long double variants are printed as hex
    // api needed to prevent premature destruction
    APInt api = CFP->getValueAPF().bitcastToAPInt();
    const uint64_t *p = api.getRawData();
    if (TD->isBigEndian()) {
      O << TAI->getData32bitsDirective(AddrSpace) << uint32_t(p[0] >> 32)
        << '\t' << TAI->getCommentString()
        << " long double most significant word\n";
      O << TAI->getData32bitsDirective(AddrSpace) << uint32_t(p[0])
        << '\t' << TAI->getCommentString()
        << " long double next word\n";
      O << TAI->getData32bitsDirective(AddrSpace) << uint32_t(p[1] >> 32)
        << '\t' << TAI->getCommentString()
        << " long double next word\n";
      O << TAI->getData32bitsDirective(AddrSpace) << uint32_t(p[1])
        << '\t' << TAI->getCommentString()
        << " long double least significant word\n";
     } else {
      O << TAI->getData32bitsDirective(AddrSpace) << uint32_t(p[1])
        << '\t' << TAI->getCommentString()
        << " long double least significant word\n";
      O << TAI->getData32bitsDirective(AddrSpace) << uint32_t(p[1] >> 32)
        << '\t' << TAI->getCommentString()
        << " long double next word\n";
      O << TAI->getData32bitsDirective(AddrSpace) << uint32_t(p[0])
        << '\t' << TAI->getCommentString()
        << " long double next word\n";
      O << TAI->getData32bitsDirective(AddrSpace) << uint32_t(p[0] >> 32)
        << '\t' << TAI->getCommentString()
        << " long double most significant word\n";
    }
    return;
  } else assert(0 && "Floating point constant type not handled");
}

void AsmPrinter::EmitGlobalConstantLargeInt(const ConstantInt *CI,
                                            unsigned AddrSpace) {
  const TargetData *TD = TM.getTargetData();
  unsigned BitWidth = CI->getBitWidth();
  assert(isPowerOf2_32(BitWidth) &&
         "Non-power-of-2-sized integers not handled!");

  // We don't expect assemblers to support integer data directives
  // for more than 64 bits, so we emit the data in at most 64-bit
  // quantities at a time.
  const uint64_t *RawData = CI->getValue().getRawData();
  for (unsigned i = 0, e = BitWidth / 64; i != e; ++i) {
    uint64_t Val;
    if (TD->isBigEndian())
      Val = RawData[e - i - 1];
    else
      Val = RawData[i];

    if (TAI->getData64bitsDirective(AddrSpace))
      O << TAI->getData64bitsDirective(AddrSpace) << Val << '\n';
    else if (TD->isBigEndian()) {
      O << TAI->getData32bitsDirective(AddrSpace) << unsigned(Val >> 32)
        << '\t' << TAI->getCommentString()
        << " Double-word most significant word " << Val << '\n';
      O << TAI->getData32bitsDirective(AddrSpace) << unsigned(Val)
        << '\t' << TAI->getCommentString()
        << " Double-word least significant word " << Val << '\n';
    } else {
      O << TAI->getData32bitsDirective(AddrSpace) << unsigned(Val)
        << '\t' << TAI->getCommentString()
        << " Double-word least significant word " << Val << '\n';
      O << TAI->getData32bitsDirective(AddrSpace) << unsigned(Val >> 32)
        << '\t' << TAI->getCommentString()
        << " Double-word most significant word " << Val << '\n';
    }
  }
}

/// EmitGlobalConstant - Print a general LLVM constant to the .s file.
void AsmPrinter::EmitGlobalConstant(const Constant *CV, unsigned AddrSpace) {
  const TargetData *TD = TM.getTargetData();
  const Type *type = CV->getType();
  unsigned Size = TD->getTypePaddedSize(type);

  if (CV->isNullValue() || isa<UndefValue>(CV)) {
    EmitZeros(Size, AddrSpace);
    return;
  } else if (const ConstantArray *CVA = dyn_cast<ConstantArray>(CV)) {
    EmitGlobalConstantArray(CVA);
    return;
  } else if (const ConstantStruct *CVS = dyn_cast<ConstantStruct>(CV)) {
    EmitGlobalConstantStruct(CVS, AddrSpace);
    return;
  } else if (const ConstantFP *CFP = dyn_cast<ConstantFP>(CV)) {
    EmitGlobalConstantFP(CFP, AddrSpace);
    return;
  } else if (const ConstantInt *CI = dyn_cast<ConstantInt>(CV)) {
    // Small integers are handled below; large integers are handled here.
    if (Size > 4) {
      EmitGlobalConstantLargeInt(CI, AddrSpace);
      return;
    }
  } else if (const ConstantVector *CP = dyn_cast<ConstantVector>(CV)) {
    EmitGlobalConstantVector(CP);
    return;
  }

  printDataDirective(type, AddrSpace);
  EmitConstantValueOnly(CV);
  if (const ConstantInt *CI = dyn_cast<ConstantInt>(CV)) {
    SmallString<40> S;
    CI->getValue().toStringUnsigned(S, 16);
    O << "\t\t\t" << TAI->getCommentString() << " 0x" << S.c_str();
  }
  O << '\n';
}

void AsmPrinter::EmitMachineConstantPoolValue(MachineConstantPoolValue *MCPV) {
  // Target doesn't support this yet!
  abort();
}

/// PrintSpecial - Print information related to the specified machine instr
/// that is independent of the operand, and may be independent of the instr
/// itself.  This can be useful for portably encoding the comment character
/// or other bits of target-specific knowledge into the asmstrings.  The
/// syntax used is ${:comment}.  Targets can override this to add support
/// for their own strange codes.
void AsmPrinter::PrintSpecial(const MachineInstr *MI, const char *Code) {
  if (!strcmp(Code, "private")) {
    O << TAI->getPrivateGlobalPrefix();
  } else if (!strcmp(Code, "comment")) {
    O << TAI->getCommentString();
  } else if (!strcmp(Code, "uid")) {
    // Assign a unique ID to this machine instruction.
    static const MachineInstr *LastMI = 0;
    static const Function *F = 0;
    static unsigned Counter = 0U-1;

    // Comparing the address of MI isn't sufficient, because machineinstrs may
    // be allocated to the same address across functions.
    const Function *ThisF = MI->getParent()->getParent()->getFunction();
    
    // If this is a new machine instruction, bump the counter.
    if (LastMI != MI || F != ThisF) {
      ++Counter;
      LastMI = MI;
      F = ThisF;
    }
    O << Counter;
  } else {
    cerr << "Unknown special formatter '" << Code
         << "' for machine instr: " << *MI;
    exit(1);
  }    
}


/// printInlineAsm - This method formats and prints the specified machine
/// instruction that is an inline asm.
void AsmPrinter::printInlineAsm(const MachineInstr *MI) const {
  unsigned NumOperands = MI->getNumOperands();
  
  // Count the number of register definitions.
  unsigned NumDefs = 0;
  for (; MI->getOperand(NumDefs).isReg() && MI->getOperand(NumDefs).isDef();
       ++NumDefs)
    assert(NumDefs != NumOperands-1 && "No asm string?");
  
  assert(MI->getOperand(NumDefs).isSymbol() && "No asm string?");

  // Disassemble the AsmStr, printing out the literal pieces, the operands, etc.
  const char *AsmStr = MI->getOperand(NumDefs).getSymbolName();

  // If this asmstr is empty, just print the #APP/#NOAPP markers.
  // These are useful to see where empty asm's wound up.
  if (AsmStr[0] == 0) {
    O << TAI->getInlineAsmStart() << "\n\t" << TAI->getInlineAsmEnd() << '\n';
    return;
  }
  
  O << TAI->getInlineAsmStart() << "\n\t";

  // The variant of the current asmprinter.
  int AsmPrinterVariant = TAI->getAssemblerDialect();

  int CurVariant = -1;            // The number of the {.|.|.} region we are in.
  const char *LastEmitted = AsmStr; // One past the last character emitted.
  
  while (*LastEmitted) {
    switch (*LastEmitted) {
    default: {
      // Not a special case, emit the string section literally.
      const char *LiteralEnd = LastEmitted+1;
      while (*LiteralEnd && *LiteralEnd != '{' && *LiteralEnd != '|' &&
             *LiteralEnd != '}' && *LiteralEnd != '$' && *LiteralEnd != '\n')
        ++LiteralEnd;
      if (CurVariant == -1 || CurVariant == AsmPrinterVariant)
        O.write(LastEmitted, LiteralEnd-LastEmitted);
      LastEmitted = LiteralEnd;
      break;
    }
    case '\n':
      ++LastEmitted;   // Consume newline character.
      O << '\n';       // Indent code with newline.
      break;
    case '$': {
      ++LastEmitted;   // Consume '$' character.
      bool Done = true;

      // Handle escapes.
      switch (*LastEmitted) {
      default: Done = false; break;
      case '$':     // $$ -> $
        if (CurVariant == -1 || CurVariant == AsmPrinterVariant)
          O << '$';
        ++LastEmitted;  // Consume second '$' character.
        break;
      case '(':             // $( -> same as GCC's { character.
        ++LastEmitted;      // Consume '(' character.
        if (CurVariant != -1) {
          cerr << "Nested variants found in inline asm string: '"
               << AsmStr << "'\n";
          exit(1);
        }
        CurVariant = 0;     // We're in the first variant now.
        break;
      case '|':
        ++LastEmitted;  // consume '|' character.
        if (CurVariant == -1)
          O << '|';       // this is gcc's behavior for | outside a variant
        else
          ++CurVariant;   // We're in the next variant.
        break;
      case ')':         // $) -> same as GCC's } char.
        ++LastEmitted;  // consume ')' character.
        if (CurVariant == -1)
          O << '}';     // this is gcc's behavior for } outside a variant
        else 
          CurVariant = -1;
        break;
      }
      if (Done) break;
      
      bool HasCurlyBraces = false;
      if (*LastEmitted == '{') {     // ${variable}
        ++LastEmitted;               // Consume '{' character.
        HasCurlyBraces = true;
      }
      
      const char *IDStart = LastEmitted;
      char *IDEnd;
      errno = 0;
      long Val = strtol(IDStart, &IDEnd, 10); // We only accept numbers for IDs.
      if (!isdigit(*IDStart) || (Val == 0 && errno == EINVAL)) {
        cerr << "Bad $ operand number in inline asm string: '" 
             << AsmStr << "'\n";
        exit(1);
      }
      LastEmitted = IDEnd;
      
      char Modifier[2] = { 0, 0 };
      
      if (HasCurlyBraces) {
        // If we have curly braces, check for a modifier character.  This
        // supports syntax like ${0:u}, which correspond to "%u0" in GCC asm.
        if (*LastEmitted == ':') {
          ++LastEmitted;    // Consume ':' character.
          if (*LastEmitted == 0) {
            cerr << "Bad ${:} expression in inline asm string: '" 
                 << AsmStr << "'\n";
            exit(1);
          }
          
          Modifier[0] = *LastEmitted;
          ++LastEmitted;    // Consume modifier character.
        }
        
        if (*LastEmitted != '}') {
          cerr << "Bad ${} expression in inline asm string: '" 
               << AsmStr << "'\n";
          exit(1);
        }
        ++LastEmitted;    // Consume '}' character.
      }
      
      if ((unsigned)Val >= NumOperands-1) {
        cerr << "Invalid $ operand number in inline asm string: '" 
             << AsmStr << "'\n";
        exit(1);
      }
      
      // Okay, we finally have a value number.  Ask the target to print this
      // operand!
      if (CurVariant == -1 || CurVariant == AsmPrinterVariant) {
        unsigned OpNo = 1;

        bool Error = false;

        // Scan to find the machine operand number for the operand.
        for (; Val; --Val) {
          if (OpNo >= MI->getNumOperands()) break;
          unsigned OpFlags = MI->getOperand(OpNo).getImm();
          OpNo += (OpFlags >> 3) + 1;
        }

        if (OpNo >= MI->getNumOperands()) {
          Error = true;
        } else {
          unsigned OpFlags = MI->getOperand(OpNo).getImm();
          ++OpNo;  // Skip over the ID number.

          if (Modifier[0]=='l')  // labels are target independent
            printBasicBlockLabel(MI->getOperand(OpNo).getMBB(), 
                                 false, false, false);
          else {
            AsmPrinter *AP = const_cast<AsmPrinter*>(this);
            if ((OpFlags & 7) == 4) {
              Error = AP->PrintAsmMemoryOperand(MI, OpNo, AsmPrinterVariant,
                                                Modifier[0] ? Modifier : 0);
            } else {
              Error = AP->PrintAsmOperand(MI, OpNo, AsmPrinterVariant,
                                          Modifier[0] ? Modifier : 0);
            }
          }
        }
        if (Error) {
          cerr << "Invalid operand found in inline asm: '"
               << AsmStr << "'\n";
          MI->dump();
          exit(1);
        }
      }
      break;
    }
    }
  }
  O << "\n\t" << TAI->getInlineAsmEnd() << '\n';
}

/// printImplicitDef - This method prints the specified machine instruction
/// that is an implicit def.
void AsmPrinter::printImplicitDef(const MachineInstr *MI) const {
  O << '\t' << TAI->getCommentString() << " implicit-def: "
    << TRI->getAsmName(MI->getOperand(0).getReg()) << '\n';
}

/// printLabel - This method prints a local label used by debug and
/// exception handling tables.
void AsmPrinter::printLabel(const MachineInstr *MI) const {
  printLabel(MI->getOperand(0).getImm());
}

void AsmPrinter::printLabel(unsigned Id) const {
  O << TAI->getPrivateGlobalPrefix() << "label" << Id << ":\n";
}

/// printDeclare - This method prints a local variable declaration used by
/// debug tables.
/// FIXME: It doesn't really print anything rather it inserts a DebugVariable
/// entry into dwarf table.
void AsmPrinter::printDeclare(const MachineInstr *MI) const {
  unsigned FI = MI->getOperand(0).getIndex();
  GlobalValue *GV = MI->getOperand(1).getGlobal();
  DW->RecordVariable(cast<GlobalVariable>(GV), FI);
}

/// PrintAsmOperand - Print the specified operand of MI, an INLINEASM
/// instruction, using the specified assembler variant.  Targets should
/// overried this to format as appropriate.
bool AsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                 unsigned AsmVariant, const char *ExtraCode) {
  // Target doesn't support this yet!
  return true;
}

bool AsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                                       unsigned AsmVariant,
                                       const char *ExtraCode) {
  // Target doesn't support this yet!
  return true;
}

/// printBasicBlockLabel - This method prints the label for the specified
/// MachineBasicBlock
void AsmPrinter::printBasicBlockLabel(const MachineBasicBlock *MBB,
                                      bool printAlign, 
                                      bool printColon,
                                      bool printComment) const {
  if (printAlign) {
    unsigned Align = MBB->getAlignment();
    if (Align)
      EmitAlignment(Log2_32(Align));
  }

  O << TAI->getPrivateGlobalPrefix() << "BB" << getFunctionNumber() << '_'
    << MBB->getNumber();
  if (printColon)
    O << ':';
  if (printComment && MBB->getBasicBlock())
    O << '\t' << TAI->getCommentString() << ' '
      << MBB->getBasicBlock()->getNameStart();
}

/// printPICJumpTableSetLabel - This method prints a set label for the
/// specified MachineBasicBlock for a jumptable entry.
void AsmPrinter::printPICJumpTableSetLabel(unsigned uid, 
                                           const MachineBasicBlock *MBB) const {
  if (!TAI->getSetDirective())
    return;
  
  O << TAI->getSetDirective() << ' ' << TAI->getPrivateGlobalPrefix()
    << getFunctionNumber() << '_' << uid << "_set_" << MBB->getNumber() << ',';
  printBasicBlockLabel(MBB, false, false, false);
  O << '-' << TAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber() 
    << '_' << uid << '\n';
}

void AsmPrinter::printPICJumpTableSetLabel(unsigned uid, unsigned uid2,
                                           const MachineBasicBlock *MBB) const {
  if (!TAI->getSetDirective())
    return;
  
  O << TAI->getSetDirective() << ' ' << TAI->getPrivateGlobalPrefix()
    << getFunctionNumber() << '_' << uid << '_' << uid2
    << "_set_" << MBB->getNumber() << ',';
  printBasicBlockLabel(MBB, false, false, false);
  O << '-' << TAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber() 
    << '_' << uid << '_' << uid2 << '\n';
}

/// printDataDirective - This method prints the asm directive for the
/// specified type.
void AsmPrinter::printDataDirective(const Type *type, unsigned AddrSpace) {
  const TargetData *TD = TM.getTargetData();
  switch (type->getTypeID()) {
  case Type::IntegerTyID: {
    unsigned BitWidth = cast<IntegerType>(type)->getBitWidth();
    if (BitWidth <= 8)
      O << TAI->getData8bitsDirective(AddrSpace);
    else if (BitWidth <= 16)
      O << TAI->getData16bitsDirective(AddrSpace);
    else if (BitWidth <= 32)
      O << TAI->getData32bitsDirective(AddrSpace);
    else if (BitWidth <= 64) {
      assert(TAI->getData64bitsDirective(AddrSpace) &&
             "Target cannot handle 64-bit constant exprs!");
      O << TAI->getData64bitsDirective(AddrSpace);
    } else {
      assert(0 && "Target cannot handle given data directive width!");
    }
    break;
  }
  case Type::PointerTyID:
    if (TD->getPointerSize() == 8) {
      assert(TAI->getData64bitsDirective(AddrSpace) &&
             "Target cannot handle 64-bit pointer exprs!");
      O << TAI->getData64bitsDirective(AddrSpace);
    } else if (TD->getPointerSize() == 2) {
      O << TAI->getData16bitsDirective(AddrSpace);
    } else if (TD->getPointerSize() == 1) {
      O << TAI->getData8bitsDirective(AddrSpace);
    } else {
      O << TAI->getData32bitsDirective(AddrSpace);
    }
    break;
  case Type::FloatTyID: case Type::DoubleTyID:
  case Type::X86_FP80TyID: case Type::FP128TyID: case Type::PPC_FP128TyID:
    assert (0 && "Should have already output floating point constant.");
  default:
    assert (0 && "Can't handle printing this type of thing");
    break;
  }
}

void AsmPrinter::printSuffixedName(const char *Name, const char *Suffix,
                                   const char *Prefix) {
  if (Name[0]=='\"')
    O << '\"';
  O << TAI->getPrivateGlobalPrefix();
  if (Prefix) O << Prefix;
  if (Name[0]=='\"')
    O << '\"';
  if (Name[0]=='\"')
    O << Name[1];
  else
    O << Name;
  O << Suffix;
  if (Name[0]=='\"')
    O << '\"';
}

void AsmPrinter::printSuffixedName(const std::string &Name, const char* Suffix) {
  printSuffixedName(Name.c_str(), Suffix);
}

void AsmPrinter::printVisibility(const std::string& Name,
                                 unsigned Visibility) const {
  if (Visibility == GlobalValue::HiddenVisibility) {
    if (const char *Directive = TAI->getHiddenDirective())
      O << Directive << Name << '\n';
  } else if (Visibility == GlobalValue::ProtectedVisibility) {
    if (const char *Directive = TAI->getProtectedDirective())
      O << Directive << Name << '\n';
  }
}

void AsmPrinter::printOffset(int64_t Offset) const {
  if (Offset > 0)
    O << '+' << Offset;
  else if (Offset < 0)
    O << Offset;
}

GCMetadataPrinter *AsmPrinter::GetOrCreateGCPrinter(GCStrategy *S) {
  if (!S->usesMetadata())
    return 0;
  
  gcp_iterator GCPI = GCMetadataPrinters.find(S);
  if (GCPI != GCMetadataPrinters.end())
    return GCPI->second;
  
  const char *Name = S->getName().c_str();
  
  for (GCMetadataPrinterRegistry::iterator
         I = GCMetadataPrinterRegistry::begin(),
         E = GCMetadataPrinterRegistry::end(); I != E; ++I)
    if (strcmp(Name, I->getName()) == 0) {
      GCMetadataPrinter *GMP = I->instantiate();
      GMP->S = S;
      GCMetadataPrinters.insert(std::make_pair(S, GMP));
      return GMP;
    }
  
  cerr << "no GCMetadataPrinter registered for GC: " << Name << "\n";
  abort();
}
