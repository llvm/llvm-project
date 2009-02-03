//===-- X86IntelAsmPrinter.cpp - Convert X86 LLVM code to Intel assembly --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to Intel format assembly language.
// This printer is the output mechanism used by `llc'.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "X86IntelAsmPrinter.h"
#include "X86InstrInfo.h"
#include "X86TargetAsmInfo.h"
#include "X86.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

STATISTIC(EmittedInsts, "Number of machine instrs printed");

static X86MachineFunctionInfo calculateFunctionInfo(const Function *F,
                                                    const TargetData *TD) {
  X86MachineFunctionInfo Info;
  uint64_t Size = 0;

  switch (F->getCallingConv()) {
  case CallingConv::X86_StdCall:
    Info.setDecorationStyle(StdCall);
    break;
  case CallingConv::X86_FastCall:
    Info.setDecorationStyle(FastCall);
    break;
  default:
    return Info;
  }

  unsigned argNum = 1;
  for (Function::const_arg_iterator AI = F->arg_begin(), AE = F->arg_end();
       AI != AE; ++AI, ++argNum) {
    const Type* Ty = AI->getType();

    // 'Dereference' type in case of byval parameter attribute
    if (F->paramHasAttr(argNum, Attribute::ByVal))
      Ty = cast<PointerType>(Ty)->getElementType();

    // Size should be aligned to DWORD boundary
    Size += ((TD->getTypePaddedSize(Ty) + 3)/4)*4;
  }

  // We're not supporting tooooo huge arguments :)
  Info.setBytesToPopOnReturn((unsigned int)Size);
  return Info;
}


/// decorateName - Query FunctionInfoMap and use this information for various
/// name decoration.
void X86IntelAsmPrinter::decorateName(std::string &Name,
                                      const GlobalValue *GV) {
  const Function *F = dyn_cast<Function>(GV);
  if (!F) return;

  // We don't want to decorate non-stdcall or non-fastcall functions right now
  unsigned CC = F->getCallingConv();
  if (CC != CallingConv::X86_StdCall && CC != CallingConv::X86_FastCall)
    return;

  FMFInfoMap::const_iterator info_item = FunctionInfoMap.find(F);

  const X86MachineFunctionInfo *Info;
  if (info_item == FunctionInfoMap.end()) {
    // Calculate apropriate function info and populate map
    FunctionInfoMap[F] = calculateFunctionInfo(F, TM.getTargetData());
    Info = &FunctionInfoMap[F];
  } else {
    Info = &info_item->second;
  }

  const FunctionType *FT = F->getFunctionType();
  switch (Info->getDecorationStyle()) {
  case None:
    break;
  case StdCall:
    // "Pure" variadic functions do not receive @0 suffix.
    if (!FT->isVarArg() || (FT->getNumParams() == 0) ||
        (FT->getNumParams() == 1 && F->hasStructRetAttr()))
      Name += '@' + utostr_32(Info->getBytesToPopOnReturn());
    break;
  case FastCall:
    // "Pure" variadic functions do not receive @0 suffix.
    if (!FT->isVarArg() || (FT->getNumParams() == 0) ||
        (FT->getNumParams() == 1 && F->hasStructRetAttr()))
      Name += '@' + utostr_32(Info->getBytesToPopOnReturn());

    if (Name[0] == '_')
      Name[0] = '@';
    else
      Name = '@' + Name;

    break;
  default:
    assert(0 && "Unsupported DecorationStyle");
  }
}

/// runOnMachineFunction - This uses the printMachineInstruction()
/// method to print assembly for each instruction.
///
bool X86IntelAsmPrinter::runOnMachineFunction(MachineFunction &MF) {
  SetupMachineFunction(MF);
  O << "\n\n";

  // Print out constants referenced by the function
  EmitConstantPool(MF.getConstantPool());

  // Print out labels for the function.
  const Function *F = MF.getFunction();
  unsigned CC = F->getCallingConv();

  // Populate function information map.  Actually, We don't want to populate
  // non-stdcall or non-fastcall functions' information right now.
  if (CC == CallingConv::X86_StdCall || CC == CallingConv::X86_FastCall)
    FunctionInfoMap[F] = *MF.getInfo<X86MachineFunctionInfo>();

  decorateName(CurrentFnName, F);

  SwitchToTextSection("_text", F);

  unsigned FnAlign = 4;
  if (F->hasFnAttr(Attribute::OptimizeForSize))
    FnAlign = 1;
  switch (F->getLinkage()) {
  default: assert(0 && "Unsupported linkage type!");
  case Function::PrivateLinkage:
  case Function::InternalLinkage:
    EmitAlignment(FnAlign);
    break;
  case Function::DLLExportLinkage:
    DLLExportedFns.insert(CurrentFnName);
    //FALLS THROUGH
  case Function::ExternalLinkage:
    O << "\tpublic " << CurrentFnName << "\n";
    EmitAlignment(FnAlign);
    break;
  }

  O << CurrentFnName << "\tproc near\n";

  // Print out code for the function.
  for (MachineFunction::const_iterator I = MF.begin(), E = MF.end();
       I != E; ++I) {
    // Print a label for the basic block if there are any predecessors.
    if (!I->pred_empty()) {
      printBasicBlockLabel(I, true, true);
      O << '\n';
    }
    for (MachineBasicBlock::const_iterator II = I->begin(), E = I->end();
         II != E; ++II) {
      // Print the assembly for the instruction.
      printMachineInstruction(II);
    }
  }

  // Print out jump tables referenced by the function.
  EmitJumpTableInfo(MF.getJumpTableInfo(), MF);

  O << CurrentFnName << "\tendp\n";

  O.flush();

  // We didn't modify anything.
  return false;
}

void X86IntelAsmPrinter::printSSECC(const MachineInstr *MI, unsigned Op) {
  unsigned char value = MI->getOperand(Op).getImm();
  assert(value <= 7 && "Invalid ssecc argument!");
  switch (value) {
  case 0: O << "eq"; break;
  case 1: O << "lt"; break;
  case 2: O << "le"; break;
  case 3: O << "unord"; break;
  case 4: O << "neq"; break;
  case 5: O << "nlt"; break;
  case 6: O << "nle"; break;
  case 7: O << "ord"; break;
  }
}

void X86IntelAsmPrinter::printOp(const MachineOperand &MO,
                                 const char *Modifier) {
  switch (MO.getType()) {
  case MachineOperand::MO_Register: {
    if (TargetRegisterInfo::isPhysicalRegister(MO.getReg())) {
      unsigned Reg = MO.getReg();
      if (Modifier && strncmp(Modifier, "subreg", strlen("subreg")) == 0) {
        MVT VT = (strcmp(Modifier,"subreg64") == 0) ?
          MVT::i64 : ((strcmp(Modifier, "subreg32") == 0) ? MVT::i32 :
                      ((strcmp(Modifier,"subreg16") == 0) ? MVT::i16 :MVT::i8));
        Reg = getX86SubSuperRegister(Reg, VT);
      }
      O << TRI->getName(Reg);
    } else
      O << "reg" << MO.getReg();
    return;
  }
  case MachineOperand::MO_Immediate:
    O << MO.getImm();
    return;
  case MachineOperand::MO_MachineBasicBlock:
    printBasicBlockLabel(MO.getMBB());
    return;
  case MachineOperand::MO_JumpTableIndex: {
    bool isMemOp  = Modifier && !strcmp(Modifier, "mem");
    if (!isMemOp) O << "OFFSET ";
    O << TAI->getPrivateGlobalPrefix() << "JTI" << getFunctionNumber()
      << "_" << MO.getIndex();
    return;
  }
  case MachineOperand::MO_ConstantPoolIndex: {
    bool isMemOp  = Modifier && !strcmp(Modifier, "mem");
    if (!isMemOp) O << "OFFSET ";
    O << "[" << TAI->getPrivateGlobalPrefix() << "CPI"
      << getFunctionNumber() << "_" << MO.getIndex();
    printOffset(MO.getOffset());
    O << "]";
    return;
  }
  case MachineOperand::MO_GlobalAddress: {
    bool isCallOp = Modifier && !strcmp(Modifier, "call");
    bool isMemOp  = Modifier && !strcmp(Modifier, "mem");
    GlobalValue *GV = MO.getGlobal();
    std::string Name = Mang->getValueName(GV);

    decorateName(Name, GV);

    if (!isMemOp && !isCallOp) O << "OFFSET ";
    if (GV->hasDLLImportLinkage()) {
      // FIXME: This should be fixed with full support of stdcall & fastcall
      // CC's
      O << "__imp_";
    }
    O << Name;
    printOffset(MO.getOffset());
    return;
  }
  case MachineOperand::MO_ExternalSymbol: {
    bool isCallOp = Modifier && !strcmp(Modifier, "call");
    if (!isCallOp) O << "OFFSET ";
    O << TAI->getGlobalPrefix() << MO.getSymbolName();
    return;
  }
  default:
    O << "<unknown operand type>"; return;
  }
}

void X86IntelAsmPrinter::printMemReference(const MachineInstr *MI, unsigned Op,
                                           const char *Modifier) {
  assert(isMem(MI, Op) && "Invalid memory reference!");

  const MachineOperand &BaseReg  = MI->getOperand(Op);
  int ScaleVal                   = MI->getOperand(Op+1).getImm();
  const MachineOperand &IndexReg = MI->getOperand(Op+2);
  const MachineOperand &DispSpec = MI->getOperand(Op+3);

  O << "[";
  bool NeedPlus = false;
  if (BaseReg.getReg()) {
    printOp(BaseReg, Modifier);
    NeedPlus = true;
  }

  if (IndexReg.getReg()) {
    if (NeedPlus) O << " + ";
    if (ScaleVal != 1)
      O << ScaleVal << "*";
    printOp(IndexReg, Modifier);
    NeedPlus = true;
  }

  if (DispSpec.isGlobal() || DispSpec.isCPI() ||
      DispSpec.isJTI()) {
    if (NeedPlus)
      O << " + ";
    printOp(DispSpec, "mem");
  } else {
    int DispVal = DispSpec.getImm();
    if (DispVal || (!BaseReg.getReg() && !IndexReg.getReg())) {
      if (NeedPlus) {
        if (DispVal > 0)
          O << " + ";
        else {
          O << " - ";
          DispVal = -DispVal;
        }
      }
      O << DispVal;
    }
  }
  O << "]";
}

void X86IntelAsmPrinter::printPICJumpTableSetLabel(unsigned uid,
                                           const MachineBasicBlock *MBB) const {
  if (!TAI->getSetDirective())
    return;

  O << TAI->getSetDirective() << ' ' << TAI->getPrivateGlobalPrefix()
    << getFunctionNumber() << '_' << uid << "_set_" << MBB->getNumber() << ',';
  printBasicBlockLabel(MBB, false, false, false);
  O << '-' << "\"L" << getFunctionNumber() << "$pb\"'\n";
}

void X86IntelAsmPrinter::printPICLabel(const MachineInstr *MI, unsigned Op) {
  O << "\"L" << getFunctionNumber() << "$pb\"\n";
  O << "\"L" << getFunctionNumber() << "$pb\":";
}

bool X86IntelAsmPrinter::printAsmMRegister(const MachineOperand &MO,
                                           const char Mode) {
  unsigned Reg = MO.getReg();
  switch (Mode) {
  default: return true;  // Unknown mode.
  case 'b': // Print QImode register
    Reg = getX86SubSuperRegister(Reg, MVT::i8);
    break;
  case 'h': // Print QImode high register
    Reg = getX86SubSuperRegister(Reg, MVT::i8, true);
    break;
  case 'w': // Print HImode register
    Reg = getX86SubSuperRegister(Reg, MVT::i16);
    break;
  case 'k': // Print SImode register
    Reg = getX86SubSuperRegister(Reg, MVT::i32);
    break;
  }

  O << '%' << TRI->getName(Reg);
  return false;
}

/// PrintAsmOperand - Print out an operand for an inline asm expression.
///
bool X86IntelAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                         unsigned AsmVariant,
                                         const char *ExtraCode) {
  // Does this asm operand have a single letter operand modifier?
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0) return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default: return true;  // Unknown modifier.
    case 'b': // Print QImode register
    case 'h': // Print QImode high register
    case 'w': // Print HImode register
    case 'k': // Print SImode register
      return printAsmMRegister(MI->getOperand(OpNo), ExtraCode[0]);
    }
  }

  printOperand(MI, OpNo);
  return false;
}

bool X86IntelAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                               unsigned OpNo,
                                               unsigned AsmVariant,
                                               const char *ExtraCode) {
  if (ExtraCode && ExtraCode[0])
    return true; // Unknown modifier.
  printMemReference(MI, OpNo);
  return false;
}

/// printMachineInstruction -- Print out a single X86 LLVM instruction
/// MI in Intel syntax to the current output stream.
///
void X86IntelAsmPrinter::printMachineInstruction(const MachineInstr *MI) {
  ++EmittedInsts;

  // Call the autogenerated instruction printer routines.
  printInstruction(MI);
}

bool X86IntelAsmPrinter::doInitialization(Module &M) {
  bool Result = AsmPrinter::doInitialization(M);

  Mang->markCharUnacceptable('.');

  O << "\t.686\n\t.model flat\n\n";

  // Emit declarations for external functions.
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (I->isDeclaration()) {
      std::string Name = Mang->getValueName(I);
      decorateName(Name, I);

      O << "\textern " ;
      if (I->hasDLLImportLinkage()) {
        O << "__imp_";
      }
      O << Name << ":near\n";
    }

  // Emit declarations for external globals.  Note that VC++ always declares
  // external globals to have type byte, and if that's good enough for VC++...
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    if (I->isDeclaration()) {
      std::string Name = Mang->getValueName(I);

      O << "\textern " ;
      if (I->hasDLLImportLinkage()) {
        O << "__imp_";
      }
      O << Name << ":byte\n";
    }
  }

  return Result;
}

bool X86IntelAsmPrinter::doFinalization(Module &M) {
  const TargetData *TD = TM.getTargetData();

  // Print out module-level global variables here.
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    if (I->isDeclaration()) continue;   // External global require no code

    // Check to see if this is a special global used by LLVM, if so, emit it.
    if (EmitSpecialLLVMGlobal(I))
      continue;

    std::string name = Mang->getValueName(I);
    Constant *C = I->getInitializer();
    unsigned Align = TD->getPreferredAlignmentLog(I);
    bool bCustomSegment = false;

    switch (I->getLinkage()) {
    case GlobalValue::CommonLinkage:
    case GlobalValue::LinkOnceLinkage:
    case GlobalValue::WeakLinkage:
      SwitchToDataSection("");
      O << name << "?\tsegment common 'COMMON'\n";
      bCustomSegment = true;
      // FIXME: the default alignment is 16 bytes, but 1, 2, 4, and 256
      // are also available.
      break;
    case GlobalValue::AppendingLinkage:
      SwitchToDataSection("");
      O << name << "?\tsegment public 'DATA'\n";
      bCustomSegment = true;
      // FIXME: the default alignment is 16 bytes, but 1, 2, 4, and 256
      // are also available.
      break;
    case GlobalValue::DLLExportLinkage:
      DLLExportedGVs.insert(name);
      // FALL THROUGH
    case GlobalValue::ExternalLinkage:
      O << "\tpublic " << name << "\n";
      // FALL THROUGH
    case GlobalValue::InternalLinkage:
      SwitchToSection(TAI->getDataSection());
      break;
    default:
      assert(0 && "Unknown linkage type!");
    }

    if (!bCustomSegment)
      EmitAlignment(Align, I);

    O << name << ":\t\t\t\t" << TAI->getCommentString()
      << " " << I->getName() << '\n';

    EmitGlobalConstant(C);

    if (bCustomSegment)
      O << name << "?\tends\n";
  }

    // Output linker support code for dllexported globals
  if (!DLLExportedGVs.empty() || !DLLExportedFns.empty()) {
    SwitchToDataSection("");
    O << "; WARNING: The following code is valid only with MASM v8.x"
      << "and (possible) higher\n"
      << "; This version of MASM is usually shipped with Microsoft "
      << "Visual Studio 2005\n"
      << "; or (possible) further versions. Unfortunately, there is no "
      << "way to support\n"
      << "; dllexported symbols in the earlier versions of MASM in fully "
      << "automatic way\n\n";
    O << "_drectve\t segment info alias('.drectve')\n";
  }

  for (StringSet<>::iterator i = DLLExportedGVs.begin(),
         e = DLLExportedGVs.end();
         i != e; ++i)
    O << "\t db ' /EXPORT:" << i->getKeyData() << ",data'\n";

  for (StringSet<>::iterator i = DLLExportedFns.begin(),
         e = DLLExportedFns.end();
         i != e; ++i)
    O << "\t db ' /EXPORT:" << i->getKeyData() << "'\n";

  if (!DLLExportedGVs.empty() || !DLLExportedFns.empty())
    O << "_drectve\t ends\n";

  // Bypass X86SharedAsmPrinter::doFinalization().
  bool Result = AsmPrinter::doFinalization(M);
  SwitchToDataSection("");
  O << "\tend\n";
  return Result;
}

void X86IntelAsmPrinter::EmitString(const ConstantArray *CVA) const {
  unsigned NumElts = CVA->getNumOperands();
  if (NumElts) {
    // ML does not have escape sequences except '' for '.  It also has a maximum
    // string length of 255.
    unsigned len = 0;
    bool inString = false;
    for (unsigned i = 0; i < NumElts; i++) {
      int n = cast<ConstantInt>(CVA->getOperand(i))->getZExtValue() & 255;
      if (len == 0)
        O << "\tdb ";

      if (n >= 32 && n <= 127) {
        if (!inString) {
          if (len > 0) {
            O << ",'";
            len += 2;
          } else {
            O << "'";
            len++;
          }
          inString = true;
        }
        if (n == '\'') {
          O << "'";
          len++;
        }
        O << char(n);
      } else {
        if (inString) {
          O << "'";
          len++;
          inString = false;
        }
        if (len > 0) {
          O << ",";
          len++;
        }
        O << n;
        len += 1 + (n > 9) + (n > 99);
      }

      if (len > 60) {
        if (inString) {
          O << "'";
          inString = false;
        }
        O << "\n";
        len = 0;
      }
    }

    if (len > 0) {
      if (inString)
        O << "'";
      O << "\n";
    }
  }
}

// Include the auto-generated portion of the assembly writer.
#include "X86GenAsmWriter1.inc"
