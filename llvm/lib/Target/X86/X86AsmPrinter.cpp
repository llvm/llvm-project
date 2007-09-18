//===-- X86AsmPrinter.cpp - Convert X86 LLVM IR to X86 assembly -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file the shared super class printer that converts from our internal
// representation of machine-dependent LLVM code to Intel and AT&T format
// assembly language.
// This printer is the output mechanism used by `llc'.
//
//===----------------------------------------------------------------------===//

#include "X86AsmPrinter.h"
#include "X86ATTAsmPrinter.h"
#include "X86COFF.h"
#include "X86IntelAsmPrinter.h"
#include "X86MachineFunctionInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CallingConv.h"
#include "llvm/Constants.h"
#include "llvm/Module.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Type.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

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

  for (Function::const_arg_iterator AI = F->arg_begin(), AE = F->arg_end();
       AI != AE; ++AI)
    // Size should be aligned to DWORD boundary
    Size += ((TD->getTypeSize(AI->getType()) + 3)/4)*4;
  
  // We're not supporting tooooo huge arguments :)
  Info.setBytesToPopOnReturn((unsigned int)Size);
  return Info;
}


/// decorateName - Query FunctionInfoMap and use this information for various
/// name decoration.
void X86SharedAsmPrinter::decorateName(std::string &Name,
                                       const GlobalValue *GV) {
  const Function *F = dyn_cast<Function>(GV);
  if (!F) return;

  // We don't want to decorate non-stdcall or non-fastcall functions right now
  unsigned CC = F->getCallingConv();
  if (CC != CallingConv::X86_StdCall && CC != CallingConv::X86_FastCall)
    return;

  // Decorate names only when we're targeting Cygwin/Mingw32 targets
  if (!Subtarget->isTargetCygMing())
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
        (FT->getNumParams() == 1 && FT->isStructReturn())) 
      Name += '@' + utostr_32(Info->getBytesToPopOnReturn());
    break;
  case FastCall:
    // "Pure" variadic functions do not receive @0 suffix.
    if (!FT->isVarArg() || (FT->getNumParams() == 0) ||
        (FT->getNumParams() == 1 && FT->isStructReturn())) 
      Name += '@' + utostr_32(Info->getBytesToPopOnReturn());

    if (Name[0] == '_') {
      Name[0] = '@';
    } else {
      Name = '@' + Name;
    }    
    break;
  default:
    assert(0 && "Unsupported DecorationStyle");
  }
}

/// doInitialization
bool X86SharedAsmPrinter::doInitialization(Module &M) {
  if (TAI->doesSupportDebugInformation()) {
    // Emit initial debug information.
    DW.BeginModule(&M);
  }

  bool Result = AsmPrinter::doInitialization(M);

  // Darwin wants symbols to be quoted if they have complex names.
  if (Subtarget->isTargetDarwin())
    Mang->setUseQuotes(true);

  return Result;
}

bool X86SharedAsmPrinter::doFinalization(Module &M) {
  // Note: this code is not shared by the Intel printer as it is too different
  // from how MASM does things.  When making changes here don't forget to look
  // at X86IntelAsmPrinter::doFinalization().
  const TargetData *TD = TM.getTargetData();
  
  // Print out module-level global variables here.
  for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I) {
    if (!I->hasInitializer())
      continue;   // External global require no code
    
    // Check to see if this is a special global used by LLVM, if so, emit it.
    if (EmitSpecialLLVMGlobal(I)) {
      if (Subtarget->isTargetDarwin() &&
          TM.getRelocationModel() == Reloc::Static) {
        if (I->getName() == "llvm.global_ctors")
          O << ".reference .constructors_used\n";
        else if (I->getName() == "llvm.global_dtors")
          O << ".reference .destructors_used\n";
      }
      continue;
    }
    
    std::string name = Mang->getValueName(I);
    Constant *C = I->getInitializer();
    const Type *Type = C->getType();
    unsigned Size = TD->getTypeSize(Type);
    unsigned Align = TD->getPreferredAlignmentLog(I);

    if (I->hasHiddenVisibility()) {
      if (const char *Directive = TAI->getHiddenDirective())
        O << Directive << name << "\n";
    } else if (I->hasProtectedVisibility()) {
      if (const char *Directive = TAI->getProtectedDirective())
        O << Directive << name << "\n";
    }
    
    if (Subtarget->isTargetELF())
      O << "\t.type\t" << name << ",@object\n";
    
    if (C->isNullValue()) {
      if (I->hasExternalLinkage()) {
        if (const char *Directive = TAI->getZeroFillDirective()) {
          O << "\t.globl\t" << name << "\n";
          O << Directive << "__DATA__, __common, " << name << ", "
            << Size << ", " << Align << "\n";
          continue;
        }
      }
      
      if (!I->hasSection() && !I->isThreadLocal() &&
          (I->hasInternalLinkage() || I->hasWeakLinkage() ||
           I->hasLinkOnceLinkage())) {
        if (Size == 0) Size = 1;   // .comm Foo, 0 is undefined, avoid it.
        if (!NoZerosInBSS && TAI->getBSSSection())
          SwitchToDataSection(TAI->getBSSSection(), I);
        else
          SwitchToDataSection(TAI->getDataSection(), I);
        if (TAI->getLCOMMDirective() != NULL) {
          if (I->hasInternalLinkage()) {
            O << TAI->getLCOMMDirective() << name << "," << Size;
            if (Subtarget->isTargetDarwin())
              O << "," << Align;
          } else
            O << TAI->getCOMMDirective()  << name << "," << Size;
        } else {
          if (!Subtarget->isTargetCygMing()) {
            if (I->hasInternalLinkage())
              O << "\t.local\t" << name << "\n";
          }
          O << TAI->getCOMMDirective()  << name << "," << Size;
          if (TAI->getCOMMDirectiveTakesAlignment())
            O << "," << (TAI->getAlignmentIsInBytes() ? (1 << Align) : Align);
        }
        O << "\t\t" << TAI->getCommentString() << " " << I->getName() << "\n";
        continue;
      }
    }

    switch (I->getLinkage()) {
    case GlobalValue::LinkOnceLinkage:
    case GlobalValue::WeakLinkage:
      if (Subtarget->isTargetDarwin()) {
        O << "\t.globl " << name << "\n"
          << "\t.weak_definition " << name << "\n";
        SwitchToDataSection(".section __DATA,__const_coal,coalesced", I);
      } else if (Subtarget->isTargetCygMing()) {
        std::string SectionName(".section\t.data$linkonce." +
                                name +
                                ",\"aw\"");
        SwitchToDataSection(SectionName.c_str(), I);
        O << "\t.globl " << name << "\n"
          << "\t.linkonce same_size\n";
      } else {
        std::string SectionName("\t.section\t.llvm.linkonce.d." +
                                name +
                                ",\"aw\",@progbits");
        SwitchToDataSection(SectionName.c_str(), I);
        O << "\t.weak\t" << name << "\n";
      }
      break;
    case GlobalValue::DLLExportLinkage:
      DLLExportedGVs.insert(Mang->makeNameProper(I->getName(),""));
      // FALL THROUGH
    case GlobalValue::AppendingLinkage:
      // FIXME: appending linkage variables should go into a section of
      // their name or something.  For now, just emit them as external.
    case GlobalValue::ExternalLinkage:
      // If external or appending, declare as a global symbol
      O << "\t.globl " << name << "\n";
      // FALL THROUGH
    case GlobalValue::InternalLinkage: {
      if (I->isConstant()) {
        const ConstantArray *CVA = dyn_cast<ConstantArray>(C);
        if (TAI->getCStringSection() && CVA && CVA->isCString()) {
          SwitchToDataSection(TAI->getCStringSection(), I);
          break;
        }
      }
      // FIXME: special handling for ".ctors" & ".dtors" sections
      if (I->hasSection() &&
          (I->getSection() == ".ctors" ||
           I->getSection() == ".dtors")) {
        std::string SectionName = ".section " + I->getSection();
        
        if (Subtarget->isTargetCygMing()) {
          SectionName += ",\"aw\"";
        } else {
          assert(!Subtarget->isTargetDarwin());
          SectionName += ",\"aw\",@progbits";
        }

        SwitchToDataSection(SectionName.c_str());
      } else {
        if (C->isNullValue() && !NoZerosInBSS && TAI->getBSSSection())
          SwitchToDataSection(I->isThreadLocal() ? TAI->getTLSBSSSection() :
                              TAI->getBSSSection(), I);
        else if (!I->isConstant())
          SwitchToDataSection(I->isThreadLocal() ? TAI->getTLSDataSection() :
                              TAI->getDataSection(), I);
        else if (I->isThreadLocal())
          SwitchToDataSection(TAI->getTLSDataSection());
        else {
          // Read-only data.
          bool HasReloc = C->ContainsRelocations();
          if (HasReloc &&
              Subtarget->isTargetDarwin() &&
              TM.getRelocationModel() != Reloc::Static)
            SwitchToDataSection("\t.const_data\n");
          else if (!HasReloc && Size == 4 &&
                   TAI->getFourByteConstantSection())
            SwitchToDataSection(TAI->getFourByteConstantSection(), I);
          else if (!HasReloc && Size == 8 &&
                   TAI->getEightByteConstantSection())
            SwitchToDataSection(TAI->getEightByteConstantSection(), I);
          else if (!HasReloc && Size == 16 &&
                   TAI->getSixteenByteConstantSection())
            SwitchToDataSection(TAI->getSixteenByteConstantSection(), I);
          else if (TAI->getReadOnlySection())
            SwitchToDataSection(TAI->getReadOnlySection(), I);
          else
            SwitchToDataSection(TAI->getDataSection(), I);
        }
      }
      
      break;
    }
    default:
      assert(0 && "Unknown linkage type!");
    }

    EmitAlignment(Align, I);
    O << name << ":\t\t\t\t" << TAI->getCommentString() << " " << I->getName()
      << "\n";
    if (TAI->hasDotTypeDotSizeDirective())
      O << "\t.size\t" << name << ", " << Size << "\n";
    // If the initializer is a extern weak symbol, remember to emit the weak
    // reference!
    if (const GlobalValue *GV = dyn_cast<GlobalValue>(C))
      if (GV->hasExternalWeakLinkage())
        ExtWeakSymbols.insert(GV);

    EmitGlobalConstant(C);
  }
  
  // Output linker support code for dllexported globals
  if (DLLExportedGVs.begin() != DLLExportedGVs.end()) {
    SwitchToDataSection(".section .drectve");
  }

  for (std::set<std::string>::iterator i = DLLExportedGVs.begin(),
         e = DLLExportedGVs.end();
         i != e; ++i) {
    O << "\t.ascii \" -export:" << *i << ",data\"\n";
  }    

  if (DLLExportedFns.begin() != DLLExportedFns.end()) {
    SwitchToDataSection(".section .drectve");
  }

  for (std::set<std::string>::iterator i = DLLExportedFns.begin(),
         e = DLLExportedFns.end();
         i != e; ++i) {
    O << "\t.ascii \" -export:" << *i << "\"\n";
  }    

  if (Subtarget->isTargetDarwin()) {
    SwitchToDataSection("");

    // Output stubs for dynamically-linked functions
    unsigned j = 1;
    for (std::set<std::string>::iterator i = FnStubs.begin(), e = FnStubs.end();
         i != e; ++i, ++j) {
      SwitchToDataSection(".section __IMPORT,__jump_table,symbol_stubs,"
                          "self_modifying_code+pure_instructions,5", 0);
      O << "L" << *i << "$stub:\n";
      O << "\t.indirect_symbol " << *i << "\n";
      O << "\thlt ; hlt ; hlt ; hlt ; hlt\n";
    }

    O << "\n";

    if (ExceptionHandling && TAI->doesSupportExceptionHandling() && MMI) {
      // Add the (possibly multiple) personalities to the set of global values.
      const std::vector<Function *>& Personalities = MMI->getPersonalities();

      for (std::vector<Function *>::const_iterator I = Personalities.begin(),
             E = Personalities.end(); I != E; ++I)
        if (*I) GVStubs.insert("_" + (*I)->getName());
    }

    // Output stubs for external and common global variables.
    if (GVStubs.begin() != GVStubs.end())
      SwitchToDataSection(
                    ".section __IMPORT,__pointers,non_lazy_symbol_pointers");
    for (std::set<std::string>::iterator i = GVStubs.begin(), e = GVStubs.end();
         i != e; ++i) {
      O << "L" << *i << "$non_lazy_ptr:\n";
      O << "\t.indirect_symbol " << *i << "\n";
      O << "\t.long\t0\n";
    }

    // Emit final debug information.
    DW.EndModule();

    // Funny Darwin hack: This flag tells the linker that no global symbols
    // contain code that falls through to other global symbols (e.g. the obvious
    // implementation of multiple entry points).  If this doesn't occur, the
    // linker can safely perform dead code stripping.  Since LLVM never
    // generates code that does this, it is always safe to set.
    O << "\t.subsections_via_symbols\n";
  } else if (Subtarget->isTargetCygMing()) {
    // Emit type information for external functions
    for (std::set<std::string>::iterator i = FnStubs.begin(), e = FnStubs.end();
         i != e; ++i) {
      O << "\t.def\t " << *i
        << ";\t.scl\t" << COFF::C_EXT
        << ";\t.type\t" << (COFF::DT_FCN << COFF::N_BTSHFT)
        << ";\t.endef\n";
    }
    
    // Emit final debug information.
    DW.EndModule();    
  } else if (Subtarget->isTargetELF()) {
    // Emit final debug information.
    DW.EndModule();
  }

  return AsmPrinter::doFinalization(M);
}

/// createX86CodePrinterPass - Returns a pass that prints the X86 assembly code
/// for a MachineFunction to the given output stream, using the given target
/// machine description.
///
FunctionPass *llvm::createX86CodePrinterPass(std::ostream &o,
                                             X86TargetMachine &tm) {
  const X86Subtarget *Subtarget = &tm.getSubtarget<X86Subtarget>();

  if (Subtarget->isFlavorIntel()) {
    return new X86IntelAsmPrinter(o, tm, tm.getTargetAsmInfo());
  } else {
    return new X86ATTAsmPrinter(o, tm, tm.getTargetAsmInfo());
  }
}
