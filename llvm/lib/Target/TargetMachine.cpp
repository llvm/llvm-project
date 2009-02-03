//===-- TargetMachine.cpp - General Target Information ---------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the general parts of a Target machine.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

//---------------------------------------------------------------------------
// Command-line options that tend to be useful on more than one back-end.
//

namespace llvm {
  bool PrintMachineCode;
  bool NoFramePointerElim;
  bool NoExcessFPPrecision;
  bool UnsafeFPMath;
  bool FiniteOnlyFPMathOption;
  bool HonorSignDependentRoundingFPMathOption;
  bool UseSoftFloat;
  bool NoZerosInBSS;
  bool ExceptionHandling;
  bool UnwindTablesMandatory;
  Reloc::Model RelocationModel;
  CodeModel::Model CMModel;
  bool PerformTailCallOpt;
  unsigned StackAlignment;
  bool RealignStack;
  bool VerboseAsm;
  bool DisableJumpTables;
  bool StrongPHIElim;
  bool DisableRedZone;
}

static cl::opt<bool, true> PrintCode("print-machineinstrs",
  cl::desc("Print generated machine code"),
  cl::location(PrintMachineCode), cl::init(false));

static cl::opt<bool, true>
DisableFPElim("disable-fp-elim",
              cl::desc("Disable frame pointer elimination optimization"),
              cl::location(NoFramePointerElim),
              cl::init(false));
static cl::opt<bool, true>
DisableExcessPrecision("disable-excess-fp-precision",
             cl::desc("Disable optimizations that may increase FP precision"),
             cl::location(NoExcessFPPrecision),
             cl::init(false));
static cl::opt<bool, true>
EnableUnsafeFPMath("enable-unsafe-fp-math",
             cl::desc("Enable optimizations that may decrease FP precision"),
             cl::location(UnsafeFPMath),
             cl::init(false));
static cl::opt<bool, true>
EnableFiniteOnlyFPMath("enable-finite-only-fp-math",
             cl::desc("Enable optimizations that assumes non- NaNs / +-Infs"),
             cl::location(FiniteOnlyFPMathOption),
             cl::init(false));
static cl::opt<bool, true>
EnableHonorSignDependentRoundingFPMath(cl::Hidden,
             "enable-sign-dependent-rounding-fp-math",
     cl::desc("Force codegen to assume rounding mode can change dynamically"),
             cl::location(HonorSignDependentRoundingFPMathOption),
             cl::init(false));

static cl::opt<bool, true>
GenerateSoftFloatCalls("soft-float",
             cl::desc("Generate software floating point library calls"),
             cl::location(UseSoftFloat),
             cl::init(false));
static cl::opt<bool, true>
DontPlaceZerosInBSS("nozero-initialized-in-bss",
            cl::desc("Don't place zero-initialized symbols into bss section"),
            cl::location(NoZerosInBSS),
            cl::init(false));
static cl::opt<bool, true>
EnableExceptionHandling("enable-eh",
             cl::desc("Emit DWARF exception handling (default if target supports)"),
             cl::location(ExceptionHandling),
             cl::init(false));
static cl::opt<bool, true>
EnableUnwindTables("unwind-tables",
             cl::desc("Generate unwinding tables for all functions"),
             cl::location(UnwindTablesMandatory),
             cl::init(false));

static cl::opt<llvm::Reloc::Model, true>
DefRelocationModel(
  "relocation-model",
  cl::desc("Choose relocation model"),
  cl::location(RelocationModel),
  cl::init(Reloc::Default),
  cl::values(
    clEnumValN(Reloc::Default, "default",
               "Target default relocation model"),
    clEnumValN(Reloc::Static, "static",
               "Non-relocatable code"),
    clEnumValN(Reloc::PIC_, "pic",
               "Fully relocatable, position independent code"),
    clEnumValN(Reloc::DynamicNoPIC, "dynamic-no-pic",
               "Relocatable external references, non-relocatable code"),
    clEnumValEnd));
static cl::opt<llvm::CodeModel::Model, true>
DefCodeModel(
  "code-model",
  cl::desc("Choose code model"),
  cl::location(CMModel),
  cl::init(CodeModel::Default),
  cl::values(
    clEnumValN(CodeModel::Default, "default",
               "Target default code model"),
    clEnumValN(CodeModel::Small, "small",
               "Small code model"),
    clEnumValN(CodeModel::Kernel, "kernel",
               "Kernel code model"),
    clEnumValN(CodeModel::Medium, "medium",
               "Medium code model"),
    clEnumValN(CodeModel::Large, "large",
               "Large code model"),
    clEnumValEnd));

static cl::opt<bool, true>
EnablePerformTailCallOpt("tailcallopt",
                         cl::desc("Turn on tail call optimization."),
                         cl::location(PerformTailCallOpt),
                         cl::init(false));

static cl::opt<unsigned, true>
OverrideStackAlignment("stack-alignment",
                       cl::desc("Override default stack alignment"),
                       cl::location(StackAlignment),
                       cl::init(0));

static cl::opt<bool, true>
EnableRealignStack("realign-stack",
                   cl::desc("Realign stack if needed"),
                   cl::location(RealignStack),
                   cl::init(true));

static cl::opt<bool, true>
AsmVerbose("asm-verbose", cl::desc("Add comments to directives."),
           cl::location(VerboseAsm),
           cl::init(false));

static cl::opt<bool, true>
DisableSwitchTables(cl::Hidden, "disable-jump-tables", 
           cl::desc("Do not generate jump tables."),
           cl::location(DisableJumpTables),
           cl::init(false));

static cl::opt<bool, true>
EnableStrongPHIElim(cl::Hidden, "strong-phi-elim",
           cl::desc("Use strong PHI elimination."),
           cl::location(StrongPHIElim),
           cl::init(false));

static cl::opt<bool, true>
DisableRedZoneOption("disable-red-zone",
           cl::desc("Do not emit code that uses the red zone."),
           cl::location(DisableRedZone),
           cl::init(false));

//---------------------------------------------------------------------------
// TargetMachine Class
//

TargetMachine::~TargetMachine() {
  delete AsmInfo;
}

/// getRelocationModel - Returns the code generation relocation model. The
/// choices are static, PIC, and dynamic-no-pic, and target default.
Reloc::Model TargetMachine::getRelocationModel() {
  return RelocationModel;
}

/// setRelocationModel - Sets the code generation relocation model.
void TargetMachine::setRelocationModel(Reloc::Model Model) {
  RelocationModel = Model;
}

/// getCodeModel - Returns the code model. The choices are small, kernel,
/// medium, large, and target default.
CodeModel::Model TargetMachine::getCodeModel() {
  return CMModel;
}

/// setCodeModel - Sets the code model.
void TargetMachine::setCodeModel(CodeModel::Model Model) {
  CMModel = Model;
}

namespace llvm {
  /// FiniteOnlyFPMath - This returns true when the -enable-finite-only-fp-math
  /// option is specified on the command line. If this returns false (default),
  /// the code generator is not allowed to assume that FP arithmetic arguments
  /// and results are never NaNs or +-Infs.
  bool FiniteOnlyFPMath() { return UnsafeFPMath || FiniteOnlyFPMathOption; }
  
  /// HonorSignDependentRoundingFPMath - Return true if the codegen must assume
  /// that the rounding mode of the FPU can change from its default.
  bool HonorSignDependentRoundingFPMath() {
    return !UnsafeFPMath && HonorSignDependentRoundingFPMathOption;
  }
}

