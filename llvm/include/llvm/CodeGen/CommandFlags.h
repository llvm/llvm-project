//===-- CommandFlags.h - Register Coalescing Interface ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains codegen-specific flags that are shared between different
// command line tools. The tools "llc" and "opt" both use this file to prevent
// flag duplication.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_COMMAND_LINE_FLAGS_H
#define LLVM_CODEGEN_COMMAND_LINE_FLAGS_H

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"

#include <string>
using namespace llvm;

cl::opt<std::string>
MArch("march", cl::desc("Architecture to generate code for (see --version)"));

cl::opt<std::string>
MCPU("mcpu",
     cl::desc("Target a specific cpu type (-mcpu=help for details)"),
     cl::value_desc("cpu-name"),
     cl::init(""));

cl::list<std::string>
MAttrs("mattr",
       cl::CommaSeparated,
       cl::desc("Target specific attributes (-mattr=help for details)"),
       cl::value_desc("a1,+a2,-a3,..."));

cl::opt<Reloc::Model>
RelocModel("relocation-model",
           cl::desc("Choose relocation model"),
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

cl::opt<llvm::CodeModel::Model>
CMModel("code-model",
        cl::desc("Choose code model"),
        cl::init(CodeModel::Default),
        cl::values(clEnumValN(CodeModel::Default, "default",
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

cl::opt<bool>
RelaxAll("mc-relax-all",
         cl::desc("When used with filetype=obj, "
                  "relax all fixups in the emitted object file"));

cl::opt<TargetMachine::CodeGenFileType>
FileType("filetype", cl::init(TargetMachine::CGFT_AssemblyFile),
  cl::desc("Choose a file type (not all types are supported by all targets):"),
  cl::values(
             clEnumValN(TargetMachine::CGFT_AssemblyFile, "asm",
                        "Emit an assembly ('.s') file"),
             clEnumValN(TargetMachine::CGFT_ObjectFile, "obj",
                        "Emit a native object ('.o') file"),
             clEnumValN(TargetMachine::CGFT_Null, "null",
                        "Emit nothing, for performance testing"),
             clEnumValEnd));

cl::opt<bool> DisableDotLoc("disable-dot-loc", cl::Hidden,
                            cl::desc("Do not use .loc entries"));

cl::opt<bool> DisableCFI("disable-cfi", cl::Hidden,
                         cl::desc("Do not use .cfi_* directives"));

cl::opt<bool> EnableDwarfDirectory("enable-dwarf-directory", cl::Hidden,
                  cl::desc("Use .file directives with an explicit directory."));

cl::opt<bool>
DisableRedZone("disable-red-zone",
               cl::desc("Do not emit code that uses the red zone."),
               cl::init(false));

cl::opt<bool>
EnableFPMAD("enable-fp-mad",
            cl::desc("Enable less precise MAD instructions to be generated"),
            cl::init(false));

cl::opt<bool>
DisableFPElim("disable-fp-elim",
              cl::desc("Disable frame pointer elimination optimization"),
              cl::init(false));

cl::opt<bool>
DisableFPElimNonLeaf("disable-non-leaf-fp-elim",
  cl::desc("Disable frame pointer elimination optimization for non-leaf funcs"),
  cl::init(false));

cl::opt<bool>
EnableUnsafeFPMath("enable-unsafe-fp-math",
                cl::desc("Enable optimizations that may decrease FP precision"),
                cl::init(false));

cl::opt<bool>
EnableNoInfsFPMath("enable-no-infs-fp-math",
                cl::desc("Enable FP math optimizations that assume no +-Infs"),
                cl::init(false));

cl::opt<bool>
EnableNoNaNsFPMath("enable-no-nans-fp-math",
                   cl::desc("Enable FP math optimizations that assume no NaNs"),
                   cl::init(false));

cl::opt<bool>
EnableHonorSignDependentRoundingFPMath("enable-sign-dependent-rounding-fp-math",
      cl::Hidden,
      cl::desc("Force codegen to assume rounding mode can change dynamically"),
      cl::init(false));

cl::opt<bool>
GenerateSoftFloatCalls("soft-float",
                    cl::desc("Generate software floating point library calls"),
                    cl::init(false));

cl::opt<llvm::FloatABI::ABIType>
FloatABIForCalls("float-abi",
                 cl::desc("Choose float ABI type"),
                 cl::init(FloatABI::Default),
                 cl::values(
                     clEnumValN(FloatABI::Default, "default",
                                "Target default float ABI type"),
                     clEnumValN(FloatABI::Soft, "soft",
                                "Soft float ABI (implied by -soft-float)"),
                     clEnumValN(FloatABI::Hard, "hard",
                                "Hard float ABI (uses FP registers)"),
                     clEnumValEnd));

cl::opt<llvm::FPOpFusion::FPOpFusionMode>
FuseFPOps("fp-contract",
          cl::desc("Enable aggresive formation of fused FP ops"),
          cl::init(FPOpFusion::Standard),
          cl::values(
              clEnumValN(FPOpFusion::Fast, "fast",
                         "Fuse FP ops whenever profitable"),
              clEnumValN(FPOpFusion::Standard, "on",
                         "Only fuse 'blessed' FP ops."),
              clEnumValN(FPOpFusion::Strict, "off",
                         "Only fuse FP ops when the result won't be effected."),
              clEnumValEnd));

cl::opt<bool>
DontPlaceZerosInBSS("nozero-initialized-in-bss",
              cl::desc("Don't place zero-initialized symbols into bss section"),
              cl::init(false));

cl::opt<bool>
EnableGuaranteedTailCallOpt("tailcallopt",
  cl::desc("Turn fastcc calls into tail calls by (potentially) changing ABI."),
  cl::init(false));

cl::opt<bool>
DisableTailCalls("disable-tail-calls",
                 cl::desc("Never emit tail calls"),
                 cl::init(false));

cl::opt<unsigned>
OverrideStackAlignment("stack-alignment",
                       cl::desc("Override default stack alignment"),
                       cl::init(0));

cl::opt<bool>
EnableRealignStack("realign-stack",
                   cl::desc("Realign stack if needed"),
                   cl::init(true));

cl::opt<std::string>
TrapFuncName("trap-func", cl::Hidden,
        cl::desc("Emit a call to trap function rather than a trap instruction"),
        cl::init(""));

cl::opt<bool>
EnablePIE("enable-pie",
          cl::desc("Assume the creation of a position independent executable."),
          cl::init(false));

cl::opt<bool>
SegmentedStacks("segmented-stacks",
                cl::desc("Use segmented stacks if possible."),
                cl::init(false));

cl::opt<bool>
UseInitArray("use-init-array",
             cl::desc("Use .init_array instead of .ctors."),
             cl::init(false));

cl::opt<std::string> StopAfter("stop-after",
                            cl::desc("Stop compilation after a specific pass"),
                            cl::value_desc("pass-name"),
                                      cl::init(""));
cl::opt<std::string> StartAfter("start-after",
                          cl::desc("Resume compilation after a specific pass"),
                          cl::value_desc("pass-name"),
                          cl::init(""));

cl::opt<unsigned>
SSPBufferSize("stack-protector-buffer-size", cl::init(8),
              cl::desc("Lower bound for a buffer to be considered for "
                       "stack protection"));
#endif
