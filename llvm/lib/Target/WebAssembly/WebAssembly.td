//- WebAssembly.td - Describe the WebAssembly Target Machine --*- tablegen -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This is a target description file for the WebAssembly architecture,
/// which is also known as "wasm".
///
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces which we are implementing
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// WebAssembly Subtarget features.
//===----------------------------------------------------------------------===//

def FeatureSIMD128 : SubtargetFeature<"simd128", "HasSIMD128", "true",
                                      "Enable 128-bit SIMD">;
def FeatureAtomics : SubtargetFeature<"atomics", "HasAtomics", "true",
                                      "Enable Atomics">;
def FeatureNontrappingFPToInt :
      SubtargetFeature<"nontrapping-fptoint",
                       "HasNontrappingFPToInt", "true",
                       "Enable non-trapping float-to-int conversion operators">;

def FeatureSignExt :
      SubtargetFeature<"sign-ext",
                       "HasSignExt", "true",
                       "Enable sign extension operators">;

def FeatureExceptionHandling :
      SubtargetFeature<"exception-handling", "HasExceptionHandling", "true",
                       "Enable Wasm exception handling">;

//===----------------------------------------------------------------------===//
// Architectures.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "WebAssemblyRegisterInfo.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "WebAssemblyInstrInfo.td"

def WebAssemblyInstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// WebAssembly Processors supported.
//===----------------------------------------------------------------------===//

// Minimal Viable Product.
def : ProcessorModel<"mvp", NoSchedModel, []>;

// Generic processor: latest stable version.
def : ProcessorModel<"generic", NoSchedModel, []>;

// Latest and greatest experimental version of WebAssembly. Bugs included!
def : ProcessorModel<"bleeding-edge", NoSchedModel,
                      [FeatureSIMD128, FeatureAtomics]>;

//===----------------------------------------------------------------------===//
// Target Declaration
//===----------------------------------------------------------------------===//

def WebAssemblyAsmParser : AsmParser {
  // The physical register names are not in the binary format or asm text
  let ShouldEmitMatchRegisterName = 0;
}

def WebAssembly : Target {
  let InstructionSet = WebAssemblyInstrInfo;
  let AssemblyParsers  = [WebAssemblyAsmParser];
}
