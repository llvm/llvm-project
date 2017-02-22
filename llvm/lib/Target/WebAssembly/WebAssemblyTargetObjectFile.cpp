//===-- WebAssemblyTargetObjectFile.cpp - WebAssembly Object Info ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file defines the functions of the WebAssembly-specific subclass
/// of TargetLoweringObjectFile.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyTargetObjectFile.h"
#include "WebAssemblyTargetMachine.h"
using namespace llvm;

void WebAssemblyTargetObjectFileELF::Initialize(MCContext &Ctx,
                                                const TargetMachine &TM) {
  TargetLoweringObjectFileELF::Initialize(Ctx, TM);
  InitializeELF(TM.Options.UseInitArray);
}

void WebAssemblyTargetObjectFile::Initialize(MCContext &Ctx,
                                             const TargetMachine &TM) {
  TargetLoweringObjectFileWasm::Initialize(Ctx, TM);
  InitializeWasm();
}
