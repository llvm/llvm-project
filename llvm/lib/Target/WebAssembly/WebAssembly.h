//===-- WebAssembly.h - Top-level interface for WebAssembly  ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the entry points for global functions defined in
/// the LLVM WebAssembly back-end.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLY_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLY_H

#include "llvm/Support/CodeGen.h"

namespace llvm {

class WebAssemblyTargetMachine;
class FunctionPass;

FunctionPass *createWebAssemblyISelDag(WebAssemblyTargetMachine &TM,
                                       CodeGenOpt::Level OptLevel);

FunctionPass *createWebAssemblyCFGStackify();

FunctionPass *createWebAssemblyRelooper();

} // end namespace llvm

#endif
