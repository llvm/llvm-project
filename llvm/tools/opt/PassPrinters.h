//=- PassPrinters.h - Utilities to print analysis info for passes -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Utilities to print analysis info for various kinds of passes.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_OPT_PASSPRINTERS_H
#define LLVM_TOOLS_OPT_PASSPRINTERS_H

namespace llvm {

class CallGraphSCCPass;
class FunctionPass;
class ModulePass;
class LoopPass;
class PassInfo;
class raw_ostream;
class RegionPass;

FunctionPass *createFunctionPassPrinter(const PassInfo *PI, raw_ostream &out);

CallGraphSCCPass *createCallGraphPassPrinter(const PassInfo *PI,
                                             raw_ostream &out);

ModulePass *createModulePassPrinter(const PassInfo *PI, raw_ostream &out);

LoopPass *createLoopPassPrinter(const PassInfo *PI, raw_ostream &out);

RegionPass *createRegionPassPrinter(const PassInfo *PI, raw_ostream &out);

} // end namespace llvm

#endif // LLVM_TOOLS_OPT_PASSPRINTERS_H
