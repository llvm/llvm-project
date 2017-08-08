//===-- MCAsmInfoWasm.cpp - Wasm asm properties -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take in general on Wasm-based targets
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmInfoWasm.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionWasm.h"
using namespace llvm;

void MCAsmInfoWasm::anchor() { }

MCAsmInfoWasm::MCAsmInfoWasm() {
  HasIdentDirective = true;
  WeakRefDirective = "\t.weak\t";
  PrivateGlobalPrefix = ".L";
  PrivateLabelPrefix = ".L";
}
