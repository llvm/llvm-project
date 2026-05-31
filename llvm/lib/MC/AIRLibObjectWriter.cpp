//===- llvm/MC/AIRLibObjectWriter.cpp - AIRLib Writer -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCAIRLibObjectWriter.h"
#include "llvm/MC/MCSection.h"

using namespace llvm;

MCAIRLibTargetWriter::~MCAIRLibTargetWriter() = default;

uint64_t AIRLibObjectWriter::writeObject() {
  auto &Asm = *this->Asm;
  // The AIR `.metallib` container is monolithic: AIREmbedderPass packs
  // the entire serialized metallib into a single MCSection. Multiple
  // non-empty sections would mean the embedder produced something this
  // writer cannot represent, so require exactly one.
  const MCSection *Payload = nullptr;
  for (const MCSection &Sec : Asm) {
    if (Asm.getSectionAddressSize(Sec) == 0)
      continue;
    assert(!Payload && "AIRLibObjectWriter: more than one non-empty section; "
                       "the .metallib container is monolithic by design");
    Payload = &Sec;
  }
  if (Payload)
    Asm.writeSectionData(W.OS, Payload);
  return 0;
}
