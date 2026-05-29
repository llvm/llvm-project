//===- llvm/MC/MetalLibObjectWriter.cpp - MetalLib Writer -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCMetalLibObjectWriter.h"
#include "llvm/MC/MCSection.h"

using namespace llvm;

MCMetalLibTargetWriter::~MCMetalLibTargetWriter() = default;

uint64_t MetalLibObjectWriter::writeObject() {
  auto &Asm = *this->Asm;
  // The Metal `.metallib` container is monolithic: MetalEmbedderPass packs
  // the entire serialized metallib into a single MCSection. Multiple
  // non-empty sections would mean the embedder produced something this
  // writer cannot represent, so require exactly one.
  const MCSection *Payload = nullptr;
  for (const MCSection &Sec : Asm) {
    if (Asm.getSectionAddressSize(Sec) == 0)
      continue;
    assert(!Payload && "MetalLibObjectWriter: more than one non-empty section; "
                       "the .metallib container is monolithic by design");
    Payload = &Sec;
  }
  if (Payload)
    Asm.writeSectionData(W.OS, Payload);
  return 0;
}
