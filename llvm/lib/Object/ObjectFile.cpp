//===- ObjectFile.cpp - File format independent object file ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a file format independent ObjectFile class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ObjectFile.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/DXContainer.h"
#include "llvm/Object/Error.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/Wasm.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <memory>
#include <system_error>

using namespace llvm;
using namespace object;

raw_ostream &object::operator<<(raw_ostream &OS, const SectionedAddress &Addr) {
  OS << "SectionedAddress{" << format_hex(Addr.Address, 10);
  if (Addr.SectionIndex != SectionedAddress::UndefSection)
    OS << ", " << Addr.SectionIndex;
  return OS << "}";
}

void ObjectFile::anchor() {}

ObjectFile::ObjectFile(unsigned int Type, MemoryBufferRef Source)
    : SymbolicFile(Type, Source) {}

bool SectionRef::containsSymbol(SymbolRef S) const {
  Expected<section_iterator> SymSec = S.getSection();
  if (!SymSec) {
    // TODO: Actually report errors helpfully.
    consumeError(SymSec.takeError());
    return false;
  }
  return *this == **SymSec;
}

Expected<uint64_t> ObjectFile::getSymbolValue(DataRefImpl Ref) const {
  uint32_t Flags;
  if (Error E = getSymbolFlags(Ref).moveInto(Flags))
    // TODO: Test this error.
    return std::move(E);

  if (Flags & SymbolRef::SF_Undefined)
    return 0;
  if (Flags & SymbolRef::SF_Common)
    return getCommonSymbolSize(Ref);
  return getSymbolValueImpl(Ref);
}

Error ObjectFile::printSymbolName(raw_ostream &OS, DataRefImpl Symb) const {
  Expected<StringRef> Name = getSymbolName(Symb);
  if (!Name)
    return Name.takeError();
  OS << *Name;
  return Error::success();
}

uint32_t ObjectFile::getSymbolAlignment(DataRefImpl DRI) const { return 0; }

bool ObjectFile::isSectionBitcode(DataRefImpl Sec) const {
  Expected<StringRef> NameOrErr = getSectionName(Sec);
  if (NameOrErr)
    return *NameOrErr == ".llvm.lto";
  consumeError(NameOrErr.takeError());
  return false;
}

bool ObjectFile::isSectionStripped(DataRefImpl Sec) const { return false; }

bool ObjectFile::isBerkeleyText(DataRefImpl Sec) const {
  return isSectionText(Sec);
}

bool ObjectFile::isBerkeleyData(DataRefImpl Sec) const {
  return isSectionData(Sec);
}

bool ObjectFile::isDebugSection(DataRefImpl Sec) const { return false; }

bool ObjectFile::hasDebugInfo() const {
  return any_of(sections(),
                [](SectionRef Sec) { return Sec.isDebugSection(); });
}

Expected<section_iterator>
ObjectFile::getRelocatedSection(DataRefImpl Sec) const {
  return section_iterator(SectionRef(Sec, this));
}

Triple ObjectFile::makeTriple() const {
  Triple TheTriple;
  auto Arch = getArch();
  TheTriple.setArch(Triple::ArchType(Arch));

  auto OS = getOS();
  if (OS != Triple::UnknownOS)
    TheTriple.setOS(OS);

  // For ARM targets, try to use the build attributes to build determine
  // the build target. Target features are also added, but later during
  // disassembly.
  if (Arch == Triple::arm || Arch == Triple::armeb)
    setARMSubArch(TheTriple);

  // TheTriple defaults to ELF, and COFF doesn't have an environment:
  // something we can do here is indicate that it is mach-o.
  if (isMachO()) {
    TheTriple.setObjectFormat(Triple::MachO);
  } else if (isCOFF()) {
    const auto COFFObj = cast<COFFObjectFile>(this);
    if (COFFObj->getArch() == Triple::thumb)
      TheTriple.setTriple("thumbv7-windows");
  } else if (isXCOFF()) {
    // XCOFF implies AIX.
    TheTriple.setOS(Triple::AIX);
    TheTriple.setObjectFormat(Triple::XCOFF);
  } else if (isGOFF()) {
    TheTriple.setOS(Triple::ZOS);
    TheTriple.setObjectFormat(Triple::GOFF);
  } else if (TheTriple.isAMDGPU()) {
    TheTriple.setVendor(Triple::AMD);
  } else if (TheTriple.isNVPTX()) {
    TheTriple.setVendor(Triple::NVIDIA);
  }

  return TheTriple;
}

bool ObjectFile::isReflectionSectionStrippable(
    llvm::binaryformat::Swift5ReflectionSectionKind ReflectionSectionKind)
    const {
  using llvm::binaryformat::Swift5ReflectionSectionKind;
  return ReflectionSectionKind == Swift5ReflectionSectionKind::fieldmd ||
         ReflectionSectionKind == Swift5ReflectionSectionKind::reflstr ||
         ReflectionSectionKind == Swift5ReflectionSectionKind::assocty;
}
