//===-- ObjDumper.cpp - Base dumper class -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements ObjDumper.
///
//===----------------------------------------------------------------------===//

#include "ObjDumper.h"
#include "Error.h"
#include "llvm-readobj.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

namespace llvm {

static inline Error createError(const Twine &Msg) {
  return createStringError(object::object_error::parse_failed, Msg);
}

ObjDumper::ObjDumper(ScopedPrinter &Writer) : W(Writer) {}

ObjDumper::~ObjDumper() {
}

static void printAsPrintable(raw_ostream &W, const uint8_t *Start, size_t Len) {
  for (size_t i = 0; i < Len; i++)
    W << (isPrint(Start[i]) ? static_cast<char>(Start[i]) : '.');
}

static std::vector<object::SectionRef>
getSectionRefsByNameOrIndex(const object::ObjectFile *Obj,
                            ArrayRef<std::string> Sections) {
  std::vector<object::SectionRef> Ret;
  std::map<std::string, bool> SecNames;
  std::map<unsigned, bool> SecIndices;
  unsigned SecIndex;
  for (StringRef Section : Sections) {
    if (!Section.getAsInteger(0, SecIndex))
      SecIndices.emplace(SecIndex, false);
    else
      SecNames.emplace(std::string(Section), false);
  }

  SecIndex = Obj->isELF() ? 0 : 1;
  for (object::SectionRef SecRef : Obj->sections()) {
    StringRef SecName = unwrapOrError(Obj->getFileName(), SecRef.getName());
    auto NameIt = SecNames.find(std::string(SecName));
    if (NameIt != SecNames.end())
      NameIt->second = true;
    auto IndexIt = SecIndices.find(SecIndex);
    if (IndexIt != SecIndices.end())
      IndexIt->second = true;
    if (NameIt != SecNames.end() || IndexIt != SecIndices.end())
      Ret.push_back(SecRef);
    SecIndex++;
  }

  for (const std::pair<const std::string, bool> &S : SecNames)
    if (!S.second)
      reportWarning(
          createError(formatv("could not find section '{0}'", S.first).str()),
          Obj->getFileName());

  for (std::pair<unsigned, bool> S : SecIndices)
    if (!S.second)
      reportWarning(
          createError(formatv("could not find section {0}", S.first).str()),
          Obj->getFileName());

  return Ret;
}

void ObjDumper::printSectionsAsString(const object::ObjectFile *Obj,
                                      ArrayRef<std::string> Sections) {
  bool First = true;
  for (object::SectionRef Section :
       getSectionRefsByNameOrIndex(Obj, Sections)) {
    StringRef SectionName =
        unwrapOrError(Obj->getFileName(), Section.getName());

    if (!First)
      W.startLine() << '\n';
    First = false;
    W.startLine() << "String dump of section '" << SectionName << "':\n";

    StringRef SectionContent =
        unwrapOrError(Obj->getFileName(), Section.getContents());

    const uint8_t *SecContent = SectionContent.bytes_begin();
    const uint8_t *CurrentWord = SecContent;
    const uint8_t *SecEnd = SectionContent.bytes_end();

    while (CurrentWord <= SecEnd) {
      size_t WordSize = strnlen(reinterpret_cast<const char *>(CurrentWord),
                                SecEnd - CurrentWord);
      if (!WordSize) {
        CurrentWord++;
        continue;
      }
      W.startLine() << format("[%6tx] ", CurrentWord - SecContent);
      printAsPrintable(W.startLine(), CurrentWord, WordSize);
      W.startLine() << '\n';
      CurrentWord += WordSize + 1;
    }
  }
}

void ObjDumper::printSectionsAsHex(const object::ObjectFile *Obj,
                                   ArrayRef<std::string> Sections) {
  bool First = true;
  for (object::SectionRef Section :
       getSectionRefsByNameOrIndex(Obj, Sections)) {
    StringRef SectionName =
        unwrapOrError(Obj->getFileName(), Section.getName());

    if (!First)
      W.startLine() << '\n';
    First = false;
    W.startLine() << "Hex dump of section '" << SectionName << "':\n";

    StringRef SectionContent =
        unwrapOrError(Obj->getFileName(), Section.getContents());
    const uint8_t *SecContent = SectionContent.bytes_begin();
    const uint8_t *SecEnd = SecContent + SectionContent.size();

    for (const uint8_t *SecPtr = SecContent; SecPtr < SecEnd; SecPtr += 16) {
      const uint8_t *TmpSecPtr = SecPtr;
      uint8_t i;
      uint8_t k;

      W.startLine() << format_hex(Section.getAddress() + (SecPtr - SecContent),
                                  10);
      W.startLine() << ' ';
      for (i = 0; TmpSecPtr < SecEnd && i < 4; ++i) {
        for (k = 0; TmpSecPtr < SecEnd && k < 4; k++, TmpSecPtr++) {
          uint8_t Val = *(reinterpret_cast<const uint8_t *>(TmpSecPtr));
          W.startLine() << format_hex_no_prefix(Val, 2);
        }
        W.startLine() << ' ';
      }

      // We need to print the correct amount of spaces to match the format.
      // We are adding the (4 - i) last rows that are 8 characters each.
      // Then, the (4 - i) spaces that are in between the rows.
      // Least, if we cut in a middle of a row, we add the remaining characters,
      // which is (8 - (k * 2)).
      if (i < 4)
        W.startLine() << format("%*c", (4 - i) * 8 + (4 - i), ' ');
      if (k < 4)
        W.startLine() << format("%*c", 8 - k * 2, ' ');

      TmpSecPtr = SecPtr;
      for (i = 0; TmpSecPtr + i < SecEnd && i < 16; ++i)
        W.startLine() << (isPrint(TmpSecPtr[i])
                              ? static_cast<char>(TmpSecPtr[i])
                              : '.');

      W.startLine() << '\n';
    }
  }
}

} // namespace llvm
