//===- RemarkLinker.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides an implementation of the remark linker.
//
//===----------------------------------------------------------------------===//

#include "llvm/Remarks/RemarkLinker.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Remarks/BitstreamRemarkContainer.h"
#include "llvm/Remarks/RemarkParser.h"
#include "llvm/Remarks/RemarkSerializer.h"
#include "llvm/Support/Error.h"

using namespace llvm;
using namespace llvm::remarks;

static Expected<StringRef>
getRemarksSectionName(const object::ObjectFile &Obj) {
  if (Obj.isMachO())
    return StringRef("__remarks");
  // ELF -> .remarks, but there is no ELF support at this point.
  return createStringError(std::errc::illegal_byte_sequence,
                           "Unsupported file format.");
}

Expected<Optional<StringRef>>
llvm::remarks::getRemarksSectionContents(const object::ObjectFile &Obj) {
  Expected<StringRef> SectionName = getRemarksSectionName(Obj);
  if (!SectionName)
    return SectionName.takeError();

  for (const object::SectionRef &Section : Obj.sections()) {
    Expected<StringRef> MaybeName = Section.getName();
    if (!MaybeName)
      return MaybeName.takeError();
    if (*MaybeName != *SectionName)
      continue;

    if (Expected<StringRef> Contents = Section.getContents())
      return *Contents;
    else
      return Contents.takeError();
  }
  return Optional<StringRef>{};
}

Remark &RemarkLinker::keep(std::unique_ptr<Remark> Remark) {
  StrTab.internalize(*Remark);
  auto Inserted = Remarks.insert(std::move(Remark));
  return **Inserted.first;
}

void RemarkLinker::setExternalFilePrependPath(StringRef PrependPathIn) {
  PrependPath = PrependPathIn;
}

// Discard remarks with no source location.
static bool shouldKeepRemark(const Remark &R) { return R.Loc.hasValue(); }

Error RemarkLinker::link(StringRef Buffer, Optional<Format> RemarkFormat) {
  if (!RemarkFormat) {
    Expected<Format> ParserFormat = magicToFormat(Buffer);
    if (!ParserFormat)
      return ParserFormat.takeError();
    RemarkFormat = *ParserFormat;
  }

  Expected<std::unique_ptr<RemarkParser>> MaybeParser =
      createRemarkParserFromMeta(
          *RemarkFormat, Buffer, /*StrTab=*/None,
          PrependPath ? Optional<StringRef>(StringRef(*PrependPath))
                      : Optional<StringRef>(None));
  if (!MaybeParser)
    return MaybeParser.takeError();

  RemarkParser &Parser = **MaybeParser;

  while (true) {
    Expected<std::unique_ptr<Remark>> Next = Parser.next();
    if (Error E = Next.takeError()) {
      if (E.isA<EndOfFileError>()) {
        consumeError(std::move(E));
        break;
      }
      return E;
    }

    assert(*Next != nullptr);

    if (shouldKeepRemark(**Next))
      keep(std::move(*Next));
  }
  return Error::success();
}

Error RemarkLinker::link(const object::ObjectFile &Obj,
                         Optional<Format> RemarkFormat) {
  Expected<Optional<StringRef>> SectionOrErr = getRemarksSectionContents(Obj);
  if (!SectionOrErr)
    return SectionOrErr.takeError();

  if (Optional<StringRef> Section = *SectionOrErr)
    return link(*Section, RemarkFormat);
  return Error::success();
}

Error RemarkLinker::serialize(raw_ostream &OS, Format RemarksFormat) const {
  Expected<std::unique_ptr<RemarkSerializer>> MaybeSerializer =
      createRemarkSerializer(RemarksFormat, SerializerMode::Standalone, OS,
                             std::move(const_cast<StringTable &>(StrTab)));
  if (!MaybeSerializer)
    return MaybeSerializer.takeError();

  std::unique_ptr<remarks::RemarkSerializer> Serializer =
      std::move(*MaybeSerializer);

  for (const Remark &R : remarks())
    Serializer->emit(R);
  return Error::success();
}
