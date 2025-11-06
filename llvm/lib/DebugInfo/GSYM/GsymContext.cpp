//===-- GsymContext.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

#include "llvm/DebugInfo/GSYM/GsymContext.h"

#include "llvm/DebugInfo/GSYM/GsymReader.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::gsym;

GsymContext::~GsymContext() = default;
GsymContext::GsymContext(std::unique_ptr<GsymReader> Reader)
    : DIContext(CK_GSYM), Reader(std::move(Reader)) {}

void GsymContext::dump(raw_ostream &OS, DIDumpOptions DumpOpts) {}

static bool fillLineInfoFromLocation(const SourceLocation &Location,
                                     DILineInfoSpecifier Specifier,
                                     DILineInfo &LineInfo) {
  // FIXME Demangle in case of DINameKind::ShortName
  if (Specifier.FNKind != DINameKind::None) {
    LineInfo.FunctionName = Location.Name.str();
  }

  switch (Specifier.FLIKind) {
  case DILineInfoSpecifier::FileLineInfoKind::RelativeFilePath:
    // We have no information to determine the relative path, so we fall back to
    // returning the absolute path.
  case DILineInfoSpecifier::FileLineInfoKind::RawValue:
  case DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath:
    if (Location.Dir.empty()) {
      if (Location.Base.empty())
        LineInfo.FileName = DILineInfo::BadString;
      else
        LineInfo.FileName = Location.Base.str();
    } else {
      SmallString<128> Path(Location.Dir);
      sys::path::append(Path, Location.Base);
      LineInfo.FileName = static_cast<std::string>(Path);
    }
    break;

  case DILineInfoSpecifier::FileLineInfoKind::BaseNameOnly:
    LineInfo.FileName = Location.Base.str();
    break;

  default:
    return false;
  }
  LineInfo.Line = Location.Line;

  // We don't have information in GSYM to fill any of the Source, Column,
  // StartFileName or StartLine attributes.

  return true;
}

std::optional<DILineInfo>
GsymContext::getLineInfoForAddress(object::SectionedAddress Address,
                                   DILineInfoSpecifier Specifier) {
  if (Address.SectionIndex != object::SectionedAddress::UndefSection)
    return {};

  auto ResultOrErr = Reader->lookup(Address.Address);

  if (!ResultOrErr) {
    consumeError(ResultOrErr.takeError());
    return {};
  }

  const auto &Result = *ResultOrErr;

  DILineInfo LineInfo;

  if (Result.Locations.empty()) {
    // No debug info for this, we just had a symbol from the symbol table.

    // FIXME Demangle in case of DINameKind::ShortName
    if (Specifier.FNKind != DINameKind::None)
      LineInfo.FunctionName = Result.FuncName.str();
  } else if (!fillLineInfoFromLocation(Result.Locations.front(), Specifier,
                                       LineInfo))
    return {};

  LineInfo.StartAddress = Result.FuncRange.start();

  return LineInfo;
}

std::optional<DILineInfo>
GsymContext::getLineInfoForDataAddress(object::SectionedAddress Address) {
  // We can't implement this, there's no such information in the GSYM file.

  return {};
}

DILineInfoTable
GsymContext::getLineInfoForAddressRange(object::SectionedAddress Address,
                                        uint64_t Size,
                                        DILineInfoSpecifier Specifier) {
  if (Size == 0)
    return DILineInfoTable();

  if (Address.SectionIndex != llvm::object::SectionedAddress::UndefSection)
    return DILineInfoTable();

  if (auto FuncInfoOrErr = Reader->getFunctionInfo(Address.Address)) {
    DILineInfoTable Table;
    if (FuncInfoOrErr->OptLineTable) {
      const gsym::LineTable &LT = *FuncInfoOrErr->OptLineTable;
      const uint64_t StartAddr = Address.Address;
      const uint64_t EndAddr = Address.Address + Size;
      for (const auto &LineEntry : LT) {
        if (StartAddr <= LineEntry.Addr && LineEntry.Addr < EndAddr) {
          // Use LineEntry.Addr, LineEntry.File (which is a file index into the
          // files tables from the GsymReader), and LineEntry.Line (source line
          // number) to add stuff to the DILineInfoTable
        }
      }
    }
    return Table;
  } else {
    consumeError(FuncInfoOrErr.takeError());
    return DILineInfoTable();
  }
}

DIInliningInfo
GsymContext::getInliningInfoForAddress(object::SectionedAddress Address,
                                       DILineInfoSpecifier Specifier) {
  auto ResultOrErr = Reader->lookup(Address.Address);

  if (!ResultOrErr)
    return {};

  const auto &Result = *ResultOrErr;

  DIInliningInfo InlineInfo;

  for (const auto &Location : Result.Locations) {
    DILineInfo LineInfo;

    if (!fillLineInfoFromLocation(Location, Specifier, LineInfo))
      return {};

    // Hm, that's probably something that should only be filled in the first or
    // last frame?
    LineInfo.StartAddress = Result.FuncRange.start();

    InlineInfo.addFrame(LineInfo);
  }

  return InlineInfo;
}

std::vector<DILocal>
GsymContext::getLocalsForAddress(object::SectionedAddress Address) {
  // We can't implement this, there's no such information in the GSYM file.

  return {};
}
