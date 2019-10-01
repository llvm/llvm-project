//===- ELFConfig.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CopyConfig.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace objcopy {
namespace elf {

static Expected<NewSymbolInfo> parseNewSymbolInfo(StringRef FlagValue,
                                                  uint8_t DefaultVisibility) {
  // Parse value given with --add-symbol option and create the
  // new symbol if possible. The value format for --add-symbol is:
  //
  // <name>=[<section>:]<value>[,<flags>]
  //
  // where:
  // <name> - symbol name, can be empty string
  // <section> - optional section name. If not given ABS symbol is created
  // <value> - symbol value, can be decimal or hexadecimal number prefixed
  //           with 0x.
  // <flags> - optional flags affecting symbol type, binding or visibility:
  //           The following are currently supported:
  //
  //           global, local, weak, default, hidden, file, section, object,
  //           indirect-function.
  //
  //           The following flags are ignored and provided for GNU
  //           compatibility only:
  //
  //           warning, debug, constructor, indirect, synthetic,
  //           unique-object, before=<symbol>.
  NewSymbolInfo SI;
  StringRef Value;
  std::tie(SI.SymbolName, Value) = FlagValue.split('=');
  if (Value.empty())
    return createStringError(
        errc::invalid_argument,
        "bad format for --add-symbol, missing '=' after '%s'",
        SI.SymbolName.str().c_str());

  if (Value.contains(':')) {
    std::tie(SI.SectionName, Value) = Value.split(':');
    if (SI.SectionName.empty() || Value.empty())
      return createStringError(
          errc::invalid_argument,
          "bad format for --add-symbol, missing section name or symbol value");
  }

  SmallVector<StringRef, 6> Flags;
  Value.split(Flags, ',');
  if (Flags[0].getAsInteger(0, SI.Value))
    return createStringError(errc::invalid_argument, "bad symbol value: '%s'",
                             Flags[0].str().c_str());

  SI.Visibility = DefaultVisibility;

  using Functor = std::function<void(void)>;
  SmallVector<StringRef, 6> UnsupportedFlags;
  for (size_t I = 1, NumFlags = Flags.size(); I < NumFlags; ++I)
    static_cast<Functor>(
        StringSwitch<Functor>(Flags[I])
            .CaseLower("global", [&SI] { SI.Bind = ELF::STB_GLOBAL; })
            .CaseLower("local", [&SI] { SI.Bind = ELF::STB_LOCAL; })
            .CaseLower("weak", [&SI] { SI.Bind = ELF::STB_WEAK; })
            .CaseLower("default", [&SI] { SI.Visibility = ELF::STV_DEFAULT; })
            .CaseLower("hidden", [&SI] { SI.Visibility = ELF::STV_HIDDEN; })
            .CaseLower("protected",
                       [&SI] { SI.Visibility = ELF::STV_PROTECTED; })
            .CaseLower("file", [&SI] { SI.Type = ELF::STT_FILE; })
            .CaseLower("section", [&SI] { SI.Type = ELF::STT_SECTION; })
            .CaseLower("object", [&SI] { SI.Type = ELF::STT_OBJECT; })
            .CaseLower("function", [&SI] { SI.Type = ELF::STT_FUNC; })
            .CaseLower("indirect-function",
                       [&SI] { SI.Type = ELF::STT_GNU_IFUNC; })
            .CaseLower("debug", [] {})
            .CaseLower("constructor", [] {})
            .CaseLower("warning", [] {})
            .CaseLower("indirect", [] {})
            .CaseLower("synthetic", [] {})
            .CaseLower("unique-object", [] {})
            .StartsWithLower("before", [] {})
            .Default([&] { UnsupportedFlags.push_back(Flags[I]); }))();
  if (!UnsupportedFlags.empty())
    return createStringError(errc::invalid_argument,
                             "unsupported flag%s for --add-symbol: '%s'",
                             UnsupportedFlags.size() > 1 ? "s" : "",
                             join(UnsupportedFlags, "', '").c_str());
  return SI;
}

Expected<ELFCopyConfig> parseConfig(const CopyConfig &Config) {
  ELFCopyConfig ELFConfig;
  if (Config.NewSymbolVisibility) {
    const uint8_t Invalid = 0xff;
    ELFConfig.NewSymbolVisibility =
        StringSwitch<uint8_t>(*Config.NewSymbolVisibility)
            .Case("default", ELF::STV_DEFAULT)
            .Case("hidden", ELF::STV_HIDDEN)
            .Case("internal", ELF::STV_INTERNAL)
            .Case("protected", ELF::STV_PROTECTED)
            .Default(Invalid);

    if (ELFConfig.NewSymbolVisibility == Invalid)
      return createStringError(errc::invalid_argument,
                               "'%s' is not a valid symbol visibility",
                               Config.NewSymbolVisibility->str().c_str());
  }

  for (StringRef Arg : Config.SymbolsToAdd) {
    Expected<elf::NewSymbolInfo> NSI = parseNewSymbolInfo(
        Arg,
        ELFConfig.NewSymbolVisibility.getValueOr((uint8_t)ELF::STV_DEFAULT));
    if (!NSI)
      return NSI.takeError();
    ELFConfig.SymbolsToAdd.push_back(*NSI);
  }

  return ELFConfig;
}

} // end namespace elf
} // end namespace objcopy
} // end namespace llvm
