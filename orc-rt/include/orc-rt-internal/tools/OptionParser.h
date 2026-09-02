//===- OptionParser.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simple orc-rt command-line parser. Supports flags, values and positionals.
// All storage owned by caller.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_INTERNAL_TOOLS_OPTIONPARSER_H
#define ORC_RT_INTERNAL_TOOLS_OPTIONPARSER_H

#include <algorithm>
#include <charconv>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "orc-rt-internal/support/StringExtras.h"
#include "orc-rt/support/Error.h"
#include "orc-rt/support/move_only_function.h"

namespace orc_rt {
namespace detail {

template <typename T> inline std::optional<T> parseValue(std::string_view Str);

template <>
inline std::optional<std::string>
parseValue<std::string>(std::string_view Str) {
  return std::string(Str);
}

template <>
inline std::optional<std::string_view>
parseValue<std::string_view>(std::string_view Str) {
  return Str;
}

template <> inline std::optional<int> parseValue<int>(std::string_view Str) {
  if (Str.empty())
    return std::nullopt;
  int Val{};
  auto Ret = std::from_chars(Str.data(), Str.data() + Str.size(), Val);
  if (Ret.ec != std::errc() || Ret.ptr != Str.data() + Str.size())
    return std::nullopt;
  return Val;
}

template <> inline std::optional<bool> parseValue<bool>(std::string_view Str) {
  if (Str.empty())
    return std::nullopt;

  if (Str == "1")
    return true;
  if (Str == "0")
    return false;

  std::string Val;
  std::transform(
      Str.begin(), Str.end(), std::back_inserter(Val),
      [](unsigned char C) { return static_cast<char>(std::tolower(C)); });

  if (Val == "true")
    return true;
  if (Val == "false")
    return false;

  return std::nullopt;
}
} // namespace detail

class OptionParser {
public:
  enum class OptionKind { Flag, Value };
  OptionParser() = default;

  OptionParser &addFlag(std::string_view Name, std::string_view Desc,
                        bool DefaultVal, bool &Val,
                        std::optional<char> ShortName = std::nullopt) {
    return addValue(Name, Desc, DefaultVal, Val, OptionKind::Flag,
                    std::move(ShortName));
  }

  template <typename T>
  OptionParser &addValue(std::string_view Name, std::string_view Desc,
                         T DefaultVal, T &Val,
                         OptionKind Kind = OptionKind::Value,
                         std::optional<char> ShortName = std::nullopt) {
    Val = DefaultVal;
    Opts.push_back({.Name = std::string(Name),
                    .ShortName = std::move(ShortName),
                    .Desc = std::string(Desc),
                    .Kind = Kind,
                    .Default = [&Val, DV = DefaultVal]() { Val = DV; },
                    .FromString = [&Val, OptName = std::string(Name)](
                                      std::string_view S) -> Error {
                      if (auto V = detail::parseValue<T>(S)) {
                        Val = *V;
                        return Error::success();
                      }
                      return make_error<StringError>(
                          std::string("Invalid value for '") + OptName +
                          "': '" + std::string(S) + "'");
                    }});
    return *this;
  }

  std::string formatHelp(std::string_view ProgramName) const {
    StringOutputStream OS;
    OS << "Usage: " << ProgramName << " [options] [positional arguments]\n\n";
    OS << "OPTIONS:\n";

    bool AnyShortNames =
        std::any_of(Opts.begin(), Opts.end(),
                    [](const Option &O) { return O.ShortName.has_value(); });

    size_t MaxWidth = 0;
    for (const auto &Opt : Opts) {
      size_t CurrentWidth = 2; // "  "
      if (AnyShortNames)
        CurrentWidth += 4;                   // "-x, "
      CurrentWidth += 2 + Opt.Name.length(); // "--name"
      if (Opt.Kind == OptionKind::Value)
        CurrentWidth += 8; // " <value>"
      MaxWidth = std::max(MaxWidth, CurrentWidth);
    }

    for (const auto &Opt : Opts) {
      std::string FlagStr = "  ";
      if (AnyShortNames) {
        if (Opt.ShortName) {
          FlagStr += "-";
          FlagStr += *Opt.ShortName;
          FlagStr += ", ";
        } else {
          FlagStr += "    "; // Pad gutter
        }
      }
      FlagStr += "--" + Opt.Name;
      if (Opt.Kind == OptionKind::Value)
        FlagStr += " <value>";

      OS << ljust(FlagStr, MaxWidth + 2) << Opt.Desc << "\n";
    }

    return std::move(OS).str();
  }

  /// Parse an argument list.
  ///
  /// Iterators should be over program arguments only, and should not include
  /// the program name as the first argument.
  template <typename I> Error parse(I Begin, I End) {
    std::for_each(Opts.begin(), Opts.end(),
                  [](const Option &O) { O.Default(); });
    Positionals.clear();
    bool AfterDashDash = false;

    for (auto It = Begin; It != End; ++It) {
      std::string_view Tok(*It);
      if (!AfterDashDash && Tok == "--") {
        AfterDashDash = true;
        continue;
      }
      if (!AfterDashDash && startsWith(Tok, "--")) {
        std::string_view K = Tok.substr(2);
        std::string_view V;
        bool HasValue = false;
        if (auto P = K.find('='); P != std::string_view::npos) {
          V = K.substr(P + 1);
          K = K.substr(0, P);
          HasValue = true;
        }
        auto FoundOpt = findOpt(K);
        if (!FoundOpt)
          return make_error<StringError>("Unknown option '" + std::string(Tok) +
                                         "'");
        if (auto Err = consumeValue(FoundOpt, V, HasValue, It, End))
          return Err;
      } else if (!AfterDashDash && startsWith(Tok, "-") && Tok.size() > 1) {
        std::string_view Group = Tok.substr(1);
        for (size_t i = 0; i < Group.size(); ++i) {
          auto FoundOpt = findOpt(Group[i]);
          if (!FoundOpt)
            return make_error<StringError>(
                std::string("Unknown short option '-") + Group[i] + "'");
          if (FoundOpt->Kind == OptionKind::Value) {
            std::string_view V = Group.substr(i + 1);
            bool HasValue = !V.empty();
            if (auto Err = consumeValue(FoundOpt, V, HasValue, It, End))
              return Err;
            break;
          } else {
            if (auto Err = FoundOpt->FromString("true"))
              return Err;
          }
        }
      } else {
        Positionals.emplace_back(Tok);
      }
    }
    return Error::success();
  }

  /// Parse main-like args: argc must be >= 1, and argv[0] must contain the
  /// program name.
  Error parseAsMainArgs(int argc, char **argv) {
    if (argc == 0)
      return make_error<StringError>("no program-name argument in argv[0]");
    return parse(argv + 1, argv + argc);
  }

  const std::vector<std::string> &positionals() const { return Positionals; }

private:
  struct Option {
    std::string Name;
    std::optional<char> ShortName;
    std::string Desc;
    OptionKind Kind{};
    move_only_function<void() const> Default;
    move_only_function<Error(std::string_view) const> FromString;
  };

  std::vector<std::string> Positionals;
  std::vector<Option> Opts;

  const Option *findOpt(std::string_view L) const {
    auto It = std::find_if(Opts.begin(), Opts.end(),
                           [&](const Option &O) { return O.Name == L; });
    return It != Opts.end() ? &(*It) : nullptr;
  }

  const Option *findOpt(char S) const {
    auto It = std::find_if(Opts.begin(), Opts.end(), [&](const Option &O) {
      return O.ShortName && *O.ShortName == S;
    });
    return It != Opts.end() ? &(*It) : nullptr;
  }

  template <typename I>
  Error consumeValue(const Option *Opt, std::string_view ExplicitV, bool HasV,
                     I &It, I End) {
    std::string_view V = ExplicitV;
    if (Opt->Kind == OptionKind::Flag) {
      if (!HasV)
        V = "true";
    } else if (!HasV) {
      if (++It == End)
        return make_error<StringError>("Option '--" + Opt->Name +
                                       "' requires a value");
      V = *It;
    }
    return Opt->FromString(V);
  }

  static bool startsWith(std::string_view S, std::string_view P) {
    return S.size() >= P.size() && S.compare(0, P.size(), P) == 0;
  }
};

} // namespace orc_rt

#endif // ORC_RT_INTERNAL_TOOLS_OPTIONPARSER_H
