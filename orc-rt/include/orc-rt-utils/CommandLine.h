#include <algorithm>
#include <charconv>
#include <functional>
#include <iomanip>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "orc-rt/Error.h"

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

class CommandLineParser {
public:
  enum class OptionKind { Flag, Value };
  CommandLineParser() = default;

  CommandLineParser &addFlag(std::string_view Name, std::string_view Desc,
                             bool DefaultVal, bool &Val,
                             std::optional<char> ShortName = std::nullopt) {
    return addValue(Name, Desc, DefaultVal, Val, OptionKind::Flag,
                    std::move(ShortName));
  }

  template <typename T>
  CommandLineParser &addValue(std::string_view Name, std::string_view Desc,
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
                                      std::string_view S) -> orc_rt::Error {
                      if (auto V = detail::parseValue<T>(S)) {
                        Val = *V;
                        return orc_rt::Error::success();
                      }
                      return orc_rt::make_error<orc_rt::StringError>(
                          std::string("Invalid value for '") + OptName +
                          "': '" + std::string(S) + "'");
                    }});
    return *this;
  }

  void printHelp(std::ostream &OS, std::string_view ProgramName) const {
    OS << "Usage: " << ProgramName << " [options] [positional arguments]\n\n";
    OS << "OPTIONS:\n";

    bool AnyShortNames = std::any_of(
        Opts.begin(), Opts.end(),
        [](const Option &O) { return O.ShortName.has_value(); });

    size_t MaxWidth = 0;
    for (const auto &Opt : Opts) {
      size_t CurrentWidth = 2; // "  "
      if (AnyShortNames)
        CurrentWidth += 4; // "-x, "
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

      OS << std::left << std::setw(MaxWidth + 2) << FlagStr << Opt.Desc << "\n";
    }
  }

  template <typename I> orc_rt::Error parse(I Begin, I End) {
    std::for_each(Opts.begin(), Opts.end(),
                  [](const Option &O) { O.Default(); });
    Positionals.clear();
    bool AfterDashDash = false;
    if (Begin != End)
      Begin++;

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
          return orc_rt::make_error<orc_rt::StringError>(
              "Unknown option '" + std::string(Tok) + "'");
        if (auto Err = consumeValue(FoundOpt, V, HasValue, It, End))
          return Err;
      } else if (!AfterDashDash && startsWith(Tok, "-") && Tok.size() > 1) {
        std::string_view Group = Tok.substr(1);
        for (size_t i = 0; i < Group.size(); ++i) {
          auto FoundOpt = findOpt(Group[i]);
          if (!FoundOpt)
            return orc_rt::make_error<orc_rt::StringError>(
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
    return orc_rt::Error::success();
  }

  orc_rt::Error parse(int argc, char **argv) {
    return parse(argv, argv + argc);
  }

  const std::vector<std::string> &positionals() const { return Positionals; }

private:
  struct Option {
    std::string Name;
    std::optional<char> ShortName;
    std::string Desc;
    OptionKind Kind{};
    std::function<void()> Default;
    std::function<orc_rt::Error(std::string_view)> FromString;
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
  orc_rt::Error consumeValue(const Option *Opt, std::string_view ExplicitV,
                             bool HasV, I &It, I End) {
    std::string_view V = ExplicitV;
    if (Opt->Kind == OptionKind::Flag) {
      if (!HasV)
        V = "true";
    } else if (!HasV) {
      if (++It == End)
        return orc_rt::make_error<orc_rt::StringError>(
            "Option '--" + Opt->Name + "' requires a value");
      V = *It;
    }
    return Opt->FromString(V);
  }

  static bool startsWith(std::string_view S, std::string_view P) {
    return S.size() >= P.size() && S.compare(0, P.size(), P) == 0;
  }
};
} // namespace orc_rt
