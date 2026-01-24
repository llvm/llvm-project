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
                             bool DefaultVal, bool &Val) {
    return addValue(Name, Desc, DefaultVal, Val, OptionKind::Flag);
  }

  template <typename T>
  CommandLineParser &addValue(std::string_view Name, std::string_view Desc,
                              T DefaultVal, T &Val,
                              OptionKind Kind = OptionKind::Value) {
    Val = DefaultVal;
    Opts.push_back({.Name = std::string(Name),
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
    size_t MaxWidth = std::accumulate(
        Opts.begin(), Opts.end(), size_t(0), [](size_t Max, const Option &Opt) {
          size_t Len = Opt.Name.length() + 2; // "--"
          if (Opt.Kind == OptionKind::Value)
            Len += 8; // "=<value>"
          return std::max(Max, Len);
        });

    std::for_each(Opts.begin(), Opts.end(), [&](const Option &Opt) {
      std::string FlagStr =
          "--" + Opt.Name + (Opt.Kind == OptionKind::Value ? "=<value>" : "");
      OS << "  " << std::left << std::setw(MaxWidth + 2) << FlagStr << Opt.Desc
         << "\n";
    });
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

      if (!AfterDashDash && beginsDashes(Tok)) {
        std::string_view K = removeLeadingDashes(Tok);
        std::string_view V;
        bool HasValue = false;

        if (auto P = K.find('='); P != std::string_view::npos) {
          V = K.substr(P + 1);
          K = K.substr(0, P);
          HasValue = true;
        }

        auto FoundOpt =
            std::find_if(Opts.begin(), Opts.end(), [&](const Option &o) {
              return std::string_view(o.Name) == K;
            });

        if (FoundOpt == Opts.end()) {
          return orc_rt::make_error<orc_rt::StringError>(
              "Unknown option '" + std::string(Tok) + "'");
        }

        if (FoundOpt->Kind == OptionKind::Flag) {
          if (!HasValue)
            V = "true";
        } else if (!HasValue) {
          if (std::next(It) == End) {
            return orc_rt::make_error<orc_rt::StringError>(
                "Option '" + std::string(K) + "' requires a value");
          }
          V = *++It;
        }

        if (auto Err = FoundOpt->FromString(V))
          return Err;

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
    std::string Desc;
    OptionKind Kind{};
    std::function<void()> Default;
    std::function<orc_rt::Error(std::string_view)> FromString;
  };

  std::vector<std::string> Positionals;
  std::vector<Option> Opts;

  static bool beginsDashes(std::string_view S) {
    return !S.empty() && S.front() == '-';
  }

  static bool startsWith(std::string_view S, std::string_view P) {
    return S.size() >= P.size() && S.compare(0, P.size(), P) == 0;
  }

  static std::string_view removeLeadingDashes(std::string_view S) {
    if (startsWith(S, "--"))
      return S.substr(2);
    if (startsWith(S, "-"))
      return S.substr(1);
    return S;
  }
};
} // namespace orc_rt
