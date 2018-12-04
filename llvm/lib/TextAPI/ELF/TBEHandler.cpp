//===- TBEHandler.cpp -----------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===-----------------------------------------------------------------------===/

#include "llvm/TextAPI/ELF/TBEHandler.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/TextAPI/ELF/ELFStub.h"

using namespace llvm;
using namespace llvm::elfabi;

LLVM_YAML_STRONG_TYPEDEF(ELFArch, ELFArchMapper)

namespace llvm {
namespace yaml {

/// YAML traits for ELFSymbolType.
template <> struct ScalarEnumerationTraits<ELFSymbolType> {
  static void enumeration(IO &IO, ELFSymbolType &SymbolType) {
    IO.enumCase(SymbolType, "NoType", ELFSymbolType::NoType);
    IO.enumCase(SymbolType, "Func", ELFSymbolType::Func);
    IO.enumCase(SymbolType, "Object", ELFSymbolType::Object);
    IO.enumCase(SymbolType, "TLS", ELFSymbolType::TLS);
    IO.enumCase(SymbolType, "Unknown", ELFSymbolType::Unknown);
    // Treat other symbol types as noise, and map to Unknown.
    if (!IO.outputting() && IO.matchEnumFallback())
      SymbolType = ELFSymbolType::Unknown;
  }
};

/// YAML traits for ELFArch.
template <> struct ScalarTraits<ELFArchMapper> {
  static void output(const ELFArchMapper &Value, void *,
                     llvm::raw_ostream &Out) {
    // Map from integer to architecture string.
    switch (Value) {
    case (ELFArch)ELF::EM_X86_64:
      Out << "x86_64";
      break;
    case (ELFArch)ELF::EM_AARCH64:
      Out << "AArch64";
      break;
    case (ELFArch)ELF::EM_NONE:
    default:
      Out << "Unknown";
    }
  }

  static StringRef input(StringRef Scalar, void *, ELFArchMapper &Value) {
    // Map from architecture string to integer.
    Value = StringSwitch<ELFArch>(Scalar)
                .Case("x86_64", ELF::EM_X86_64)
                .Case("AArch64", ELF::EM_AARCH64)
                .Case("Unknown", ELF::EM_NONE)
                .Default(ELF::EM_NONE);

    // Returning empty StringRef indicates successful parse.
    return StringRef();
  }

  // Don't place quotation marks around architecture value.
  static QuotingType mustQuote(StringRef) { return QuotingType::None; }
};

/// YAML traits for TbeVersion.
template <> struct ScalarTraits<VersionTuple> {
  static void output(const VersionTuple &Value, void *,
                     llvm::raw_ostream &Out) {
    Out << Value.getAsString();
  }

  static StringRef input(StringRef Scalar, void *, VersionTuple &Value) {
    if (Value.tryParse(Scalar))
      return StringRef("Can't parse version: invalid version format.");

    if (Value > TBEVersionCurrent)
      return StringRef("Unsupported TBE version.");

    // Returning empty StringRef indicates successful parse.
    return StringRef();
  }

  // Don't place quotation marks around version value.
  static QuotingType mustQuote(StringRef) { return QuotingType::None; }
};

/// YAML traits for ELFSymbol.
template <> struct MappingTraits<ELFSymbol> {
  static void mapping(IO &IO, ELFSymbol &Symbol) {
    IO.mapRequired("Type", Symbol.Type);
    // The need for symbol size depends on the symbol type.
    if (Symbol.Type == ELFSymbolType::NoType) {
      IO.mapOptional("Size", Symbol.Size, (uint64_t)0);
    } else if (Symbol.Type == ELFSymbolType::Func) {
      Symbol.Size = 0;
    } else {
      IO.mapRequired("Size", Symbol.Size);
    }
    IO.mapOptional("Undefined", Symbol.Undefined, false);
    IO.mapOptional("Warning", Symbol.Warning);
  }

  // Compacts symbol information into a single line.
  static const bool flow = true;
};

/// YAML traits for set of ELFSymbols.
template <> struct CustomMappingTraits<std::set<ELFSymbol>> {
  static void inputOne(IO &IO, StringRef Key, std::set<ELFSymbol> &Set) {
    ELFSymbol Sym(Key.str());
    IO.mapRequired(Key.str().c_str(), Sym);
    Set.insert(Sym);
  }

  static void output(IO &IO, std::set<ELFSymbol> &Set) {
    for (auto &Sym : Set)
      IO.mapRequired(Sym.Name.c_str(), const_cast<ELFSymbol &>(Sym));
  }
};

/// YAML traits for generic string vectors (i.e. list of needed libraries).
template <> struct SequenceTraits<std::vector<std::string>> {
  static size_t size(IO &IO, std::vector<std::string> &List) {
    return List.size();
  }

  static std::string &element(IO &IO, std::vector<std::string> &List,
                              size_t Index) {
    if (Index >= List.size())
      List.resize(Index + 1);
    return List[Index];
  }

  // Compacts list of needed libraries into a single line.
  static const bool flow = true;
};

/// YAML traits for ELFStub objects.
template <> struct MappingTraits<ELFStub> {
  static void mapping(IO &IO, ELFStub &Stub) {
    if (!IO.mapTag("!tapi-tbe", true))
      IO.setError("Not a .tbe YAML file.");
    IO.mapRequired("TbeVersion", Stub.TbeVersion);
    IO.mapRequired("SoName", Stub.SoName);
    IO.mapRequired("Arch", (ELFArchMapper &)Stub.Arch);
    IO.mapOptional("NeededLibs", Stub.NeededLibs);
    IO.mapRequired("Symbols", Stub.Symbols);
  }
};

} // end namespace yaml
} // end namespace llvm

std::unique_ptr<ELFStub> TBEHandler::readFile(StringRef Buf) {
  yaml::Input YamlIn(Buf);
  std::unique_ptr<ELFStub> Stub(new ELFStub());
  YamlIn >> *Stub;
  if (YamlIn.error())
    return nullptr;
  return Stub;
}

Error TBEHandler::writeFile(raw_ostream &OS, const ELFStub &Stub) {
  yaml::Output YamlOut(OS, NULL, /*WrapColumn =*/0);

  YamlOut << const_cast<ELFStub &>(Stub);
  return Error::success();
}
