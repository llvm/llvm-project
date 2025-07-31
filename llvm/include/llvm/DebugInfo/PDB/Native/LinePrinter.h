//===- LinePrinter.h ------------------------------------------ *- C++ --*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_NATIVE_LINEPRINTER_H
#define LLVM_DEBUGINFO_PDB_NATIVE_LINEPRINTER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/DebugInfo/PDB/Native/FormatUtil.h"
#include "llvm/Support/BinaryStreamRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/raw_ostream.h"

#include <list>

// Container for filter options to control which elements will be printed.
struct FilterOptions {
  std::list<std::string> ExcludeTypes;
  std::list<std::string> ExcludeSymbols;
  std::list<std::string> ExcludeCompilands;
  std::list<std::string> IncludeTypes;
  std::list<std::string> IncludeSymbols;
  std::list<std::string> IncludeCompilands;
  uint32_t PaddingThreshold;
  uint32_t SizeThreshold;
  std::optional<uint32_t> DumpModi;
  std::optional<uint32_t> ParentRecurseDepth;
  std::optional<uint32_t> ChildrenRecurseDepth;
  std::optional<uint32_t> SymbolOffset;
  bool JustMyCode;
};

namespace llvm {
namespace msf {
class MSFStreamLayout;
} // namespace msf
namespace pdb {

class ClassLayout;
class PDBFile;
class SymbolGroup;

class LinePrinter {
  friend class WithColor;

public:
  LLVM_ABI LinePrinter(int Indent, bool UseColor, raw_ostream &Stream,
                       const FilterOptions &Filters);

  LLVM_ABI void Indent(uint32_t Amount = 0);
  LLVM_ABI void Unindent(uint32_t Amount = 0);
  LLVM_ABI void NewLine();

  LLVM_ABI void printLine(const Twine &T);
  LLVM_ABI void print(const Twine &T);
  template <typename... Ts> void formatLine(const char *Fmt, Ts &&...Items) {
    printLine(formatv(Fmt, std::forward<Ts>(Items)...));
  }
  template <typename... Ts> void format(const char *Fmt, Ts &&...Items) {
    print(formatv(Fmt, std::forward<Ts>(Items)...));
  }

  LLVM_ABI void formatBinary(StringRef Label, ArrayRef<uint8_t> Data,
                             uint64_t StartOffset);
  LLVM_ABI void formatBinary(StringRef Label, ArrayRef<uint8_t> Data,
                             uint64_t BaseAddr, uint64_t StartOffset);

  LLVM_ABI void formatMsfStreamData(StringRef Label, PDBFile &File,
                                    uint32_t StreamIdx, StringRef StreamPurpose,
                                    uint64_t Offset, uint64_t Size);
  LLVM_ABI void formatMsfStreamData(StringRef Label, PDBFile &File,
                                    const msf::MSFStreamLayout &Stream,
                                    BinarySubstreamRef Substream);
  LLVM_ABI void formatMsfStreamBlocks(PDBFile &File,
                                      const msf::MSFStreamLayout &Stream);

  bool hasColor() const { return UseColor; }
  raw_ostream &getStream() { return OS; }
  int getIndentLevel() const { return CurrentIndent; }

  LLVM_ABI bool IsClassExcluded(const ClassLayout &Class);
  LLVM_ABI bool IsTypeExcluded(llvm::StringRef TypeName, uint64_t Size);
  LLVM_ABI bool IsSymbolExcluded(llvm::StringRef SymbolName);
  LLVM_ABI bool IsCompilandExcluded(llvm::StringRef CompilandName);

  const FilterOptions &getFilters() const { return Filters; }

private:
  template <typename Iter>
  void SetFilters(std::list<Regex> &List, Iter Begin, Iter End) {
    List.clear();
    for (; Begin != End; ++Begin)
      List.emplace_back(StringRef(*Begin));
  }

  raw_ostream &OS;
  int IndentSpaces;
  int CurrentIndent;
  bool UseColor;
  const FilterOptions &Filters;

  std::list<Regex> ExcludeCompilandFilters;
  std::list<Regex> ExcludeTypeFilters;
  std::list<Regex> ExcludeSymbolFilters;

  std::list<Regex> IncludeCompilandFilters;
  std::list<Regex> IncludeTypeFilters;
  std::list<Regex> IncludeSymbolFilters;
};

struct PrintScope {
  explicit PrintScope(LinePrinter &P, uint32_t IndentLevel)
      : P(P), IndentLevel(IndentLevel) {}
  explicit PrintScope(const PrintScope &Other, uint32_t LabelWidth)
      : P(Other.P), IndentLevel(Other.IndentLevel), LabelWidth(LabelWidth) {}

  LinePrinter &P;
  uint32_t IndentLevel;
  uint32_t LabelWidth = 0;
};

inline PrintScope withLabelWidth(const PrintScope &Scope, uint32_t W) {
  return PrintScope{Scope, W};
}

struct AutoIndent {
  explicit AutoIndent(LinePrinter &L, uint32_t Amount = 0)
      : L(&L), Amount(Amount) {
    L.Indent(Amount);
  }
  explicit AutoIndent(const PrintScope &Scope) {
    L = &Scope.P;
    Amount = Scope.IndentLevel;
  }
  ~AutoIndent() {
    if (L)
      L->Unindent(Amount);
  }

  LinePrinter *L = nullptr;
  uint32_t Amount = 0;
};

template <class T>
inline raw_ostream &operator<<(LinePrinter &Printer, const T &Item) {
  return Printer.getStream() << Item;
}

enum class PDB_ColorItem {
  None,
  Address,
  Type,
  Comment,
  Padding,
  Keyword,
  Offset,
  Identifier,
  Path,
  SectionHeader,
  LiteralValue,
  Register,
};

class WithColor {
public:
  LLVM_ABI WithColor(LinePrinter &P, PDB_ColorItem C);
  LLVM_ABI ~WithColor();

  raw_ostream &get() { return OS; }

private:
  void applyColor(PDB_ColorItem C);
  raw_ostream &OS;
  bool UseColor;
};
} // namespace pdb
} // namespace llvm

#endif
