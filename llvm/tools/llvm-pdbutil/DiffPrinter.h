//===- DiffPrinter.h ------------------------------------------ *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_DIFFPRINTER_H
#define LLVM_TOOLS_LLVMPDBDUMP_DIFFPRINTER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/PDB/Native/RawConstants.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <list>
#include <unordered_set>

namespace std {
template <> struct hash<llvm::pdb::PdbRaw_FeatureSig> {
  typedef llvm::pdb::PdbRaw_FeatureSig argument_type;
  typedef std::size_t result_type;
  result_type operator()(argument_type Item) const {
    return std::hash<uint32_t>{}(uint32_t(Item));
  }
};
} // namespace std

namespace llvm {
namespace pdb {

class PDBFile;

enum class DiffResult { UNSPECIFIED, IDENTICAL, EQUIVALENT, DIFFERENT };

struct IdenticalDiffProvider {
  template <typename T, typename U>
  DiffResult compare(const T &Left, const U &Right) {
    return (Left == Right) ? DiffResult::IDENTICAL : DiffResult::DIFFERENT;
  }

  template <typename T> std::string format(const T &Item, bool Right) {
    return formatv("{0}", Item).str();
  }
};

struct EquivalentDiffProvider {
  template <typename T, typename U>
  DiffResult compare(const T &Left, const U &Right) {
    return (Left == Right) ? DiffResult::IDENTICAL : DiffResult::EQUIVALENT;
  }

  template <typename T> std::string format(const T &Item, bool Right) {
    return formatv("{0}", Item).str();
  }
};

class DiffPrinter {
public:
  DiffPrinter(uint32_t Indent, StringRef Header, uint32_t PropertyWidth,
              uint32_t FieldWidth, bool Result, bool Values,
              raw_ostream &Stream);
  ~DiffPrinter();

  template <typename T, typename U> struct Identical {};

  template <typename Provider = IdenticalDiffProvider, typename T, typename U>
  void print(StringRef Property, const T &Left, const U &Right,
             Provider P = Provider()) {
    std::string L = P.format(Left, false);
    std::string R = P.format(Right, true);

    DiffResult Result = P.compare(Left, Right);
    printExplicit(Property, Result, L, R);
  }

  void printExplicit(StringRef Property, DiffResult C, StringRef Left,
                     StringRef Right);

  template <typename T, typename U>
  void printExplicit(StringRef Property, DiffResult C, const T &Left,
                     const U &Right) {
    std::string L = formatv("{0}", Left).str();
    std::string R = formatv("{0}", Right).str();
    printExplicit(Property, C, StringRef(L), StringRef(R));
  }

  template <typename T, typename U>
  void diffUnorderedArray(StringRef Property, ArrayRef<T> Left,
                          ArrayRef<U> Right) {
    std::unordered_set<T> LS(Left.begin(), Left.end());
    std::unordered_set<U> RS(Right.begin(), Right.end());
    std::string Count1 = formatv("{0} element(s)", Left.size());
    std::string Count2 = formatv("{0} element(s)", Right.size());
    print(std::string(Property) + "s (set)", Count1, Count2);
    for (const auto &L : LS) {
      auto Iter = RS.find(L);
      std::string Text = formatv("{0}", L).str();
      if (Iter == RS.end()) {
        print(Property, Text, "(not present)");
        continue;
      }
      print(Property, Text, Text);
      RS.erase(Iter);
    }
    for (const auto &R : RS) {
      auto Iter = LS.find(R);
      std::string Text = formatv("{0}", R).str();
      if (Iter == LS.end()) {
        print(Property, "(not present)", Text);
        continue;
      }
      print(Property, Text, Text);
    }
  }

  template <typename ValueProvider = IdenticalDiffProvider, typename T,
            typename U>
  void diffUnorderedMap(StringRef Property, const StringMap<T> &Left,
                        const StringMap<U> &Right,
                        ValueProvider P = ValueProvider()) {
    StringMap<U> RightCopy(Right);

    std::string Count1 = formatv("{0} element(s)", Left.size());
    std::string Count2 = formatv("{0} element(s)", Right.size());
    print(std::string(Property) + "s (map)", Count1, Count2);

    for (const auto &L : Left) {
      auto Iter = RightCopy.find(L.getKey());
      if (Iter == RightCopy.end()) {
        printExplicit(L.getKey(), DiffResult::DIFFERENT, L.getValue(),
                      "(not present)");
        continue;
      }

      print(L.getKey(), L.getValue(), Iter->getValue(), P);
      RightCopy.erase(Iter);
    }

    for (const auto &R : RightCopy) {
      printExplicit(R.getKey(), DiffResult::DIFFERENT, "(not present)",
                    R.getValue());
    }
  }

  void printFullRow(StringRef Text);

private:
  uint32_t tableWidth() const;

  void printHeaderRow();
  void printSeparatorRow();
  void newLine(char InitialChar = '|');
  void printValue(StringRef Value, DiffResult C, AlignStyle Style,
                  uint32_t Width, bool Force);
  void printResult(DiffResult Result);

  bool PrintResult;
  bool PrintValues;
  uint32_t Indent;
  uint32_t PropertyWidth;
  uint32_t FieldWidth;
  raw_ostream &OS;
};
} // namespace pdb
} // namespace llvm

#endif
