//===- FormatUtil.h ------------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLDBTEST_FORMATUTIL_H
#define LLVM_TOOLS_LLDBTEST_FORMATUTIL_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <list>

namespace lldb_private {

class LinePrinter {
  llvm::raw_ostream &OS;
  int IndentSpaces;
  int CurrentIndent;

public:
  LinePrinter(int Indent, llvm::raw_ostream &Stream);

  void Indent(uint32_t Amount = 0);
  void Unindent(uint32_t Amount = 0);
  void NewLine();

  void printLine(const llvm::Twine &T);
  void print(const llvm::Twine &T);
  template <typename... Ts> void formatLine(const char *Fmt, Ts &&... Items) {
    printLine(llvm::formatv(Fmt, std::forward<Ts>(Items)...));
  }
  template <typename... Ts> void format(const char *Fmt, Ts &&... Items) {
    print(llvm::formatv(Fmt, std::forward<Ts>(Items)...));
  }

  void formatBinary(llvm::StringRef Label, llvm::ArrayRef<uint8_t> Data,
                    uint32_t StartOffset);
  void formatBinary(llvm::StringRef Label, llvm::ArrayRef<uint8_t> Data,
                    uint64_t BaseAddr, uint32_t StartOffset);

  llvm::raw_ostream &getStream() { return OS; }
  int getIndentLevel() const { return CurrentIndent; }
};

struct AutoIndent {
  explicit AutoIndent(LinePrinter &L, uint32_t Amount = 0)
      : L(&L), Amount(Amount) {
    L.Indent(Amount);
  }
  ~AutoIndent() {
    if (L)
      L->Unindent(Amount);
  }

  LinePrinter *L = nullptr;
  uint32_t Amount = 0;
};

template <class T>
inline llvm::raw_ostream &operator<<(LinePrinter &Printer, const T &Item) {
  Printer.getStream() << Item;
  return Printer.getStream();
}

} // namespace lldb_private

#endif
