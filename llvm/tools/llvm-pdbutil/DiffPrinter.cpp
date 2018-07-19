
#include "DiffPrinter.h"

#include "llvm/Support/FormatAdapters.h"

using namespace llvm;
using namespace llvm::pdb;

namespace {
struct Colorize {
  Colorize(raw_ostream &OS, DiffResult Result) : OS(OS) {
    if (!OS.has_colors())
      return;
    switch (Result) {
    case DiffResult::IDENTICAL:
      OS.changeColor(raw_ostream::Colors::GREEN, false);
      break;
    case DiffResult::EQUIVALENT:
      OS.changeColor(raw_ostream::Colors::YELLOW, true);
      break;
    default:
      OS.changeColor(raw_ostream::Colors::RED, false);
      break;
    }
  }

  ~Colorize() {
    if (OS.has_colors())
      OS.resetColor();
  }

  raw_ostream &OS;
};
}

DiffPrinter::DiffPrinter(uint32_t Indent, StringRef Header,
                         uint32_t PropertyWidth, uint32_t FieldWidth,
                         bool Result, bool Fields, raw_ostream &Stream)
    : PrintResult(Result), PrintValues(Fields), Indent(Indent),
      PropertyWidth(PropertyWidth), FieldWidth(FieldWidth), OS(Stream) {
  printHeaderRow();
  printFullRow(Header);
}

DiffPrinter::~DiffPrinter() {}

uint32_t DiffPrinter::tableWidth() const {
  // `|`
  uint32_t W = 1;

  // `<width>|`
  W += PropertyWidth + 1;

  if (PrintResult) {
    // ` I |`
    W += 4;
  }

  if (PrintValues) {
    // `<width>|<width>|`
    W += 2 * (FieldWidth + 1);
  }
  return W;
}

void DiffPrinter::printFullRow(StringRef Text) {
  newLine();
  printValue(Text, DiffResult::UNSPECIFIED, AlignStyle::Center,
             tableWidth() - 2, true);
  printSeparatorRow();
}

void DiffPrinter::printSeparatorRow() {
  newLine();
  OS << formatv("{0}", fmt_repeat('-', PropertyWidth));
  if (PrintResult) {
    OS << '+';
    OS << formatv("{0}", fmt_repeat('-', 3));
  }
  if (PrintValues) {
    OS << '+';
    OS << formatv("{0}", fmt_repeat('-', FieldWidth));
    OS << '+';
    OS << formatv("{0}", fmt_repeat('-', FieldWidth));
  }
  OS << '|';
}

void DiffPrinter::printHeaderRow() {
  newLine('-');
  OS << formatv("{0}", fmt_repeat('-', tableWidth() - 1));
}

void DiffPrinter::newLine(char InitialChar) {
  OS << "\n";
  OS.indent(Indent) << InitialChar;
}

void DiffPrinter::printExplicit(StringRef Property, DiffResult C,
                                StringRef Left, StringRef Right) {
  newLine();
  printValue(Property, DiffResult::UNSPECIFIED, AlignStyle::Right,
             PropertyWidth, true);
  printResult(C);
  printValue(Left, C, AlignStyle::Center, FieldWidth, false);
  printValue(Right, C, AlignStyle::Center, FieldWidth, false);
  printSeparatorRow();
}

void DiffPrinter::printResult(DiffResult Result) {
  if (!PrintResult)
    return;
  switch (Result) {
  case DiffResult::DIFFERENT:
    printValue("D", Result, AlignStyle::Center, 3, true);
    break;
  case DiffResult::EQUIVALENT:
    printValue("E", Result, AlignStyle::Center, 3, true);
    break;
  case DiffResult::IDENTICAL:
    printValue("I", Result, AlignStyle::Center, 3, true);
    break;
  case DiffResult::UNSPECIFIED:
    printValue(" ", Result, AlignStyle::Center, 3, true);
    break;
  }
}

void DiffPrinter::printValue(StringRef Value, DiffResult C, AlignStyle Style,
                             uint32_t Width, bool Force) {
  if (!Force && !PrintValues)
    return;

  if (Style == AlignStyle::Right)
    --Width;

  std::string FormattedItem =
      formatv("{0}", fmt_align(Value, Style, Width)).str();
  if (C != DiffResult::UNSPECIFIED) {
    Colorize Color(OS, C);
    OS << FormattedItem;
  } else
    OS << FormattedItem;
  if (Style == AlignStyle::Right)
    OS << ' ';
  OS << '|';
}
