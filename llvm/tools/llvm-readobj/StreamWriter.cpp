#include "StreamWriter.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Format.h"
#include <cctype>

using namespace llvm::support;

namespace llvm {

raw_ostream &operator<<(raw_ostream &OS, const HexNumber& Value) {
  OS << "0x" << to_hexString(Value.Value);
  return OS;
}

const std::string to_hexString(uint64_t Value, bool UpperCase) {
  std::string number;
  llvm::raw_string_ostream stream(number);
  stream << format_hex_no_prefix(Value, 1, UpperCase);
  return stream.str();
}

const std::string to_string(uint64_t Value) {
  std::string number;
  llvm::raw_string_ostream stream(number);
  stream << format_decimal(Value, 1);
  return stream.str();
}

void StreamWriter::printBinaryImpl(StringRef Label, StringRef Str,
                                   ArrayRef<uint8_t> Data, bool Block) {
  if (Data.size() > 16)
    Block = true;

  if (Block) {
    startLine() << Label;
    if (Str.size() > 0)
      OS << ": " << Str;
    OS << " (\n";
    for (size_t addr = 0, end = Data.size(); addr < end; addr += 16) {
      startLine() << format("  %04" PRIX64 ": ", uint64_t(addr));
      // Dump line of hex.
      for (size_t i = 0; i < 16; ++i) {
        if (i != 0 && i % 4 == 0)
          OS << ' ';
        if (addr + i < end)
          OS << hexdigit((Data[addr + i] >> 4) & 0xF, false)
             << hexdigit(Data[addr + i] & 0xF, false);
        else
          OS << "  ";
      }
      // Print ascii.
      OS << "  |";
      for (std::size_t i = 0; i < 16 && addr + i < end; ++i) {
        if (std::isprint(Data[addr + i] & 0xFF))
          OS << Data[addr + i];
        else
          OS << ".";
      }
      OS << "|\n";
    }

    startLine() << ")\n";
  } else {
    startLine() << Label << ":";
    if (Str.size() > 0)
      OS << " " << Str;
    OS << " (";
    for (size_t i = 0; i < Data.size(); ++i) {
      if (i > 0)
        OS << " ";

      OS << format("%02X", static_cast<int>(Data[i]));
    }
    OS << ")\n";
  }
}

} // namespace llvm
