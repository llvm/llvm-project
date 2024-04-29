//===- LLDBPropertyDefEmitter.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Produce the list of source languages header file fragment for the SBAPI.
//
//===----------------------------------------------------------------------===//

#include <fstream>
#include <llvm/ADT/StringRef.h>
#include <regex>

namespace lldb_private {
int EmitSBAPIDWARFEnum(int argc, char **argv) {
  std::string InputFilename;
  std::string OutputFilename;
  std::string DepFilename;
  // This command line option parser is as robust as the worst shell script.
  for (int i = 0; i < argc; ++i) {
    if (llvm::StringRef(argv[i]).ends_with("Dwarf.def"))
      InputFilename = std::string(argv[i]);
    if (llvm::StringRef(argv[i]) == "-o" && i + 1 < argc)
      OutputFilename = std::string(argv[i + 1]);
    if (llvm::StringRef(argv[i]) == "-d" && i + 1 < argc)
      DepFilename = std::string(argv[i + 1]);
  }
  std::ifstream input(InputFilename);
  std::ofstream output(OutputFilename);
  output
      << R"(//===-- SBLanguages.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBLANGUAGE_H
#define LLDB_API_SBLANGUAGE_H
/// Used by \ref SBExpressionOptions.
/// These enumerations use the same language enumerations as the DWARF
/// specification for ease of use and consistency.
enum SBSourceLanguageName : uint16_t {
)";
  std::string line;
  std::regex macro_regex(R"(^ *HANDLE_DW_LNAME *\( *([^,]+), ([^,]+), )"
                         "\"(.*)\",.*\\).*",
                         std::regex::extended);
  while (std::getline(input, line)) {
    std::smatch match;
    if (!std::regex_match(line, match, macro_regex))
      continue;

    output << "  /// " << match[3] << ".\n";
    output << "  eLanguageName" << match[2] << " = " << match[1] << ",\n";
  }
  output << "};\n\n";
  output << "#endif\n";
  // Emit the dependencies file.
  std::ofstream(DepFilename) << OutputFilename << ": " << InputFilename << '\n';
  return 0;
}
} // namespace lldb_private
