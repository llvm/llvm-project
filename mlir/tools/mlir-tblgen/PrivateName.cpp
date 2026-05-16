//===- PrivateName.cpp - Private name obfuscation for mlir-tblgen ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Plumbing for the `--mlir-private-name-obfuscator` mlir-tblgen flag. This
// is opt-in release-engineering tooling for downstream MLIR builds; the
// rest of the mlir-tblgen code generator does not need to know about it.
//
// When the user passes `--mlir-private-name-obfuscator=<cmd>`, mlir-tblgen
// pipes each name marked `isPrivate` into `<cmd>` via stdin, reads the
// first whitespace-delimited token of stdout, and uses that token (with a
// leading `_`) as the obfuscated form of the name. Results are cached for
// the lifetime of the process.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/PrivateName.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdio>
#include <string>

#ifdef _WIN32
#define MLIR_TBLGEN_POPEN _popen
#define MLIR_TBLGEN_PCLOSE _pclose
#else
#define MLIR_TBLGEN_POPEN popen
#define MLIR_TBLGEN_PCLOSE pclose
#endif

using namespace mlir;
using namespace llvm;

namespace {
llvm::cl::opt<std::string> privateNameObfuscator(
    "mlir-private-name-obfuscator",
    llvm::cl::desc(
        "Shell command used to obfuscate the dialect namespace, op mnemonic, "
        "AttrDef/TypeDef mnemonic, and pass argument/name of private "
        "TableGen items. The set of private dialects is controlled by "
        "`--mlir-private-dialects`; passes are toggled individually via "
        "`let isPrivate = 1;` in ODS. For each name, mlir-tblgen runs "
        "`printf %s <name> | <cmd>` and uses the first whitespace-delimited "
        "token of stdout (prefixed with `_`) as the obfuscated form. An "
        "empty value disables obfuscation."),
    llvm::cl::init(""));

llvm::cl::list<std::string> privateDialects(
    "mlir-private-dialects", llvm::cl::CommaSeparated,
    llvm::cl::desc(
        "Comma-separated list of dialect namespaces whose ops, attributes, "
        "and types are obfuscated when `--mlir-private-name-obfuscator` is "
        "also set. Op argument attribute keys are not obfuscated."),
    llvm::cl::cb<void, std::string>([](const std::string &name) {
      mlir::tblgen::addPrivateDialect(name);
    }));

llvm::cl::opt<bool> privatePasses(
    "mlir-private-passes",
    llvm::cl::desc(
        "Treat all passes as private. When `--mlir-private-name-obfuscator` "
        "is also set, pass arguments / names are replaced with opaque "
        "identifiers and pass / option / statistic descriptions are emitted "
        "as empty strings."),
    llvm::cl::init(false));

/// Process-wide cache of `name -> obfuscated(name)`.
llvm::StringMap<std::string> &obfuscationCache() {
  static llvm::StringMap<std::string> cache;
  return cache;
}

/// Appends `s` to `out`, single-quoted with embedded single quotes escaped
/// using the POSIX `'\''` idiom.
void appendShellSingleQuoted(std::string &out, StringRef s) {
  out.push_back('\'');
  for (char c : s) {
    if (c == '\'')
      out.append("'\\''");
    else
      out.push_back(c);
  }
  out.push_back('\'');
}

/// Invokes `printf '%s' <name> | <obfuscator>` and returns `_` + first
/// whitespace-delimited token of stdout.
std::string runPrivateNameObfuscator(StringRef name) {
  std::string cmd = "printf '%s' ";
  appendShellSingleQuoted(cmd, name);
  cmd.append(" | ");
  cmd.append(privateNameObfuscator);

  FILE *f = MLIR_TBLGEN_POPEN(cmd.c_str(), "r");
  if (!f)
    llvm::report_fatal_error(
        "--mlir-private-name-obfuscator: failed to spawn obfuscator command");

  std::string out;
  char buf[256];
  while (size_t n = std::fread(buf, 1, sizeof(buf), f))
    out.append(buf, n);
  int rc = MLIR_TBLGEN_PCLOSE(f);
  if (rc != 0)
    llvm::report_fatal_error("--mlir-private-name-obfuscator: obfuscator "
                             "command exited with non-zero status");

  StringRef rest = StringRef(out).ltrim();
  size_t end = rest.find_first_of(" \t\r\n");
  StringRef token = (end == StringRef::npos) ? rest : rest.substr(0, end);
  if (token.empty())
    llvm::report_fatal_error("--mlir-private-name-obfuscator: obfuscator "
                             "command produced empty output");

  std::string result;
  result.reserve(1 + token.size());
  result.push_back('_');
  result.append(token.data(), token.size());
  return result;
}
} // namespace

bool mlir::tblgen::obfuscatePrivateNamesEnabled() {
  return !privateNameObfuscator.empty();
}

bool mlir::tblgen::arePassesPrivate() { return privatePasses; }

StringRef mlir::tblgen::obfuscatePrivateName(StringRef name) {
  if (name.empty())
    return name;
  auto &cache = obfuscationCache();
  if (auto it = cache.find(name); it != cache.end())
    return it->second;
  std::string obf = runPrivateNameObfuscator(name);
  auto inserted = cache.try_emplace(name, std::move(obf));
  return inserted.first->second;
}

std::string mlir::tblgen::maybeObfuscateDotted(StringRef name, bool isPrivate) {
  if (!isPrivate || !obfuscatePrivateNamesEnabled())
    return std::string(name);
  size_t dot = name.find('.');
  if (dot == StringRef::npos)
    return obfuscatePrivateName(name).str();
  StringRef dialect = name.substr(0, dot);
  StringRef rest = name.substr(dot + 1);
  std::string result = obfuscatePrivateName(dialect).str();
  result.push_back('.');
  result += obfuscatePrivateName(rest).str();
  return result;
}
