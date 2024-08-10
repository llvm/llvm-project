//===- toyc.cpp - The Toy Compiler ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the entry point for the Toy compiler.
//
//===----------------------------------------------------------------------===//

#include "toy/AST.h"
#include "toy/Lexer.h"
#include "toy/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>
#include <system_error>

using namespace toy;
namespace cl = llvm::cl;

// clang-format off
// Cratels: 处理 option
// clang-format on
static cl::opt<std::string> inputFilename(
    // clang-format off
    // Cratels: 根据位置而不是根据前缀来进行 option 的解析。第一个不是根据前缀来解析的 option 会给它。这也意味着最多只能有一个 positional 的参数
    // clang-format on
    cl::Positional, cl::desc("<input toy file>"), cl::init("-"),
    cl::value_desc("filename"));

// static cl::opt<std::string> userName("name", cl::desc("User name"),
//                                      cl::init("-"),
//                                      cl::value_desc("user name"));
namespace {
enum Action { None, DumpAST };
} // namespace

static cl::opt<enum Action> emitAction(
    // clang-format off
    // Cratels: prefix 的前缀为 emit，解析--emit=后面的值给 emitAction
    // clang-format on
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")));

/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
// clang-format off
// Cratels: ModuleAST代指一个 EntryPoint，是一个 AST 的基本块
// clang-format on
std::unique_ptr<toy::ModuleAST> parseInputFile(llvm::StringRef filename) {
  // clang-format off
  // Cratels: 接受输入文件路径或者直接输入文本内容
  // clang-format on
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }

  // clang-format off
  // Cratels: 获得文件内容
  // clang-format on
  auto buffer = fileOrErr.get()->getBuffer();

  llvm::outs() << "文件内容：" << buffer << "\n";

  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  Parser parser(lexer);
  return parser.parseModule();
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  // clang-format off
  // Cratels: 文件名
  // clang-format on
  llvm::outs() << inputFilename << "\n";

  // if (userName != "Cratels") {
  //   return -1;
  // }

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  switch (emitAction) {
  case Action::DumpAST:
    dump(*moduleAST);
    return 0;
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }

  return 0;
}
