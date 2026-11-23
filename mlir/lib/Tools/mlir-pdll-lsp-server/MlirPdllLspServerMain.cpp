//===- MlirPdllLspServerMain.cpp - MLIR PDLL Language Server main ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-pdll-lsp-server/MlirPdllLspServerMain.h"
#include "LSPServer.h"
#include "PDLLServer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LSP/Logging.h"
#include "llvm/Support/LSP/Transport.h"
#include "llvm/Support/Program.h"

using namespace mlir;
using namespace mlir::lsp;

using llvm::lsp::JSONStreamStyle;
using llvm::lsp::Logger;

LogicalResult mlir::MlirPdllLspServerMain(int argc, char **argv) {
  llvm::cl::opt<JSONStreamStyle> inputStyle{
      "input-style",
      llvm::cl::desc("Input JSON stream encoding"),
      llvm::cl::values(clEnumValN(JSONStreamStyle::Standard, "standard",
                                  "usual LSP protocol"),
                       clEnumValN(JSONStreamStyle::Delimited, "delimited",
                                  "messages delimited by `// -----` lines, "
                                  "with // comment support")),
      llvm::cl::init(JSONStreamStyle::Standard),
      llvm::cl::Hidden,
  };
  llvm::cl::opt<bool> litTest{
      "lit-test",
      llvm::cl::desc(
          "Abbreviation for -input-style=delimited -pretty -log=verbose. "
          "Intended to simplify lit tests"),
      llvm::cl::init(false),
  };
  llvm::cl::opt<Logger::Level> logLevel{
      "log",
      llvm::cl::desc("Verbosity of log messages written to stderr"),
      llvm::cl::values(
          clEnumValN(Logger::Level::Error, "error", "Error messages only"),
          clEnumValN(Logger::Level::Info, "info",
                     "High level execution tracing"),
          clEnumValN(Logger::Level::Debug, "verbose", "Low level details")),
      llvm::cl::init(Logger::Level::Info),
  };
  llvm::cl::opt<bool> prettyPrint{
      "pretty",
      llvm::cl::desc("Pretty-print JSON output"),
      llvm::cl::init(false),
  };
  llvm::cl::list<std::string> extraIncludeDirs(
      "pdll-extra-dir", llvm::cl::desc("Extra directory of include files"),
      llvm::cl::value_desc("directory"), llvm::cl::Prefix);
  llvm::cl::list<std::string> compilationDatabases(
      "pdll-compilation-database",
      llvm::cl::desc("Compilation YAML databases containing additional "
                     "compilation information for .pdll files"));

  llvm::cl::ParseCommandLineOptions(argc, argv, "PDLL LSP Language Server");

  if (litTest) {
    inputStyle = JSONStreamStyle::Delimited;
    logLevel = Logger::Level::Debug;
    prettyPrint = true;
  }

  // Configure the logger.
  Logger::setLogLevel(logLevel);

  // Configure the transport used for communication.
  llvm::sys::ChangeStdinToBinary();
  llvm::lsp::JSONTransport transport(stdin, llvm::outs(), inputStyle,
                                     prettyPrint);

  // Configure the servers and start the main language server.
  PDLLServer::Options options(compilationDatabases, extraIncludeDirs);
  PDLLServer server(options);
  return runPdllLSPServer(server, transport);
}
