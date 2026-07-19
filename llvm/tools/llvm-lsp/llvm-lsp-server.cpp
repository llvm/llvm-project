//===-- llvm-lsp-server.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/BasicBlock.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LSP/Logging.h"
#include "llvm/Support/Program.h"

#include "llvm-lsp-server.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace llvm;

static cl::OptionCategory LlvmLspServerCategory("llvm-lsp-server options");

static cl::opt<lsp::Logger::Level> LogLevel(
    "log-level", cl::desc("Log level"), cl::init(lsp::Logger::Level::Info),
    cl::values(clEnumValN(lsp::Logger::Level::Info, "info", "Info"),
               clEnumValN(lsp::Logger::Level::Debug, "debug", "Debug"),
               clEnumValN(lsp::Logger::Level::Error, "error", "Error")),
    cl::cat(LlvmLspServerCategory));

llvm::Error LspServer::run() {
  registerMessageHandlers();
  return Transport.run(MessageHandler);
}

void LspServer::sendInfo(const std::string &Message) {
  ShowMessageSender(lsp::ShowMessageParams(lsp::MessageType::Info, Message));
}

void LspServer::sendError(const std::string &Message) {
  ShowMessageSender(lsp::ShowMessageParams(lsp::MessageType::Error, Message));
}

void LspServer::handleRequestInitialize(
    const lsp::InitializeParams &Params,
    lsp::Callback<llvm::json::Value> Reply) {

  // clang-format off
  json::Object ResponseParams{
    {"capabilities",
      json::Object{
          {"textDocumentSync",
          json::Object{
              {"openClose", true},
              {"change", 0}, // We dont want to sync the documents.
          }
        },
        {"referencesProvider", true},
        {"documentSymbolProvider", true},
      }
    }
  };
  // clang-format on
  Reply(json::Value(std::move(ResponseParams)));
}

void LspServer::handleRequestShutdown(const lsp::NoParams &Params,
                                      lsp::Callback<std::nullptr_t> Reply) {
  // Do cleanup if needed
  ShutDownRequested = true;
  Reply(nullptr);
}


bool LspServer::registerMessageHandlers() {
  MessageHandler.method("initialize", this,
                        &LspServer::handleRequestInitialize);

  // Handle recieving messages
  MessageHandler.notification(
      "textDocument/didOpen", this,
      &LspServer::handleNotificationTextDocumentDidOpen);
  MessageHandler.method("textDocument/references", this,
                        &LspServer::handleRequestGetReferences);
  MessageHandler.method("textDocument/documentSymbol", this,
                        &LspServer::handleRequestTextDocumentDocumentSymbol);

  // Setup posting of messages
  ShowMessageSender =
      MessageHandler.outgoingNotification<lsp::ShowMessageParams>(
          "window/showMessage");

  // Return true to indicate handlers were registered successfully
  return true;
}

int main(int argc, char **argv) {
  cl::HideUnrelatedOptions(LlvmLspServerCategory);
  cl::ParseCommandLineOptions(argc, argv, "LLVM LSP Language Server");

  llvm::sys::ChangeStdinToBinary();
  lsp::JSONTransport Transport(stdin, llvm::outs());

  LspServer LS(Transport);

  lsp::Logger::setLogLevel(LogLevel);

  auto LSResult = LS.run();
  if (!LSResult)
    lsp::Logger::error("Error while running Language Server: {}", LSResult);

  return LS.getExitCode();
}
