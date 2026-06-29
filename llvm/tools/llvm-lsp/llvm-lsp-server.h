//===-- llvm-lsp-server.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_LSP_SERVER_H
#define LLVM_TOOLS_LLVM_LSP_SERVER_H

#include <sstream>

#include "IRDocument.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LSP/Protocol.h"
#include "llvm/Support/LSP/Transport.h"

namespace llvm {

class LspServer {
  lsp::MessageHandler MessageHandler;

  bool ShutDownRequested = false;

  std::unordered_map<std::string, std::unique_ptr<IRDocument>> OpenDocuments;
  std::unordered_map<std::string, std::string> SVGToIRMap;

public:
  LspServer(lsp::JSONTransport &Transport)
      : MessageHandler(Transport), Transport(Transport) {
    lsp::Logger::info("Starting LLVM LSP Server");
  }

  // Runs LSP server
  llvm::Error run();

  // Sends a message to client as INFO notification
  void sendInfo(const std::string &Message);

  // Sends a message to client as ERROR notification
  void sendError(const std::string &Message);

  // The process exit code, should be success only if the State is Exitted
  int getExitCode() { return 1 - ShutDownRequested; }

private:
  // ---------- Functions to handle various RPC calls -----------------------

  // initialize
  void handleRequestInitialize(const lsp::InitializeParams &Params,
                               lsp::Callback<llvm::json::Value> Reply);

  // shutdown
  void handleRequestShutdown(const lsp::NoParams &Params,
                             lsp::Callback<std::nullptr_t> Reply);

  // textDocument/didOpen
  void handleNotificationTextDocumentDidOpen(
      const lsp::DidOpenTextDocumentParams &Params);

  // textDocument/references
  void
  handleRequestGetReferences(const lsp::ReferenceParams &Params,
                             lsp::Callback<std::vector<lsp::Location>> Reply);

  // textDocument/documentSymbol
  void handleRequestTextDocumentDocumentSymbol(
      const lsp::DocumentSymbolParams &Params,
      lsp::Callback<std::vector<lsp::DocumentSymbol>> Reply);

  // Identifies RPC Call and dispatches the handling to other methods
  bool registerMessageHandlers();

  lsp::OutgoingNotification<lsp::ShowMessageParams> ShowMessageSender;

  lsp::JSONTransport &Transport;
};

} // namespace llvm

#endif // LLVM_TOOLS_LLVM_LSP_SERVER_H
