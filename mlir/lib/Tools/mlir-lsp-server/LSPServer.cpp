//===- LSPServer.cpp - MLIR Language Server -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LSPServer.h"
#include "../lsp-server-support/Logging.h"
#include "../lsp-server-support/Protocol.h"
#include "../lsp-server-support/Transport.h"
#include "MLIRServer.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringMap.h"

#define DEBUG_TYPE "mlir-lsp-server"

using namespace mlir;
using namespace mlir::lsp;

//===----------------------------------------------------------------------===//
// LSPServer::Impl
//===----------------------------------------------------------------------===//

struct LSPServer::Impl {
  Impl(MLIRServer &server, JSONTransport &transport)
      : server(server), transport(transport) {}

  //===--------------------------------------------------------------------===//
  // Initialization

  void onInitialize(const InitializeParams &params,
                    Callback<llvm::json::Value> reply);
  void onInitialized(const InitializedParams &params);
  void onShutdown(const NoParams &params, Callback<std::nullptr_t> reply);

  //===--------------------------------------------------------------------===//
  // Document Change

  void onDocumentDidOpen(const DidOpenTextDocumentParams &params);
  void onDocumentDidClose(const DidCloseTextDocumentParams &params);
  void onDocumentDidChange(const DidChangeTextDocumentParams &params);

  //===--------------------------------------------------------------------===//
  // Definitions and References

  void onGoToDefinition(const TextDocumentPositionParams &params,
                        Callback<std::vector<Location>> reply);
  void onReference(const ReferenceParams &params,
                   Callback<std::vector<Location>> reply);

  //===--------------------------------------------------------------------===//
  // Hover

  void onHover(const TextDocumentPositionParams &params,
               Callback<Optional<Hover>> reply);

  //===--------------------------------------------------------------------===//
  // Document Symbols

  void onDocumentSymbol(const DocumentSymbolParams &params,
                        Callback<std::vector<DocumentSymbol>> reply);

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  MLIRServer &server;
  JSONTransport &transport;

  /// An outgoing notification used to send diagnostics to the client when they
  /// are ready to be processed.
  OutgoingNotification<PublishDiagnosticsParams> publishDiagnostics;

  /// Used to indicate that the 'shutdown' request was received from the
  /// Language Server client.
  bool shutdownRequestReceived = false;
};

//===----------------------------------------------------------------------===//
// Initialization

void LSPServer::Impl::onInitialize(const InitializeParams &params,
                                   Callback<llvm::json::Value> reply) {
  // Send a response with the capabilities of this server.
  llvm::json::Object serverCaps{
      {"textDocumentSync",
       llvm::json::Object{
           {"openClose", true},
           {"change", (int)TextDocumentSyncKind::Full},
           {"save", true},
       }},
      {"definitionProvider", true},
      {"referencesProvider", true},
      {"hoverProvider", true},

      // For now we only support documenting symbols when the client supports
      // hierarchical symbols.
      {"documentSymbolProvider",
       params.capabilities.hierarchicalDocumentSymbol},
  };

  llvm::json::Object result{
      {{"serverInfo",
        llvm::json::Object{{"name", "mlir-lsp-server"}, {"version", "0.0.0"}}},
       {"capabilities", std::move(serverCaps)}}};
  reply(std::move(result));
}
void LSPServer::Impl::onInitialized(const InitializedParams &) {}
void LSPServer::Impl::onShutdown(const NoParams &,
                                 Callback<std::nullptr_t> reply) {
  shutdownRequestReceived = true;
  reply(nullptr);
}

//===----------------------------------------------------------------------===//
// Document Change

void LSPServer::Impl::onDocumentDidOpen(
    const DidOpenTextDocumentParams &params) {
  PublishDiagnosticsParams diagParams(params.textDocument.uri,
                                      params.textDocument.version);
  server.addOrUpdateDocument(params.textDocument.uri, params.textDocument.text,
                             params.textDocument.version,
                             diagParams.diagnostics);

  // Publish any recorded diagnostics.
  publishDiagnostics(diagParams);
}
void LSPServer::Impl::onDocumentDidClose(
    const DidCloseTextDocumentParams &params) {
  Optional<int64_t> version = server.removeDocument(params.textDocument.uri);
  if (!version)
    return;

  // Empty out the diagnostics shown for this document. This will clear out
  // anything currently displayed by the client for this document (e.g. in the
  // "Problems" pane of VSCode).
  publishDiagnostics(
      PublishDiagnosticsParams(params.textDocument.uri, *version));
}
void LSPServer::Impl::onDocumentDidChange(
    const DidChangeTextDocumentParams &params) {
  // TODO: We currently only support full document updates, we should refactor
  // to avoid this.
  if (params.contentChanges.size() != 1)
    return;
  PublishDiagnosticsParams diagParams(params.textDocument.uri,
                                      params.textDocument.version);
  server.addOrUpdateDocument(
      params.textDocument.uri, params.contentChanges.front().text,
      params.textDocument.version, diagParams.diagnostics);

  // Publish any recorded diagnostics.
  publishDiagnostics(diagParams);
}

//===----------------------------------------------------------------------===//
// Definitions and References

void LSPServer::Impl::onGoToDefinition(const TextDocumentPositionParams &params,
                                       Callback<std::vector<Location>> reply) {
  std::vector<Location> locations;
  server.getLocationsOf(params.textDocument.uri, params.position, locations);
  reply(std::move(locations));
}

void LSPServer::Impl::onReference(const ReferenceParams &params,
                                  Callback<std::vector<Location>> reply) {
  std::vector<Location> locations;
  server.findReferencesOf(params.textDocument.uri, params.position, locations);
  reply(std::move(locations));
}

//===----------------------------------------------------------------------===//
// Hover

void LSPServer::Impl::onHover(const TextDocumentPositionParams &params,
                              Callback<Optional<Hover>> reply) {
  reply(server.findHover(params.textDocument.uri, params.position));
}

//===----------------------------------------------------------------------===//
// Document Symbols

void LSPServer::Impl::onDocumentSymbol(
    const DocumentSymbolParams &params,
    Callback<std::vector<DocumentSymbol>> reply) {
  std::vector<DocumentSymbol> symbols;
  server.findDocumentSymbols(params.textDocument.uri, symbols);
  reply(std::move(symbols));
}

//===----------------------------------------------------------------------===//
// LSPServer
//===----------------------------------------------------------------------===//

LSPServer::LSPServer(MLIRServer &server, JSONTransport &transport)
    : impl(std::make_unique<Impl>(server, transport)) {}
LSPServer::~LSPServer() = default;

LogicalResult LSPServer::run() {
  MessageHandler messageHandler(impl->transport);

  // Initialization
  messageHandler.method("initialize", impl.get(), &Impl::onInitialize);
  messageHandler.notification("initialized", impl.get(), &Impl::onInitialized);
  messageHandler.method("shutdown", impl.get(), &Impl::onShutdown);

  // Document Changes
  messageHandler.notification("textDocument/didOpen", impl.get(),
                              &Impl::onDocumentDidOpen);
  messageHandler.notification("textDocument/didClose", impl.get(),
                              &Impl::onDocumentDidClose);
  messageHandler.notification("textDocument/didChange", impl.get(),
                              &Impl::onDocumentDidChange);

  // Definitions and References
  messageHandler.method("textDocument/definition", impl.get(),
                        &Impl::onGoToDefinition);
  messageHandler.method("textDocument/references", impl.get(),
                        &Impl::onReference);

  // Hover
  messageHandler.method("textDocument/hover", impl.get(), &Impl::onHover);

  // Document Symbols
  messageHandler.method("textDocument/documentSymbol", impl.get(),
                        &Impl::onDocumentSymbol);

  // Diagnostics
  impl->publishDiagnostics =
      messageHandler.outgoingNotification<PublishDiagnosticsParams>(
          "textDocument/publishDiagnostics");

  // Run the main loop of the transport.
  LogicalResult result = success();
  if (llvm::Error error = impl->transport.run(messageHandler)) {
    Logger::error("Transport error: {0}", error);
    llvm::consumeError(std::move(error));
    result = failure();
  } else {
    result = success(impl->shutdownRequestReceived);
  }
  return result;
}
