//===- LSPServer.cpp - MLIR Language Server -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LSPServer.h"
#include "MLIRServer.h"
#include "lsp/Logging.h"
#include "lsp/Protocol.h"
#include "lsp/Transport.h"
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

  MLIRServer &server;
  JSONTransport &transport;

  /// Used to indicate that the 'shutdown' request was received from the
  /// Language Server client.
  bool shutdownRequestReceived = false;
};

//===----------------------------------------------------------------------===//
// Initialization

void LSPServer::Impl::onInitialize(const InitializeParams &params,
                                   Callback<llvm::json::Value> reply) {
  llvm::json::Object serverCaps{
      {"textDocumentSync",
       llvm::json::Object{
           {"openClose", true},
           {"change", (int)TextDocumentSyncKind::Full},
           {"save", true},
       }},
      {"definitionProvider", true},
      {"referencesProvider", true},
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
  server.addOrUpdateDocument(params.textDocument.uri, params.textDocument.text);
}
void LSPServer::Impl::onDocumentDidClose(
    const DidCloseTextDocumentParams &params) {
  server.removeDocument(params.textDocument.uri);
}
void LSPServer::Impl::onDocumentDidChange(
    const DidChangeTextDocumentParams &params) {
  // TODO: We currently only support full document updates, we should refactor
  // to avoid this.
  if (params.contentChanges.size() != 1)
    return;
  server.addOrUpdateDocument(params.textDocument.uri,
                             params.contentChanges.front().text);
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
// LSPServer
//===----------------------------------------------------------------------===//

LSPServer::LSPServer(MLIRServer &server, JSONTransport &transport)
    : impl(std::make_unique<Impl>(server, transport)) {}
LSPServer::~LSPServer() {}

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
