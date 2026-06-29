//===-- llvm-lsp-server.cpp -------------------------------------*- C++ -*-===//
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

#include "IRDocument.h"
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

static lsp::Position llvmFileLocToLspPosition(const FileLoc &Pos) {
  return lsp::Position(Pos.Line, Pos.Col);
}

static lsp::Range llvmFileLocRangeToLspRange(const FileLocRange &Range) {
  return lsp::Range(llvmFileLocToLspPosition(Range.Start),
                    llvmFileLocToLspPosition(Range.End));
}

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

void LspServer::handleNotificationTextDocumentDidOpen(
    const lsp::DidOpenTextDocumentParams &Params) {
  StringRef Filepath = Params.textDocument.uri.file();

  // Prepare IRDocument for Queries
  lsp::Logger::info("Creating IRDocument for {}", Filepath.str());
  OpenDocuments[Filepath.str()] = std::make_unique<IRDocument>(Filepath.str());
}

void LspServer::handleRequestGetReferences(
    const lsp::ReferenceParams &Params,
    lsp::Callback<std::vector<lsp::Location>> Reply) {
  auto Filepath = Params.textDocument.uri.file();
  auto Line = Params.position.line;
  auto Character = Params.position.character;
  assert(Line >= 0);
  assert(Character >= 0);
  std::stringstream SS;
  std::vector<lsp::Location> Result;
  const auto &Doc = OpenDocuments[Filepath.str()];
  if (Instruction *MaybeI = Doc->getInstructionAtLocation(Line, Character)) {
    auto TryAddReference = [&Result, &Params, &Doc](Instruction *I) {
      auto MaybeInstLocation = Doc->ParserContext.getInstructionLocation(I);
      if (!MaybeInstLocation)
        return;
      Result.emplace_back(
          lsp::Location(Params.textDocument.uri,
                        llvmFileLocRangeToLspRange(MaybeInstLocation.value())));
    };
    TryAddReference(MaybeI);
    for (User *U : MaybeI->users()) {
      if (auto *UserInst = dyn_cast<Instruction>(U)) {
        TryAddReference(UserInst);
      }
    }
  }

  Reply(std::move(Result));
}

void LspServer::handleRequestTextDocumentDocumentSymbol(
    const lsp::DocumentSymbolParams &Params,
    lsp::Callback<std::vector<lsp::DocumentSymbol>> Reply) {
  if (OpenDocuments.find(Params.textDocument.uri.file().str()) ==
      OpenDocuments.end()) {
    lsp::Logger::error(
        "Document in textDocument/documentSymbol request not open: {}",
        Params.textDocument.uri.file());
    return Reply(
        make_error<lsp::LSPError>(formatv("Did not open file previously {}",
                                          Params.textDocument.uri.file()),
                                  lsp::ErrorCode::InvalidParams));
  }
  auto &Doc = OpenDocuments[Params.textDocument.uri.file().str()];
  std::vector<lsp::DocumentSymbol> Result;
  for (const auto &Fn : Doc->getFunctions()) {
    lsp::DocumentSymbol Func;
    Func.name = Fn.getNameOrAsOperand();
    Func.kind = lsp::SymbolKind::Function;
    auto MaybeLoc = Doc->ParserContext.getFunctionLocation(&Fn);
    if (!MaybeLoc)
      continue;
    Func.range = llvmFileLocRangeToLspRange(*MaybeLoc);
    // FIXME: Should set the range of the function name in the definition, but
    // we currently don't know where it is
    Func.selectionRange = Func.range;
    for (const auto &BB : Fn) {
      lsp::DocumentSymbol Block;
      Block.name = BB.getNameOrAsOperand();
      // Using namespace as there is no block kind, and namespace is the closest
      Block.kind = lsp::SymbolKind::Namespace;
      Block.detail = "basic block";
      auto MaybeLoc = Doc->ParserContext.getBlockLocation(&BB);
      if (!MaybeLoc)
        continue;
      Block.range = llvmFileLocRangeToLspRange(*MaybeLoc);
      // FIXME: Should set the range of the basic block label, but we currently
      // don't know where it is
      Block.selectionRange = Block.range;
      for (const auto &I : BB) {
        lsp::DocumentSymbol Inst;
        Inst.name = I.getNameOrAsOperand();
        Inst.kind = lsp::SymbolKind::Variable;
        {
          raw_string_ostream Ss(Inst.detail);
          I.print(Ss);
        }
        auto MaybeLoc = Doc->ParserContext.getInstructionLocation(&I);
        if (!MaybeLoc)
          continue;
        Inst.range = llvmFileLocRangeToLspRange(*MaybeLoc);
        Inst.selectionRange = Inst.range;
        Block.children.emplace_back(std::move(Inst));
      }
      Func.children.emplace_back(std::move(Block));
    }
    Result.emplace_back(std::move(Func));
  }
  Reply(std::move(Result));
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
