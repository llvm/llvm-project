//===-- StoringDiagnosticConsumer.h -----------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2023 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StoringDiagnosticConsumer_h_
#define liblldb_StoringDiagnosticConsumer_h_

#include "Plugins/ExpressionParser/Swift/SwiftDiagnostic.h"
#include "Plugins/Language/Swift/LogChannelSwift.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/StreamString.h"

#include "swift/AST/DiagnosticEngine.h"
#include "swift/AST/DiagnosticsClangImporter.h"
#include "swift/AST/DiagnosticsSema.h"
#include "swift/Basic/DiagnosticOptions.h"
#include "swift/Basic/SourceManager.h"
#include "swift/Frontend/PrintingDiagnosticConsumer.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Process.h"

namespace lldb_private {

class StoringDiagnosticConsumer : public swift::DiagnosticConsumer {
public:
  StoringDiagnosticConsumer(SwiftASTContext &ast_context)
      : m_ast_context(ast_context) {
    m_ast_context.GetDiagnosticEngine().resetHadAnyError();
    m_ast_context.GetDiagnosticEngine().addConsumer(*this);
  }

  ~StoringDiagnosticConsumer() {
    m_ast_context.GetDiagnosticEngine().takeConsumers();
  }

  /// Consume a Diagnostic from the Swift compiler.
  void handleDiagnostic(swift::SourceManager &source_mgr,
                        const swift::DiagnosticInfo &info) override {
    llvm::StringRef buffer_name = "<anonymous>";
    unsigned buffer_id = 0;
    std::pair<unsigned, unsigned> line_col = {0, 0};

    llvm::SmallString<256> text;
    {
      llvm::raw_svector_ostream out(text);
      swift::DiagnosticEngine::formatDiagnosticText(out, info.FormatString,
                                                    info.FormatArgs);
    }

    swift::SourceLoc source_loc = info.Loc;
    if (source_loc.isValid()) {
      buffer_id = source_mgr.findBufferContainingLoc(source_loc);
      buffer_name = source_mgr.getDisplayNameForLoc(source_loc);
      line_col = source_mgr.getPresumedLineAndColumnForLoc(source_loc);
    }

    bool use_fixits = false;
    // Determine what kind of diagnostic we're emitting, and whether
    // we want to use its fixits:
    lldb::Severity severity;
    llvm::SourceMgr::DiagKind source_mgr_kind;
    std::string kind_str;
    switch (info.Kind) {
    case swift::DiagnosticKind::Error:
      severity = lldb::eSeverityError;
      source_mgr_kind = llvm::SourceMgr::DK_Error;
      kind_str = "error: ";
      use_fixits = true;
      break;
    case swift::DiagnosticKind::Warning:
      severity = lldb::eSeverityWarning;
      source_mgr_kind = llvm::SourceMgr::DK_Warning;
      kind_str = "warning: ";
      break;
    case swift::DiagnosticKind::Note:
      severity = lldb::eSeverityInfo;
      source_mgr_kind = llvm::SourceMgr::DK_Note;
      kind_str = "note: ";
      break;
    case swift::DiagnosticKind::Remark:
      severity = lldb::eSeverityInfo;
      source_mgr_kind = llvm::SourceMgr::DK_Remark;
      kind_str = "remark: ";
      break;
    }

    std::string formatted_text, raw_text;
    if (!line_col.first) {
      use_fixits = false;
      raw_text = formatted_text = text.str();
    } else {
      // Swift may insert note diagnostics after an error diagnostic with fixits
      // related to that error. Check if the latest inserted diagnostic is an
      // error one, and that the diagnostic being processed is a note one that
      // points to the same error, and if so, copy the fixits from the note
      // diagnostic to the error one. There may be subsequent notes with fixits
      // related to the same error, but we only copy the first one as the fixits
      // are mutually exclusive (for example, one may suggest inserting a '?'
      // and the next may suggest inserting '!')
      if (info.Kind == swift::DiagnosticKind::Note &&
          !m_raw_swift_diagnostics.empty()) {
        auto &last_diagnostic = m_raw_swift_diagnostics.back();
        if (last_diagnostic.detail.severity == lldb::eSeverityError &&
            last_diagnostic.fixits.empty() &&
            last_diagnostic.buffer_id == buffer_id &&
            last_diagnostic.detail.source_location &&
            last_diagnostic.detail.source_location->column == line_col.second &&
            last_diagnostic.detail.source_location->line == line_col.first)
          last_diagnostic.fixits.insert(last_diagnostic.fixits.end(),
                                        info.FixIts.begin(), info.FixIts.end());
      }

      // Translate ranges.
      llvm::SmallVector<llvm::SMRange, 2> ranges;
      for (auto R : info.Ranges)
        ranges.push_back(getRawRange(source_mgr, R));

      // Translate fix-its.
      llvm::SmallVector<llvm::SMFixIt, 2> fix_its;
      for (swift::DiagnosticInfo::FixIt F : info.FixIts)
        fix_its.push_back(getRawFixIt(source_mgr, F));

      // Display the diagnostic.
      auto message = source_mgr.GetMessage(source_loc, source_mgr_kind, text,
                                           ranges, fix_its);
      {
        llvm::raw_string_ostream os(formatted_text);
        source_mgr.getLLVMSourceMgr().PrintMessage(os, message);
      }
      size_t kind_pos = formatted_text.find(kind_str);
      formatted_text = formatted_text.substr(0, kind_pos) +
                       formatted_text.substr(kind_pos + kind_str.size());
      raw_text = text.str();
    }
    if (info.Kind == swift::DiagnosticKind::Remark) {
      if (info.ID == swift::diag::module_loaded.ID) {
        // Divert module import remarks into the logs.
        LLDB_LOG(GetLog(LLDBLog::Types), "{0} Module import remark: {1}",
                 m_ast_context.GetDescription(), formatted_text);
      }
      return;
    }
    // FXIME: This is a heuristic.
    uint16_t len = info.Ranges.size() ? info.Ranges.front().getByteLength() : 0;
    bool in_user_input = false;
    bool hidden = false;
    [&]() {
      if (buffer_name == "repl.swift") {
        in_user_input = true;
        return;
      }
      if (buffer_name != "<REPL>" && buffer_name != "<EXPR>" &&
          !buffer_name.starts_with("<user expression"))
        return;

      unsigned buffer_id = source_mgr.findBufferContainingLoc(info.Loc);
      llvm::SourceMgr &src_mgr = source_mgr.getLLVMSourceMgr();
      auto *buffer = src_mgr.getMemoryBuffer(buffer_id);
      if (!buffer)
        return;

      // Clang uses the much more elegant #line mechanism to find the
      // user code, but doing this here could interfere with
      // ExprOptions.PoundLineLine mechanism used Swift
      // Playgrounds. So instead we find the markers from both ends.
      llvm::StringRef buffer_data = buffer->getBuffer();
      size_t pos = buffer_data.find("__LLDB_USER_START__");
      if (pos == llvm::StringRef::npos)
        return;
      auto start_loc =
          llvm::SMLoc::getFromPointer(buffer->getBufferStart() + pos);
      unsigned start_line = src_mgr.FindLineNumber(start_loc, buffer_id);

      pos = buffer_data.rfind("__LLDB_USER_END__");
      if (pos == llvm::StringRef::npos)
        return;
      auto end_loc =
          llvm::SMLoc::getFromPointer(buffer->getBufferStart() + pos);
      unsigned end_line = src_mgr.FindLineNumber(end_loc, buffer_id);

      unsigned diag_line =
          source_mgr.getLineAndColumnInBuffer(info.Loc, buffer_id).first;

      in_user_input = start_line < diag_line && diag_line < end_line;
      hidden = !in_user_input;
    }();
    DiagnosticDetail::SourceLocation loc = {FileSpec{buffer_name.str()},
                                            line_col.first,
                                            (uint16_t)line_col.second,
                                            len,
                                            hidden,
                                            in_user_input};
    DiagnosticDetail detail = {loc, severity, raw_text, formatted_text};
    RawDiagnostic diagnostic(
        detail, buffer_id,
        use_fixits ? info.FixIts : llvm::ArrayRef<swift::Diagnostic::FixIt>());
    if (info.ID == swift::diag::error_from_clang.ID ||
        info.ID == swift::diag::bridging_header_error.ID) {
      if (m_raw_clang_diagnostics.empty() ||
          m_raw_clang_diagnostics.back().detail.rendered !=
              diagnostic.detail.rendered) {
        m_raw_clang_diagnostics.push_back(std::move(diagnostic));
        if (info.Kind == swift::DiagnosticKind::Error) {
          m_num_clang_errors++;
          // Any errors from clang could be related module import
          // issues which shoud be surfaced in the health log channel.
          LLDB_LOG(GetLog(LLDBLog::Types), "{0} Clang error: {1}",
                   m_ast_context.GetDescription(), formatted_text);
          LLDB_LOG(lldb_private::GetSwiftHealthLog(), "{0} Clang error: {1}",
                   m_ast_context.GetDescription(), formatted_text);
        }
      }
    } else {
      m_raw_swift_diagnostics.push_back(std::move(diagnostic));
      if (info.Kind == swift::DiagnosticKind::Error)
        m_num_swift_errors++;
    }
  }

  void Clear() {
    m_raw_swift_diagnostics.clear();
    m_num_swift_errors = 0;
    // Don't reset Clang errors. ClangImporter's DiagnosticEngine doesn't either.
    // Don't reset LLDB diagnostics.
  }

  unsigned HasDiagnostics() {
    return !m_raw_clang_diagnostics.empty() ||
           !m_raw_swift_diagnostics.empty() || !m_diagnostics.empty();
  }

  unsigned NumClangErrors() { return m_num_clang_errors; }

  static lldb::Severity SeverityForKind(swift::DiagnosticKind kind) {
    switch (kind) {
    case swift::DiagnosticKind::Error:
      return lldb::eSeverityError;
    case swift::DiagnosticKind::Warning:
      return lldb::eSeverityWarning;
    case swift::DiagnosticKind::Note:
    case swift::DiagnosticKind::Remark:
      return lldb::eSeverityInfo;
    }

    llvm_unreachable("Unhandled DiagnosticKind in switch.");
  }

  // Forward and consume the stored diagnostics to \c diagnostic_manager.
  void PrintDiagnostics(DiagnosticManager &diagnostic_manager,
                        SwiftASTContext::DiagnosticCursor cursor,
                        uint32_t bufferID = UINT32_MAX, uint32_t first_line = 0,
                        uint32_t last_line = UINT32_MAX) {
    // Move all diagnostics starting at the cursor into diagnostic_manager.
    bool added_one_diagnostic = false;
    for (size_t i = cursor.lldb; i < m_diagnostics.size(); ++i) {
      diagnostic_manager.AddDiagnostic(std::move(m_diagnostics[i]));
      added_one_diagnostic = true;
    }
    m_diagnostics.resize(std::min(m_diagnostics.size(), cursor.lldb));

    // We often make expressions and wrap them in some code.  When we
    // see errors we want the line numbers to be correct so we correct
    // them below. LLVM stores in SourceLoc objects as character
    // offsets so there is no way to get LLVM to move its error line
    // numbers around by adjusting the source location, we must do it
    // manually. We also want to use the same error formatting as LLVM
    // and Clang, so we must muck with the string.

    auto format_diagnostic = [&](const RawDiagnostic &diagnostic,
                                 const DiagnosticOrigin origin) {
      const lldb::Severity severity = diagnostic.detail.severity;
      const auto &diag_loc = diagnostic.detail.source_location;

      // Make sure the error line is in range or in another file.
      if (diagnostic.buffer_id == bufferID && diag_loc && diag_loc->file &&
          (diag_loc->line < first_line || diag_loc->line > last_line))
        return;

      // Heuristic to skip expected expression warnings.
      if (diagnostic.detail.message.find("$__lldb") != std::string::npos)
        return;

      // Diagnose global errors.
      if (severity == lldb::eSeverityError &&
          (!diag_loc || diag_loc->line == 0)) {
        diagnostic_manager.AddDiagnostic(std::make_unique<Diagnostic>(
            origin, diagnostic.buffer_id, diagnostic.detail));
        added_one_diagnostic = true;
        return;
      }

      DiagnosticDetail detail = diagnostic.detail;
      if (diag_loc) {
        // Need to remap the error/warning to a different line.
        DiagnosticDetail::SourceLocation new_loc;
        new_loc = *diag_loc;
        if (new_loc.line)
          new_loc.line -= first_line - 1;
        detail.source_location = new_loc;
      }

      auto new_diagnostic =
          std::make_unique<SwiftDiagnostic>(detail, 0, diagnostic.buffer_id);
      for (auto fixit : diagnostic.fixits)
        new_diagnostic->AddFixIt(fixit);

      diagnostic_manager.AddDiagnostic(std::move(new_diagnostic));
      if (diagnostic.detail.severity == lldb::eSeverityError)
        added_one_diagnostic = true;
    };

    for (size_t i = cursor.clang; i < m_raw_clang_diagnostics.size(); ++i)
      format_diagnostic(m_raw_clang_diagnostics[i], eDiagnosticOriginClang);

    for (size_t i = cursor.swift; i < m_raw_swift_diagnostics.size(); ++i)
      format_diagnostic(m_raw_swift_diagnostics[i], eDiagnosticOriginSwift);

    if (added_one_diagnostic)
      return;

    // We found no error that is newer than the cursor position.
    // ClangImporter errors often happen outside the current transaction.
    for (const RawDiagnostic &diagnostic : m_raw_clang_diagnostics)
      diagnostic_manager.AddDiagnostic(std::make_unique<Diagnostic>(
          eDiagnosticOriginClang, 0, diagnostic.detail));

    // If we printed a clang error, ignore Swift warnings, which are
    // expected: The default lldb_expr wrapper produces some warnings.
    if (!m_num_clang_errors || m_num_swift_errors)
      for (const RawDiagnostic &diagnostic : m_raw_swift_diagnostics)
        diagnostic_manager.AddDiagnostic(std::make_unique<Diagnostic>(
            eDiagnosticOriginClang, 0, diagnostic.detail));
  }
  void AddDiagnostic(std::unique_ptr<Diagnostic> diagnostic) {
    if (diagnostic)
      m_diagnostics.push_back(std::move(diagnostic));
  }

private:
  // We don't currently use lldb_private::Diagostic or any of the lldb
  // DiagnosticManager machinery to store diagnostics as they
  // occur. Instead, we store them in raw form using this struct, then
  // transcode them to SwiftDiagnostics in PrintDiagnostic.
  struct RawDiagnostic {
    RawDiagnostic() = default;
    RawDiagnostic(DiagnosticDetail detail, unsigned buffer_id,
                  llvm::ArrayRef<swift::Diagnostic::FixIt> fixits)
        : detail(detail), fixits(fixits), buffer_id(buffer_id) {}
    RawDiagnostic(const RawDiagnostic &other) = default;
    DiagnosticDetail detail;
    std::vector<swift::DiagnosticInfo::FixIt> fixits;
    /// Only stored for comparison in HandleDiagnostic. There is no
    /// guarantee that the SourceMgr is still alive in a stored diagnostic.
    unsigned buffer_id;
  };

  SwiftASTContext &m_ast_context;
  /// Stores all diagnostics coming from the Swift compiler.
  std::vector<RawDiagnostic> m_raw_swift_diagnostics;
  std::vector<RawDiagnostic> m_raw_clang_diagnostics;
  /// Stores diagnostics coming from LLDB.
  std::vector<std::unique_ptr<Diagnostic>> m_diagnostics;

  unsigned m_num_swift_errors = 0;
  unsigned m_num_clang_errors = 0;

  friend class SwiftASTContext::ScopedDiagnostics;
};

} // namespace lldb_private

#endif
