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

class ANSIColorStringStream : public llvm::raw_string_ostream {
public:
  ANSIColorStringStream(bool colorize)
      : llvm::raw_string_ostream(m_buffer), m_colorize(colorize) {}
  /// Changes the foreground color of text that will be output from
  /// this point forward.
  ///
  /// \param colors      ANSI color to use, the special SAVEDCOLOR can
  ///                    be used to change only the bold attribute, and
  ///                    keep colors untouched.
  /// \param bold        bold/brighter text, default false
  /// \param bg          if true change the background,
  ///                    default: change foreground
  /// \returns           itself so it can be used within << invocations.
  raw_ostream &changeColor(enum Colors colors, bool bold = false,
                           bool bg = false) override {
    if (llvm::sys::Process::ColorNeedsFlush())
      flush();
    const char *colorcode;
    if (colors == SAVEDCOLOR)
      colorcode = llvm::sys::Process::OutputBold(bg);
    else
      colorcode =
          llvm::sys::Process::OutputColor(static_cast<char>(colors), bold, bg);
    if (colorcode) {
      size_t len = strlen(colorcode);
      write(colorcode, len);
    }
    return *this;
  }

  /// Resets the colors to terminal defaults. Call this when you are
  /// done outputting colored text, or before program exit.
  raw_ostream &resetColor() override {
    if (llvm::sys::Process::ColorNeedsFlush())
      flush();
    const char *colorcode = llvm::sys::Process::ResetColor();
    if (colorcode) {
      size_t len = strlen(colorcode);
      write(colorcode, len);
    }
    return *this;
  }

  /// Reverses the forground and background colors.
  raw_ostream &reverseColor() override {
    if (llvm::sys::Process::ColorNeedsFlush())
      flush();
    const char *colorcode = llvm::sys::Process::OutputReverse();
    if (colorcode) {
      size_t len = strlen(colorcode);
      write(colorcode, len);
    }
    return *this;
  }

  /// This function determines if this stream is connected to a "tty"
  /// or "console" window. That is, the output would be displayed to
  /// the user rather than being put on a pipe or stored in a file.
  bool is_displayed() const override { return m_colorize; }

  /// This function determines if this stream is displayed and
  /// supports colors.
  bool has_colors() const override { return m_colorize; }

protected:
  std::string m_buffer;
  bool m_colorize;
};


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
    llvm::StringRef bufferName = "<anonymous>";
    unsigned bufferID = 0;
    std::pair<unsigned, unsigned> line_col = {0, 0};

    llvm::SmallString<256> text;
    {
      llvm::raw_svector_ostream out(text);
      swift::DiagnosticEngine::formatDiagnosticText(out, info.FormatString,
                                                    info.FormatArgs);
    }

    swift::SourceLoc source_loc = info.Loc;
    if (source_loc.isValid()) {
      bufferID = source_mgr.findBufferContainingLoc(source_loc);
      bufferName = source_mgr.getDisplayNameForLoc(source_loc);
      line_col = source_mgr.getPresumedLineAndColumnForLoc(source_loc);
    }

    bool use_fixits = false;
    std::string formatted_text;
    if (!line_col.first) {
      formatted_text = text.str();
    } else {
      ANSIColorStringStream os(m_colorize);

      // Determine what kind of diagnostic we're emitting, and whether
      // we want to use its fixits:
      llvm::SourceMgr::DiagKind source_mgr_kind;
      switch (info.Kind) {
      case swift::DiagnosticKind::Error:
        source_mgr_kind = llvm::SourceMgr::DK_Error;
        use_fixits = true;
        break;
      case swift::DiagnosticKind::Warning:
        source_mgr_kind = llvm::SourceMgr::DK_Warning;
        break;
      case swift::DiagnosticKind::Note:
        source_mgr_kind = llvm::SourceMgr::DK_Note;
        break;
      case swift::DiagnosticKind::Remark:
        source_mgr_kind = llvm::SourceMgr::DK_Remark;
        break;
      }

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
        if (last_diagnostic.kind == swift::DiagnosticKind::Error &&
            last_diagnostic.fixits.empty() &&
            last_diagnostic.bufferID == bufferID &&
            last_diagnostic.column == line_col.second &&
            last_diagnostic.line == line_col.first)
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
      source_mgr.getLLVMSourceMgr().PrintMessage(os, message);

      // str() implicitly flushes the stram.
      std::string &s = os.str();
      formatted_text = !s.empty() ? std::move(s) : std::string(text);
    }
    if (info.Kind == swift::DiagnosticKind::Remark) {
      if (info.ID == swift::diag::module_loaded.ID) {
        // Divert module import remarks into the logs.
        LLDB_LOG(GetLog(LLDBLog::Types), "{0} Module import remark: {1}",
                 m_ast_context.GetDescription(), formatted_text);
      }
      return;
    }
    RawDiagnostic diagnostic(
        formatted_text, info.Kind, bufferName.str(), bufferID, line_col.first,
        line_col.second,
        use_fixits ? info.FixIts : llvm::ArrayRef<swift::Diagnostic::FixIt>());
    if (info.ID == swift::diag::error_from_clang.ID ||
        info.ID == swift::diag::bridging_header_error.ID) {
      if (m_raw_clang_diagnostics.empty() ||
          m_raw_clang_diagnostics.back() != diagnostic) {
        m_raw_clang_diagnostics.push_back(std::move(diagnostic));
        if (info.Kind == swift::DiagnosticKind::Error)
          m_num_clang_errors++;
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

  static DiagnosticSeverity SeverityForKind(swift::DiagnosticKind kind) {
    switch (kind) {
    case swift::DiagnosticKind::Error:
      return eDiagnosticSeverityError;
    case swift::DiagnosticKind::Warning:
      return eDiagnosticSeverityWarning;
    case swift::DiagnosticKind::Note:
    case swift::DiagnosticKind::Remark:
      return eDiagnosticSeverityRemark;
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
      const DiagnosticSeverity severity = SeverityForKind(diagnostic.kind);

      // Make sure the error line is in range or in another file.
      if (diagnostic.bufferID == bufferID && !diagnostic.bufferName.empty() &&
          (diagnostic.line < first_line || diagnostic.line > last_line))
        return;

      // Heuristic to skip expected expression warnings.
      if (diagnostic.description.find("$__lldb") != std::string::npos)
        return;

      // Diagnose global errors.
      if (severity == eDiagnosticSeverityError && diagnostic.line == 0) {
        diagnostic_manager.AddDiagnostic(diagnostic.description.c_str(),
                                         severity, origin);
        added_one_diagnostic = true;
        return;
      }

      // Need to remap the error/warning to a different line.
      StreamString match;
      match.Printf("%s:%u:", diagnostic.bufferName.c_str(), diagnostic.line);
      const size_t match_len = match.GetString().size();
      size_t match_pos = diagnostic.description.find(match.GetString().str());
      if (match_pos == std::string::npos)
        return;

      // We have some <file>:<line>:" instances that need to be updated.
      StreamString fixed_description;
      size_t start_pos = 0;
      do {
        if (match_pos > start_pos)
          fixed_description.Printf(
              "%s",
              diagnostic.description.substr(start_pos, match_pos).c_str());
        fixed_description.Printf("%s:%u:", diagnostic.bufferName.c_str(),
                                 diagnostic.line - first_line + 1);
        start_pos = match_pos + match_len;
        match_pos =
            diagnostic.description.find(match.GetString().str(), start_pos);
      } while (match_pos != std::string::npos);

      // Append any last remaining text.
      if (start_pos < diagnostic.description.size())
        fixed_description.Printf(
            "%s",
            diagnostic.description
                .substr(start_pos, diagnostic.description.size() - start_pos)
                .c_str());

      auto new_diagnostic = std::make_unique<SwiftDiagnostic>(
          fixed_description.GetData(), severity, origin, bufferID);
      for (auto fixit : diagnostic.fixits)
        new_diagnostic->AddFixIt(fixit);

      diagnostic_manager.AddDiagnostic(std::move(new_diagnostic));
      if (diagnostic.kind == swift::DiagnosticKind::Error)
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
      diagnostic_manager.AddDiagnostic(diagnostic.description.c_str(),
                                       SeverityForKind(diagnostic.kind),
                                       eDiagnosticOriginClang);

    // If we printed a clang error, ignore Swift warnings, which are
    // expected: The default lldb_expr wrapper produces some warnings.
    if (!m_num_clang_errors || m_num_swift_errors)
      for (const RawDiagnostic &diagnostic : m_raw_swift_diagnostics)
        diagnostic_manager.AddDiagnostic(diagnostic.description.c_str(),
                                         SeverityForKind(diagnostic.kind),
                                         eDiagnosticOriginSwift);
  }

  bool GetColorize() const { return m_colorize; }

  bool SetColorize(bool b) {
    const bool old = m_colorize;
    m_colorize = b;
    return old;
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
    RawDiagnostic(std::string in_desc, swift::DiagnosticKind in_kind,
                  std::string in_bufferName, unsigned in_bufferID,
                  uint32_t in_line, uint32_t in_column,
                  llvm::ArrayRef<swift::Diagnostic::FixIt> in_fixits)
        : description(in_desc), bufferName(in_bufferName), fixits(in_fixits),
          kind(in_kind), bufferID(in_bufferID), line(in_line),
          column(in_column) {}
    RawDiagnostic(const RawDiagnostic &other) = default;
    bool operator==(const RawDiagnostic& other) {
      return kind == other.kind && line == other.line &&
             bufferID == other.bufferID && column == other.column &&
             bufferName == other.bufferName && description == other.description;
    }
    bool operator!=(const RawDiagnostic &other) { return !(*this == other); }

    std::string description;
    std::string bufferName;
    std::vector<swift::DiagnosticInfo::FixIt> fixits;
    swift::DiagnosticKind kind = swift::DiagnosticKind::Error;
    /// Only stored for comparison in HandleDiagnostic. There is no
    /// guarantee that the SourceMgr is still alive in a stored diagnostic.
    unsigned bufferID;
    uint32_t line;
    uint32_t column;
  };

  SwiftASTContext &m_ast_context;
  /// Stores all diagnostics coming from the Swift compiler.
  std::vector<RawDiagnostic> m_raw_swift_diagnostics;
  std::vector<RawDiagnostic> m_raw_clang_diagnostics;
  /// Stores diagnostics coming from LLDB.
  std::vector<std::unique_ptr<Diagnostic>> m_diagnostics;

  unsigned m_num_swift_errors = 0;
  unsigned m_num_clang_errors = 0;
  bool m_colorize = false;

  friend class SwiftASTContext::ScopedDiagnostics;
};

} // namespace lldb_private

#endif
