//===-- SwiftREPL.h ---------------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SwiftREPL_h_
#define liblldb_SwiftREPL_h_

#include "lldb/Utility/Status.h"
#include "lldb/Expression/REPL.h"
#include "lldb/lldb-public.h"

#include <string>
#include <vector>

namespace lldb_private {

class IRExecutionUnit;

//----------------------------------------------------------------------
/// @class SwiftREPL SwiftREPL.h "lldb/Expression/SwiftREPL.h"
/// @brief Encapsulates a swift REPL session.
//----------------------------------------------------------------------
class SwiftREPL : public REPL {
public:
  SwiftREPL(Target &target);
  ~SwiftREPL();

  static void Initialize();

  static void Terminate();

protected:
  static lldb::REPLSP CreateInstance(Status &error, lldb::LanguageType language,
                                     Debugger *debugger, Target *target,
                                     const char *repl_options);

  static lldb::REPLSP CreateInstanceFromTarget(Status &error, Target &target,
                                               const char *repl_options);

  static lldb::REPLSP CreateInstanceFromDebugger(Status &error,
                                                 Debugger &debugger,
                                                 const char *repl_options);

  static void
  EnumerateSupportedLanguages(std::set<lldb::LanguageType> &languages);

  Status DoInitialization() override;

  ConstString GetSourceFileBasename() override;

  const char *GetAutoIndentCharacters() override { return "}:"; }

  bool SourceIsComplete(const std::string &source) override;

  lldb::offset_t GetDesiredIndentation(const StringList &lines,
                                       int cursor_position,
                                       int tab_size) override;

  lldb::LanguageType GetLanguage() override;

  bool PrintOneVariable(Debugger &debugger, lldb::StreamFileSP &output_sp,
                        lldb::ValueObjectSP &valobj_sp,
                        ExpressionVariable *var = nullptr) override;

  int CompleteCode(const std::string &current_code,
                   StringList &matches) override;

public:
  static bool classof(const REPL *repl) {
    return repl->getKind() == LLVMCastKind::eKindSwift;
  }

private:
  lldb::SwiftASTContextSP m_swift_ast_sp;
  bool m_swift_repl_initialized = false;
};
}

#endif // liblldb_SwiftREPL_h_
