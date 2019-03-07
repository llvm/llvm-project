//===-- SwiftExpressionSourceCode.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SwiftExpressionSourceCode.h"

#include "Plugins/ExpressionParser/Swift/SwiftASTManipulator.h"
#include "lldb/Target/Language.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/StreamString.h"

#include "llvm/ADT/StringRef.h"

using namespace lldb;
using namespace lldb_private;

/// Format the OS name the way that Swift availability attributes do.
static llvm::StringRef getAvailabilityName(llvm::Triple::OSType os) {
  switch (os) {
  case llvm::Triple::MacOSX: return "macOS";
  case llvm::Triple::IOS: return "iOS";
  case llvm::Triple::TvOS: return "tvOS";
  case llvm::Triple::WatchOS: return "watchOS";
  default:
    return llvm::Triple::getOSTypeName(os);
  }
}
uint32_t SwiftExpressionSourceCode::GetNumBodyLines() {
  if (m_num_body_lines == 0)
    // 2 = <one for zero indexing> + <one for the body start marker>
    m_num_body_lines = 2 + std::count(m_body.begin(), m_body.end(), '\n');
  return m_num_body_lines;
}

bool SwiftExpressionSourceCode::GetText(
               std::string &text, 
               lldb::LanguageType wrapping_language,
               bool needs_object_ptr,
               bool static_method,
               bool is_class,
               bool weak_self,
               const EvaluateExpressionOptions &options,
               ExecutionContext &exe_ctx,
               const Expression::SwiftGenericInfo &generic_info,
               uint32_t &first_body_line) const
  {
  Target *target = exe_ctx.GetTargetPtr();


  if (m_wrap) {
    const char *body = m_body.c_str();
    const char *pound_file = options.GetPoundLineFilePath();
    const uint32_t pound_line = options.GetPoundLineLine();
    StreamString pound_body;
    if (pound_file && pound_line) {
      if (wrapping_language == eLanguageTypeSwift) {
        pound_body.Printf("#sourceLocation(file: \"%s\", line: %u)\n%s",
                          pound_file, pound_line, body);
      } else {
        pound_body.Printf("#line %u \"%s\"\n%s", pound_line, pound_file, body);
      }
      body = pound_body.GetString().data();
    }

    if (wrapping_language != eLanguageTypeSwift) {
      return false;
    }

    StreamString wrap_stream;


    // First construct a tagged form of the user expression so we can find it
    // later:
    std::string tagged_body;
    llvm::SmallString<16> buffer;
    llvm::raw_svector_ostream os_vers(buffer);

    auto arch_spec = target->GetArchitecture();
    auto triple = arch_spec.GetTriple();
    if (triple.isOSDarwin()) {
      if (auto process_sp = exe_ctx.GetProcessSP()) {
        os_vers << getAvailabilityName(triple.getOS()) << " ";
        auto platform = target->GetPlatform();
        bool is_simulator =
            platform->GetPluginName().GetStringRef().endswith("-simulator");
        if (is_simulator) {
          // The simulators look like the host OS to Process, but Platform
          // can the version out of an environment variable.
          os_vers << platform->GetOSVersion(process_sp.get()).getAsString();
        } else {
          llvm::VersionTuple version = process_sp->GetHostOSVersion();
          os_vers << version.getAsString();
        }
      }
    }
    SwiftPersistentExpressionState *persistent_state =
      llvm::cast<SwiftPersistentExpressionState>(target->GetPersistentExpressionStateForLanguage(lldb::eLanguageTypeSwift));
    std::vector<swift::ValueDecl *> persistent_results;
    // Check if we have already declared the playground stub debug functions
    persistent_state->GetSwiftPersistentDecls(ConstString("__builtin_log_with_id"), {},
                                              persistent_results);

    size_t num_persistent_results = persistent_results.size();
    bool need_to_declare_log_functions = num_persistent_results == 0;
    EvaluateExpressionOptions localOptions(options);

    localOptions.SetPreparePlaygroundStubFunctions(need_to_declare_log_functions);

    SwiftASTManipulator::WrapExpression(wrap_stream, m_body.c_str(),
                                        needs_object_ptr, static_method,
                                        is_class, weak_self,
                                        localOptions,
                                        generic_info,
                                        os_vers.str(),
                                        first_body_line);

    text = wrap_stream.GetString();
  } else {
    text.append(m_body);
  }

  return true;
}

bool SwiftExpressionSourceCode::GetOriginalBodyBounds(
    std::string transformed_text,
    size_t &start_loc, size_t &end_loc) {
  const char *start_marker;
  const char *end_marker;

  start_marker = SwiftASTManipulator::GetUserCodeStartMarker();
  end_marker = SwiftASTManipulator::GetUserCodeEndMarker();

  start_loc = transformed_text.find(start_marker);
  if (start_loc == std::string::npos)
    return false;
  start_loc += strlen(start_marker);
  end_loc = transformed_text.find(end_marker);
  return end_loc != std::string::npos;
  return false;
}
