//===-- OptionValuePathMappings.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/OptionValuePathMappings.h"

#include "lldb/Host/FileSystem.h"
#include "lldb/Host/StringConvert.h"
#include "lldb/Utility/Args.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Stream.h"

using namespace lldb;
using namespace lldb_private;
namespace {
static bool VerifyPathExists(const char *path) {
  if (path && path[0])
    return FileSystem::Instance().Exists(path);
  else
    return false;
}
}

void OptionValuePathMappings::DumpValue(const ExecutionContext *exe_ctx,
                                        Stream &strm, uint32_t dump_mask) {
  if (dump_mask & eDumpOptionType)
    strm.Printf("(%s)", GetTypeAsCString());
  if (dump_mask & eDumpOptionValue) {
    if (dump_mask & eDumpOptionType)
      strm.Printf(" =%s", (m_path_mappings.GetSize() > 0) ? "\n" : "");
    m_path_mappings.Dump(&strm);
  }
}

Status OptionValuePathMappings::SetValueFromString(llvm::StringRef value,
                                                   VarSetOperationType op) {
  Status error;
  Args args(value.str());
  const size_t argc = args.GetArgumentCount();

  switch (op) {
  case eVarSetOperationClear:
    Clear();
    NotifyValueChanged();
    break;

  case eVarSetOperationReplace:
    // Must be at least one index + 1 pair of paths, and the pair count must be
    // even
    if (argc >= 3 && (((argc - 1) & 1) == 0)) {
      uint32_t idx =
          StringConvert::ToUInt32(args.GetArgumentAtIndex(0), UINT32_MAX);
      const uint32_t count = m_path_mappings.GetSize();
      if (idx > count) {
        error.SetErrorStringWithFormat(
            "invalid file list index %u, index must be 0 through %u", idx,
            count);
      } else {
        bool changed = false;
        for (size_t i = 1; i < argc; idx++, i += 2) {
          const char *orginal_path = args.GetArgumentAtIndex(i);
          const char *replace_path = args.GetArgumentAtIndex(i + 1);
          if (VerifyPathExists(replace_path)) {
            ConstString a(orginal_path);
            ConstString b(replace_path);
            if (!m_path_mappings.Replace(a, b, idx, m_notify_changes))
              m_path_mappings.Append(a, b, m_notify_changes);
            changed = true;
          } else {
            std::string previousError =
                error.Fail() ? std::string(error.AsCString()) + "\n" : "";
            error.SetErrorStringWithFormat(
                "%sthe replacement path doesn't exist: \"%s\"",
                previousError.c_str(), replace_path);
          }
        }
        if (changed)
          NotifyValueChanged();
      }
    } else {
      error.SetErrorString("replace operation takes an array index followed by "
                           "one or more path pairs");
    }
    break;

  case eVarSetOperationAssign:
    if (argc < 2 || (argc & 1)) {
      error.SetErrorString("assign operation takes one or more path pairs");
      break;
    }
    m_path_mappings.Clear(m_notify_changes);
    // Fall through to append case
    LLVM_FALLTHROUGH;
  case eVarSetOperationAppend:
    if (argc < 2 || (argc & 1)) {
      error.SetErrorString("append operation takes one or more path pairs");
      break;
    } else {
      bool changed = false;
      for (size_t i = 0; i < argc; i += 2) {
        const char *orginal_path = args.GetArgumentAtIndex(i);
        const char *replace_path = args.GetArgumentAtIndex(i + 1);
        if (VerifyPathExists(replace_path)) {
          ConstString a(orginal_path);
          ConstString b(replace_path);
          m_path_mappings.Append(a, b, m_notify_changes);
          m_value_was_set = true;
          changed = true;
        } else {
          std::string previousError =
              error.Fail() ? std::string(error.AsCString()) + "\n" : "";
          error.SetErrorStringWithFormat(
              "%sthe replacement path doesn't exist: \"%s\"",
              previousError.c_str(), replace_path);
        }
      }
      if (changed)
        NotifyValueChanged();
    }
    break;

  case eVarSetOperationInsertBefore:
  case eVarSetOperationInsertAfter:
    // Must be at least one index + 1 pair of paths, and the pair count must be
    // even
    if (argc >= 3 && (((argc - 1) & 1) == 0)) {
      uint32_t idx =
          StringConvert::ToUInt32(args.GetArgumentAtIndex(0), UINT32_MAX);
      const uint32_t count = m_path_mappings.GetSize();
      if (idx > count) {
        error.SetErrorStringWithFormat(
            "invalid file list index %u, index must be 0 through %u", idx,
            count);
      } else {
        bool changed = false;
        if (op == eVarSetOperationInsertAfter)
          ++idx;
        for (size_t i = 1; i < argc; i += 2) {
          const char *orginal_path = args.GetArgumentAtIndex(i);
          const char *replace_path = args.GetArgumentAtIndex(i + 1);
          if (VerifyPathExists(replace_path)) {
            ConstString a(orginal_path);
            ConstString b(replace_path);
            m_path_mappings.Insert(a, b, idx, m_notify_changes);
            changed = true;
            idx++;
          } else {
            std::string previousError =
                error.Fail() ? std::string(error.AsCString()) + "\n" : "";
            error.SetErrorStringWithFormat(
                "%sthe replacement path doesn't exist: \"%s\"",
                previousError.c_str(), replace_path);
          }
        }
        if (changed)
          NotifyValueChanged();
      }
    } else {
      error.SetErrorString("insert operation takes an array index followed by "
                           "one or more path pairs");
    }
    break;

  case eVarSetOperationRemove:
    if (argc > 0) {
      std::vector<int> remove_indexes;
      for (size_t i = 0; i < argc; ++i) {
        int idx =
            StringConvert::ToSInt32(args.GetArgumentAtIndex(i), INT32_MAX);
        if (idx < 0 || idx >= (int)m_path_mappings.GetSize()) {
          error.SetErrorStringWithFormat(
              "invalid array index '%s', aborting remove operation",
              args.GetArgumentAtIndex(i));
          break;
        } else
          remove_indexes.push_back(idx);
      }

      // Sort and then erase in reverse so indexes are always valid
      llvm::sort(remove_indexes.begin(), remove_indexes.end());
      for (auto index : llvm::reverse(remove_indexes))
        m_path_mappings.Remove(index, m_notify_changes);
      NotifyValueChanged();
    } else {
      error.SetErrorString("remove operation takes one or more array index");
    }
    break;

  case eVarSetOperationInvalid:
    error = OptionValue::SetValueFromString(value, op);
    break;
  }
  return error;
}

lldb::OptionValueSP OptionValuePathMappings::DeepCopy() const {
  return OptionValueSP(new OptionValuePathMappings(*this));
}
