#include "LibCxx.h"
#include "LibCxxStringInfoExtractor.h"

#include "lldb/DataFormatters/FormattersHelpers.h"
#include <unordered_map>

using namespace lldb;
using namespace lldb_private;

namespace {

class StringFrontend : public SyntheticChildrenFrontEnd {

public:
  StringFrontend(ValueObject &valobj, const char *prefix = "")
      : SyntheticChildrenFrontEnd(valobj), m_prefix(prefix) {}

  llvm::Expected<uint32_t> CalculateNumChildren() override {
    return m_size + m_special_members_count;
  }

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override {

    if (idx < m_special_members_count) {
      return m_backend.GetChildMemberWithName(ConstString("__r_"),
                                              /*can_create=*/true);
    }

    idx -= m_special_members_count;

    if (!m_str_data_ptr || idx > m_size || !m_element_size) {
      return {};
    }

    auto char_it = m_chars.find(idx);
    if (char_it != m_chars.end()) {
      return char_it->second;
    }

    uint64_t offset = idx * m_element_size;
    uint64_t address = m_str_data_ptr->GetValueAsUnsigned(0);

    if (!address) {
      return {};
    }

    StreamString name;
    name.Printf("[%" PRIu64 "]", (uint64_t)idx);

    m_chars[idx] = CreateValueObjectFromAddress(
        name.GetString(), address + offset, m_backend.GetExecutionContextRef(),
        m_element_type);

    return m_chars[idx];
  }

  size_t GetIndexOfChildWithName(ConstString name) override {
    if (name == "__r_") {
      return 0;
    }
    return formatters::ExtractIndexFromString(name.GetCString()) +
           m_special_members_count;
  }

  ChildCacheState Update() override {

    clear();

    auto string_info = ExtractLibcxxStringInfo(m_backend);
    if (!string_info)
      return ChildCacheState::eRefetch;
    std::tie(m_size, m_str_data_ptr) = *string_info;

    m_element_type = m_backend.GetCompilerType().GetTypeTemplateArgument(0);
    m_element_size = m_element_type.GetByteSize(nullptr).value_or(0);

    if (m_str_data_ptr->IsArrayType()) {
      // this means the string is in short-mode and the
      // data is stored inline in array,
      // so we need address of this array
      Status status;
      m_str_data_ptr = m_str_data_ptr->AddressOf(status);
    }

    return ChildCacheState::eReuse;
  }

  bool MightHaveChildren() override { return true; }

  bool SetValueFromCString(const char *value_str, Status &error) override {

    ValueObjectSP expr_value_sp;

    std::unique_lock<std::recursive_mutex> lock;
    ExecutionContext exe_ctx(m_backend.GetExecutionContextRef(), lock);

    Target *target = exe_ctx.GetTargetPtr();
    StackFrame *frame = exe_ctx.GetFramePtr();

    if (target && frame) {
      EvaluateExpressionOptions options;
      options.SetUseDynamic(frame->CalculateTarget()->GetPreferDynamicValue());
      options.SetIgnoreBreakpoints(true);

      if (target->GetLanguage() != eLanguageTypeUnknown)
        options.SetLanguage(target->GetLanguage());
      else
        options.SetLanguage(frame->GetLanguage());
      StreamString expr;
      expr.Printf("%s = %s\"%s\"", m_backend.GetName().AsCString(), m_prefix,
                  value_str);
      ExpressionResults result = target->EvaluateExpression(
          expr.GetString(), frame, expr_value_sp, options);
      if (result != eExpressionCompleted)
        error.SetErrorStringWithFormat("Expression (%s) can't be evaluated.",
                                       expr.GetData());
    }

    return error.Success();
  }

private:
  void clear() {
    m_size = 0;
    m_element_size = 0;
    m_str_data_ptr = nullptr;
    m_element_type.Clear();
    m_chars.clear();
  }

  std::unordered_map<uint32_t, ValueObjectSP> m_chars;
  ValueObjectSP m_str_data_ptr;
  CompilerType m_element_type;
  uint32_t m_size = 0;
  uint32_t m_element_size = 0;
  const char *m_prefix = "";
  static const uint32_t m_special_members_count =
      1; // __r_ member needed for correct summaries
};

} // namespace

SyntheticChildrenFrontEnd *formatters::LibcxxStdStringSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (valobj_sp)
    return new StringFrontend(*valobj_sp);
  return nullptr;
}

SyntheticChildrenFrontEnd *formatters::LibcxxStdWStringSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (valobj_sp)
    return new StringFrontend(*valobj_sp, "L");
  return nullptr;
}

SyntheticChildrenFrontEnd *
formatters::LibcxxStdU16StringSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (valobj_sp)
    return new StringFrontend(*valobj_sp, "u");
  return nullptr;
}

SyntheticChildrenFrontEnd *
formatters::LibcxxStdU32StringSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (valobj_sp)
    return new StringFrontend(*valobj_sp, "U");
  return nullptr;
}
