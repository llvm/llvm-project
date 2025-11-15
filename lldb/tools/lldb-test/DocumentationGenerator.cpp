//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DocumentationGenerator.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/DataFormatters/FormatManager.h"
#include "lldb/Interpreter/OptionValueArch.h"
#include "lldb/Interpreter/OptionValueBoolean.h"
#include "lldb/Interpreter/OptionValueChar.h"
#include "lldb/Interpreter/OptionValueEnumeration.h"
#include "lldb/Interpreter/OptionValueFileSpec.h"
#include "lldb/Interpreter/OptionValueFormat.h"
#include "lldb/Interpreter/OptionValueFormatEntity.h"
#include "lldb/Interpreter/OptionValueLanguage.h"
#include "lldb/Interpreter/OptionValueProperties.h"
#include "lldb/Interpreter/OptionValueRegex.h"
#include "lldb/Interpreter/OptionValueSInt64.h"
#include "lldb/Interpreter/OptionValueString.h"
#include "lldb/Interpreter/OptionValueUInt64.h"
#include "lldb/Target/Language.h"
#include "lldb/Utility/DataExtractor.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"

#include <cstdio>

namespace {

using namespace lldb;
using namespace lldb_private;

constexpr llvm::StringRef SETTINGS_HEADER = R"(
# Settings

This page lists all available settings in LLDB.
Settings can be set using `settings set <name> <value>`.
Values can be added to arrays and dictionaries with `settings append -- <name> <value>`.

## Root
)";

/// Print the fields for one option value.
/// These fields are at the bottom of the directive.
void printMdOptionFields(Stream &os, const OptionValue &val) {
  switch (val.GetType()) {
  case OptionValue::eTypeArch:
    os << ":default: "
       << val.GetAsArch()->GetDefaultValue().GetArchitectureName() << '\n';
    break;
  case OptionValue::eTypeBoolean:
    os << ":default: ";
    os.PutCString(val.GetAsBoolean()->GetDefaultValue() ? "true" : "false");
    os << '\n';
    break;
  case OptionValue::eTypeChar:
    os << ":default: " << val.GetAsChar()->GetCurrentValue() << '\n';
    break;
  case OptionValue::eTypeEnum: {
    const auto &enumerations = val.GetAsEnumeration()->Enumerations();
    auto default_value = val.GetAsEnumeration()->GetDefaultValue();
    llvm::StringRef default_str;

    for (const auto &entry : enumerations) {
      os << ":enum " << entry.cstring << ": ";
      if (entry.value.description)
        os << entry.value.description;
      os << '\n';
      if (entry.value.value == default_value)
        default_str = entry.cstring;
    }

    if (!default_str.empty())
      os << ":default: " << default_str << '\n';
  } break;
  case OptionValue::eTypeFileSpec: {
    std::string path = val.GetAsFileSpec()->GetDefaultValue().GetPath(false);

    // Some defaults include the user's home directory. This should show as '~'
    // in the documentation.
    llvm::SmallString<64> user_home_dir;
    if (FileSystem::Instance().GetHomeDirectory(user_home_dir)) {
      std::string home_path = FileSpec(user_home_dir.c_str()).GetPath(false);
      if (llvm::StringRef(path).starts_with(home_path))
        path.replace(0, user_home_dir.size(), "~");
    }

    if (!path.empty())
      os << ":default: " << path << '\n';
  } break;
  case OptionValue::eTypeFormat:
    os << ":default: "
       << FormatManager::GetFormatAsCString(
              val.GetAsFormat()->GetCurrentValue())
       << '\n';
    break;
  case OptionValue::eTypeFormatEntity:
    os << ":default: "
       << val.GetAsFormatEntity()->GetEscapedDefaultFormatStr() << '\n';
    break;
  case OptionValue::eTypeLanguage:
    os << ":default: "
       << Language::GetNameForLanguageType(
              val.GetAsLanguage()->GetDefaultValue())
       << '\n';
    break;
  case OptionValue::eTypeRegex:
    os << ":default: " << val.GetAsRegex()->GetDefaultValue() << '\n';
    break;
  case OptionValue::eTypeSInt64: {
    os << ":default: "
       << llvm::formatv("{}", val.GetAsSInt64()->GetDefaultValue()).str()
       << '\n';

    int64_t min = val.GetAsSInt64()->GetMinimumValue();
    if (min != 0)
      os << ":minimum: " << llvm::formatv("{}", min).str() << '\n';

    int64_t max = val.GetAsSInt64()->GetMaximumValue();
    if (max != std::numeric_limits<int64_t>::max())
      os << ":maximum: " << llvm::formatv("{}", max).str() << '\n';
  } break;
  case OptionValue::eTypeUInt64: {
    os << ":default: "
       << llvm::formatv("{}", val.GetAsUInt64()->GetDefaultValue()).str()
       << '\n';

    uint64_t min = val.GetAsUInt64()->GetMinimumValue();
    if (min != 0)
      os << ":minimum: " << llvm::formatv("{}", min).str() << '\n';

    uint64_t max = val.GetAsUInt64()->GetMaximumValue();
    if (max != std::numeric_limits<uint64_t>::max())
      os << ":maximum: " << llvm::formatv("{}", max).str() << '\n';
  } break;
  case OptionValue::eTypeString: {
    llvm::StringRef default_val = val.GetAsString()->GetDefaultValueAsRef();
    if (!default_val.empty())
      os << ":default: " << val.GetAsString()->GetDefaultValueAsRef()
         << '\n';
  } break;
  default:
    break;
  }
}

void printMdOptionValueProperty(Stream &os, llvm::StringRef prefix,
                                const Property &prop) {
  OptionValueSP value_sp = prop.GetValue();
  if (!value_sp || value_sp->GetType() == OptionValue::eTypeProperties)
    return;

  os << "```{lldbsetting} ";
  if (!prefix.empty())
    os << prefix << '.';
  os << prop.GetName() << '\n';
  os << ":type: \"" << value_sp->GetTypeAsCString() << "\"\n\n";

  os << prop.GetDescription().trim() << "\n\n";

  printMdOptionFields(os, *value_sp);
  os << "```\n";
}

void printMdOptionProperties(Stream &os, uint8_t level, llvm::StringRef name,
                             const OptionValueProperties &props) {

  if (level > 0)
    os << std::string(level + 2, '#') << ' ' << props.GetName() << "\n\n";

  for (size_t i = 0; i < props.GetNumProperties(); i++) {
    const Property *prop = props.GetPropertyAtIndex(i);
    if (prop)
      printMdOptionValueProperty(os, name, *prop);
  }

  // put properties last
  for (size_t i = 0; i < props.GetNumProperties(); i++) {
    const Property *prop = props.GetPropertyAtIndex(i);
    if (!prop || !prop->GetValue() ||
        prop->GetValue()->GetType() != OptionValue::eTypeProperties)
      continue;

    std::string full_path;
    if (level > 0)
      full_path = name.str() + '.';
    full_path.append(prop->GetName());

    printMdOptionProperties(os, level + 1, full_path,
                            *prop->GetValue()->GetAsProperties());
  }
}

} // namespace

int lldb_private::generateMarkdownDocs(Debugger &dbg,
                                       llvm::StringRef output_dir) {
  std::string output_file = (output_dir + "/settings.md").str();
  StreamFile os(output_file.c_str(),
               File::eOpenOptionWriteOnly | File::eOpenOptionCanCreate |
                   File::eOpenOptionTruncate);
  os << SETTINGS_HEADER;
  printMdOptionProperties(os, 0, "", *dbg.GetValueProperties());
  return 0;
}
