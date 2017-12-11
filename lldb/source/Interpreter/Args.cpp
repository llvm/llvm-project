//===-- Args.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <cstdlib>
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/DataFormatters/FormatManager.h"
#include "lldb/Host/OptionParser.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Utility/StreamString.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"

using namespace lldb;
using namespace lldb_private;


// A helper function for argument parsing.
// Parses the initial part of the first argument using normal double quote
// rules:
// backslash escapes the double quote and itself. The parsed string is appended
// to the second
// argument. The function returns the unparsed portion of the string, starting
// at the closing
// quote.
static llvm::StringRef ParseDoubleQuotes(llvm::StringRef quoted,
                                         std::string &result) {
  // Inside double quotes, '\' and '"' are special.
  static const char *k_escapable_characters = "\"\\";
  while (true) {
    // Skip over over regular characters and append them.
    size_t regular = quoted.find_first_of(k_escapable_characters);
    result += quoted.substr(0, regular);
    quoted = quoted.substr(regular);

    // If we have reached the end of string or the closing quote, we're done.
    if (quoted.empty() || quoted.front() == '"')
      break;

    // We have found a backslash.
    quoted = quoted.drop_front();

    if (quoted.empty()) {
      // A lone backslash at the end of string, let's just append it.
      result += '\\';
      break;
    }

    // If the character after the backslash is not a whitelisted escapable
    // character, we
    // leave the character sequence untouched.
    if (strchr(k_escapable_characters, quoted.front()) == nullptr)
      result += '\\';

    result += quoted.front();
    quoted = quoted.drop_front();
  }

  return quoted;
}

static size_t ArgvToArgc(const char **argv) {
  if (!argv)
    return 0;
  size_t count = 0;
  while (*argv++)
    ++count;
  return count;
}

// A helper function for SetCommandString. Parses a single argument from the
// command string, processing quotes and backslashes in a shell-like manner.
// The function returns a tuple consisting of the parsed argument, the quote
// char used, and the unparsed portion of the string starting at the first
// unqouted, unescaped whitespace character.
static std::tuple<std::string, char, llvm::StringRef>
ParseSingleArgument(llvm::StringRef command) {
  // Argument can be split into multiple discontiguous pieces, for example:
  //  "Hello ""World"
  // this would result in a single argument "Hello World" (without the quotes)
  // since the quotes would be removed and there is not space between the
  // strings.
  std::string arg;

  // Since we can have multiple quotes that form a single command
  // in a command like: "Hello "world'!' (which will make a single
  // argument "Hello world!") we remember the first quote character
  // we encounter and use that for the quote character.
  char first_quote_char = '\0';

  bool arg_complete = false;
  do {
    // Skip over over regular characters and append them.
    size_t regular = command.find_first_of(" \t\"'`\\");
    arg += command.substr(0, regular);
    command = command.substr(regular);

    if (command.empty())
      break;

    char special = command.front();
    command = command.drop_front();
    switch (special) {
    case '\\':
      if (command.empty()) {
        arg += '\\';
        break;
      }

      // If the character after the backslash is not a whitelisted escapable
      // character, we
      // leave the character sequence untouched.
      if (strchr(" \t\\'\"`", command.front()) == nullptr)
        arg += '\\';

      arg += command.front();
      command = command.drop_front();

      break;

    case ' ':
    case '\t':
      // We are not inside any quotes, we just found a space after an
      // argument. We are done.
      arg_complete = true;
      break;

    case '"':
    case '\'':
    case '`':
      // We found the start of a quote scope.
      if (first_quote_char == '\0')
        first_quote_char = special;

      if (special == '"')
        command = ParseDoubleQuotes(command, arg);
      else {
        // For single quotes, we simply skip ahead to the matching quote
        // character
        // (or the end of the string).
        size_t quoted = command.find(special);
        arg += command.substr(0, quoted);
        command = command.substr(quoted);
      }

      // If we found a closing quote, skip it.
      if (!command.empty())
        command = command.drop_front();

      break;
    }
  } while (!arg_complete);

  return std::make_tuple(arg, first_quote_char, command);
}

Args::ArgEntry::ArgEntry(llvm::StringRef str, char quote) : quote(quote) {
  size_t size = str.size();
  ptr.reset(new char[size + 1]);

  ::memcpy(data(), str.data() ? str.data() : "", size);
  ptr[size] = 0;
  ref = llvm::StringRef(c_str(), size);
}

//----------------------------------------------------------------------
// Args constructor
//----------------------------------------------------------------------
Args::Args(llvm::StringRef command) { SetCommandString(command); }

Args::Args(const Args &rhs) { *this = rhs; }

Args::Args(const StringList &list) : Args() {
  for(size_t i = 0; i < list.GetSize(); ++i)
    AppendArgument(list[i]);
}

Args &Args::operator=(const Args &rhs) {
  Clear();

  m_argv.clear();
  m_entries.clear();
  for (auto &entry : rhs.m_entries) {
    m_entries.emplace_back(entry.ref, entry.quote);
    m_argv.push_back(m_entries.back().data());
  }
  m_argv.push_back(nullptr);
  return *this;
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
Args::~Args() {}

void Args::Dump(Stream &s, const char *label_name) const {
  if (!label_name)
    return;

  int i = 0;
  for (auto &entry : m_entries) {
    s.Indent();
    s.Format("{0}[{1}]=\"{2}\"\n", label_name, i++, entry.ref);
  }
  s.Format("{0}[{1}]=NULL\n", label_name, i);
  s.EOL();
}

bool Args::GetCommandString(std::string &command) const {
  command.clear();

  for (size_t i = 0; i < m_entries.size(); ++i) {
    if (i > 0)
      command += ' ';
    command += m_entries[i].ref;
  }

  return !m_entries.empty();
}

bool Args::GetQuotedCommandString(std::string &command) const {
  command.clear();

  for (size_t i = 0; i < m_entries.size(); ++i) {
    if (i > 0)
      command += ' ';

    if (m_entries[i].quote) {
      command += m_entries[i].quote;
      command += m_entries[i].ref;
      command += m_entries[i].quote;
    } else {
      command += m_entries[i].ref;
    }
  }

  return !m_entries.empty();
}

void Args::SetCommandString(llvm::StringRef command) {
  Clear();
  m_argv.clear();

  static const char *k_space_separators = " \t";
  command = command.ltrim(k_space_separators);
  std::string arg;
  char quote;
  while (!command.empty()) {
    std::tie(arg, quote, command) = ParseSingleArgument(command);
    m_entries.emplace_back(arg, quote);
    m_argv.push_back(m_entries.back().data());
    command = command.ltrim(k_space_separators);
  }
  m_argv.push_back(nullptr);
}

void Args::UpdateArgsAfterOptionParsing() {
  assert(!m_argv.empty());
  assert(m_argv.back() == nullptr);

  // Now m_argv might be out of date with m_entries, so we need to fix that.
  // This happens because getopt_long_only may permute the order of the
  // arguments in argv, so we need to re-order the quotes and the refs array
  // to match.
  for (size_t i = 0; i < m_argv.size() - 1; ++i) {
    const char *argv = m_argv[i];
    auto pos =
        std::find_if(m_entries.begin() + i, m_entries.end(),
                     [argv](const ArgEntry &D) { return D.c_str() == argv; });
    assert(pos != m_entries.end());
    size_t distance = std::distance(m_entries.begin(), pos);
    if (i == distance)
      continue;

    assert(distance > i);

    std::swap(m_entries[i], m_entries[distance]);
    assert(m_entries[i].ref.data() == m_argv[i]);
  }
  m_entries.resize(m_argv.size() - 1);
}

size_t Args::GetArgumentCount() const { return m_entries.size(); }

const char *Args::GetArgumentAtIndex(size_t idx) const {
  if (idx < m_argv.size())
    return m_argv[idx];
  return nullptr;
}

char Args::GetArgumentQuoteCharAtIndex(size_t idx) const {
  if (idx < m_entries.size())
    return m_entries[idx].quote;
  return '\0';
}

char **Args::GetArgumentVector() {
  assert(!m_argv.empty());
  // TODO: functions like execve and posix_spawnp exhibit undefined behavior
  // when argv or envp is null.  So the code below is actually wrong.  However,
  // other code in LLDB depends on it being null.  The code has been acting this
  // way for some time, so it makes sense to leave it this way until someone
  // has the time to come along and fix it.
  return (m_argv.size() > 1) ? m_argv.data() : nullptr;
}

const char **Args::GetConstArgumentVector() const {
  assert(!m_argv.empty());
  return (m_argv.size() > 1) ? const_cast<const char **>(m_argv.data())
                             : nullptr;
}

void Args::Shift() {
  // Don't pop the last NULL terminator from the argv array
  if (m_entries.empty())
    return;
  m_argv.erase(m_argv.begin());
  m_entries.erase(m_entries.begin());
}

void Args::Unshift(llvm::StringRef arg_str, char quote_char) {
  InsertArgumentAtIndex(0, arg_str, quote_char);
}

void Args::AppendArguments(const Args &rhs) {
  assert(m_argv.size() == m_entries.size() + 1);
  assert(m_argv.back() == nullptr);
  m_argv.pop_back();
  for (auto &entry : rhs.m_entries) {
    m_entries.emplace_back(entry.ref, entry.quote);
    m_argv.push_back(m_entries.back().data());
  }
  m_argv.push_back(nullptr);
}

void Args::AppendArguments(const char **argv) {
  size_t argc = ArgvToArgc(argv);

  assert(m_argv.size() == m_entries.size() + 1);
  assert(m_argv.back() == nullptr);
  m_argv.pop_back();
  for (auto arg : llvm::makeArrayRef(argv, argc)) {
    m_entries.emplace_back(arg, '\0');
    m_argv.push_back(m_entries.back().data());
  }

  m_argv.push_back(nullptr);
}

void Args::AppendArgument(llvm::StringRef arg_str, char quote_char) {
  InsertArgumentAtIndex(GetArgumentCount(), arg_str, quote_char);
}

void Args::InsertArgumentAtIndex(size_t idx, llvm::StringRef arg_str,
                                 char quote_char) {
  assert(m_argv.size() == m_entries.size() + 1);
  assert(m_argv.back() == nullptr);

  if (idx > m_entries.size())
    return;
  m_entries.emplace(m_entries.begin() + idx, arg_str, quote_char);
  m_argv.insert(m_argv.begin() + idx, m_entries[idx].data());
}

void Args::ReplaceArgumentAtIndex(size_t idx, llvm::StringRef arg_str,
                                  char quote_char) {
  assert(m_argv.size() == m_entries.size() + 1);
  assert(m_argv.back() == nullptr);

  if (idx >= m_entries.size())
    return;

  if (arg_str.size() > m_entries[idx].ref.size()) {
    m_entries[idx] = ArgEntry(arg_str, quote_char);
    m_argv[idx] = m_entries[idx].data();
  } else {
    const char *src_data = arg_str.data() ? arg_str.data() : "";
    ::memcpy(m_entries[idx].data(), src_data, arg_str.size());
    m_entries[idx].ptr[arg_str.size()] = 0;
    m_entries[idx].ref = m_entries[idx].ref.take_front(arg_str.size());
  }
}

void Args::DeleteArgumentAtIndex(size_t idx) {
  if (idx >= m_entries.size())
    return;

  m_argv.erase(m_argv.begin() + idx);
  m_entries.erase(m_entries.begin() + idx);
}

void Args::SetArguments(size_t argc, const char **argv) {
  Clear();

  auto args = llvm::makeArrayRef(argv, argc);
  m_entries.resize(argc);
  m_argv.resize(argc + 1);
  for (size_t i = 0; i < args.size(); ++i) {
    char quote =
        ((args[i][0] == '\'') || (args[i][0] == '"') || (args[i][0] == '`'))
            ? args[i][0]
            : '\0';

    m_entries[i] = ArgEntry(args[i], quote);
    m_argv[i] = m_entries[i].data();
  }
}

void Args::SetArguments(const char **argv) {
  SetArguments(ArgvToArgc(argv), argv);
}

Status Args::ParseOptions(Options &options, ExecutionContext *execution_context,
                          PlatformSP platform_sp, bool require_validation) {
  StreamString sstr;
  Status error;
  Option *long_options = options.GetLongOptions();
  if (long_options == nullptr) {
    error.SetErrorStringWithFormat("invalid long options");
    return error;
  }

  for (int i = 0; long_options[i].definition != nullptr; ++i) {
    if (long_options[i].flag == nullptr) {
      if (isprint8(long_options[i].val)) {
        sstr << (char)long_options[i].val;
        switch (long_options[i].definition->option_has_arg) {
        default:
        case OptionParser::eNoArgument:
          break;
        case OptionParser::eRequiredArgument:
          sstr << ':';
          break;
        case OptionParser::eOptionalArgument:
          sstr << "::";
          break;
        }
      }
    }
  }
  std::unique_lock<std::mutex> lock;
  OptionParser::Prepare(lock);
  int val;
  while (1) {
    int long_options_index = -1;
    val = OptionParser::Parse(GetArgumentCount(), GetArgumentVector(),
                              sstr.GetString(), long_options,
                              &long_options_index);
    if (val == -1)
      break;

    // Did we get an error?
    if (val == '?') {
      error.SetErrorStringWithFormat("unknown or ambiguous option");
      break;
    }
    // The option auto-set itself
    if (val == 0)
      continue;

    ((Options *)&options)->OptionSeen(val);

    // Lookup the long option index
    if (long_options_index == -1) {
      for (int i = 0; long_options[i].definition || long_options[i].flag ||
                      long_options[i].val;
           ++i) {
        if (long_options[i].val == val) {
          long_options_index = i;
          break;
        }
      }
    }
    // Call the callback with the option
    if (long_options_index >= 0 &&
        long_options[long_options_index].definition) {
      const OptionDefinition *def = long_options[long_options_index].definition;

      if (!platform_sp) {
        // User did not pass in an explicit platform.  Try to grab
        // from the execution context.
        TargetSP target_sp =
            execution_context ? execution_context->GetTargetSP() : TargetSP();
        platform_sp = target_sp ? target_sp->GetPlatform() : PlatformSP();
      }
      OptionValidator *validator = def->validator;

      if (!platform_sp && require_validation) {
        // Caller requires validation but we cannot validate as we
        // don't have the mandatory platform against which to
        // validate.
        error.SetErrorString("cannot validate options: "
                             "no platform available");
        return error;
      }

      bool validation_failed = false;
      if (platform_sp) {
        // Ensure we have an execution context, empty or not.
        ExecutionContext dummy_context;
        ExecutionContext *exe_ctx_p =
            execution_context ? execution_context : &dummy_context;
        if (validator && !validator->IsValid(*platform_sp, *exe_ctx_p)) {
          validation_failed = true;
          error.SetErrorStringWithFormat("Option \"%s\" invalid.  %s",
                                         def->long_option,
                                         def->validator->LongConditionString());
        }
      }

      // As long as validation didn't fail, we set the option value.
      if (!validation_failed)
        error = options.SetOptionValue(
            long_options_index,
            (def->option_has_arg == OptionParser::eNoArgument)
                ? nullptr
                : OptionParser::GetOptionArgument(),
            execution_context);
    } else {
      error.SetErrorStringWithFormat("invalid option with value '%i'", val);
    }
    if (error.Fail())
      break;
  }

  // Update our ARGV now that get options has consumed all the options
  m_argv.erase(m_argv.begin(), m_argv.begin() + OptionParser::GetOptionIndex());
  UpdateArgsAfterOptionParsing();
  return error;
}

void Args::Clear() {
  m_entries.clear();
  m_argv.clear();
  m_argv.push_back(nullptr);
}

lldb::addr_t Args::StringToAddress(const ExecutionContext *exe_ctx,
                                   llvm::StringRef s, lldb::addr_t fail_value,
                                   Status *error_ptr) {
  bool error_set = false;
  if (s.empty()) {
    if (error_ptr)
      error_ptr->SetErrorStringWithFormat("invalid address expression \"%s\"",
                                          s.str().c_str());
    return fail_value;
  }

  llvm::StringRef sref = s;

  lldb::addr_t addr = LLDB_INVALID_ADDRESS;
  if (!s.getAsInteger(0, addr)) {
    if (error_ptr)
      error_ptr->Clear();
    return addr;
  }

  // Try base 16 with no prefix...
  if (!s.getAsInteger(16, addr)) {
    if (error_ptr)
      error_ptr->Clear();
    return addr;
  }

  Target *target = nullptr;
  if (!exe_ctx || !(target = exe_ctx->GetTargetPtr())) {
    if (error_ptr)
      error_ptr->SetErrorStringWithFormat("invalid address expression \"%s\"",
                                          s.str().c_str());
    return fail_value;
  }

  lldb::ValueObjectSP valobj_sp;
  EvaluateExpressionOptions options;
  options.SetCoerceToId(false);
  options.SetUnwindOnError(true);
  options.SetKeepInMemory(false);
  options.SetTryAllThreads(true);

  ExpressionResults expr_result =
      target->EvaluateExpression(s, exe_ctx->GetFramePtr(), valobj_sp, options);

  bool success = false;
  if (expr_result == eExpressionCompleted) {
    if (valobj_sp)
      valobj_sp = valobj_sp->GetQualifiedRepresentationIfAvailable(
          valobj_sp->GetDynamicValueType(), true);
    // Get the address to watch.
    if (valobj_sp)
      addr = valobj_sp->GetValueAsUnsigned(fail_value, &success);
    if (success) {
      if (error_ptr)
        error_ptr->Clear();
      return addr;
    } else {
      if (error_ptr) {
        error_set = true;
        error_ptr->SetErrorStringWithFormat(
            "address expression \"%s\" resulted in a value whose type "
            "can't be converted to an address: %s",
            s.str().c_str(), valobj_sp->GetTypeName().GetCString());
      }
    }

  } else {
    // Since the compiler can't handle things like "main + 12" we should
    // try to do this for now. The compiler doesn't like adding offsets
    // to function pointer types.
    static RegularExpression g_symbol_plus_offset_regex(
        "^(.*)([-\\+])[[:space:]]*(0x[0-9A-Fa-f]+|[0-9]+)[[:space:]]*$");
    RegularExpression::Match regex_match(3);
    if (g_symbol_plus_offset_regex.Execute(sref, &regex_match)) {
      uint64_t offset = 0;
      bool add = true;
      std::string name;
      std::string str;
      if (regex_match.GetMatchAtIndex(s, 1, name)) {
        if (regex_match.GetMatchAtIndex(s, 2, str)) {
          add = str[0] == '+';

          if (regex_match.GetMatchAtIndex(s, 3, str)) {
            if (!llvm::StringRef(str).getAsInteger(0, offset)) {
              Status error;
              addr = StringToAddress(exe_ctx, name.c_str(),
                                     LLDB_INVALID_ADDRESS, &error);
              if (addr != LLDB_INVALID_ADDRESS) {
                if (add)
                  return addr + offset;
                else
                  return addr - offset;
              }
            }
          }
        }
      }
    }

    if (error_ptr) {
      error_set = true;
      error_ptr->SetErrorStringWithFormat(
          "address expression \"%s\" evaluation failed", s.str().c_str());
    }
  }

  if (error_ptr) {
    if (!error_set)
      error_ptr->SetErrorStringWithFormat("invalid address expression \"%s\"",
                                          s.str().c_str());
  }
  return fail_value;
}

const char *Args::StripSpaces(std::string &s, bool leading, bool trailing,
                              bool return_null_if_empty) {
  static const char *k_white_space = " \t\v";
  if (!s.empty()) {
    if (leading) {
      size_t pos = s.find_first_not_of(k_white_space);
      if (pos == std::string::npos)
        s.clear();
      else if (pos > 0)
        s.erase(0, pos);
    }

    if (trailing) {
      size_t rpos = s.find_last_not_of(k_white_space);
      if (rpos != std::string::npos && rpos + 1 < s.size())
        s.erase(rpos + 1);
    }
  }
  if (return_null_if_empty && s.empty())
    return nullptr;
  return s.c_str();
}

bool Args::StringToBoolean(llvm::StringRef ref, bool fail_value,
                           bool *success_ptr) {
  if (success_ptr)
    *success_ptr = true;
  ref = ref.trim();
  if (ref.equals_lower("false") || ref.equals_lower("off") ||
      ref.equals_lower("no") || ref.equals_lower("0")) {
    return false;
  } else if (ref.equals_lower("true") || ref.equals_lower("on") ||
             ref.equals_lower("yes") || ref.equals_lower("1")) {
    return true;
  }
  if (success_ptr)
    *success_ptr = false;
  return fail_value;
}

char Args::StringToChar(llvm::StringRef s, char fail_value, bool *success_ptr) {
  if (success_ptr)
    *success_ptr = false;
  if (s.size() != 1)
    return fail_value;

  if (success_ptr)
    *success_ptr = true;
  return s[0];
}

bool Args::StringToVersion(llvm::StringRef string, uint32_t &major,
                           uint32_t &minor, uint32_t &update) {
  major = UINT32_MAX;
  minor = UINT32_MAX;
  update = UINT32_MAX;

  if (string.empty())
    return false;

  llvm::StringRef major_str, minor_str, update_str;

  std::tie(major_str, minor_str) = string.split('.');
  std::tie(minor_str, update_str) = minor_str.split('.');
  if (major_str.getAsInteger(10, major))
    return false;
  if (!minor_str.empty() && minor_str.getAsInteger(10, minor))
    return false;
  if (!update_str.empty() && update_str.getAsInteger(10, update))
    return false;

  return true;
}

const char *Args::GetShellSafeArgument(const FileSpec &shell,
                                       const char *unsafe_arg,
                                       std::string &safe_arg) {
  struct ShellDescriptor {
    ConstString m_basename;
    const char *m_escapables;
  };

  static ShellDescriptor g_Shells[] = {{ConstString("bash"), " '\"<>()&"},
                                       {ConstString("tcsh"), " '\"<>()&$"},
                                       {ConstString("sh"), " '\"<>()&"}};

  // safe minimal set
  const char *escapables = " '\"";

  if (auto basename = shell.GetFilename()) {
    for (const auto &Shell : g_Shells) {
      if (Shell.m_basename == basename) {
        escapables = Shell.m_escapables;
        break;
      }
    }
  }

  safe_arg.assign(unsafe_arg);
  size_t prev_pos = 0;
  while (prev_pos < safe_arg.size()) {
    // Escape spaces and quotes
    size_t pos = safe_arg.find_first_of(escapables, prev_pos);
    if (pos != std::string::npos) {
      safe_arg.insert(pos, 1, '\\');
      prev_pos = pos + 2;
    } else
      break;
  }
  return safe_arg.c_str();
}

int64_t Args::StringToOptionEnum(llvm::StringRef s,
                                 OptionEnumValueElement *enum_values,
                                 int32_t fail_value, Status &error) {
  error.Clear();
  if (!enum_values) {
    error.SetErrorString("invalid enumeration argument");
    return fail_value;
  }

  if (s.empty()) {
    error.SetErrorString("empty enumeration string");
    return fail_value;
  }

  for (int i = 0; enum_values[i].string_value != nullptr; i++) {
    llvm::StringRef this_enum(enum_values[i].string_value);
    if (this_enum.startswith(s))
      return enum_values[i].value;
  }

  StreamString strm;
  strm.PutCString("invalid enumeration value, valid values are: ");
  for (int i = 0; enum_values[i].string_value != nullptr; i++) {
    strm.Printf("%s\"%s\"", i > 0 ? ", " : "", enum_values[i].string_value);
  }
  error.SetErrorString(strm.GetString());
  return fail_value;
}

lldb::ScriptLanguage
Args::StringToScriptLanguage(llvm::StringRef s, lldb::ScriptLanguage fail_value,
                             bool *success_ptr) {
  if (success_ptr)
    *success_ptr = true;

  if (s.equals_lower("python"))
    return eScriptLanguagePython;
  if (s.equals_lower("default"))
    return eScriptLanguageDefault;
  if (s.equals_lower("none"))
    return eScriptLanguageNone;

  if (success_ptr)
    *success_ptr = false;
  return fail_value;
}

Status Args::StringToFormat(const char *s, lldb::Format &format,
                            size_t *byte_size_ptr) {
  format = eFormatInvalid;
  Status error;

  if (s && s[0]) {
    if (byte_size_ptr) {
      if (isdigit(s[0])) {
        char *format_char = nullptr;
        unsigned long byte_size = ::strtoul(s, &format_char, 0);
        if (byte_size != ULONG_MAX)
          *byte_size_ptr = byte_size;
        s = format_char;
      } else
        *byte_size_ptr = 0;
    }

    const bool partial_match_ok = true;
    if (!FormatManager::GetFormatFromCString(s, partial_match_ok, format)) {
      StreamString error_strm;
      error_strm.Printf(
          "Invalid format character or name '%s'. Valid values are:\n", s);
      for (Format f = eFormatDefault; f < kNumFormats; f = Format(f + 1)) {
        char format_char = FormatManager::GetFormatAsFormatChar(f);
        if (format_char)
          error_strm.Printf("'%c' or ", format_char);

        error_strm.Printf("\"%s\"", FormatManager::GetFormatAsCString(f));
        error_strm.EOL();
      }

      if (byte_size_ptr)
        error_strm.PutCString(
            "An optional byte size can precede the format character.\n");
      error.SetErrorString(error_strm.GetString());
    }

    if (error.Fail())
      return error;
  } else {
    error.SetErrorStringWithFormat("%s option string", s ? "empty" : "invalid");
  }
  return error;
}

lldb::Encoding Args::StringToEncoding(llvm::StringRef s,
                                      lldb::Encoding fail_value) {
  return llvm::StringSwitch<lldb::Encoding>(s)
      .Case("uint", eEncodingUint)
      .Case("sint", eEncodingSint)
      .Case("ieee754", eEncodingIEEE754)
      .Case("vector", eEncodingVector)
      .Default(fail_value);
}

uint32_t Args::StringToGenericRegister(llvm::StringRef s) {
  if (s.empty())
    return LLDB_INVALID_REGNUM;
  uint32_t result = llvm::StringSwitch<uint32_t>(s)
                        .Case("pc", LLDB_REGNUM_GENERIC_PC)
                        .Case("sp", LLDB_REGNUM_GENERIC_SP)
                        .Case("fp", LLDB_REGNUM_GENERIC_FP)
                        .Cases("ra", "lr", LLDB_REGNUM_GENERIC_RA)
                        .Case("flags", LLDB_REGNUM_GENERIC_FLAGS)
                        .Case("arg1", LLDB_REGNUM_GENERIC_ARG1)
                        .Case("arg2", LLDB_REGNUM_GENERIC_ARG2)
                        .Case("arg3", LLDB_REGNUM_GENERIC_ARG3)
                        .Case("arg4", LLDB_REGNUM_GENERIC_ARG4)
                        .Case("arg5", LLDB_REGNUM_GENERIC_ARG5)
                        .Case("arg6", LLDB_REGNUM_GENERIC_ARG6)
                        .Case("arg7", LLDB_REGNUM_GENERIC_ARG7)
                        .Case("arg8", LLDB_REGNUM_GENERIC_ARG8)
                        .Default(LLDB_INVALID_REGNUM);
  return result;
}

void Args::AddOrReplaceEnvironmentVariable(llvm::StringRef env_var_name,
                                           llvm::StringRef new_value) {
  if (env_var_name.empty())
    return;

  // Build the new entry.
  std::string var_string(env_var_name);
  if (!new_value.empty()) {
    var_string += "=";
    var_string += new_value;
  }

  size_t index = 0;
  if (ContainsEnvironmentVariable(env_var_name, &index)) {
    ReplaceArgumentAtIndex(index, var_string);
    return;
  }

  // We didn't find it.  Append it instead.
  AppendArgument(var_string);
}

bool Args::ContainsEnvironmentVariable(llvm::StringRef env_var_name,
                                       size_t *argument_index) const {
  // Validate args.
  if (env_var_name.empty())
    return false;

  // Check each arg to see if it matches the env var name.
  for (auto arg : llvm::enumerate(m_entries)) {
    llvm::StringRef name, value;
    std::tie(name, value) = arg.value().ref.split('=');
    if (name != env_var_name)
      continue;

    if (argument_index)
      *argument_index = arg.index();
    return true;
  }

  // We didn't find a match.
  return false;
}

size_t Args::FindArgumentIndexForOption(Option *long_options,
                                        int long_options_index) const {
  char short_buffer[3];
  char long_buffer[255];
  ::snprintf(short_buffer, sizeof(short_buffer), "-%c",
             long_options[long_options_index].val);
  ::snprintf(long_buffer, sizeof(long_buffer), "--%s",
             long_options[long_options_index].definition->long_option);

  for (auto entry : llvm::enumerate(m_entries)) {
    if (entry.value().ref.startswith(short_buffer) ||
        entry.value().ref.startswith(long_buffer))
      return entry.index();
  }

  return size_t(-1);
}

bool Args::IsPositionalArgument(const char *arg) {
  if (arg == nullptr)
    return false;

  bool is_positional = true;
  const char *cptr = arg;

  if (cptr[0] == '%') {
    ++cptr;
    while (isdigit(cptr[0]))
      ++cptr;
    if (cptr[0] != '\0')
      is_positional = false;
  } else
    is_positional = false;

  return is_positional;
}

std::string Args::ParseAliasOptions(Options &options,
                                    CommandReturnObject &result,
                                    OptionArgVector *option_arg_vector,
                                    llvm::StringRef raw_input_string) {
  std::string result_string(raw_input_string);
  StreamString sstr;
  int i;
  Option *long_options = options.GetLongOptions();

  if (long_options == nullptr) {
    result.AppendError("invalid long options");
    result.SetStatus(eReturnStatusFailed);
    return result_string;
  }

  for (i = 0; long_options[i].definition != nullptr; ++i) {
    if (long_options[i].flag == nullptr) {
      sstr << (char)long_options[i].val;
      switch (long_options[i].definition->option_has_arg) {
      default:
      case OptionParser::eNoArgument:
        break;
      case OptionParser::eRequiredArgument:
        sstr << ":";
        break;
      case OptionParser::eOptionalArgument:
        sstr << "::";
        break;
      }
    }
  }

  std::unique_lock<std::mutex> lock;
  OptionParser::Prepare(lock);
  result.SetStatus(eReturnStatusSuccessFinishNoResult);
  int val;
  while (1) {
    int long_options_index = -1;
    val = OptionParser::Parse(GetArgumentCount(), GetArgumentVector(),
                              sstr.GetString(), long_options,
                              &long_options_index);

    if (val == -1)
      break;

    if (val == '?') {
      result.AppendError("unknown or ambiguous option");
      result.SetStatus(eReturnStatusFailed);
      break;
    }

    if (val == 0)
      continue;

    options.OptionSeen(val);

    // Look up the long option index
    if (long_options_index == -1) {
      for (int j = 0; long_options[j].definition || long_options[j].flag ||
                      long_options[j].val;
           ++j) {
        if (long_options[j].val == val) {
          long_options_index = j;
          break;
        }
      }
    }

    // See if the option takes an argument, and see if one was supplied.
    if (long_options_index == -1) {
      result.AppendErrorWithFormat("Invalid option with value '%c'.\n", val);
      result.SetStatus(eReturnStatusFailed);
      return result_string;
    }

    StreamString option_str;
    option_str.Printf("-%c", val);
    const OptionDefinition *def = long_options[long_options_index].definition;
    int has_arg =
        (def == nullptr) ? OptionParser::eNoArgument : def->option_has_arg;

    const char *option_arg = nullptr;
    switch (has_arg) {
    case OptionParser::eRequiredArgument:
      if (OptionParser::GetOptionArgument() == nullptr) {
        result.AppendErrorWithFormat(
            "Option '%s' is missing argument specifier.\n",
            option_str.GetData());
        result.SetStatus(eReturnStatusFailed);
        return result_string;
      }
      LLVM_FALLTHROUGH;
    case OptionParser::eOptionalArgument:
      option_arg = OptionParser::GetOptionArgument();
      LLVM_FALLTHROUGH;
    case OptionParser::eNoArgument:
      break;
    default:
      result.AppendErrorWithFormat("error with options table; invalid value "
                                   "in has_arg field for option '%c'.\n",
                                   val);
      result.SetStatus(eReturnStatusFailed);
      return result_string;
    }
    if (!option_arg)
      option_arg = "<no-argument>";
    option_arg_vector->emplace_back(option_str.GetString(), has_arg,
                                    option_arg);

    // Find option in the argument list; also see if it was supposed to take
    // an argument and if one was supplied.  Remove option (and argument, if
    // given) from the argument list.  Also remove them from the
    // raw_input_string, if one was passed in.
    size_t idx = FindArgumentIndexForOption(long_options, long_options_index);
    if (idx == size_t(-1))
      continue;

    if (!result_string.empty()) {
      auto tmp_arg = m_entries[idx].ref;
      size_t pos = result_string.find(tmp_arg);
      if (pos != std::string::npos)
        result_string.erase(pos, tmp_arg.size());
    }
    ReplaceArgumentAtIndex(idx, llvm::StringRef());
    if ((long_options[long_options_index].definition->option_has_arg !=
         OptionParser::eNoArgument) &&
        (OptionParser::GetOptionArgument() != nullptr) &&
        (idx + 1 < GetArgumentCount()) &&
        (m_entries[idx + 1].ref == OptionParser::GetOptionArgument())) {
      if (result_string.size() > 0) {
        auto tmp_arg = m_entries[idx + 1].ref;
        size_t pos = result_string.find(tmp_arg);
        if (pos != std::string::npos)
          result_string.erase(pos, tmp_arg.size());
      }
      ReplaceArgumentAtIndex(idx + 1, llvm::StringRef());
    }
  }
  return result_string;
}

void Args::ParseArgsForCompletion(Options &options,
                                  OptionElementVector &option_element_vector,
                                  uint32_t cursor_index) {
  StreamString sstr;
  Option *long_options = options.GetLongOptions();
  option_element_vector.clear();

  if (long_options == nullptr) {
    return;
  }

  // Leading : tells getopt to return a : for a missing option argument AND
  // to suppress error messages.

  sstr << ":";
  for (int i = 0; long_options[i].definition != nullptr; ++i) {
    if (long_options[i].flag == nullptr) {
      sstr << (char)long_options[i].val;
      switch (long_options[i].definition->option_has_arg) {
      default:
      case OptionParser::eNoArgument:
        break;
      case OptionParser::eRequiredArgument:
        sstr << ":";
        break;
      case OptionParser::eOptionalArgument:
        sstr << "::";
        break;
      }
    }
  }

  std::unique_lock<std::mutex> lock;
  OptionParser::Prepare(lock);
  OptionParser::EnableError(false);

  int val;
  auto opt_defs = options.GetDefinitions();

  // Fooey... OptionParser::Parse permutes the GetArgumentVector to move the
  // options to the front. So we have to build another Arg and pass that to
  // OptionParser::Parse so it doesn't change the one we have.

  std::vector<char *> dummy_vec = m_argv;

  bool failed_once = false;
  uint32_t dash_dash_pos = -1;

  while (1) {
    bool missing_argument = false;
    int long_options_index = -1;

    val = OptionParser::Parse(dummy_vec.size() - 1, &dummy_vec[0],
                              sstr.GetString(), long_options,
                              &long_options_index);

    if (val == -1) {
      // When we're completing a "--" which is the last option on line,
      if (failed_once)
        break;

      failed_once = true;

      // If this is a bare  "--" we mark it as such so we can complete it
      // successfully later.
      // Handling the "--" is a little tricky, since that may mean end of
      // options or arguments, or the
      // user might want to complete options by long name.  I make this work by
      // checking whether the
      // cursor is in the "--" argument, and if so I assume we're completing the
      // long option, otherwise
      // I let it pass to OptionParser::Parse which will terminate the option
      // parsing.
      // Note, in either case we continue parsing the line so we can figure out
      // what other options
      // were passed.  This will be useful when we come to restricting
      // completions based on what other
      // options we've seen on the line.

      if (static_cast<size_t>(OptionParser::GetOptionIndex()) <
              dummy_vec.size() - 1 &&
          (strcmp(dummy_vec[OptionParser::GetOptionIndex() - 1], "--") == 0)) {
        dash_dash_pos = OptionParser::GetOptionIndex() - 1;
        if (static_cast<size_t>(OptionParser::GetOptionIndex() - 1) ==
            cursor_index) {
          option_element_vector.push_back(
              OptionArgElement(OptionArgElement::eBareDoubleDash,
                               OptionParser::GetOptionIndex() - 1,
                               OptionArgElement::eBareDoubleDash));
          continue;
        } else
          break;
      } else
        break;
    } else if (val == '?') {
      option_element_vector.push_back(
          OptionArgElement(OptionArgElement::eUnrecognizedArg,
                           OptionParser::GetOptionIndex() - 1,
                           OptionArgElement::eUnrecognizedArg));
      continue;
    } else if (val == 0) {
      continue;
    } else if (val == ':') {
      // This is a missing argument.
      val = OptionParser::GetOptionErrorCause();
      missing_argument = true;
    }

    ((Options *)&options)->OptionSeen(val);

    // Look up the long option index
    if (long_options_index == -1) {
      for (int j = 0; long_options[j].definition || long_options[j].flag ||
                      long_options[j].val;
           ++j) {
        if (long_options[j].val == val) {
          long_options_index = j;
          break;
        }
      }
    }

    // See if the option takes an argument, and see if one was supplied.
    if (long_options_index >= 0) {
      int opt_defs_index = -1;
      for (size_t i = 0; i < opt_defs.size(); i++) {
        if (opt_defs[i].short_option != val)
          continue;
        opt_defs_index = i;
        break;
      }

      const OptionDefinition *def = long_options[long_options_index].definition;
      int has_arg =
          (def == nullptr) ? OptionParser::eNoArgument : def->option_has_arg;
      switch (has_arg) {
      case OptionParser::eNoArgument:
        option_element_vector.push_back(OptionArgElement(
            opt_defs_index, OptionParser::GetOptionIndex() - 1, 0));
        break;
      case OptionParser::eRequiredArgument:
        if (OptionParser::GetOptionArgument() != nullptr) {
          int arg_index;
          if (missing_argument)
            arg_index = -1;
          else
            arg_index = OptionParser::GetOptionIndex() - 1;

          option_element_vector.push_back(OptionArgElement(
              opt_defs_index, OptionParser::GetOptionIndex() - 2, arg_index));
        } else {
          option_element_vector.push_back(OptionArgElement(
              opt_defs_index, OptionParser::GetOptionIndex() - 1, -1));
        }
        break;
      case OptionParser::eOptionalArgument:
        if (OptionParser::GetOptionArgument() != nullptr) {
          option_element_vector.push_back(OptionArgElement(
              opt_defs_index, OptionParser::GetOptionIndex() - 2,
              OptionParser::GetOptionIndex() - 1));
        } else {
          option_element_vector.push_back(OptionArgElement(
              opt_defs_index, OptionParser::GetOptionIndex() - 2,
              OptionParser::GetOptionIndex() - 1));
        }
        break;
      default:
        // The options table is messed up.  Here we'll just continue
        option_element_vector.push_back(
            OptionArgElement(OptionArgElement::eUnrecognizedArg,
                             OptionParser::GetOptionIndex() - 1,
                             OptionArgElement::eUnrecognizedArg));
        break;
      }
    } else {
      option_element_vector.push_back(
          OptionArgElement(OptionArgElement::eUnrecognizedArg,
                           OptionParser::GetOptionIndex() - 1,
                           OptionArgElement::eUnrecognizedArg));
    }
  }

  // Finally we have to handle the case where the cursor index points at a
  // single "-".  We want to mark that in
  // the option_element_vector, but only if it is not after the "--".  But it
  // turns out that OptionParser::Parse just ignores
  // an isolated "-".  So we have to look it up by hand here.  We only care if
  // it is AT the cursor position.
  // Note, a single quoted dash is not the same as a single dash...

  const ArgEntry &cursor = m_entries[cursor_index];
  if ((static_cast<int32_t>(dash_dash_pos) == -1 ||
       cursor_index < dash_dash_pos) &&
      cursor.quote == '\0' && cursor.ref == "-") {
    option_element_vector.push_back(
        OptionArgElement(OptionArgElement::eBareDash, cursor_index,
                         OptionArgElement::eBareDash));
  }
}

void Args::EncodeEscapeSequences(const char *src, std::string &dst) {
  dst.clear();
  if (src) {
    for (const char *p = src; *p != '\0'; ++p) {
      size_t non_special_chars = ::strcspn(p, "\\");
      if (non_special_chars > 0) {
        dst.append(p, non_special_chars);
        p += non_special_chars;
        if (*p == '\0')
          break;
      }

      if (*p == '\\') {
        ++p; // skip the slash
        switch (*p) {
        case 'a':
          dst.append(1, '\a');
          break;
        case 'b':
          dst.append(1, '\b');
          break;
        case 'f':
          dst.append(1, '\f');
          break;
        case 'n':
          dst.append(1, '\n');
          break;
        case 'r':
          dst.append(1, '\r');
          break;
        case 't':
          dst.append(1, '\t');
          break;
        case 'v':
          dst.append(1, '\v');
          break;
        case '\\':
          dst.append(1, '\\');
          break;
        case '\'':
          dst.append(1, '\'');
          break;
        case '"':
          dst.append(1, '"');
          break;
        case '0':
          // 1 to 3 octal chars
          {
            // Make a string that can hold onto the initial zero char,
            // up to 3 octal digits, and a terminating NULL.
            char oct_str[5] = {'\0', '\0', '\0', '\0', '\0'};

            int i;
            for (i = 0; (p[i] >= '0' && p[i] <= '7') && i < 4; ++i)
              oct_str[i] = p[i];

            // We don't want to consume the last octal character since
            // the main for loop will do this for us, so we advance p by
            // one less than i (even if i is zero)
            p += i - 1;
            unsigned long octal_value = ::strtoul(oct_str, nullptr, 8);
            if (octal_value <= UINT8_MAX) {
              dst.append(1, (char)octal_value);
            }
          }
          break;

        case 'x':
          // hex number in the format
          if (isxdigit(p[1])) {
            ++p; // Skip the 'x'

            // Make a string that can hold onto two hex chars plus a
            // NULL terminator
            char hex_str[3] = {*p, '\0', '\0'};
            if (isxdigit(p[1])) {
              ++p; // Skip the first of the two hex chars
              hex_str[1] = *p;
            }

            unsigned long hex_value = strtoul(hex_str, nullptr, 16);
            if (hex_value <= UINT8_MAX)
              dst.append(1, (char)hex_value);
          } else {
            dst.append(1, 'x');
          }
          break;

        default:
          // Just desensitize any other character by just printing what
          // came after the '\'
          dst.append(1, *p);
          break;
        }
      }
    }
  }
}

void Args::ExpandEscapedCharacters(const char *src, std::string &dst) {
  dst.clear();
  if (src) {
    for (const char *p = src; *p != '\0'; ++p) {
      if (isprint8(*p))
        dst.append(1, *p);
      else {
        switch (*p) {
        case '\a':
          dst.append("\\a");
          break;
        case '\b':
          dst.append("\\b");
          break;
        case '\f':
          dst.append("\\f");
          break;
        case '\n':
          dst.append("\\n");
          break;
        case '\r':
          dst.append("\\r");
          break;
        case '\t':
          dst.append("\\t");
          break;
        case '\v':
          dst.append("\\v");
          break;
        case '\'':
          dst.append("\\'");
          break;
        case '"':
          dst.append("\\\"");
          break;
        case '\\':
          dst.append("\\\\");
          break;
        default: {
          // Just encode as octal
          dst.append("\\0");
          char octal_str[32];
          snprintf(octal_str, sizeof(octal_str), "%o", *p);
          dst.append(octal_str);
        } break;
        }
      }
    }
  }
}

std::string Args::EscapeLLDBCommandArgument(const std::string &arg,
                                            char quote_char) {
  const char *chars_to_escape = nullptr;
  switch (quote_char) {
  case '\0':
    chars_to_escape = " \t\\'\"`";
    break;
  case '\'':
    chars_to_escape = "";
    break;
  case '"':
    chars_to_escape = "$\"`\\";
    break;
  default:
    assert(false && "Unhandled quote character");
  }

  std::string res;
  res.reserve(arg.size());
  for (char c : arg) {
    if (::strchr(chars_to_escape, c))
      res.push_back('\\');
    res.push_back(c);
  }
  return res;
}
