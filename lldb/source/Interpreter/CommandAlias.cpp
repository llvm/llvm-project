//===-- CommandAlias.cpp ------------------------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/CommandAlias.h"

#include "llvm/Support/ErrorHandling.h"

#include "lldb/Core/StreamString.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/Options.h"

using namespace lldb;
using namespace lldb_private;

static bool ProcessAliasOptionsArgs(lldb::CommandObjectSP &cmd_obj_sp,
                                    const char *options_args,
                                    OptionArgVectorSP &option_arg_vector_sp) {
  bool success = true;
  OptionArgVector *option_arg_vector = option_arg_vector_sp.get();

  if (!options_args || (strlen(options_args) < 1))
    return true;

  std::string options_string(options_args);
  Args args(options_args);
  CommandReturnObject result;
  // Check to see if the command being aliased can take any command options.
  Options *options = cmd_obj_sp->GetOptions();
  if (options) {
    // See if any options were specified as part of the alias;  if so, handle
    // them appropriately.
    ExecutionContext exe_ctx =
        cmd_obj_sp->GetCommandInterpreter().GetExecutionContext();
    options->NotifyOptionParsingStarting(&exe_ctx);
    args.Unshift("dummy_arg");
    args.ParseAliasOptions(*options, result, option_arg_vector, options_string);
    args.Shift();
    if (result.Succeeded())
      options->VerifyPartialOptions(result);
    if (!result.Succeeded() &&
        result.GetStatus() != lldb::eReturnStatusStarted) {
      result.AppendError("Unable to create requested alias.\n");
      return false;
    }
  }

  if (!options_string.empty()) {
    if (cmd_obj_sp->WantsRawCommandString())
      option_arg_vector->push_back(
          OptionArgPair("<argument>", OptionArgValue(-1, options_string)));
    else {
      const size_t argc = args.GetArgumentCount();
      for (size_t i = 0; i < argc; ++i)
        if (strcmp(args.GetArgumentAtIndex(i), "") != 0)
          option_arg_vector->push_back(OptionArgPair(
              "<argument>",
              OptionArgValue(-1, std::string(args.GetArgumentAtIndex(i)))));
    }
  }

  return success;
}

CommandAlias::CommandAlias(CommandInterpreter &interpreter,
                           lldb::CommandObjectSP cmd_sp,
                           const char *options_args, const char *name,
                           const char *help, const char *syntax, uint32_t flags)
    : CommandObject(interpreter, name, help, syntax, flags),
      m_underlying_command_sp(),
      m_option_string(options_args ? options_args : ""),
      m_option_args_sp(new OptionArgVector),
      m_is_dashdash_alias(eLazyBoolCalculate), m_did_set_help(false),
      m_did_set_help_long(false) {
  if (ProcessAliasOptionsArgs(cmd_sp, options_args, m_option_args_sp)) {
    m_underlying_command_sp = cmd_sp;
    for (int i = 0;
         auto cmd_entry = m_underlying_command_sp->GetArgumentEntryAtIndex(i);
         i++) {
      m_arguments.push_back(*cmd_entry);
    }
    if (!help || !help[0]) {
      StreamString sstr;
      StreamString translation_and_help;
      GetAliasExpansion(sstr);

      translation_and_help.Printf("(%s)  %s", sstr.GetData(),
                                  GetUnderlyingCommand()->GetHelp());
      SetHelp(translation_and_help.GetData());
    }
  }
}

bool CommandAlias::WantsRawCommandString() {
  if (IsValid())
    return m_underlying_command_sp->WantsRawCommandString();
  return false;
}

bool CommandAlias::WantsCompletion() {
  if (IsValid())
    return m_underlying_command_sp->WantsCompletion();
  return false;
}

int CommandAlias::HandleCompletion(Args &input, int &cursor_index,
                                   int &cursor_char_position,
                                   int match_start_point,
                                   int max_return_elements, bool &word_complete,
                                   StringList &matches) {
  if (IsValid())
    return m_underlying_command_sp->HandleCompletion(
        input, cursor_index, cursor_char_position, match_start_point,
        max_return_elements, word_complete, matches);
  return -1;
}

int CommandAlias::HandleArgumentCompletion(
    Args &input, int &cursor_index, int &cursor_char_position,
    OptionElementVector &opt_element_vector, int match_start_point,
    int max_return_elements, bool &word_complete, StringList &matches) {
  if (IsValid())
    return m_underlying_command_sp->HandleArgumentCompletion(
        input, cursor_index, cursor_char_position, opt_element_vector,
        match_start_point, max_return_elements, word_complete, matches);
  return -1;
}

Options *CommandAlias::GetOptions() {
  if (IsValid())
    return m_underlying_command_sp->GetOptions();
  return nullptr;
}

bool CommandAlias::Execute(const char *args_string,
                           CommandReturnObject &result) {
  llvm_unreachable("CommandAlias::Execute is not to be called");
}

void CommandAlias::GetAliasExpansion(StreamString &help_string) {
  const char *command_name = m_underlying_command_sp->GetCommandName();
  help_string.Printf("'%s", command_name);

  if (m_option_args_sp) {
    OptionArgVector *options = m_option_args_sp.get();
    for (size_t i = 0; i < options->size(); ++i) {
      OptionArgPair cur_option = (*options)[i];
      std::string opt = cur_option.first;
      OptionArgValue value_pair = cur_option.second;
      std::string value = value_pair.second;
      if (opt.compare("<argument>") == 0) {
        help_string.Printf(" %s", value.c_str());
      } else {
        help_string.Printf(" %s", opt.c_str());
        if ((value.compare("<no-argument>") != 0) &&
            (value.compare("<need-argument") != 0)) {
          help_string.Printf(" %s", value.c_str());
        }
      }
    }
  }

  help_string.Printf("'");
}

bool CommandAlias::IsDashDashCommand() {
  if (m_is_dashdash_alias == eLazyBoolCalculate) {
    m_is_dashdash_alias = eLazyBoolNo;
    if (IsValid()) {
      for (const OptionArgPair &opt_arg : *GetOptionArguments()) {
        if (opt_arg.first == "<argument>" && !opt_arg.second.second.empty() &&
            llvm::StringRef(opt_arg.second.second).endswith("--")) {
          m_is_dashdash_alias = eLazyBoolYes;
          break;
        }
      }
      // if this is a nested alias, it may be adding arguments on top of an
      // already dash-dash alias
      if ((m_is_dashdash_alias == eLazyBoolNo) && IsNestedAlias())
        m_is_dashdash_alias =
            (GetUnderlyingCommand()->IsDashDashCommand() ? eLazyBoolYes
                                                         : eLazyBoolNo);
    }
  }
  return (m_is_dashdash_alias == eLazyBoolYes);
}

bool CommandAlias::IsNestedAlias() {
  if (GetUnderlyingCommand())
    return GetUnderlyingCommand()->IsAlias();
  return false;
}

std::pair<lldb::CommandObjectSP, OptionArgVectorSP> CommandAlias::Desugar() {
  auto underlying = GetUnderlyingCommand();
  if (!underlying)
    return {nullptr, nullptr};

  if (underlying->IsAlias()) {
    auto desugared = ((CommandAlias *)underlying.get())->Desugar();
    auto options = GetOptionArguments();
    options->insert(options->begin(), desugared.second->begin(),
                    desugared.second->end());
    return {desugared.first, options};
  }

  return {underlying, GetOptionArguments()};
}

// allow CommandAlias objects to provide their own help, but fallback to the
// info
// for the underlying command if no customization has been provided
void CommandAlias::SetHelp(const char *str) {
  this->CommandObject::SetHelp(str);
  m_did_set_help = true;
}

void CommandAlias::SetHelpLong(const char *str) {
  this->CommandObject::SetHelpLong(str);
  m_did_set_help_long = true;
}

const char *CommandAlias::GetHelp() {
  if (!m_cmd_help_short.empty() || m_did_set_help)
    return m_cmd_help_short.c_str();
  if (IsValid())
    return m_underlying_command_sp->GetHelp();
  return nullptr;
}

const char *CommandAlias::GetHelpLong() {
  if (!m_cmd_help_long.empty() || m_did_set_help_long)
    return m_cmd_help_long.c_str();
  if (IsValid())
    return m_underlying_command_sp->GetHelpLong();
  return nullptr;
}
