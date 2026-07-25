//===-- StackFrameRecognizer.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/StackFrameRecognizer.h"
#include "lldb/Core/Module.h"
#include "lldb/Interpreter/Interfaces/ScriptedStackFrameRecognizerInterface.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/Utility/ScriptedMetadata.h"

using namespace lldb;
using namespace lldb_private;

class ScriptedRecognizedStackFrame : public RecognizedStackFrame {
  bool m_hidden;
  lldb::StackFrameSP m_most_relevant_frame;
  lldb::ValueObjectSP m_exception;

public:
  ScriptedRecognizedStackFrame(ValueObjectListSP args, bool hidden,
                               lldb::StackFrameSP most_relevant_frame,
                               lldb::ValueObjectSP exception,
                               std::string stop_desc)
      : m_hidden(hidden), m_most_relevant_frame(std::move(most_relevant_frame)),
        m_exception(std::move(exception)) {
    m_arguments = std::move(args);
    m_stop_desc = std::move(stop_desc);
  }
  bool ShouldHide() override { return m_hidden; }
  lldb::StackFrameSP GetMostRelevantFrame() override {
    return m_most_relevant_frame;
  }
  lldb::ValueObjectSP GetExceptionObject() override { return m_exception; }
};

ScriptedStackFrameRecognizer::ScriptedStackFrameRecognizer(
    ScriptInterpreter *interpreter, const char *pclass)
    : m_python_class(pclass) {
  if (!interpreter)
    return;

  m_interface_sp = interpreter->CreateScriptedStackFrameRecognizerInterface();
  if (!m_interface_sp)
    return;

  ScriptedMetadata scripted_metadata(m_python_class, nullptr);
  auto obj_or_err = m_interface_sp->CreatePluginObject(scripted_metadata);
  if (!obj_or_err) {
    llvm::consumeError(obj_or_err.takeError());
    m_interface_sp.reset();
  }
}

RecognizedStackFrameSP
ScriptedStackFrameRecognizer::RecognizeFrame(lldb::StackFrameSP frame) {
  if (!m_interface_sp)
    return RecognizedStackFrameSP();

  ValueObjectListSP args = m_interface_sp->GetRecognizedArguments(frame);
  auto args_synthesized = std::make_shared<ValueObjectList>();
  if (args) {
    for (const auto &o : args->GetObjects())
      args_synthesized->Append(ValueObjectRecognizerSynthesizedValue::Create(
          *o, eValueTypeVariableArgument));
  }

  bool hidden = m_interface_sp->ShouldHide(frame);
  lldb::StackFrameSP most_relevant =
      m_interface_sp->SelectMostRelevantFrame(frame);
  lldb::ValueObjectSP exception = m_interface_sp->GetException(frame);
  std::string stop_desc = m_interface_sp->GetStopDescription(frame);

  return RecognizedStackFrameSP(new ScriptedRecognizedStackFrame(
      args_synthesized, hidden, std::move(most_relevant), std::move(exception),
      std::move(stop_desc)));
}

void StackFrameRecognizerManager::BumpGeneration() {
  uint32_t n = m_generation;
  n = (n + 1) & ((1 << 16) - 1);
  m_generation = n;
}

void StackFrameRecognizerManager::AddRecognizer(
    StackFrameRecognizerSP recognizer, ConstString module,
    llvm::ArrayRef<ConstString> symbols,
    Mangled::NamePreference symbol_mangling, bool first_instruction_only) {
  m_recognizers.push_front({(uint32_t)m_recognizers.size(), recognizer, false,
                            module, RegularExpressionSP(), symbols,
                            RegularExpressionSP(), symbol_mangling,
                            first_instruction_only, true});
  BumpGeneration();
}

void StackFrameRecognizerManager::AddRecognizer(
    StackFrameRecognizerSP recognizer, RegularExpressionSP module,
    RegularExpressionSP symbol, Mangled::NamePreference symbol_mangling,
    bool first_instruction_only) {
  m_recognizers.push_front({(uint32_t)m_recognizers.size(), recognizer, true,
                            ConstString(), module, std::vector<ConstString>(),
                            symbol, symbol_mangling, first_instruction_only,
                            true});
  BumpGeneration();
}

void StackFrameRecognizerManager::ForEach(
    const std::function<void(
        uint32_t, bool, std::string, std::string, llvm::ArrayRef<ConstString>,
        Mangled::NamePreference name_preference, bool)> &callback) {
  for (auto entry : m_recognizers) {
    if (entry.is_regexp) {
      std::string module_name;
      std::string symbol_name;

      if (entry.module_regexp)
        module_name = entry.module_regexp->GetText().str();
      if (entry.symbol_regexp)
        symbol_name = entry.symbol_regexp->GetText().str();

      callback(entry.recognizer_id, entry.enabled, entry.recognizer->GetName(),
               module_name, llvm::ArrayRef(ConstString(symbol_name)),
               entry.symbol_mangling, true);
    } else {
      callback(entry.recognizer_id, entry.enabled, entry.recognizer->GetName(),
               entry.module.GetCString(), entry.symbols, entry.symbol_mangling,
               false);
    }
  }
}

bool StackFrameRecognizerManager::SetEnabledForID(uint32_t recognizer_id,
                                                  bool enabled) {
  auto found =
      llvm::find_if(m_recognizers, [recognizer_id](const RegisteredEntry &e) {
        return e.recognizer_id == recognizer_id;
      });
  if (found == m_recognizers.end())
    return false;
  found->enabled = enabled;
  BumpGeneration();
  return true;
}

bool StackFrameRecognizerManager::RemoveRecognizerWithID(
    uint32_t recognizer_id) {
  auto found =
      llvm::find_if(m_recognizers, [recognizer_id](const RegisteredEntry &e) {
        return e.recognizer_id == recognizer_id;
      });
  if (found == m_recognizers.end())
    return false;
  m_recognizers.erase(found);
  BumpGeneration();
  return true;
}

void StackFrameRecognizerManager::RemoveAllRecognizers() {
  BumpGeneration();
  m_recognizers.clear();
}

StackFrameRecognizerSP
StackFrameRecognizerManager::GetRecognizerForFrame(StackFrameSP frame) {
  const SymbolContext &symctx = frame->GetSymbolContext(
      eSymbolContextModule | eSymbolContextFunction | eSymbolContextSymbol);
  ModuleSP module_sp = symctx.module_sp;
  if (!module_sp)
    return StackFrameRecognizerSP();
  llvm::StringRef module_name = module_sp->GetFileSpec().GetFilename();
  const Symbol *symbol = symctx.symbol;
  if (!symbol)
    return StackFrameRecognizerSP();
  Address start_addr = symbol->GetAddress();
  Address current_addr = frame->GetFrameCodeAddress();

  for (const auto &entry : m_recognizers) {
    if (!entry.enabled)
      continue;

    if (entry.module)
      if (entry.module != module_name)
        continue;

    if (entry.module_regexp)
      if (!entry.module_regexp->Execute(module_name))
        continue;

    ConstString function_name = symctx.GetFunctionName(entry.symbol_mangling);

    if (!entry.symbols.empty())
      if (!llvm::is_contained(entry.symbols, function_name))
        continue;

    if (entry.symbol_regexp)
      if (!entry.symbol_regexp->Execute(function_name.GetStringRef()))
        continue;

    if (entry.first_instruction_only)
      if (start_addr != current_addr)
        continue;

    return entry.recognizer;
  }
  return StackFrameRecognizerSP();
}

RecognizedStackFrameSP
StackFrameRecognizerManager::RecognizeFrame(StackFrameSP frame) {
  auto recognizer = GetRecognizerForFrame(frame);
  if (!recognizer)
    return RecognizedStackFrameSP();
  return recognizer->RecognizeFrame(frame);
}
