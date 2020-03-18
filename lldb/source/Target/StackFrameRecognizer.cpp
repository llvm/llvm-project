//===-- StackFrameRecognizer.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <vector>
#include "lldb/Core/Module.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/StackFrameRecognizer.h"
#include "lldb/Utility/RegularExpression.h"

using namespace lldb;
using namespace lldb_private;

class ScriptedRecognizedStackFrame : public RecognizedStackFrame {
public:
  ScriptedRecognizedStackFrame(ValueObjectListSP args) {
    m_arguments = args;
  }
};

ScriptedStackFrameRecognizer::ScriptedStackFrameRecognizer(
    ScriptInterpreter *interpreter, const char *pclass)
    : m_interpreter(interpreter), m_python_class(pclass) {
  m_python_object_sp =
      m_interpreter->CreateFrameRecognizer(m_python_class.c_str());
}

RecognizedStackFrameSP
ScriptedStackFrameRecognizer::RecognizeFrame(lldb::StackFrameSP frame) {
  if (!m_python_object_sp || !m_interpreter)
    return RecognizedStackFrameSP();

  ValueObjectListSP args =
      m_interpreter->GetRecognizedArguments(m_python_object_sp, frame);
  auto args_synthesized = ValueObjectListSP(new ValueObjectList());
  for (const auto &o : args->GetObjects()) {
    args_synthesized->Append(ValueObjectRecognizerSynthesizedValue::Create(
        *o, eValueTypeVariableArgument));
  }

  return RecognizedStackFrameSP(
      new ScriptedRecognizedStackFrame(args_synthesized));
}

class StackFrameRecognizerManagerImpl {
public:
  void AddRecognizer(StackFrameRecognizerSP recognizer, ConstString module,
                     llvm::ArrayRef<ConstString> symbols,
                     bool first_instruction_only) {
    m_recognizers.push_front({(uint32_t)m_recognizers.size(), false, recognizer,
                              false, module, RegularExpressionSP(), symbols,
                              RegularExpressionSP(), first_instruction_only});
  }

  void AddRecognizer(StackFrameRecognizerSP recognizer,
                     RegularExpressionSP module, RegularExpressionSP symbol,
                     bool first_instruction_only) {
    m_recognizers.push_front(
        {(uint32_t)m_recognizers.size(), false, recognizer, true, ConstString(),
         module, std::vector<ConstString>(), symbol, first_instruction_only});
  }

  void ForEach(std::function<
               void(uint32_t recognized_id, std::string recognizer_name,
                    std::string module, llvm::ArrayRef<ConstString> symbols,
                    bool regexp)> const &callback) {
    for (auto entry : m_recognizers) {
      if (entry.is_regexp) {
        std::string module_name;
        std::string symbol_name;

        if (entry.module_regexp)
          module_name = entry.module_regexp->GetText().str();
        if (entry.symbol_regexp)
          symbol_name = entry.symbol_regexp->GetText().str();

        callback(entry.recognizer_id, entry.recognizer->GetName(), module_name,
                 llvm::makeArrayRef(ConstString(symbol_name)), true);

      } else {
        callback(entry.recognizer_id, entry.recognizer->GetName(),
                 entry.module.GetCString(), entry.symbols, false);
      }
    }
  }

  bool RemoveRecognizerWithID(uint32_t recognizer_id) {
    if (recognizer_id >= m_recognizers.size()) return false;
    if (m_recognizers[recognizer_id].deleted) return false;
    m_recognizers[recognizer_id].deleted = true;
    return true;
  }

  void RemoveAllRecognizers() {
    m_recognizers.clear();
  }

  StackFrameRecognizerSP GetRecognizerForFrame(StackFrameSP frame) {
    const SymbolContext &symctx = frame->GetSymbolContext(
        eSymbolContextModule | eSymbolContextFunction | eSymbolContextSymbol);
    ConstString function_name = symctx.GetFunctionName();
    ModuleSP module_sp = symctx.module_sp;
    if (!module_sp) return StackFrameRecognizerSP();
    ConstString module_name = module_sp->GetFileSpec().GetFilename();
    Symbol *symbol = symctx.symbol;
    if (!symbol) return StackFrameRecognizerSP();
    Address start_addr = symbol->GetAddress();
    Address current_addr = frame->GetFrameCodeAddress();

    for (auto entry : m_recognizers) {
      if (entry.deleted) continue;
      if (entry.module)
        if (entry.module != module_name) continue;

      if (entry.module_regexp)
        if (!entry.module_regexp->Execute(module_name.GetStringRef())) continue;

      if (!entry.symbols.empty())
        if (!llvm::is_contained(entry.symbols, function_name))
          continue;

      if (entry.symbol_regexp)
        if (!entry.symbol_regexp->Execute(function_name.GetStringRef()))
          continue;

      if (entry.first_instruction_only)
        if (start_addr != current_addr) continue;

      return entry.recognizer;
    }
    return StackFrameRecognizerSP();
  }

  RecognizedStackFrameSP RecognizeFrame(StackFrameSP frame) {
    auto recognizer = GetRecognizerForFrame(frame);
    if (!recognizer) return RecognizedStackFrameSP();
    return recognizer->RecognizeFrame(frame);
  }

 private:
  struct RegisteredEntry {
    uint32_t recognizer_id;
    bool deleted;
    StackFrameRecognizerSP recognizer;
    bool is_regexp;
    ConstString module;
    RegularExpressionSP module_regexp;
    std::vector<ConstString> symbols;
    RegularExpressionSP symbol_regexp;
    bool first_instruction_only;
  };

  std::deque<RegisteredEntry> m_recognizers;
};

StackFrameRecognizerManagerImpl &GetStackFrameRecognizerManagerImpl() {
  static StackFrameRecognizerManagerImpl instance =
      StackFrameRecognizerManagerImpl();
  return instance;
}

void StackFrameRecognizerManager::AddRecognizer(
    StackFrameRecognizerSP recognizer, ConstString module,
    llvm::ArrayRef<ConstString> symbols, bool first_instruction_only) {
  GetStackFrameRecognizerManagerImpl().AddRecognizer(
      recognizer, module, symbols, first_instruction_only);
}

void StackFrameRecognizerManager::AddRecognizer(
    StackFrameRecognizerSP recognizer, RegularExpressionSP module,
    RegularExpressionSP symbol, bool first_instruction_only) {
  GetStackFrameRecognizerManagerImpl().AddRecognizer(recognizer, module, symbol,
                                                     first_instruction_only);
}

void StackFrameRecognizerManager::ForEach(
    std::function<void(uint32_t recognized_id, std::string recognizer_name,
                       std::string module, llvm::ArrayRef<ConstString> symbols,
                       bool regexp)> const &callback) {
  GetStackFrameRecognizerManagerImpl().ForEach(callback);
}

void StackFrameRecognizerManager::RemoveAllRecognizers() {
  GetStackFrameRecognizerManagerImpl().RemoveAllRecognizers();
}

bool StackFrameRecognizerManager::RemoveRecognizerWithID(uint32_t recognizer_id) {
  return GetStackFrameRecognizerManagerImpl().RemoveRecognizerWithID(recognizer_id);
}

RecognizedStackFrameSP StackFrameRecognizerManager::RecognizeFrame(
    StackFrameSP frame) {
  return GetStackFrameRecognizerManagerImpl().RecognizeFrame(frame);
}

StackFrameRecognizerSP StackFrameRecognizerManager::GetRecognizerForFrame(
    lldb::StackFrameSP frame) {
  return GetStackFrameRecognizerManagerImpl().GetRecognizerForFrame(frame);
}
