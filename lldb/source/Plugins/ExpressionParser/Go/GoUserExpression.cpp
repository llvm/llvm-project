//===-- GoUserExpression.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <stdio.h>
#if HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

// C++ Includes
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

// Other libraries and framework includes
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

// Project includes
#include "GoUserExpression.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Core/ValueObjectRegister.h"
#include "lldb/Expression/DiagnosticManager.h"
#include "lldb/Expression/ExpressionVariable.h"
#include "lldb/Symbol/GoASTContext.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/ThreadPlanCallUserExpression.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/DataEncoder.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/lldb-private.h"

#include "Plugins/ExpressionParser/Go/GoAST.h"
#include "Plugins/ExpressionParser/Go/GoParser.h"

using namespace lldb_private;
using namespace lldb;

class GoUserExpression::GoInterpreter {
public:
  GoInterpreter(ExecutionContext &exe_ctx, const char *expr)
      : m_exe_ctx(exe_ctx), m_frame(exe_ctx.GetFrameSP()), m_parser(expr) {
    if (m_frame) {
      const SymbolContext &ctx =
          m_frame->GetSymbolContext(eSymbolContextFunction);
      ConstString fname = ctx.GetFunctionName();
      if (fname.GetLength() > 0) {
        size_t dot = fname.GetStringRef().find('.');
        if (dot != llvm::StringRef::npos)
          m_package = llvm::StringRef(fname.AsCString(), dot);
      }
    }
  }

  void set_use_dynamic(DynamicValueType use_dynamic) {
    m_use_dynamic = use_dynamic;
  }

  bool Parse();
  lldb::ValueObjectSP Evaluate(ExecutionContext &exe_ctx);
  lldb::ValueObjectSP EvaluateStatement(const GoASTStmt *s);
  lldb::ValueObjectSP EvaluateExpr(const GoASTExpr *e);

  ValueObjectSP VisitBadExpr(const GoASTBadExpr *e) {
    m_parser.GetError(m_error);
    return nullptr;
  }

  ValueObjectSP VisitParenExpr(const GoASTParenExpr *e);
  ValueObjectSP VisitIdent(const GoASTIdent *e);
  ValueObjectSP VisitStarExpr(const GoASTStarExpr *e);
  ValueObjectSP VisitSelectorExpr(const GoASTSelectorExpr *e);
  ValueObjectSP VisitBasicLit(const GoASTBasicLit *e);
  ValueObjectSP VisitIndexExpr(const GoASTIndexExpr *e);
  ValueObjectSP VisitUnaryExpr(const GoASTUnaryExpr *e);
  ValueObjectSP VisitCallExpr(const GoASTCallExpr *e);

  ValueObjectSP VisitTypeAssertExpr(const GoASTTypeAssertExpr *e) {
    return NotImplemented(e);
  }

  ValueObjectSP VisitBinaryExpr(const GoASTBinaryExpr *e) {
    return NotImplemented(e);
  }

  ValueObjectSP VisitArrayType(const GoASTArrayType *e) {
    return NotImplemented(e);
  }

  ValueObjectSP VisitChanType(const GoASTChanType *e) {
    return NotImplemented(e);
  }

  ValueObjectSP VisitCompositeLit(const GoASTCompositeLit *e) {
    return NotImplemented(e);
  }

  ValueObjectSP VisitEllipsis(const GoASTEllipsis *e) {
    return NotImplemented(e);
  }

  ValueObjectSP VisitFuncType(const GoASTFuncType *e) {
    return NotImplemented(e);
  }

  ValueObjectSP VisitFuncLit(const GoASTFuncLit *e) {
    return NotImplemented(e);
  }

  ValueObjectSP VisitInterfaceType(const GoASTInterfaceType *e) {
    return NotImplemented(e);
  }

  ValueObjectSP VisitKeyValueExpr(const GoASTKeyValueExpr *e) {
    return NotImplemented(e);
  }

  ValueObjectSP VisitMapType(const GoASTMapType *e) {
    return NotImplemented(e);
  }

  ValueObjectSP VisitSliceExpr(const GoASTSliceExpr *e) {
    return NotImplemented(e);
  }

  ValueObjectSP VisitStructType(const GoASTStructType *e) {
    return NotImplemented(e);
  }

  CompilerType EvaluateType(const GoASTExpr *e);

  Status &error() { return m_error; }

private:
  std::nullptr_t NotImplemented(const GoASTExpr *e) {
    m_error.SetErrorStringWithFormat("%s node not implemented",
                                     e->GetKindName());
    return nullptr;
  }

  ExecutionContext m_exe_ctx;
  lldb::StackFrameSP m_frame;
  GoParser m_parser;
  DynamicValueType m_use_dynamic;
  Status m_error;
  llvm::StringRef m_package;
  std::vector<std::unique_ptr<GoASTStmt>> m_statements;
};

VariableSP FindGlobalVariable(TargetSP target, llvm::Twine name) {
  ConstString fullname(name.str());
  VariableList variable_list;
  if (!target) {
    return nullptr;
  }
  const uint32_t match_count =
      target->GetImages().FindGlobalVariables(fullname, 1, variable_list);
  if (match_count == 1) {
    return variable_list.GetVariableAtIndex(0);
  }
  return nullptr;
}

CompilerType LookupType(TargetSP target, ConstString name) {
  if (!target)
    return CompilerType();
  SymbolContext sc;
  TypeList type_list;
  llvm::DenseSet<SymbolFile *> searched_symbol_files;
  uint32_t num_matches = target->GetImages().FindTypes(
      sc, name, false, 2, searched_symbol_files, type_list);
  if (num_matches > 0) {
    return type_list.GetTypeAtIndex(0)->GetFullCompilerType();
  }
  return CompilerType();
}

GoUserExpression::GoUserExpression(ExecutionContextScope &exe_scope,
                                   llvm::StringRef expr, llvm::StringRef prefix,
                                   lldb::LanguageType language,
                                   ResultType desired_type,
                                   const EvaluateExpressionOptions &options)
    : UserExpression(exe_scope, expr, prefix, language, desired_type, options) {
}

bool GoUserExpression::Parse(DiagnosticManager &diagnostic_manager,
                             ExecutionContext &exe_ctx,
                             lldb_private::ExecutionPolicy execution_policy,
                             bool keep_result_in_memory,
                             bool generate_debug_info) {
  InstallContext(exe_ctx);
  m_interpreter.reset(new GoInterpreter(exe_ctx, GetUserText()));
  if (m_interpreter->Parse())
    return true;
  const char *error_cstr = m_interpreter->error().AsCString();
  if (error_cstr && error_cstr[0])
    diagnostic_manager.PutString(eDiagnosticSeverityError, error_cstr);
  else
    diagnostic_manager.Printf(eDiagnosticSeverityError,
                              "expression can't be interpreted or run");
  return false;
}

lldb::ExpressionResults
GoUserExpression::DoExecute(DiagnosticManager &diagnostic_manager,
                            ExecutionContext &exe_ctx,
                            const EvaluateExpressionOptions &options,
                            lldb::UserExpressionSP &shared_ptr_to_me,
                            lldb::ExpressionVariableSP &result) {
  Log *log(lldb_private::GetLogIfAnyCategoriesSet(LIBLLDB_LOG_EXPRESSIONS |
                                                  LIBLLDB_LOG_STEP));

  lldb_private::ExecutionPolicy execution_policy = options.GetExecutionPolicy();
  lldb::ExpressionResults execution_results = lldb::eExpressionSetupError;

  Process *process = exe_ctx.GetProcessPtr();
  Target *target = exe_ctx.GetTargetPtr();

  if (target == nullptr || process == nullptr ||
      process->GetState() != lldb::eStateStopped) {
    if (execution_policy == eExecutionPolicyAlways) {
      if (log)
        log->Printf("== [GoUserExpression::Evaluate] Expression may not run, "
                    "but is not constant ==");

      diagnostic_manager.PutString(eDiagnosticSeverityError,
                                   "expression needed to run but couldn't");

      return execution_results;
    }
  }

  m_interpreter->set_use_dynamic(options.GetUseDynamic());
  ValueObjectSP result_val_sp = m_interpreter->Evaluate(exe_ctx);
  Status err = m_interpreter->error();
  m_interpreter.reset();

  if (!result_val_sp) {
    const char *error_cstr = err.AsCString();
    if (error_cstr && error_cstr[0])
      diagnostic_manager.PutString(eDiagnosticSeverityError, error_cstr);
    else
      diagnostic_manager.PutString(eDiagnosticSeverityError,
                                   "expression can't be interpreted or run");
    return lldb::eExpressionDiscarded;
  }
  result.reset(new ExpressionVariable(ExpressionVariable::eKindGo));
  result->m_live_sp = result->m_frozen_sp = result_val_sp;
  result->m_flags |= ExpressionVariable::EVIsProgramReference;
  PersistentExpressionState *pv =
      target->GetPersistentExpressionStateForLanguage(eLanguageTypeGo);
  if (pv != nullptr) {
    result->SetName(pv->GetNextPersistentVariableName(
        *target, pv->GetPersistentVariablePrefix()));
    pv->AddVariable(result);
  }
  return lldb::eExpressionCompleted;
}

bool GoUserExpression::GoInterpreter::Parse() {
  for (std::unique_ptr<GoASTStmt> stmt(m_parser.Statement()); stmt;
       stmt.reset(m_parser.Statement())) {
    if (m_parser.Failed())
      break;
    m_statements.emplace_back(std::move(stmt));
  }
  if (m_parser.Failed() || !m_parser.AtEOF())
    m_parser.GetError(m_error);

  return m_error.Success();
}

ValueObjectSP
GoUserExpression::GoInterpreter::Evaluate(ExecutionContext &exe_ctx) {
  m_exe_ctx = exe_ctx;
  ValueObjectSP result;
  for (const std::unique_ptr<GoASTStmt> &stmt : m_statements) {
    result = EvaluateStatement(stmt.get());
    if (m_error.Fail())
      return nullptr;
  }
  return result;
}

ValueObjectSP GoUserExpression::GoInterpreter::EvaluateStatement(
    const lldb_private::GoASTStmt *stmt) {
  ValueObjectSP result;
  switch (stmt->GetKind()) {
  case GoASTNode::eBlockStmt: {
    const GoASTBlockStmt *block = llvm::cast<GoASTBlockStmt>(stmt);
    for (size_t i = 0; i < block->NumList(); ++i)
      result = EvaluateStatement(block->GetList(i));
    break;
  }
  case GoASTNode::eBadStmt:
    m_parser.GetError(m_error);
    break;
  case GoASTNode::eExprStmt: {
    const GoASTExprStmt *expr = llvm::cast<GoASTExprStmt>(stmt);
    return EvaluateExpr(expr->GetX());
  }
  default:
    m_error.SetErrorStringWithFormat("%s node not supported",
                                     stmt->GetKindName());
  }
  return result;
}

ValueObjectSP GoUserExpression::GoInterpreter::EvaluateExpr(
    const lldb_private::GoASTExpr *e) {
  if (e)
    return e->Visit<ValueObjectSP>(this);
  return ValueObjectSP();
}

ValueObjectSP GoUserExpression::GoInterpreter::VisitParenExpr(
    const lldb_private::GoASTParenExpr *e) {
  return EvaluateExpr(e->GetX());
}

ValueObjectSP GoUserExpression::GoInterpreter::VisitIdent(const GoASTIdent *e) {
  ValueObjectSP val;
  if (m_frame) {
    VariableSP var_sp;
    std::string varname = e->GetName().m_value.str();
    if (varname.size() > 1 && varname[0] == '$') {
      RegisterContextSP reg_ctx_sp = m_frame->GetRegisterContext();
      const RegisterInfo *reg =
          reg_ctx_sp->GetRegisterInfoByName(varname.c_str() + 1);
      if (reg) {
        std::string type;
        switch (reg->encoding) {
        case lldb::eEncodingSint:
          type.append("int");
          break;
        case lldb::eEncodingUint:
          type.append("uint");
          break;
        case lldb::eEncodingIEEE754:
          type.append("float");
          break;
        default:
          m_error.SetErrorString("Invalid register encoding");
          return nullptr;
        }
        switch (reg->byte_size) {
        case 8:
          type.append("64");
          break;
        case 4:
          type.append("32");
          break;
        case 2:
          type.append("16");
          break;
        case 1:
          type.append("8");
          break;
        default:
          m_error.SetErrorString("Invalid register size");
          return nullptr;
        }
        ValueObjectSP regVal = ValueObjectRegister::Create(
            m_frame.get(), reg_ctx_sp, reg->kinds[eRegisterKindLLDB]);
        CompilerType goType =
            LookupType(m_frame->CalculateTarget(), ConstString(type));
        if (regVal) {
          regVal = regVal->Cast(goType);
          return regVal;
        }
      }
      m_error.SetErrorString("Invalid register name");
      return nullptr;
    }
    VariableListSP var_list_sp(m_frame->GetInScopeVariableList(false));
    if (var_list_sp) {
      var_sp = var_list_sp->FindVariable(ConstString(varname));
      if (var_sp)
        val = m_frame->GetValueObjectForFrameVariable(var_sp, m_use_dynamic);
      else {
        // When a variable is on the heap instead of the stack, go records a
        // variable '&x' instead of 'x'.
        var_sp = var_list_sp->FindVariable(ConstString("&" + varname));
        if (var_sp) {
          val = m_frame->GetValueObjectForFrameVariable(var_sp, m_use_dynamic);
          if (val)
            val = val->Dereference(m_error);
          if (m_error.Fail())
            return nullptr;
        }
      }
    }
    if (!val) {
      m_error.Clear();
      TargetSP target = m_frame->CalculateTarget();
      if (!target) {
        m_error.SetErrorString("No target");
        return nullptr;
      }
      var_sp =
          FindGlobalVariable(target, m_package + "." + e->GetName().m_value);
      if (var_sp)
        return m_frame->TrackGlobalVariable(var_sp, m_use_dynamic);
    }
  }
  if (!val)
    m_error.SetErrorStringWithFormat("Unknown variable %s",
                                     e->GetName().m_value.str().c_str());
  return val;
}

ValueObjectSP
GoUserExpression::GoInterpreter::VisitStarExpr(const GoASTStarExpr *e) {
  ValueObjectSP target = EvaluateExpr(e->GetX());
  if (!target)
    return nullptr;
  return target->Dereference(m_error);
}

ValueObjectSP GoUserExpression::GoInterpreter::VisitSelectorExpr(
    const lldb_private::GoASTSelectorExpr *e) {
  ValueObjectSP target = EvaluateExpr(e->GetX());
  if (target) {
    if (target->GetCompilerType().IsPointerType()) {
      target = target->Dereference(m_error);
      if (m_error.Fail())
        return nullptr;
    }
    ConstString field(e->GetSel()->GetName().m_value);
    ValueObjectSP result = target->GetChildMemberWithName(field, true);
    if (!result)
      m_error.SetErrorStringWithFormat("Unknown child %s", field.AsCString());
    return result;
  }
  if (const GoASTIdent *package = llvm::dyn_cast<GoASTIdent>(e->GetX())) {
    if (VariableSP global = FindGlobalVariable(
            m_exe_ctx.GetTargetSP(), package->GetName().m_value + "." +
                                         e->GetSel()->GetName().m_value)) {
      if (m_frame) {
        m_error.Clear();
        return m_frame->GetValueObjectForFrameVariable(global, m_use_dynamic);
      }
    }
  }
  if (const GoASTBasicLit *packageLit =
          llvm::dyn_cast<GoASTBasicLit>(e->GetX())) {
    if (packageLit->GetValue().m_type == GoLexer::LIT_STRING) {
      std::string value = packageLit->GetValue().m_value.str();
      value = value.substr(1, value.size() - 2);
      if (VariableSP global = FindGlobalVariable(
              m_exe_ctx.GetTargetSP(),
              value + "." + e->GetSel()->GetName().m_value)) {
        if (m_frame) {
          m_error.Clear();
          return m_frame->TrackGlobalVariable(global, m_use_dynamic);
        }
      }
    }
  }
  // EvaluateExpr should have already set m_error.
  return target;
}

ValueObjectSP GoUserExpression::GoInterpreter::VisitBasicLit(
    const lldb_private::GoASTBasicLit *e) {
  std::string value = e->GetValue().m_value.str();
  if (e->GetValue().m_type != GoLexer::LIT_INTEGER) {
    m_error.SetErrorStringWithFormat("Unsupported literal %s", value.c_str());
    return nullptr;
  }
  errno = 0;
  int64_t intvalue = strtol(value.c_str(), nullptr, 0);
  if (errno != 0) {
    m_error.SetErrorToErrno();
    return nullptr;
  }
  DataBufferSP buf(new DataBufferHeap(sizeof(intvalue), 0));
  TargetSP target = m_exe_ctx.GetTargetSP();
  if (!target) {
    m_error.SetErrorString("No target");
    return nullptr;
  }
  ByteOrder order = target->GetArchitecture().GetByteOrder();
  uint8_t addr_size = target->GetArchitecture().GetAddressByteSize();
  DataEncoder enc(buf, order, addr_size);
  enc.PutU64(0, static_cast<uint64_t>(intvalue));
  DataExtractor data(buf, order, addr_size);

  CompilerType type = LookupType(target, ConstString("int64"));
  return ValueObject::CreateValueObjectFromData(llvm::StringRef(), data,
                                                m_exe_ctx, type);
}

ValueObjectSP GoUserExpression::GoInterpreter::VisitIndexExpr(
    const lldb_private::GoASTIndexExpr *e) {
  ValueObjectSP target = EvaluateExpr(e->GetX());
  if (!target)
    return nullptr;
  ValueObjectSP index = EvaluateExpr(e->GetIndex());
  if (!index)
    return nullptr;
  bool is_signed;
  if (!index->GetCompilerType().IsIntegerType(is_signed)) {
    m_error.SetErrorString("Unsupported index");
    return nullptr;
  }
  size_t idx;
  if (is_signed)
    idx = index->GetValueAsSigned(0);
  else
    idx = index->GetValueAsUnsigned(0);
  if (GoASTContext::IsGoSlice(target->GetCompilerType())) {
    target = target->GetStaticValue();
    ValueObjectSP cap =
        target->GetChildMemberWithName(ConstString("cap"), true);
    if (cap) {
      uint64_t capval = cap->GetValueAsUnsigned(0);
      if (idx >= capval) {
        m_error.SetErrorStringWithFormat("Invalid index %" PRIu64
                                         " , cap = %" PRIu64,
                                         uint64_t(idx), capval);
        return nullptr;
      }
    }
    target = target->GetChildMemberWithName(ConstString("array"), true);
    if (target && m_use_dynamic != eNoDynamicValues) {
      ValueObjectSP dynamic = target->GetDynamicValue(m_use_dynamic);
      if (dynamic)
        target = dynamic;
    }
    if (!target)
      return nullptr;
    return target->GetSyntheticArrayMember(idx, true);
  }
  return target->GetChildAtIndex(idx, true);
}

ValueObjectSP
GoUserExpression::GoInterpreter::VisitUnaryExpr(const GoASTUnaryExpr *e) {
  ValueObjectSP x = EvaluateExpr(e->GetX());
  if (!x)
    return nullptr;
  switch (e->GetOp()) {
  case GoLexer::OP_AMP: {
    CompilerType type = x->GetCompilerType().GetPointerType();
    uint64_t address = x->GetAddressOf();
    return ValueObject::CreateValueObjectFromAddress(llvm::StringRef(), address,
                                                     m_exe_ctx, type);
  }
  case GoLexer::OP_PLUS:
    return x;
  default:
    m_error.SetErrorStringWithFormat(
        "Operator %s not supported",
        GoLexer::LookupToken(e->GetOp()).str().c_str());
    return nullptr;
  }
}

CompilerType GoUserExpression::GoInterpreter::EvaluateType(const GoASTExpr *e) {
  TargetSP target = m_exe_ctx.GetTargetSP();
  if (auto *id = llvm::dyn_cast<GoASTIdent>(e)) {
    CompilerType result =
        LookupType(target, ConstString(id->GetName().m_value));
    if (result.IsValid())
      return result;
    std::string fullname = (m_package + "." + id->GetName().m_value).str();
    result = LookupType(target, ConstString(fullname));
    if (!result)
      m_error.SetErrorStringWithFormat("Unknown type %s", fullname.c_str());
    return result;
  }
  if (auto *sel = llvm::dyn_cast<GoASTSelectorExpr>(e)) {
    std::string package;
    if (auto *pkg_node = llvm::dyn_cast<GoASTIdent>(sel->GetX())) {
      package = pkg_node->GetName().m_value.str();
    } else if (auto *str_node = llvm::dyn_cast<GoASTBasicLit>(sel->GetX())) {
      if (str_node->GetValue().m_type == GoLexer::LIT_STRING) {
        package = str_node->GetValue().m_value.substr(1).str();
        package.resize(package.length() - 1);
      }
    }
    if (package.empty()) {
      m_error.SetErrorStringWithFormat("Invalid %s in type expression",
                                       sel->GetX()->GetKindName());
      return CompilerType();
    }
    std::string fullname =
        (package + "." + sel->GetSel()->GetName().m_value).str();
    CompilerType result = LookupType(target, ConstString(fullname));
    if (!result)
      m_error.SetErrorStringWithFormat("Unknown type %s", fullname.c_str());
    return result;
  }
  if (auto *star = llvm::dyn_cast<GoASTStarExpr>(e)) {
    CompilerType elem = EvaluateType(star->GetX());
    return elem.GetPointerType();
  }
  if (auto *paren = llvm::dyn_cast<GoASTParenExpr>(e))
    return EvaluateType(paren->GetX());
  if (auto *array = llvm::dyn_cast<GoASTArrayType>(e)) {
    CompilerType elem = EvaluateType(array->GetElt());
  }

  m_error.SetErrorStringWithFormat("Invalid %s in type expression",
                                   e->GetKindName());
  return CompilerType();
}

ValueObjectSP GoUserExpression::GoInterpreter::VisitCallExpr(
    const lldb_private::GoASTCallExpr *e) {
  ValueObjectSP x = EvaluateExpr(e->GetFun());
  if (x || e->NumArgs() != 1) {
    m_error.SetErrorStringWithFormat("Code execution not supported");
    return nullptr;
  }
  m_error.Clear();
  CompilerType type = EvaluateType(e->GetFun());
  if (!type) {
    return nullptr;
  }
  ValueObjectSP value = EvaluateExpr(e->GetArgs(0));
  if (!value)
    return nullptr;
  // TODO: Handle special conversions
  return value->Cast(type);
}

GoPersistentExpressionState::GoPersistentExpressionState()
    : PersistentExpressionState(eKindGo) {}

void GoPersistentExpressionState::RemovePersistentVariable(
    lldb::ExpressionVariableSP variable) {
  RemoveVariable(variable);

  const char *name = variable->GetName().AsCString();

  if (*(name++) != '$')
    return;
  if (*(name++) != 'g')
    return;
  if (*(name++) != 'o')
    return;

  if (strtoul(name, nullptr, 0) == m_next_persistent_variable_id - 1)
    m_next_persistent_variable_id--;
}
