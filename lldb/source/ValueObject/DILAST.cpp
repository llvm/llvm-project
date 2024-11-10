//===-- DILAST.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBType.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/ValueObject/DILAST.h"
#include "lldb/ValueObject/ValueObjectRegister.h"
#include "lldb/ValueObject/ValueObjectVariable.h"
#include "llvm/ADT/StringRef.h"

#include <vector>

namespace lldb_private {

lldb::ValueObjectSP DILGetSPWithLock(lldb::ValueObjectSP in_valobj_sp,
                                     lldb::DynamicValueType use_dynamic,
                                     bool use_synthetic) {
  Process::StopLocker stop_locker;
  std::unique_lock<std::recursive_mutex> lock;
  Status error;

  if (!in_valobj_sp) {
    error= Status("invalid value object");
    return in_valobj_sp;
  }

  lldb::ValueObjectSP value_sp = in_valobj_sp;

  Target *target = value_sp->GetTargetSP().get();
  // If this ValueObject holds an error, then it is valuable for that.
  if (value_sp->GetError().Fail())
    return value_sp;

  if (!target)
    return lldb::ValueObjectSP();

  lock = std::unique_lock<std::recursive_mutex>(target->GetAPIMutex());

  lldb::ProcessSP process_sp(value_sp->GetProcessSP());
  if (process_sp && !stop_locker.TryLock(&process_sp->GetRunLock())) {
    // We don't allow people to play around with ValueObject if the process
    // is running. If you want to look at values, pause the process, then
    // look.
    error = Status("process must be stopped.");
    return lldb::ValueObjectSP();
  }

  if (use_dynamic != lldb::eNoDynamicValues) {
    lldb::ValueObjectSP dynamic_sp = value_sp->GetDynamicValue(use_dynamic);
    if (dynamic_sp)
      value_sp = dynamic_sp;
  }

  if (use_synthetic) {
    lldb::ValueObjectSP synthetic_sp = value_sp->GetSyntheticValue();
    if (synthetic_sp)
      value_sp = synthetic_sp;
  }

  if (!value_sp)
    error= Status("invalid value object");

  return value_sp;
}

BinaryOpKind
clang_token_kind_to_binary_op_kind(clang::tok::TokenKind token_kind) {
  switch (token_kind) {
  case clang::tok::star:
    return BinaryOpKind::Mul;
  case clang::tok::slash:
    return BinaryOpKind::Div;
  case clang::tok::percent:
    return BinaryOpKind::Rem;
  case clang::tok::plus:
    return BinaryOpKind::Add;
  case clang::tok::minus:
    return BinaryOpKind::Sub;
  case clang::tok::lessless:
    return BinaryOpKind::Shl;
  case clang::tok::greatergreater:
    return BinaryOpKind::Shr;
  case clang::tok::less:
    return BinaryOpKind::LT;
  case clang::tok::greater:
    return BinaryOpKind::GT;
  case clang::tok::lessequal:
    return BinaryOpKind::LE;
  case clang::tok::greaterequal:
    return BinaryOpKind::GE;
  case clang::tok::equalequal:
    return BinaryOpKind::EQ;
  case clang::tok::exclaimequal:
    return BinaryOpKind::NE;
  case clang::tok::amp:
    return BinaryOpKind::And;
  case clang::tok::caret:
    return BinaryOpKind::Xor;
  case clang::tok::pipe:
    return BinaryOpKind::Or;
  case clang::tok::ampamp:
    return BinaryOpKind::LAnd;
  case clang::tok::pipepipe:
    return BinaryOpKind::LOr;
  case clang::tok::equal:
    return BinaryOpKind::Assign;
  case clang::tok::starequal:
    return BinaryOpKind::MulAssign;
  case clang::tok::slashequal:
    return BinaryOpKind::DivAssign;
  case clang::tok::percentequal:
    return BinaryOpKind::RemAssign;
  case clang::tok::plusequal:
    return BinaryOpKind::AddAssign;
  case clang::tok::minusequal:
    return BinaryOpKind::SubAssign;
  case clang::tok::lesslessequal:
    return BinaryOpKind::ShlAssign;
  case clang::tok::greatergreaterequal:
    return BinaryOpKind::ShrAssign;
  case clang::tok::ampequal:
    return BinaryOpKind::AndAssign;
  case clang::tok::caretequal:
    return BinaryOpKind::XorAssign;
  case clang::tok::pipeequal:
    return BinaryOpKind::OrAssign;

  default:
    break;
  }
  llvm_unreachable("did you add an element to BinaryOpKind?");
}

bool binary_op_kind_is_comp_assign(BinaryOpKind kind) {
  switch (kind) {
  case BinaryOpKind::Assign:
  case BinaryOpKind::MulAssign:
  case BinaryOpKind::DivAssign:
  case BinaryOpKind::RemAssign:
  case BinaryOpKind::AddAssign:
  case BinaryOpKind::SubAssign:
  case BinaryOpKind::ShlAssign:
  case BinaryOpKind::ShrAssign:
  case BinaryOpKind::AndAssign:
  case BinaryOpKind::XorAssign:
  case BinaryOpKind::OrAssign:
    return true;

  default:
    return false;
  }
}

CompilerType DILASTNode::result_type_deref() const {
  auto type = result_type();
  return type.IsReferenceType() ? type.GetNonReferenceType() : type;
}

static std::unordered_map<std::string, CompilerType> context_args;

bool IsContextVar(const std::string &name) {
  return context_args.find(name) != context_args.end();
}

static lldb::ValueObjectSP
LookupStaticIdentifier(lldb::TargetSP target_sp,
                       const llvm::StringRef &name_ref,
                       ConstString unqualified_name) {
  // List global variable with the same "basename". There can be many matches
  // from other scopes (namespaces, classes), so we do additional filtering
  // later.
  std::vector<lldb::ValueObjectSP> values;
  VariableList variable_list;
  ConstString name(name_ref);
  target_sp->GetImages().FindGlobalVariables(
      name, (size_t)std::numeric_limits<uint32_t>::max, variable_list);
  if (!variable_list.Empty()) {
    ExecutionContextScope *exe_scope = target_sp->GetProcessSP().get();
    if (exe_scope == nullptr)
      exe_scope = target_sp.get();
    for (const lldb::VariableSP &var_sp : variable_list) {
      lldb::ValueObjectSP valobj_sp(
          ValueObjectVariable::Create(exe_scope, var_sp));
      if (valobj_sp)
        values.push_back(valobj_sp);
    }
  }

  // Find the corrent variable by matching the name. lldb::SBValue::GetName()
  // can return strings like "::globarVar", "ns::i" or "int const ns::foo"
  // depending on the version and the platform.
  for (uint32_t i = 0; i < values.size(); ++i) {
    lldb::ValueObjectSP val = values[i];
    llvm::StringRef val_name_sstr = val->GetName().GetStringRef();
    llvm::StringRef name_sstr = name.GetStringRef();

    if (val->GetVariable() && val->GetVariable()->NameMatches(unqualified_name))
      return val;

    if (val_name_sstr == name_sstr ||
        val_name_sstr == llvm::formatv("::{0}", name_sstr).str() ||
        val_name_sstr.ends_with(llvm::formatv(" {0}", name_sstr).str()) ||
        val_name_sstr.ends_with(llvm::formatv("*{0}", name_sstr).str()) ||
        val_name_sstr.ends_with(llvm::formatv("&{0}", name_sstr).str()))
      return val;
  }
  lldb::ValueObjectSP empty_obj_sp;
  return empty_obj_sp;
}

struct EnumMember {
  CompilerType type;
  ConstString name;
  llvm::APSInt value;
};

static std::vector<EnumMember> GetEnumMembers(CompilerType type) {
  std::vector<EnumMember> enum_member_list;
  if (type.IsValid()) {
    type.ForEachEnumerator(
        [&enum_member_list](const CompilerType &integer_type, ConstString name,
                            const llvm::APSInt &value) -> bool {
          EnumMember enum_member = {integer_type, name, value};
          enum_member_list.push_back(enum_member);
          return true; // Keep iterating
        });
  }
  return enum_member_list;
}

CompilerType
ResolveTypeByName(const std::string &name,
                  std::shared_ptr<ExecutionContextScope> ctx_scope) {
  // Internally types don't have global scope qualifier in their names and
  // LLDB doesn't support queries with it too.
  llvm::StringRef name_ref(name);
  bool global_scope = false;

  if (name_ref.starts_with("::")) {
    name_ref = name_ref.drop_front(2);
    global_scope = true;
  }

  std::vector<CompilerType> result_type_list;
  lldb::TargetSP target_sp = ctx_scope->CalculateTarget();
  const char *type_name = name_ref.data();
  if (type_name && type_name[0] && target_sp) {
    ModuleList &images = target_sp->GetImages();
    ConstString const_type_name(type_name);
    TypeQuery query(type_name);
    TypeResults results;
    images.FindTypes(nullptr, query, results);
    for (const lldb::TypeSP &type_sp : results.GetTypeMap().Types())
      if (type_sp)
        result_type_list.push_back(type_sp->GetFullCompilerType());

    if (auto process_sp = target_sp->GetProcessSP()) {
      for (auto *runtime : process_sp->GetLanguageRuntimes()) {
        if (auto *vendor = runtime->GetDeclVendor()) {
          auto types = vendor->FindTypes(const_type_name, UINT32_MAX);
          for (auto type : types)
            result_type_list.push_back(type);
        }
      }
    }

    if (result_type_list.size() == 0) {
      for (auto type_system_sp : target_sp->GetScratchTypeSystems())
        if (auto compiler_type =
                type_system_sp->GetBuiltinTypeByName(const_type_name))
          result_type_list.push_back(compiler_type);
    }
  }

  // We've found multiple types, try finding the "correct" one.
  CompilerType full_match;
  std::vector<CompilerType> partial_matches;

  for (uint32_t i = 0; i < result_type_list.size(); ++i) {
    CompilerType type = result_type_list[i];
    llvm::StringRef type_name_ref = type.GetTypeName().GetStringRef();
    ;

    if (type_name_ref == name_ref)
      full_match = type;
    else if (type_name_ref.ends_with(name_ref))
      partial_matches.push_back(type);
  }

  if (global_scope) {
    // Look only for full matches when looking for a globally qualified type.
    if (full_match.IsValid())
      return full_match;
  } else {
    // We're looking for type, but there may be multiple candidates and which
    // one is correct may depend on the currect scope. For now just pick the
    // most "probable" type.

    // Full match is always correct if we're currently in the global scope.
    if (full_match.IsValid())
      return full_match;

    // If we have partial matches, pick a "random" one.
    if (partial_matches.size() > 0)
      return partial_matches.back();
  }

  CompilerType empty_type;
  return empty_type;
}

static lldb::VariableSP DILFindVariable(ConstString name,
                                        VariableList *variable_list) {
  lldb::VariableSP exact_match;
  std::vector<lldb::VariableSP> possible_matches;

  typedef std::vector<lldb::VariableSP> collection;
  typedef collection::iterator iterator;

  iterator pos, end = variable_list->end();
  for (pos = variable_list->begin(); pos != end; ++pos) {
    llvm::StringRef str_ref_name = pos->get()->GetName().GetStringRef();
    // Check for global vars, which might start with '::'.
    if (str_ref_name.size() > 2 && str_ref_name[0] == ':' &&
        str_ref_name[1] == ':')
      str_ref_name = str_ref_name.drop_front(2);

    ConstString tmp_name(str_ref_name);
    if (tmp_name == name)
      possible_matches.push_back(*pos);
    else if (pos->get()->NameMatches(name))
      possible_matches.push_back(*pos);
  }

  // Look for exact matches (favors local vars over global vars)
  for (auto var_sp : possible_matches)
    if (var_sp->GetName() == name) {
      exact_match = var_sp;
      break;
    }

  if (!exact_match)
    // Look for a global var exact match.
    for (auto var_sp : possible_matches) {
      llvm::StringRef str_ref_name = var_sp->GetName().GetStringRef();
      if (str_ref_name.size() > 2 && str_ref_name[0] == ':' &&
          str_ref_name[1] == ':')
        str_ref_name = str_ref_name.drop_front(2);
      ConstString tmp_name(str_ref_name);
      if (tmp_name == name) {
        exact_match = var_sp;
        break;
      }
    }

  // Take any match at this point.
  if (!exact_match && possible_matches.size() > 0)
    exact_match = possible_matches[0];

  return exact_match;
}

std::unique_ptr<IdentifierInfo>
LookupIdentifier(const std::string &name,
                 std::shared_ptr<ExecutionContextScope> ctx_scope,
                 lldb::DynamicValueType use_dynamic, CompilerType *scope_ptr) {
  auto context_arg = context_args.find(name);
  if (context_arg != context_args.end())
    return IdentifierInfo::FromContextArg(context_arg->second);

  ConstString name_str(name);
  llvm::StringRef name_ref = name_str.GetStringRef();

  // Support $rax as a special syntax for accessing registers.
  // Will return an invalid value in case the requested register doesn't exist.
  if (name_ref.starts_with("$")) {
    lldb::ValueObjectSP value_sp;
    const char *reg_name = name_ref.drop_front(1).data();
    Target *target = ctx_scope->CalculateTarget().get();
    Process *process = ctx_scope->CalculateProcess().get();
    if (target && process) {
      Process::StopLocker stop_locker;
      if (stop_locker.TryLock(&process->GetRunLock())) {
        StackFrame *stack_frame = ctx_scope->CalculateStackFrame().get();
        if (stack_frame) {
          lldb::RegisterContextSP reg_ctx(stack_frame->GetRegisterContext());
          if (reg_ctx) {
            if (const RegisterInfo *reg_info =
                    reg_ctx->GetRegisterInfoByName(reg_name))
              value_sp =
                  ValueObjectRegister::Create(stack_frame, reg_ctx, reg_info);
          }
        }
      }
    }
    return IdentifierInfo::FromValue(value_sp);
  }

  // Internally values don't have global scope qualifier in their names and
  // LLDB doesn't support queries with it too.
  bool global_scope = false;
  if (name_ref.starts_with("::")) {
    name_ref = name_ref.drop_front(2);
    global_scope = true;
  }

  // If the identifier doesn't refer to the global scope and doesn't have any
  // other scope qualifiers, try looking among the local and instance variables.
  if (!global_scope && !name_ref.contains("::")) {
    if (!scope_ptr || !scope_ptr->IsValid()) {
      // Lookup in the current frame.
      lldb::StackFrameSP frame = ctx_scope->CalculateStackFrame();
      // Try looking for a local variable in current scope.
      lldb::ValueObjectSP value_sp;
      lldb::VariableListSP var_list_sp(frame->GetInScopeVariableList(true));
      VariableList *variable_list = var_list_sp.get();
      if (variable_list) {
        lldb::VariableSP var_sp =
            DILFindVariable(ConstString(name_ref), variable_list);
        if (var_sp)
          value_sp = frame->GetValueObjectForFrameVariable(var_sp, use_dynamic);
      }
      if (!value_sp)
        value_sp = frame->FindVariable(ConstString(name_ref));

      bool use_synthetic = false;
      lldb::ValueObjectSP value(
          DILGetSPWithLock(value_sp, use_dynamic, use_synthetic));
      if (value)
        // Force static value, otherwise we can end up with the "real" type.
        return IdentifierInfo::FromValue(value);

      // Try looking for an instance variable (class member).
      ConstString this_string("this");
      value = frame->FindVariable(this_string);
      if (value)
        value = value->GetChildMemberWithName(name_ref.data());

      if (value)
        // Force static value, otherwise we can end up with the "real" type.
        return IdentifierInfo::FromValue(value->GetStaticValue());

    } else {
      // In a "value" scope `this` refers to the scope object itself.
      if (name_ref == "this")
        return IdentifierInfo::FromThisKeyword(scope_ptr->GetPointerType());

      // Lookup the variable as a member of the current scope value.
      lldb::ValueObjectSP empty_sp;
      bool use_synthetic = false;
      auto [member, path] =
          GetMemberInfo(empty_sp, *scope_ptr, name_ref.data(), use_synthetic);
      if (member)
        return IdentifierInfo::FromMemberPath(member.type, std::move(path));
    }
  }

  // Try looking for a global or static variable.

  lldb::ValueObjectSP value;
  if (!global_scope) {
    // Try looking for static member of the current scope value, e.g.
    // `ScopeType::NAME`. NAME can include nested struct (`Nested::SUBNAME`),
    // but it cannot be part of the global scope (start with "::").
    const char *type_name = "";
    if (scope_ptr)
      type_name = scope_ptr->GetCanonicalType().GetTypeName().AsCString();
    std::string name_with_type_prefix =
        llvm::formatv("{0}::{1}", type_name, name_ref).str();
    value = LookupStaticIdentifier(ctx_scope->CalculateTarget(),
                                   name_with_type_prefix, name_str);
  }

  // Lookup a regular global variable.
  if (!value)
    value = LookupStaticIdentifier(ctx_scope->CalculateTarget(), name_ref,
                                   name_str);

  // Try looking up enum value.
  if (!value && name_ref.contains("::")) {
    auto [enum_typename, enumerator_name] = name_ref.rsplit("::");

    auto type = ResolveTypeByName(enum_typename.str(), ctx_scope);
    std::vector<EnumMember> enum_members = GetEnumMembers(type);

    for (size_t i = 0; i < enum_members.size(); i++) {
      EnumMember enum_member = enum_members[i];
      if (enum_member.name == enumerator_name) {
        uint64_t bytes = enum_member.value.getZExtValue();
        uint64_t byte_size = 0;
        if (auto temp = type.GetByteSize(ctx_scope.get()))
          byte_size = temp.value();
        lldb::TargetSP target_sp = ctx_scope->CalculateTarget();
        lldb::DataExtractorSP data_sp = std::make_shared<DataExtractor>(
            &bytes, byte_size, target_sp->GetArchitecture().GetByteOrder(),
            static_cast<uint8_t>(
                target_sp->GetArchitecture().GetAddressByteSize()));
        ExecutionContext exe_ctx(
            ExecutionContextRef(ExecutionContext(target_sp.get(), false)));
        value = ValueObject::CreateValueObjectFromData("result", *data_sp,
                                                       exe_ctx, type);
        break;
      }
    }
  }

  // Last resort, lookup as a register (e.g. `rax` or `rip`).
  if (!value) {
    Target *target = ctx_scope->CalculateTarget().get();
    Process *process = ctx_scope->CalculateProcess().get();
    if (target && process) {
      Process::StopLocker stop_locker;
      if (stop_locker.TryLock(&process->GetRunLock())) {
        StackFrame *stack_frame = ctx_scope->CalculateStackFrame().get();
        if (stack_frame) {
          lldb::RegisterContextSP reg_ctx(stack_frame->GetRegisterContext());
          if (reg_ctx) {
            if (const RegisterInfo *reg_info =
                    reg_ctx->GetRegisterInfoByName(name_ref.data()))
              value =
                  ValueObjectRegister::Create(stack_frame, reg_ctx, reg_info);
          }
        }
      }
    }
  }

  // Force static value, otherwise we can end up with the "real" type.
  return IdentifierInfo::FromValue(value);
}

void DILErrorNode::Accept(DILVisitor *v) const { v->Visit(this); }

void LiteralNode::Accept(DILVisitor *v) const { v->Visit(this); }

void IdentifierNode::Accept(DILVisitor *v) const { v->Visit(this); }

void SizeOfNode::Accept(DILVisitor *v) const { v->Visit(this); }

void BuiltinFunctionCallNode::Accept(DILVisitor *v) const { v->Visit(this); }

void CStyleCastNode::Accept(DILVisitor *v) const { v->Visit(this); }

void CxxStaticCastNode::Accept(DILVisitor *v) const { return v->Visit(this); }

void CxxReinterpretCastNode::Accept(DILVisitor *v) const { v->Visit(this); }

void MemberOfNode::Accept(DILVisitor *v) const { v->Visit(this); }

void ArraySubscriptNode::Accept(DILVisitor *v) const { v->Visit(this); }

void BinaryOpNode::Accept(DILVisitor *v) const { v->Visit(this); }

void UnaryOpNode::Accept(DILVisitor *v) const { v->Visit(this); }

void TernaryOpNode::Accept(DILVisitor *v) const { v->Visit(this); }

void SmartPtrToPtrDecay::Accept(DILVisitor *v) const { v->Visit(this); }

} // namespace lldb_private
