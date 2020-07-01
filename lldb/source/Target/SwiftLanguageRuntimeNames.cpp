//===-- SwiftLanguageRuntimeDynamicTypeResolution.cpp ---------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "SwiftLanguageRuntimeImpl.h"
#include "lldb/Target/SwiftLanguageRuntime.h"

#include "swift/Demangling/Demangle.h"
#include "swift/Demangling/Demangler.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/ThreadPlanRunToAddress.h"
#include "lldb/Target/ThreadPlanStepInRange.h"
#include "lldb/Target/ThreadPlanStepOverRange.h"
#include "lldb/Utility/Log.h"

using namespace lldb;
using namespace lldb_private;
namespace lldb_private {

static const char *g_dollar_tau_underscore = u8"$\u03C4_";
static const char *g_tau_underscore = g_dollar_tau_underscore + 1;

static bool IsSymbolARuntimeThunk(const Symbol &symbol) {
  llvm::StringRef symbol_name =
      symbol.GetMangled().GetMangledName().GetStringRef();
  if (symbol_name.empty())
    return false;

  swift::Demangle::Context demangle_ctx;
  return demangle_ctx.isThunkSymbol(symbol_name);
}

namespace {

enum class ThunkKind {
  Unknown = 0,
  AllocatingInit,
  PartialApply,
  ObjCAttribute,
  Reabstraction,
  ProtocolConformance,
};

enum class ThunkAction {
  Unknown = 0,
  GetThunkTarget,
  StepIntoConformance,
  StepThrough
};

} // namespace

static ThunkKind GetThunkKind(llvm::StringRef symbol_name) {
  swift::Demangle::Node::Kind kind;
  swift::Demangle::Context demangle_ctx;
  if (!demangle_ctx.isThunkSymbol(symbol_name))
    return ThunkKind::Unknown;

  swift::Demangle::NodePointer nodes =
      demangle_ctx.demangleSymbolAsNode(symbol_name);
  size_t num_global_children = nodes->getNumChildren();
  if (num_global_children == 0)
    return ThunkKind::Unknown;

  if (nodes->getKind() != swift::Demangle::Node::Kind::Global)
    return ThunkKind::Unknown;
  if (nodes->getNumChildren() == 0)
    return ThunkKind::Unknown;

  swift::Demangle::NodePointer node_ptr = nodes->getFirstChild();
  kind = node_ptr->getKind();
  switch (kind) {
  case swift::Demangle::Node::Kind::ObjCAttribute:
    return ThunkKind::ObjCAttribute;
    break;
  case swift::Demangle::Node::Kind::ProtocolWitness:
    if (node_ptr->getNumChildren() == 0)
      return ThunkKind::Unknown;
    if (node_ptr->getFirstChild()->getKind() ==
        swift::Demangle::Node::Kind::ProtocolConformance)
      return ThunkKind::ProtocolConformance;
    break;
  case swift::Demangle::Node::Kind::ReabstractionThunkHelper:
    return ThunkKind::Reabstraction;
  case swift::Demangle::Node::Kind::PartialApplyForwarder:
    return ThunkKind::PartialApply;
  case swift::Demangle::Node::Kind::Allocator:
    if (node_ptr->getNumChildren() == 0)
      return ThunkKind::Unknown;
    if (node_ptr->getFirstChild()->getKind() ==
        swift::Demangle::Node::Kind::Class)
      return ThunkKind::AllocatingInit;
    break;
  default:
    break;
  }

  return ThunkKind::Unknown;
}
static const char *GetThunkKindName(ThunkKind kind) {
  switch (kind) {
  case ThunkKind::Unknown:
    return "Unknown";
  case ThunkKind::AllocatingInit:
    return "StepThrough";
  case ThunkKind::PartialApply:
    return "GetThunkTarget";
  case ThunkKind::ObjCAttribute:
    return "GetThunkTarget";
  case ThunkKind::Reabstraction:
    return "GetThunkTarget";
  case ThunkKind::ProtocolConformance:
    return "StepIntoConformance";
  }
}

static ThunkAction GetThunkAction(ThunkKind kind) {
  switch (kind) {
  case ThunkKind::Unknown:
    return ThunkAction::Unknown;
  case ThunkKind::AllocatingInit:
    return ThunkAction::StepThrough;
  case ThunkKind::PartialApply:
    return ThunkAction::GetThunkTarget;
  case ThunkKind::ObjCAttribute:
    return ThunkAction::GetThunkTarget;
  case ThunkKind::Reabstraction:
    return ThunkAction::StepThrough;
  case ThunkKind::ProtocolConformance:
    return ThunkAction::StepIntoConformance;
  }
}

static lldb::ThreadPlanSP GetStepThroughTrampolinePlan(Thread &thread,
                                                       bool stop_others) {
  // Here are the trampolines we have at present.
  // 1) The thunks from protocol invocations to the call in the actual object
  //    implementing the protocol.
  // 2) Thunks for going from Swift ObjC classes to their actual method
  //    invocations.
  // 3) Thunks that retain captured objects in closure invocations.

  ThreadPlanSP new_thread_plan_sp;

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_STEP));
  StackFrameSP stack_sp = thread.GetStackFrameAtIndex(0);
  if (!stack_sp)
    return new_thread_plan_sp;

  SymbolContext sc = stack_sp->GetSymbolContext(eSymbolContextEverything);
  Symbol *symbol = sc.symbol;

  // Note, I don't really need to consult IsSymbolARuntimeThunk here,
  // but it is fast to do and keeps this list and the one in
  // IsSymbolARuntimeThunk in sync.
  if (!symbol || !IsSymbolARuntimeThunk(*symbol))
    return new_thread_plan_sp;

  // Only do this if you are at the beginning of the thunk function:
  lldb::addr_t cur_addr = thread.GetRegisterContext()->GetPC();
  lldb::addr_t symbol_addr =
      symbol->GetAddress().GetLoadAddress(&thread.GetProcess()->GetTarget());

  if (symbol_addr != cur_addr)
    return new_thread_plan_sp;

  Address target_address;
  ConstString symbol_mangled_name = symbol->GetMangled().GetMangledName();
  const char *symbol_name = symbol_mangled_name.AsCString();

  ThunkKind thunk_kind = GetThunkKind(symbol_mangled_name.GetStringRef());
  ThunkAction thunk_action = GetThunkAction(thunk_kind);

  switch (thunk_action) {
  case ThunkAction::Unknown:
    return new_thread_plan_sp;
  case ThunkAction::GetThunkTarget: {
    swift::Demangle::Context demangle_ctx;
    std::string thunk_target = demangle_ctx.getThunkTarget(symbol_name);
    if (thunk_target.empty()) {
      if (log)
        log->Printf("Stepped to thunk \"%s\" (kind: %s) but could not "
                    "find the thunk target. ",
                    symbol_name, GetThunkKindName(thunk_kind));
      return new_thread_plan_sp;
    }
    if (log)
      log->Printf(
          "Stepped to thunk \"%s\" (kind: %s) stepping to target: \"%s\".",
          symbol_name, GetThunkKindName(thunk_kind), thunk_target.c_str());

    ModuleList modules = thread.GetProcess()->GetTarget().GetImages();
    SymbolContextList sc_list;
    modules.FindFunctionSymbols(ConstString(thunk_target),
                                eFunctionNameTypeFull, sc_list);
    if (sc_list.GetSize() == 1) {
      SymbolContext sc;
      sc_list.GetContextAtIndex(0, sc);

      if (sc.symbol)
        target_address = sc.symbol->GetAddress();
    }
  } break;
  case ThunkAction::StepIntoConformance: {
    // The TTW symbols encode the protocol conformance requirements
    // and it is possible to go to the AST and get it to replay the
    // logic that it used to determine what to dispatch to.  But that
    // ties us too closely to the logic of the compiler, and these
    // thunks are quite simple, they just do a little retaining, and
    // then call the correct function.
    // So for simplicity's sake, I'm just going to get the base name
    // of the function this protocol thunk is preparing to call, then
    // step into through the thunk, stopping if I end up in a frame
    // with that function name.
    swift::Demangle::Context demangle_ctx;
    swift::Demangle::NodePointer demangled_nodes =
        demangle_ctx.demangleSymbolAsNode(symbol_mangled_name.GetStringRef());

    // Now find the ProtocolWitness node in the demangled result.

    swift::Demangle::NodePointer witness_node = demangled_nodes;
    bool found_witness_node = false;
    while (witness_node) {
      if (witness_node->getKind() ==
          swift::Demangle::Node::Kind::ProtocolWitness) {
        found_witness_node = true;
        break;
      }
      witness_node = witness_node->getFirstChild();
    }
    if (!found_witness_node) {
      if (log)
        log->Printf("Stepped into witness thunk \"%s\" but could not "
                    "find the ProtocolWitness node in the demangled "
                    "nodes.",
                    symbol_name);
      return new_thread_plan_sp;
    }

    size_t num_children = witness_node->getNumChildren();
    if (num_children < 2) {
      if (log)
        log->Printf("Stepped into witness thunk \"%s\" but the "
                    "ProtocolWitness node doesn't have enough nodes.",
                    symbol_name);
      return new_thread_plan_sp;
    }

    swift::Demangle::NodePointer function_node = witness_node->getChild(1);
    if (function_node == nullptr ||
        function_node->getKind() != swift::Demangle::Node::Kind::Function) {
      if (log)
        log->Printf("Stepped into witness thunk \"%s\" but could not "
                    "find the function in the ProtocolWitness node.",
                    symbol_name);
      return new_thread_plan_sp;
    }

    // Okay, now find the name of this function.
    num_children = function_node->getNumChildren();
    swift::Demangle::NodePointer name_node(nullptr);
    for (size_t i = 0; i < num_children; i++) {
      if (function_node->getChild(i)->getKind() ==
          swift::Demangle::Node::Kind::Identifier) {
        name_node = function_node->getChild(i);
        break;
      }
    }

    if (!name_node) {
      if (log)
        log->Printf("Stepped into witness thunk \"%s\" but could not "
                    "find the Function name in the function node.",
                    symbol_name);
      return new_thread_plan_sp;
    }

    std::string function_name(name_node->getText());
    if (function_name.empty()) {
      if (log)
        log->Printf("Stepped into witness thunk \"%s\" but the Function "
                    "name was empty.",
                    symbol_name);
      return new_thread_plan_sp;
    }

    // We have to get the address range of the thunk symbol, and make a
    // "step through range stepping in"
    AddressRange sym_addr_range(sc.symbol->GetAddress(),
                                sc.symbol->GetByteSize());
    new_thread_plan_sp.reset(new ThreadPlanStepInRange(
        thread, sym_addr_range, sc, function_name.c_str(), eOnlyDuringStepping,
        eLazyBoolNo, eLazyBoolNo));
    return new_thread_plan_sp;

  } break;
  case ThunkAction::StepThrough: {
    if (log)
      log->Printf("Stepping through thunk: %s kind: %s", symbol_name,
                  GetThunkKindName(thunk_kind));
    AddressRange sym_addr_range(sc.symbol->GetAddress(),
                                sc.symbol->GetByteSize());
    new_thread_plan_sp.reset(new ThreadPlanStepInRange(
        thread, sym_addr_range, sc, nullptr, eOnlyDuringStepping, eLazyBoolNo,
        eLazyBoolNo));
    return new_thread_plan_sp;
  } break;
  }

  if (target_address.IsValid()) {
    new_thread_plan_sp.reset(
        new ThreadPlanRunToAddress(thread, target_address, stop_others));
  }

  return new_thread_plan_sp;
}

bool SwiftLanguageRuntime::IsSwiftMangledName(llvm::StringRef name) {
  return swift::Demangle::isSwiftSymbol(name);
}

void SwiftLanguageRuntime::GetGenericParameterNamesForFunction(
    const SymbolContext &const_sc,
    llvm::DenseMap<SwiftLanguageRuntime::ArchetypePath, StringRef> &dict) {
  // This terrifying cast avoids having too many differences with llvm.org.
  SymbolContext &sc = const_cast<SymbolContext &>(const_sc);

  // While building the Symtab itself the symbol context is incomplete.
  // Note that calling sc.module_sp->FindFunctions() here is too early and
  // would mess up the loading process.
  if (!sc.function && sc.module_sp && sc.symbol)
    return;

  Block *block = sc.GetFunctionBlock();
  if (!block)
    return;

  bool can_create = true;
  VariableListSP var_list = block->GetBlockVariableList(can_create);
  if (!var_list)
    return;

  for (unsigned i = 0; i < var_list->GetSize(); ++i) {
    VariableSP var_sp = var_list->GetVariableAtIndex(i);
    StringRef name = var_sp->GetName().GetStringRef();
    if (!name.consume_front(g_dollar_tau_underscore))
      continue;

    uint64_t depth;
    if (name.consumeInteger(10, depth))
      continue;

    if (!name.consume_front("_"))
      continue;

    uint64_t index;
    if (name.consumeInteger(10, index))
      continue;

    if (!name.empty())
      continue;

    Type *archetype = var_sp->GetType();
    if (!archetype)
      continue;

    dict.insert({{depth, index}, archetype->GetName().GetStringRef()});
  }
}

std::string
SwiftLanguageRuntime::DemangleSymbolAsString(StringRef symbol, DemangleMode mode,
                                             const SymbolContext *sc) {
  bool did_init = false;
  llvm::DenseMap<ArchetypePath, StringRef> dict;
  swift::Demangle::DemangleOptions options;
  switch (mode) {
  case eSimplified:
    options = swift::Demangle::DemangleOptions::SimplifiedUIDemangleOptions();
    break;
  case eTypeName:
    options.DisplayModuleNames = true;
    options.ShowPrivateDiscriminators = false;
    options.DisplayExtensionContexts = false;
    options.DisplayLocalNameContexts = false;
    break;
  case eDisplayTypeName:
    options = swift::Demangle::DemangleOptions::SimplifiedUIDemangleOptions();
    options.DisplayStdlibModule = false;
    options.DisplayObjCModule = false;
    options.QualifyEntities = true;
    options.DisplayModuleNames = true;
    options.DisplayLocalNameContexts = false;
    break;    
  }

  if (sc) {
    // Resolve generic parameters in the current function.
    options.GenericParameterName = [&](uint64_t depth, uint64_t index) {
      if (!did_init) {
        GetGenericParameterNamesForFunction(*sc, dict);
        did_init = true;
      }
      auto it = dict.find({depth, index});
      if (it != dict.end())
        return it->second.str();
      return swift::Demangle::genericParameterName(depth, index);
    };
  } else {
    // Print generic generic parameter names.
    options.GenericParameterName = [&](uint64_t depth, uint64_t index) {
      std::string name;
      {
        llvm::raw_string_ostream s(name);
        s << g_tau_underscore << depth << '_' << index;
      }
      return name;
    };
  }
  return swift::Demangle::demangleSymbolAsString(symbol, options);
}

bool SwiftLanguageRuntime::IsSwiftClassName(const char *name) {
  return swift::Demangle::isClass(name);
}

void SwiftLanguageRuntime::MethodName::Clear() {
  m_full.Clear();
  m_basename = llvm::StringRef();
  m_context = llvm::StringRef();
  m_arguments = llvm::StringRef();
  m_qualifiers = llvm::StringRef();
  m_template_args = llvm::StringRef();
  m_metatype_ref = llvm::StringRef();
  m_return_type = llvm::StringRef();
  m_type = eTypeInvalid;
  m_parsed = false;
  m_parse_error = false;
}

static bool StringHasAllOf(const llvm::StringRef &s, const char *which) {
  for (const char *c = which; *c != 0; c++) {
    if (s.find(*c) == llvm::StringRef::npos)
      return false;
  }
  return true;
}

static bool StringHasAnyOf(const llvm::StringRef &s,
                           std::initializer_list<const char *> which,
                           size_t &where) {
  for (const char *item : which) {
    size_t where_item = s.find(item);
    if (where_item != llvm::StringRef::npos) {
      where = where_item;
      return true;
    }
  }
  where = llvm::StringRef::npos;
  return false;
}

static bool UnpackTerminatedSubstring(const llvm::StringRef &s,
                                      const char start, const char stop,
                                      llvm::StringRef &dest) {
  size_t pos_of_start = s.find(start);
  if (pos_of_start == llvm::StringRef::npos)
    return false;
  size_t pos_of_stop = s.rfind(stop);
  if (pos_of_stop == llvm::StringRef::npos)
    return false;
  size_t token_count = 1;
  size_t idx = pos_of_start + 1;
  while (idx < s.size()) {
    if (s[idx] == start)
      ++token_count;
    if (s[idx] == stop) {
      if (token_count == 1) {
        dest = s.slice(pos_of_start, idx + 1);
        return true;
      }
    }
    idx++;
  }
  return false;
}

static bool UnpackQualifiedName(const llvm::StringRef &s, llvm::StringRef &decl,
                                llvm::StringRef &basename, bool &was_operator) {
  size_t pos_of_dot = s.rfind('.');
  if (pos_of_dot == llvm::StringRef::npos)
    return false;
  decl = s.substr(0, pos_of_dot);
  basename = s.substr(pos_of_dot + 1);
  size_t idx_of_operator;
  was_operator = StringHasAnyOf(basename, {"@infix", "@prefix", "@postfix"},
                                idx_of_operator);
  if (was_operator)
    basename = basename.substr(0, idx_of_operator - 1);
  return !decl.empty() && !basename.empty();
}

static bool ParseLocalDeclName(const swift::Demangle::NodePointer &node,
                               StreamString &identifier,
                               swift::Demangle::Node::Kind &parent_kind,
                               swift::Demangle::Node::Kind &kind) {
  swift::Demangle::Node::iterator end = node->end();
  for (swift::Demangle::Node::iterator pos = node->begin(); pos != end; ++pos) {
    swift::Demangle::NodePointer child = *pos;

    swift::Demangle::Node::Kind child_kind = child->getKind();
    switch (child_kind) {
    case swift::Demangle::Node::Kind::Number:
      break;

    default:
      if (child->hasText()) {
        identifier.PutCString(child->getText());
        return true;
      }
      break;
    }
  }
  return false;
}

static bool ParseFunction(const swift::Demangle::NodePointer &node,
                          StreamString &identifier,
                          swift::Demangle::Node::Kind &parent_kind,
                          swift::Demangle::Node::Kind &kind) {
  swift::Demangle::Node::iterator end = node->end();
  swift::Demangle::Node::iterator pos = node->begin();
  // First child is the function's scope
  parent_kind = (*pos)->getKind();
  ++pos;
  // Second child is either the type (no identifier)
  if (pos != end) {
    switch ((*pos)->getKind()) {
    case swift::Demangle::Node::Kind::Type:
      break;

    case swift::Demangle::Node::Kind::LocalDeclName:
      if (ParseLocalDeclName(*pos, identifier, parent_kind, kind))
        return true;
      else
        return false;
      break;

    default:
    case swift::Demangle::Node::Kind::InfixOperator:
    case swift::Demangle::Node::Kind::PostfixOperator:
    case swift::Demangle::Node::Kind::PrefixOperator:
    case swift::Demangle::Node::Kind::Identifier:
      if ((*pos)->hasText())
        identifier.PutCString((*pos)->getText());
      return true;
    }
  }
  return false;
}

static bool ParseGlobal(const swift::Demangle::NodePointer &node,
                        StreamString &identifier,
                        swift::Demangle::Node::Kind &parent_kind,
                        swift::Demangle::Node::Kind &kind) {
  swift::Demangle::Node::iterator end = node->end();
  for (swift::Demangle::Node::iterator pos = node->begin(); pos != end; ++pos) {
    swift::Demangle::NodePointer child = *pos;
    if (child) {
      kind = child->getKind();
      switch (child->getKind()) {
      case swift::Demangle::Node::Kind::Allocator:
        identifier.PutCString("__allocating_init");
        ParseFunction(child, identifier, parent_kind, kind);
        return true;

      case swift::Demangle::Node::Kind::Constructor:
        identifier.PutCString("init");
        ParseFunction(child, identifier, parent_kind, kind);
        return true;

      case swift::Demangle::Node::Kind::Deallocator:
        identifier.PutCString("__deallocating_deinit");
        ParseFunction(child, identifier, parent_kind, kind);
        return true;

      case swift::Demangle::Node::Kind::Destructor:
        identifier.PutCString("deinit");
        ParseFunction(child, identifier, parent_kind, kind);
        return true;

      case swift::Demangle::Node::Kind::Getter:
      case swift::Demangle::Node::Kind::Setter:
      case swift::Demangle::Node::Kind::Function:
        return ParseFunction(child, identifier, parent_kind, kind);

      // Ignore these, they decorate a function at the same level, but don't
      // contain any text
      case swift::Demangle::Node::Kind::ObjCAttribute:
        break;

      default:
        return false;
      }
    }
  }
  return false;
}

bool SwiftLanguageRuntime::MethodName::ExtractFunctionBasenameFromMangled(
    ConstString mangled, ConstString &basename, bool &is_method) {
  bool success = false;
  swift::Demangle::Node::Kind kind = swift::Demangle::Node::Kind::Global;
  swift::Demangle::Node::Kind parent_kind = swift::Demangle::Node::Kind::Global;
  if (mangled) {
    const char *mangled_cstr = mangled.GetCString();
    const size_t mangled_cstr_len = mangled.GetLength();

    if (mangled_cstr_len > 3) {
      llvm::StringRef mangled_ref(mangled_cstr, mangled_cstr_len);

      // Only demangle swift functions
      // This is a no-op right now for the new mangling, because you
      // have to demangle the whole name to figure this out anyway.
      // I'm leaving the test here in case we actually need to do this
      // only to functions.
      swift::Demangle::Context demangle_ctx;
      swift::Demangle::NodePointer node =
          demangle_ctx.demangleSymbolAsNode(mangled_ref);
      StreamString identifier;
      if (node) {
        switch (node->getKind()) {
        case swift::Demangle::Node::Kind::Global:
          success = ParseGlobal(node, identifier, parent_kind, kind);
          break;

        default:
          break;
        }

        if (!identifier.GetString().empty()) {
          basename = ConstString(identifier.GetString());
        }
      }
    }
  }
  if (success) {
    switch (kind) {
    case swift::Demangle::Node::Kind::Allocator:
    case swift::Demangle::Node::Kind::Constructor:
    case swift::Demangle::Node::Kind::Deallocator:
    case swift::Demangle::Node::Kind::Destructor:
      is_method = true;
      break;

    case swift::Demangle::Node::Kind::Getter:
    case swift::Demangle::Node::Kind::Setter:
      // don't handle getters and setters right now...
      return false;

    case swift::Demangle::Node::Kind::Function:
      switch (parent_kind) {
      case swift::Demangle::Node::Kind::BoundGenericClass:
      case swift::Demangle::Node::Kind::BoundGenericEnum:
      case swift::Demangle::Node::Kind::BoundGenericStructure:
      case swift::Demangle::Node::Kind::Class:
      case swift::Demangle::Node::Kind::Enum:
      case swift::Demangle::Node::Kind::Structure:
        is_method = true;
        break;

      default:
        break;
      }
      break;

    default:
      break;
    }
  }
  return success;
}

void SwiftLanguageRuntime::MethodName::Parse() {
  if (!m_parsed && m_full) {
    m_parse_error = false;
    m_parsed = true;
    llvm::StringRef full(m_full.GetCString());
    bool was_operator = false;

    if (full.find("::") != llvm::StringRef::npos) {
      // :: is not an allowed operator in Swift (func ::(...) { fails to
      // compile)
      // but it's a very legitimate token in C++ - as a defense, reject anything
      // with a :: in it as invalid Swift
      m_parse_error = true;
      return;
    }

    if (StringHasAllOf(full, ".:()")) {
      const size_t open_paren = full.find(" (");
      llvm::StringRef funcname = full.substr(0, open_paren);
      UnpackQualifiedName(funcname, m_context, m_basename, was_operator);
      if (was_operator)
        m_type = eTypeOperator;
      // check for obvious constructor/destructor cases
      else if (m_basename.equals("__deallocating_destructor"))
        m_type = eTypeDeallocator;
      else if (m_basename.equals("__allocating_constructor"))
        m_type = eTypeAllocator;
      else if (m_basename.equals("init"))
        m_type = eTypeConstructor;
      else if (m_basename.equals("destructor"))
        m_type = eTypeDestructor;
      else
        m_type = eTypeUnknownMethod;

      const size_t idx_of_colon =
          full.find(':', open_paren == llvm::StringRef::npos ? 0 : open_paren);
      full = full.substr(idx_of_colon + 2);
      if (full.empty())
        return;
      if (full[0] == '<') {
        if (UnpackTerminatedSubstring(full, '<', '>', m_template_args)) {
          full = full.substr(m_template_args.size());
        } else {
          m_parse_error = true;
          return;
        }
      }
      if (full.empty())
        return;
      if (full[0] == '(') {
        if (UnpackTerminatedSubstring(full, '(', ')', m_metatype_ref)) {
          full = full.substr(m_template_args.size());
          if (full[0] == '<') {
            if (UnpackTerminatedSubstring(full, '<', '>', m_template_args)) {
              full = full.substr(m_template_args.size());
            } else {
              m_parse_error = true;
              return;
            }
          }
        } else {
          m_parse_error = true;
          return;
        }
      }
      if (full.empty())
        return;
      if (full[0] == '(') {
        if (UnpackTerminatedSubstring(full, '(', ')', m_arguments)) {
          full = full.substr(m_template_args.size());
        } else {
          m_parse_error = true;
          return;
        }
      }
      if (full.empty())
        return;
      size_t idx_of_ret = full.find("->");
      if (idx_of_ret == llvm::StringRef::npos) {
        full = full.substr(idx_of_ret);
        if (full.empty()) {
          m_parse_error = true;
          return;
        }
        if (full[0] == ' ')
          full = full.substr(1);
        m_return_type = full;
      }
    } else if (full.find('.') != llvm::StringRef::npos) {
      // this is probably just a full name (module.type.func)
      UnpackQualifiedName(full, m_context, m_basename, was_operator);
      if (was_operator)
        m_type = eTypeOperator;
      else
        m_type = eTypeUnknownMethod;
    } else {
      // this is most probably just a basename
      m_basename = full;
      m_type = eTypeUnknownMethod;
    }
  }
}

llvm::StringRef SwiftLanguageRuntime::MethodName::GetBasename() {
  if (!m_parsed)
    Parse();
  return m_basename;
}

bool SwiftLanguageRuntime::GetTargetOfPartialApply(SymbolContext &curr_sc,
                                                   ConstString &apply_name,
                                                   SymbolContext &sc) {
  if (!curr_sc.module_sp)
    return false;

  SymbolContextList sc_list;
  swift::Demangle::Context demangle_ctx;
  // Make sure this is a partial apply:

  std::string apply_target =
      demangle_ctx.getThunkTarget(apply_name.GetStringRef());
  if (!apply_target.empty()) {
    curr_sc.module_sp->FindFunctions(ConstString(apply_target), CompilerDeclContext(),
                                     eFunctionNameTypeFull, true, false,
                                     sc_list);
    size_t num_symbols = sc_list.GetSize();
    if (num_symbols == 0)
      return false;

    CompileUnit *curr_cu = curr_sc.comp_unit;

    size_t num_found = 0;
    for (size_t i = 0; i < num_symbols; i++) {
      SymbolContext tmp_sc;
      if (sc_list.GetContextAtIndex(i, tmp_sc)) {
        if (tmp_sc.comp_unit && curr_cu && tmp_sc.comp_unit == curr_cu) {
          sc = tmp_sc;
          num_found++;
        } else if (curr_sc.module_sp == tmp_sc.module_sp) {
          sc = tmp_sc;
          num_found++;
        }
      }
    }
    if (num_found == 1)
      return true;
    else {
      sc.Clear(false);
      return false;
    }
  } else {
    return false;
  }
}

lldb::ThreadPlanSP
SwiftLanguageRuntime::GetStepThroughTrampolinePlan(Thread &thread,
                                                   bool stop_others) {
  return ::GetStepThroughTrampolinePlan(thread, stop_others);
}
  
} // namespace lldb_private
