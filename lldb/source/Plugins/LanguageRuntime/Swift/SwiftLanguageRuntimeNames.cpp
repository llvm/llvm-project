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

#include "Plugins/TypeSystem/Swift/SwiftASTContext.h"
#include "Plugins/TypeSystem/Swift/SwiftDemangle.h"
#include "SwiftLanguageRuntime.h"

#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/DemangledNameInfo.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/ThreadPlanRunToAddress.h"
#include "lldb/Target/ThreadPlanStepInRange.h"
#include "lldb/Target/ThreadPlanStepOverRange.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "swift/ABI/Task.h"
#include "swift/Demangling/Demangle.h"
#include "swift/Demangling/Demangler.h"

#include "Plugins/Language/Swift/SwiftMangled.h"
#include "Plugins/Process/Utility/RegisterContext_x86.h"
#include "Utility/ARM64_DWARF_Registers.h"
#include "swift/Demangling/ManglingFlavor.h"
#include "llvm/ADT/SmallSet.h"

using namespace lldb;
using namespace lldb_private;
namespace lldb_private {

static const char *g_dollar_tau_underscore = u8"$\u03C4_";
static const char *g_tau_underscore = g_dollar_tau_underscore + 1;

namespace {

enum class ThunkKind {
  Unknown = 0,
  AllocatingInit,
  PartialApply,
  ObjCAttribute,
  NonObjCAttributeOnCtor,
  Reabstraction,
  ProtocolConformance,
};

enum class ThunkAction {
  Unknown = 0,
  GetThunkTarget,
  StepIntoConformance,
  StepIntoAllocatingInit,
  StepThrough,
  RunToObjcCInteropCtor,
};

} // namespace

static swift::Demangle::NodePointer
childAtPath(swift::Demangle::NodePointer node,
            llvm::ArrayRef<swift::Demangle::Node::Kind> path) {
  if (!node || path.empty())
    return node;

  auto current_step = path.front();
  for (auto *child : *node)
    if (child && child->getKind() == current_step)
      return childAtPath(child, path.drop_front());
  return nullptr;
}

static bool hasChild(swift::Demangle::NodePointer node,
                     swift::Demangle::Node::Kind kind) {
  return childAtPath(node, {kind});
}

static bool IsSwiftAsyncFunctionSymbol(swift::Demangle::NodePointer node) {
  using namespace swift::Demangle;
  if (!node || !node->hasChildren() || node->getKind() != Node::Kind::Global)
    return false;
  if (hasChild(node, Node::Kind::AsyncSuspendResumePartialFunction))
    return false;

  // Peel off a ProtocolWitness node. If it exists, there will be a single
  // instance, before Static nodes.
  if (NodePointer witness = childAtPath(node, Node::Kind::ProtocolWitness))
    node = witness;

  // Peel off a Static node. If it exists, there will be a single instance.
  if (NodePointer static_node = childAtPath(node, Node::Kind::Static))
    node = static_node;

  // Get the {Implicit, Explicit} Closure or Function node.
  // For nested closures in Swift, the demangle tree is inverted: the
  // inner-most closure is the top-most closure node.
  NodePointer func_node = [&] {
    if (NodePointer func = childAtPath(node, Node::Kind::Function))
      return func;
    if (NodePointer implicit = childAtPath(node, Node::Kind::ImplicitClosure))
      return implicit;
    return childAtPath(node, Node::Kind::ExplicitClosure);
  }();

  using Kind = Node::Kind;
  static const llvm::SmallVector<llvm::SmallVector<Kind>>
      async_annotation_paths = {
          {Kind::Type, Kind::FunctionType, Kind::AsyncAnnotation},
          {Kind::Type, Kind::NoEscapeFunctionType, Kind::AsyncAnnotation},
          {Kind::Type, Kind::DependentGenericType, Kind::Type,
           Kind::FunctionType, Kind::AsyncAnnotation},
      };
  return llvm::any_of(async_annotation_paths,
                      [func_node](llvm::ArrayRef<Kind> path) {
                        return childAtPath(func_node, path);
                      });
}

/// Returns true if closure1 and closure2 have the same number, type, and
/// parent closures / function.
static bool
AreFuncletsOfSameAsyncClosure(swift::Demangle::NodePointer closure1,
                              swift::Demangle::NodePointer closure2) {
  using namespace swift::Demangle;
  NodePointer closure1_number = childAtPath(closure1, Node::Kind::Number);
  NodePointer closure2_number = childAtPath(closure2, Node::Kind::Number);
  if (!Node::deepEquals(closure1_number, closure2_number))
    return false;

  NodePointer closure1_type = childAtPath(closure1, Node::Kind::Type);
  NodePointer closure2_type = childAtPath(closure2, Node::Kind::Type);
  if (!Node::deepEquals(closure1_type, closure2_type))
    return false;

  // Because the tree is inverted, a parent closure (in swift code) is a child
  // *node* (in the demangle tree). Check that any such parents are identical.
  NodePointer explicit_closure1_parent =
      childAtPath(closure1, Node::Kind::ExplicitClosure);
  NodePointer explicit_closure2_parent =
      childAtPath(closure2, Node::Kind::ExplicitClosure);
  NodePointer implicit_closure1_parent =
      childAtPath(closure1, Node::Kind::ImplicitClosure);
  NodePointer implicit_closure2_parent =
      childAtPath(closure2, Node::Kind::ImplicitClosure);
  if (!Node::deepEquals(explicit_closure1_parent, explicit_closure2_parent) ||
      !Node::deepEquals(implicit_closure1_parent, implicit_closure2_parent))
    return false;

  // If there are no ExplicitClosure as parents, there may still be a
  // Function. Also check that they are identical.
  NodePointer closure1_function = childAtPath(closure1, Node::Kind::Function);
  NodePointer closure2_function = childAtPath(closure2, Node::Kind::Function);
  return Node::deepEquals(closure1_function, closure2_function);
}

std::optional<uint64_t>
SwiftLanguageRuntime::GetFuncletNumber(swift::Demangle::NodePointer node) {
  if (!node)
    return {};
  NodePointer node_id =
      childAtPath(node, {Node::Kind::AsyncSuspendResumePartialFunction,
                         Node::Kind::Number});
  if (!node_id)
    node_id = childAtPath(node, {Node::Kind::AsyncAwaitResumePartialFunction,
                                 Node::Kind::Number});
  if (node_id) {
    if (node_id->hasIndex())
      return node_id->getIndex();

    // This should never happen, log it if it does:
    std::string node_dump = getNodeTreeAsString(node);
    LLDB_LOGF(GetLog(LLDBLog::Demangle),
              "%s::Found a number node without a number:\n%s", __FUNCTION__,
              node_dump.c_str());
    return {};
  }

  // If this is an async function and there was no number, then this is an entry
  // funclet. Assign it number 0.
  if (SwiftLanguageRuntime::IsAnySwiftAsyncFunctionSymbol(node))
    return 0;
  return {};
}

std::optional<uint64_t>
SwiftLanguageRuntime::GetFuncletNumber(llvm::StringRef name) {
  using namespace swift::Demangle;
  Context ctx;
  NodePointer node = SwiftLanguageRuntime::DemangleSymbolAsNode(name, ctx);
  return GetFuncletNumber(node);
}

SwiftLanguageRuntime::FuncletComparisonResult
SwiftLanguageRuntime::AreFuncletsOfSameAsyncFunction(llvm::StringRef name1,
                                                     llvm::StringRef name2) {
  using namespace swift::Demangle;
  Context ctx;
  NodePointer node1 = DemangleSymbolAsNode(name1, ctx);
  NodePointer node2 = DemangleSymbolAsNode(name2, ctx);
  return AreFuncletsOfSameAsyncFunction(node1, node2);
}

SwiftLanguageRuntime::FuncletComparisonResult
SwiftLanguageRuntime::AreFuncletsOfSameAsyncFunction(
    swift::Demangle::NodePointer node1, swift::Demangle::NodePointer node2) {
  if (!IsAnySwiftAsyncFunctionSymbol(node1) ||
      !IsAnySwiftAsyncFunctionSymbol(node2))
    return FuncletComparisonResult::NotBothFunclets;

  // Peel off ProtocolWitnes/Static nodes, in this order.
  for (auto wrapper : {Node::Kind::ProtocolWitness, Node::Kind::Static}) {
    NodePointer wrapper1 = childAtPath(node1, wrapper);
    NodePointer wrapper2 = childAtPath(node2, wrapper);
    if (wrapper1 || wrapper2) {
      if (!wrapper1 | !wrapper2)
        return FuncletComparisonResult::DifferentAsyncFunctions;
      node1 = wrapper1;
      node2 = wrapper2;
    }
  }

  // If there are closures involved, do the closure-specific comparison.
  for (auto closure_kind :
       {Node::Kind::ImplicitClosure, Node::Kind::ExplicitClosure}) {
    NodePointer closure1 = childAtPath(node1, closure_kind);
    NodePointer closure2 = childAtPath(node2, closure_kind);
    if (closure1 || closure2) {
      if (!closure1 || !closure2)
        return FuncletComparisonResult::DifferentAsyncFunctions;
      return AreFuncletsOfSameAsyncClosure(closure1, closure2)
                 ? FuncletComparisonResult::SameAsyncFunction
                 : FuncletComparisonResult::DifferentAsyncFunctions;
    }
  }

  // Otherwise, find the corresponding function and compare the two.
  NodePointer function1 = childAtPath(node1, Node::Kind::Function);
  NodePointer function2 = childAtPath(node2, Node::Kind::Function);

  // If we fail to find a function node, conservatively fail.
  if (!function1 || !function2)
    return FuncletComparisonResult::NotBothFunclets;

  return Node::deepEquals(function1, function2)
             ? FuncletComparisonResult::SameAsyncFunction
             : FuncletComparisonResult::DifferentAsyncFunctions;
}

bool SwiftLanguageRuntime::IsSwiftAsyncFunctionSymbol(llvm::StringRef name) {
  if (!IsSwiftMangledName(name))
    return false;
  using namespace swift::Demangle;
  Context ctx;
  NodePointer node = SwiftLanguageRuntime::DemangleSymbolAsNode(name, ctx);
  return ::IsSwiftAsyncFunctionSymbol(node);
}

bool SwiftLanguageRuntime::IsSwiftAsyncAwaitResumePartialFunctionSymbol(
    llvm::StringRef name) {
  if (!IsSwiftMangledName(name))
    return false;
  using namespace swift::Demangle;
  Context ctx;
  NodePointer node = SwiftLanguageRuntime::DemangleSymbolAsNode(name, ctx);
  return hasChild(node, Node::Kind::AsyncAwaitResumePartialFunction);
}

bool SwiftLanguageRuntime::IsAnySwiftAsyncFunctionSymbol(llvm::StringRef name) {
  if (!IsSwiftMangledName(name))
    return false;
  using namespace swift::Demangle;
  Context ctx;
  NodePointer node = SwiftLanguageRuntime::DemangleSymbolAsNode(name, ctx);
  return IsAnySwiftAsyncFunctionSymbol(node);
}

bool SwiftLanguageRuntime::IsAnySwiftAsyncFunctionSymbol(
    swift::Demangle::NodePointer node) {
  using namespace swift::Demangle;
  if (!node || node->getKind() != Node::Kind::Global || !node->getNumChildren())
    return false;
  auto marker = node->getFirstChild()->getKind();
  return (marker == Node::Kind::AsyncSuspendResumePartialFunction) ||
         (marker == Node::Kind::AsyncAwaitResumePartialFunction) ||
         ::IsSwiftAsyncFunctionSymbol(node);
}

static ThunkKind GetThunkKind(Symbol *symbol) {
  auto symbol_name = symbol->GetMangled().GetMangledName().GetStringRef();

  using namespace swift::Demangle;
  Context demangle_ctx;
  NodePointer nodes =
      SwiftLanguageRuntime::DemangleSymbolAsNode(symbol_name, demangle_ctx);
  if (!nodes)
    return ThunkKind::Unknown;

  size_t num_global_children = nodes->getNumChildren();
  if (num_global_children == 0)
    return ThunkKind::Unknown;

  if (nodes->getKind() != Node::Kind::Global)
    return ThunkKind::Unknown;
  if (nodes->getNumChildren() == 0)
    return ThunkKind::Unknown;

  if (!demangle_ctx.isThunkSymbol(symbol_name))
    return ThunkKind::Unknown;

  NodePointer main_node = nodes->getFirstChild();
  switch (main_node->getKind()) {
  case Node::Kind::ObjCAttribute:
    return ThunkKind::ObjCAttribute;
  case Node::Kind::NonObjCAttribute:
    if (hasChild(nodes, Node::Kind::Constructor))
      return ThunkKind::NonObjCAttributeOnCtor;
    break;
  case Node::Kind::ProtocolWitness:
    if (hasChild(main_node, Node::Kind::ProtocolConformance))
      return ThunkKind::ProtocolConformance;
    break;
  case Node::Kind::ReabstractionThunkHelper:
    return ThunkKind::Reabstraction;
  case Node::Kind::PartialApplyForwarder:
    return ThunkKind::PartialApply;
  case Node::Kind::Allocator:
    if (hasChild(main_node, Node::Kind::Class))
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
    return "StepIntoAllocatingInit";
  case ThunkKind::PartialApply:
    return "GetThunkTarget";
  case ThunkKind::ObjCAttribute:
    return "GetThunkTarget";
  case ThunkKind::NonObjCAttributeOnCtor:
    return "RunToObjcCInteropCtor";
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
    return ThunkAction::StepIntoAllocatingInit;
  case ThunkKind::PartialApply:
    return ThunkAction::GetThunkTarget;
  case ThunkKind::ObjCAttribute:
    return ThunkAction::GetThunkTarget;
  case ThunkKind::Reabstraction:
    return ThunkAction::StepThrough;
  case ThunkKind::ProtocolConformance:
    return ThunkAction::StepIntoConformance;
  case ThunkKind::NonObjCAttributeOnCtor:
    return ThunkAction::RunToObjcCInteropCtor;
  }
}

/// Given a thread that is stopped at the start of swift_task_switch, create a
/// thread plan that runs to the address of the resume function.
static ThreadPlanSP
CreateRunThroughTaskSwitchThreadPlan(Thread &thread,
                                     unsigned resume_fn_generic_regnum) {
  RegisterContextSP reg_ctx =
      thread.GetStackFrameAtIndex(0)->GetRegisterContext();
  unsigned resume_fn_reg = reg_ctx->ConvertRegisterKindToRegisterNumber(
      RegisterKind::eRegisterKindGeneric, resume_fn_generic_regnum);
  uint64_t resume_fn_ptr = reg_ctx->ReadRegisterAsUnsigned(resume_fn_reg, 0);
  if (!resume_fn_ptr)
    return {};

  resume_fn_ptr = thread.GetProcess()->FixCodeAddress(resume_fn_ptr);
  return std::make_shared<ThreadPlanRunToAddress>(thread, resume_fn_ptr,
                                                  /*stop_others*/ false);
}

/// Creates a thread plan to step over swift runtime functions that can trigger
/// a task switch, like `async_task_switch` or `swift_asyncLet_get`.
static ThreadPlanSP
CreateRunThroughTaskSwitchingTrampolines(Thread &thread,
                                         llvm::StringRef trampoline_name) {
  // The signature for `swift_task_switch` is as follows:
  //   SWIFT_CC(swiftasync)
  //   void swift_task_switch(
  //     SWIFT_ASYNC_CONTEXT AsyncContext *resumeContext,
  //     TaskContinuationFunction *resumeFunction,
  //     ExecutorRef newExecutor);
  //
  // The async context given as the first argument is not passed using the
  // calling convention's first register, it's passed in the platform's async
  // context register. This means the `resumeFunction` parameter uses the
  // first ABI register (ex: x86-64: rdi, arm64: x0).
  if (trampoline_name == "swift_task_switch")
    return CreateRunThroughTaskSwitchThreadPlan(thread,
                                                LLDB_REGNUM_GENERIC_ARG1);
  // The signature for `swift_asyncLet_get` and `swift_asyncLet_finish` are the
  // same. Like `task_switch`, the async context (first argument) uses the async
  // context register, and not the arg1 register; as such, the continuation
  // funclet can be found in arg3.
  //
  // swift_asyncLet_get(SWIFT_ASYNC_CONTEXT AsyncContext *,
  //                         AsyncLet *,
  //                         void *,
  //                         TaskContinuationFunction *,
  if (trampoline_name == "swift_asyncLet_get" ||
      trampoline_name == "swift_asyncLet_get_throwing" ||
      trampoline_name == "swift_asyncLet_finish")
    return CreateRunThroughTaskSwitchThreadPlan(thread,
                                                LLDB_REGNUM_GENERIC_ARG3);
  return nullptr;
}

/// Demangle `symbol_name` and extracts the text at the node described by
/// `node_path`, if it exists; otherwise, returns an empty string.
static std::string FindClassName(StringRef symbol_name,
                                 llvm::ArrayRef<Node::Kind> node_path) {
  swift::Demangle::Context ctx;
  NodePointer demangled_node =
      SwiftLanguageRuntime::DemangleSymbolAsNode(symbol_name, ctx);

  if (!demangled_node) {
    std::string symbol_name_str = symbol_name.str();
    LLDB_LOGF(GetLog(LLDBLog::Step),
              "SwiftLanguageRuntime: failed to demangle %s.",
              symbol_name_str.c_str());
    return "";
  }
  NodePointer class_node = childAtPath(demangled_node, node_path);
  if (!class_node || !class_node->hasText()) {
    std::string node_str = getNodeTreeAsString(demangled_node);
    LLDB_LOGF(GetLog(LLDBLog::Step),
              "SwiftLanguageRuntime: failed to extract name from "
              "demangle node: %s",
              node_str.c_str());
    return "";
  }
  return class_node->getText().str();
}

/// If sc_list is non-empty, returns a plan that runs to any of its addresses.
/// Otherwise, returns nullptr.
static ThreadPlanSP
CreateThreadPlanRunToSCInList(Thread &thread, const SymbolContextList &sc_list,
                              bool stop_others) {
  std::vector<addr_t> load_addresses;
  Target &target = thread.GetProcess()->GetTarget();
  for (const SymbolContext &sc : sc_list) {
    const Symbol *ctor_symbol = sc.symbol;
    if (ctor_symbol)
      load_addresses.push_back(ctor_symbol->GetLoadAddress(&target));
  }

  if (load_addresses.empty()) {
    LLDB_LOG(GetLog(LLDBLog::Step),
             "SwiftLanguageRuntime: empty sc_list found.");
    return {};
  }
  return std::make_shared<ThreadPlanRunToAddress>(thread, load_addresses,
                                                  stop_others);
}

/// Search all modules for `target_func` and creates a RunToAddress plan
/// targeting all symbols found.
static ThreadPlanSP CreateRunToAddressPlan(StringRef target_func,
                                           Thread &thread, bool stop_others) {
  ModuleList modules = thread.GetProcess()->GetTarget().GetImages();
  SymbolContextList sc_list;
  modules.FindFunctionSymbols(ConstString(target_func), eFunctionNameTypeFull,
                              sc_list);
  return CreateThreadPlanRunToSCInList(thread, sc_list, stop_others);
}

static lldb::ThreadPlanSP GetStepThroughTrampolinePlan(Thread &thread,
                                                       bool stop_others) {
  // Here are the trampolines we have at present.
  // 1) The thunks from protocol invocations to the call in the actual object
  //    implementing the protocol.
  // 2) Thunks for going from Swift ObjC classes to their actual method
  //    invocations.
  // 3) Thunks that retain captured objects in closure invocations.
  // 4) Task switches for async functions.

  Log *log(GetLog(LLDBLog::Step));
  StackFrameSP stack_sp = thread.GetStackFrameAtIndex(0);
  if (!stack_sp)
    return nullptr;

  SymbolContext sc = stack_sp->GetSymbolContext(eSymbolContextEverything);
  Symbol *symbol = sc.symbol;

  if (!symbol)
    return nullptr;

  // Only do this if you are at the beginning of the thunk function:
  lldb::addr_t cur_addr = thread.GetRegisterContext()->GetPC();
  lldb::addr_t symbol_addr =
      symbol->GetAddress().GetLoadAddress(&thread.GetProcess()->GetTarget());

  if (symbol_addr != cur_addr)
    return nullptr;

  Mangled &mangled_symbol_name = symbol->GetMangled();
  const char *symbol_name = mangled_symbol_name.GetMangledName().AsCString();

  if (ThreadPlanSP thread_plan = CreateRunThroughTaskSwitchingTrampolines(
          thread, mangled_symbol_name.GetDemangledName()))
    return thread_plan;

  ThunkKind thunk_kind = GetThunkKind(symbol);
  ThunkAction thunk_action = GetThunkAction(thunk_kind);

  switch (thunk_action) {
  case ThunkAction::Unknown:
    return nullptr;
  case ThunkAction::GetThunkTarget: {
    swift::Demangle::Context demangle_ctx;
    std::string thunk_target = demangle_ctx.getThunkTarget(symbol_name);
    if (thunk_target.empty()) {
      if (log)
        log->Printf("Stepped to thunk \"%s\" (kind: %s) but could not "
                    "find the thunk target. ",
                    symbol_name, GetThunkKindName(thunk_kind));
      return nullptr;
    }
    if (log)
      log->Printf(
          "Stepped to thunk \"%s\" (kind: %s) stepping to target: \"%s\".",
          symbol_name, GetThunkKindName(thunk_kind), thunk_target.c_str());
    return CreateRunToAddressPlan(thunk_target, thread, stop_others);
  }
  case ThunkAction::RunToObjcCInteropCtor: {
    static constexpr auto class_path = {
        Node::Kind::Constructor, Node::Kind::Class, Node::Kind::Identifier};
    std::string class_name = FindClassName(symbol_name, class_path);
    if (class_name.empty()) {
      LLDB_LOGF(log,
                "SwiftLanguageRuntime: could not derive class name from symbol "
                "\"%s\".",
                symbol_name);
      return nullptr;
    }
    std::string ctor_name = llvm::formatv("{0} init", class_name);
    LLDB_LOGF(log,
              "SwiftLanguageRuntime: running to objective C constructor \"%s\" "
              "from swift.",
              ctor_name.c_str());

    SymbolContextList sc_list;
    ModuleFunctionSearchOptions options{/*include_symbols*/ true,
                                        /*include_inlines*/ true};
    ModuleList modules = thread.GetProcess()->GetTarget().GetImages();
    modules.FindFunctions(RegularExpression(ctor_name), options, sc_list);

    return CreateThreadPlanRunToSCInList(thread, sc_list, stop_others);
  }
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
    swift::Demangle::Context ctx;
    auto *demangled_nodes =
        SwiftLanguageRuntime::DemangleSymbolAsNode(symbol_name, ctx);

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
      return nullptr;
    }

    size_t num_children = witness_node->getNumChildren();
    if (num_children < 2) {
      if (log)
        log->Printf("Stepped into witness thunk \"%s\" but the "
                    "ProtocolWitness node doesn't have enough nodes.",
                    symbol_name);
      return nullptr;
    }

    swift::Demangle::NodePointer function_node = witness_node->getChild(1);
    if (function_node == nullptr ||
        function_node->getKind() != swift::Demangle::Node::Kind::Function) {
      if (log)
        log->Printf("Stepped into witness thunk \"%s\" but could not "
                    "find the function in the ProtocolWitness node.",
                    symbol_name);
      return nullptr;
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
      return nullptr;
    }

    std::string function_name(name_node->getText());
    if (function_name.empty()) {
      if (log)
        log->Printf("Stepped into witness thunk \"%s\" but the Function "
                    "name was empty.",
                    symbol_name);
      return nullptr;
    }

    // We have to get the address range of the thunk symbol, and make a
    // "step through range stepping in"
    AddressRange sym_addr_range(sc.symbol->GetAddress(),
                                sc.symbol->GetByteSize());
    return std::make_shared<ThreadPlanStepInRange>(
        thread, sym_addr_range, sc, function_name.c_str(), eOnlyDuringStepping,
        eLazyBoolNo, eLazyBoolNo);
  }
  case ThunkAction::StepIntoAllocatingInit: {
    LLDB_LOGF(log, "Stepping into allocating init: \"%s\"", symbol_name);
    using Kind = Node::Kind;
    static constexpr auto class_path = {Kind::Allocator, Kind::Class,
                                        Kind::Identifier};
    std::string class_name = FindClassName(symbol_name, class_path);
    if (class_name.empty())
      return nullptr;

    ModuleFunctionSearchOptions options{/*include_symbols*/ true,
                                        /*include_inlines*/ true};
    std::string ctor_name = llvm::formatv("{0}.init", class_name);
    SymbolContextList sc_list;
    sc.module_sp->FindFunctions(RegularExpression(ctor_name), options, sc_list);
    return CreateThreadPlanRunToSCInList(thread, sc_list, stop_others);
  }
  case ThunkAction::StepThrough: {
    if (log)
      log->Printf("Stepping through thunk: %s kind: %s", symbol_name,
                  GetThunkKindName(thunk_kind));
    AddressRange sym_addr_range(sc.symbol->GetAddress(),
                                sc.symbol->GetByteSize());
    ThreadPlanSP new_plan_sp = std::make_shared<ThreadPlanStepInRange>(
        thread, sym_addr_range, sc, nullptr, eOnlyDuringStepping, eLazyBoolNo,
        eLazyBoolNo);
    return new_plan_sp;
  }
  }

  return nullptr;
}

bool SwiftLanguageRuntime::IsSymbolARuntimeThunk(const Symbol &symbol) {
  llvm::StringRef symbol_name =
      symbol.GetMangled().GetMangledName().GetStringRef();
  if (symbol_name.empty())
    return false;
  swift::Demangle::Context demangle_ctx;
  if (demangle_ctx.isThunkSymbol(symbol_name))
    return true;

  // These are not Thunks in the sense that they don't jump to *user* code.
  // But they are language symbols that should be identified as something to act
  // on; here, the default action of stepping out is appropriate.
  Node *node = demangle_ctx.demangleSymbolAsNode(symbol_name);
  return node && node->getKind() == Node::Kind::Global &&
         hasChild(node, Node::Kind::TypeMetadataAccessFunction);
}

bool SwiftLanguageRuntime::IsSwiftMangledName(llvm::StringRef name) {
  // Old-style mangling uses a "_T" prefix. This can lead to false positives
  // with other symbols that just so happen to start with "_T". To prevent this,
  // only return true for select old-style mangled names. The known cases to are
  // ObjC classes and protocols. Classes are prefixed with either "_TtC" or
  // "_TtGC" (generic classes). Protocols are prefixed with "_TtP". Other "_T"
  // prefixed symbols are not considered to be Swift symbols.
  if (name.starts_with("_T"))
    return name.starts_with("_TtC") || name.starts_with("_TtGC") ||
           name.starts_with("_TtP");
  return swift::Demangle::isSwiftSymbol(name);
}

void SwiftLanguageRuntime::GetGenericParameterNamesForFunction(
    const SymbolContext &const_sc, const ExecutionContext *exe_ctx,
    swift::Mangle::ManglingFlavor flavor,
    llvm::DenseMap<SwiftLanguageRuntime::ArchetypePath, llvm::StringRef>
        &dict) {
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
    llvm::StringRef name = var_sp->GetName().GetStringRef();
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

    ConstString type_name;

    // Try to get bind the dynamic type from the exe_ctx.
    while (exe_ctx) {
      auto *frame = exe_ctx->GetFramePtr();
      auto *target = exe_ctx->GetTargetPtr();
      auto *process = exe_ctx->GetProcessPtr();
      auto *runtime = SwiftLanguageRuntime::Get(process);
      if (!frame || !target || !process || !runtime)
        break;
      auto type_system_or_err =
        target->GetScratchTypeSystemForLanguage(eLanguageTypeSwift);
      if (!type_system_or_err) {
        llvm::consumeError(type_system_or_err.takeError());
        break;
      }
      auto ts =
          llvm::dyn_cast_or_null<TypeSystemSwift>(type_system_or_err->get());
      if (!ts)
        break;
      CompilerType generic_type =
          ts->CreateGenericTypeParamType(depth, index, flavor);
      llvm::Expected<CompilerType> bound_type_or_err =
          runtime->BindGenericTypeParameters(*frame, generic_type);
      if (!bound_type_or_err) {
        LLDB_LOG_ERROR(GetLog(LLDBLog::Expressions | LLDBLog::Types),
                       bound_type_or_err.takeError(), "{0}");
        break;
      }

      type_name = bound_type_or_err->GetDisplayTypeName();
      break;
    }

    // Otherwise return the static archetype name from the debug info.
    if (!type_name) {
      Type *archetype = var_sp->GetType();
      if (!archetype)
        continue;
      type_name = archetype->GetName();
    }
    dict.insert({{depth, index}, type_name.GetStringRef()});
  }
}

std::pair<std::string, std::optional<DemangledNameInfo>>
SwiftLanguageRuntime::DemangleSymbolAsString(llvm::StringRef symbol,
                                             DemangleMode mode, bool tracking,
                                             const SymbolContext *sc,
                                             const ExecutionContext *exe_ctx) {
  bool did_init = false;
  llvm::DenseMap<ArchetypePath, llvm::StringRef> dict;
  swift::Demangle::DemangleOptions options;
  switch (mode) {
  case eSimplified:
    options = swift::Demangle::DemangleOptions::SimplifiedUIDemangleOptions();
    options.ShowAsyncResumePartial = false;
    options.ShowClosureSignature = false;
    break;
  case eTypeName:
    options.DisplayModuleNames = true;
    options.ShowPrivateDiscriminators = false;
    options.DisplayExtensionContexts = false;
    options.DisplayLocalNameContexts = false;
    options.ShowFunctionArgumentTypes = true;
    break;
  case eDisplayTypeName:
    options = swift::Demangle::DemangleOptions::SimplifiedUIDemangleOptions();
    options.DisplayStdlibModule = false;
    options.DisplayObjCModule = false;
    options.QualifyEntities = true;
    options.DisplayModuleNames = true;
    options.DisplayLocalNameContexts = false;
    options.DisplayDebuggerGeneratedModule = false;
    options.ShowFunctionArgumentTypes = true;
    options.ShowClosureSignature = false;
    break;
  }

  if (sc) {
    // Resolve generic parameters in the current function.
    options.GenericParameterName = [&](uint64_t depth, uint64_t index) {
      if (!did_init) {
        GetGenericParameterNamesForFunction(
            *sc, exe_ctx, SwiftLanguageRuntime::GetManglingFlavor(symbol),
            dict);
        did_init = true;
      }
      auto it = dict.find({depth, index});
      if (it != dict.end())
        return it->second.str();
      return swift::Demangle::genericParameterName(depth, index);
    };
  } else {
    // Print generic parameter names.
    options.GenericParameterName = [&](uint64_t depth, uint64_t index) {
      std::string name;
      {
        llvm::raw_string_ostream s(name);
        s << g_tau_underscore << depth << '_' << index;
      }
      return name;
    };
  }
  if (tracking) {
    TrackingNodePrinter printer = TrackingNodePrinter(options);
    swift::Demangle::demangleSymbolAsString(symbol, printer);
    return std::pair<std::string, std::optional<DemangledNameInfo>>(
        printer.takeString(), printer.takeInfo());
  }
  return std::pair<std::string, std::optional<DemangledNameInfo>>(
      swift::Demangle::demangleSymbolAsString(symbol, options), std::nullopt);
}

std::string SwiftLanguageRuntime::DemangleSymbolAsString(
    llvm::StringRef symbol, DemangleMode mode, const SymbolContext *sc,
    const ExecutionContext *exe_ctx) {
  return DemangleSymbolAsString(symbol, mode, false, sc, exe_ctx).first;
}

std::pair<std::string, DemangledNameInfo>
SwiftLanguageRuntime::TrackedDemangleSymbolAsString(
    llvm::StringRef symbol, DemangleMode mode, const SymbolContext *sc,
    const ExecutionContext *exe_ctx) {
  auto demangledData = DemangleSymbolAsString(symbol, mode, true, sc, exe_ctx);
  return std::pair<std::string, DemangledNameInfo>(demangledData.first,
                                                   *demangledData.second);
}

swift::Demangle::NodePointer
SwiftLanguageRuntime::DemangleSymbolAsNode(llvm::StringRef symbol,
                                           swift::Demangle::Context &ctx) {
  LLDB_LOGF(GetLog(LLDBLog::Demangle), "demangle swift as node: '%s'",
            symbol.str().data());
  return ctx.demangleSymbolAsNode(symbol);
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
      swift::Demangle::Context ctx;
      auto *node = SwiftLanguageRuntime::DemangleSymbolAsNode(mangled_ref, ctx);
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

    if (full.find("+") != llvm::StringRef::npos ||
        full.find("-") != llvm::StringRef::npos ||
        full.find("[") != llvm::StringRef::npos) {
      // Swift identifiers cannot contain +, -, or [. Objective-C expressions
      // will frequently begin with one of these characters, so reject these
      // defensively.
      m_parse_error = true;
      return;
    }

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
      else if (m_basename == "__deallocating_destructor")
        m_type = eTypeDeallocator;
      else if (m_basename == "__allocating_constructor")
        m_type = eTypeAllocator;
      else if (m_basename == "init")
        m_type = eTypeConstructor;
      else if (m_basename == "destructor")
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
    ModuleFunctionSearchOptions function_options;
    function_options.include_symbols = true;
    function_options.include_inlines = false;
    curr_sc.module_sp->FindFunctions(
        ConstString(apply_target), CompilerDeclContext(), eFunctionNameTypeFull,
        function_options, sc_list);
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

std::optional<SwiftLanguageRuntime::GenericSignature>
SwiftLanguageRuntime::GetGenericSignature(llvm::StringRef function_name,
                                          TypeSystemSwiftTypeRef &ts) {
  GenericSignature signature;
  unsigned num_generic_params = 0;

  auto flavor = SwiftLanguageRuntime::GetManglingFlavor(function_name);
  // Walk to the function type.
  swift::Demangle::Context ctx;
  auto *node = SwiftLanguageRuntime::DemangleSymbolAsNode(function_name, ctx);
  if (!node)
    return {};
  if (node->getKind() != swift::Demangle::Node::Kind::Global)
    return {};
  if (node->getNumChildren() != 1)
    return {};
  node = node->getFirstChild();
  for (auto child : *node)
    if (child->getKind() == swift::Demangle::Node::Kind::Type) {
      node = child;
      break;
    }
  if (node->getKind() != swift::Demangle::Node::Kind::Type)
    return {};
  if (node->getNumChildren() != 1)
    return {};
  node = node->getFirstChild();

  // Collect all the generic parameters.
  // Build a sorted map of (depth, index) -> <idx in signature.generic_params>.
  std::map<std::pair<unsigned, unsigned>, unsigned> param_idx;
  ForEachGenericParameter(node, [&](unsigned depth, unsigned index) {
    param_idx[{depth, index}] = 0;
  });
  num_generic_params = param_idx.size();
  unsigned i = 0;
  for (auto &p : param_idx) {
    param_idx[p.first] = i;
    signature.generic_params.emplace_back(p.first.first, p.first.second,
                                          num_generic_params);
    // Every generic parameter has the same shape as itself.
    signature.generic_params.back().same_shape.set(i);
    ++i;
  }

  // Collect the same shape requirements and store them in the
  // same_shape bit vector.
  if (node->getKind() != swift::Demangle::Node::Kind::DependentGenericType)
    return {};
  if (node->getNumChildren() != 2)
    return {};
  auto sig_node = node->getFirstChild();
  if (sig_node->getKind() !=
      swift::Demangle::Node::Kind::DependentGenericSignature)
    return {};
  for (auto child : *sig_node) {
    if (child->getKind() ==
            swift::Demangle::Node::Kind::DependentGenericParamCount &&
        child->hasIndex()) {
      signature.dependent_generic_param_count = child->getIndex();
      if (signature.dependent_generic_param_count > num_generic_params)
        return {};
      continue;
    }
    if (child->getKind() ==
        swift::Demangle::Node::Kind::DependentGenericSameShapeRequirement) {
      if (child->getNumChildren() != 2)
        return {};
      llvm::SmallVector<unsigned, 2> idx;
      ForEachGenericParameter(child, [&](unsigned depth, unsigned index) {
        idx.push_back(param_idx[{depth, index}]);
      });
      if (idx.size() != 2)
        return {};

      signature.generic_params[idx[0]].same_shape.set(idx[1]);
      signature.generic_params[idx[1]].same_shape.set(idx[0]);
    }
  }

  // Collect the shapes of the packs.
  node = node->getLastChild();
  if (node->getKind() != swift::Demangle::Node::Kind::Type)
    return {};
  bool error = false;
  // For each pack_expansion...
  swift::Demangle::NodePointer type_node = nullptr;
  TypeSystemSwiftTypeRef::PreOrderTraversal(
      node, [&](swift::Demangle::NodePointer node) {
        if (node->getKind() == swift::Demangle::Node::Kind::PackExpansion) {
          if (node->getNumChildren() != 2) {
            error = true;
            return false;
          }
          unsigned n = 0;
          // Store the shape of each pack expansion as index into
          // signature.generic_params.
          ForEachGenericParameter(
              node->getLastChild(), [&](unsigned depth, unsigned index) {
                unsigned idx = param_idx[{depth, index}];
                signature.pack_expansions.push_back({num_generic_params, idx});
                ++n;
              });
          if (n != 1)
            error = true;

          // Record the generic parameters used in this expansion.
          ForEachGenericParameter(
              node->getFirstChild(), [&](unsigned depth, unsigned index) {
                unsigned idx = param_idx[{depth, index}];
                signature.pack_expansions.back().generic_params.set(idx);
              });

          // Store the various type packs.
          swift::Demangle::Demangler dem;
          auto mangling = swift::Demangle::mangleNode(type_node, flavor);
          if (mangling.isSuccess())
            signature.pack_expansions.back().mangled_type =
                ts.RemangleAsType(dem, type_node, flavor).GetMangledTypeName();

          // Assuming that there are no nested pack_expansions.
          return false;
        }
        type_node = node;
        return true;
      });

  if (error)
    return {};

  // Build the maps associating value and type packs with their count
  // arguments.
  unsigned next_count = 0;
  unsigned sentinel = num_generic_params;
  // Lists all shape inidices that were already processed.
  llvm::BitVector skip(num_generic_params);
  // Count argument for each shape.
  llvm::SmallVector<unsigned, 4> value_pack_count(num_generic_params, sentinel);
  // For each pack_expansion (= value pack) ...
  for (unsigned j = 0; j < signature.pack_expansions.size(); ++j) {
    unsigned shape_idx = signature.pack_expansions[j].shape;
    unsigned count = value_pack_count[shape_idx];
    // If this pack_expansion doesn't share the shape of a previous
    // argument, allocate a new count argument.
    if (count == sentinel) {
      count = next_count++;
      // Store the count argument for this shape.
      value_pack_count[shape_idx] = count;
    }
    signature.count_for_value_pack.push_back(count);

    if (skip[shape_idx])
      continue;

    // All type packs used in this expansion share same count argument.
    for (unsigned p : signature.pack_expansions[j].generic_params.set_bits())
      if (signature.generic_params[p].same_shape[shape_idx])
        signature.count_for_type_pack.push_back(count);

    // Mark all pack_expansions with the same shape for skipping.
    auto &shape = signature.generic_params[shape_idx];
    skip |= shape.same_shape;
  }
  signature.num_counts = next_count;
  assert(signature.count_for_value_pack.size() ==
         signature.pack_expansions.size());

  // Fill in the is_pack field for all generic parameters.
  for (auto pack_expansion : signature.pack_expansions) {
    unsigned shape_idx = pack_expansion.shape;
    auto &param = signature.generic_params[shape_idx];
    param.is_pack = true;
    for (unsigned idx : param.same_shape.set_bits()) {
      auto &sibling = signature.generic_params[idx];
      sibling.is_pack = true;
    }
  }

  return signature;
}

/// Returns true if a function called `symbol_name` exists in `module`.
static bool SymbolExists(StringRef symbol_name, Module &module) {
  SymbolContextList sc_list;
  Module::LookupInfo lookup_info(ConstString(symbol_name),
                                 lldb::FunctionNameType::eFunctionNameTypeFull,
                                 lldb::eLanguageTypeSwift);
  module.FindFunctions(lookup_info, CompilerDeclContext(),
                       ModuleFunctionSearchOptions(), sc_list);
  return !sc_list.IsEmpty();
}

/// There are two possible variants of constructors: allocating
/// (Kind::Allocator) and initializing (Kind::Constructor). When mangling a
/// closure, the "context" is arbitrarily chosen to be the Kind::Constructor
/// variant. This function checks for the existence of the Kind::Constructor,
/// given by the symbol `ctor_kind_mangled_name` in the module. If it exists,
/// `ctor_kind_mangled_name` is returned. Otherwise, creates and returns the
/// symbol name for the `Kind::Allocating` variant.
static std::string
HandleCtorAndAllocatorVariants(StringRef ctor_kind_mangled_name, Module &module,
                               Node &top_level_node, Node &ctor_node) {
  if (SymbolExists(ctor_kind_mangled_name, module))
    return ctor_kind_mangled_name.str();

  using Kind = Node::Kind;
  NodeFactory factory;

  // Create a kind::Allocator node with all the children of the
  // kind::Constructor node.
  Node *allocating_ctor = factory.createNode(Kind::Allocator);
  if (!allocating_ctor)
    return "";
  for (auto *child : ctor_node)
    allocating_ctor->addChild(child, factory);
  swift_demangle::ReplaceChildWith(top_level_node, ctor_node, *allocating_ctor);

  if (auto mangled = swift::Demangle::mangleNode(&top_level_node);
      mangled.isSuccess())
    return std::move(mangled.result());
  return "";
}

std::string SwiftLanguageRuntime::GetParentNameIfClosure(Function &func) {
  using Kind = Node::Kind;
  swift::Demangle::Context ctx;
  ConstString name = func.GetMangled().GetMangledName();
  auto *node = SwiftLanguageRuntime::DemangleSymbolAsNode(name, ctx);
  if (!node || node->getKind() != Node::Kind::Global)
    return "";

  // Replace the top level closure node with the child function-like node, and
  // attempt to remangle. If successful, this produces the parent function-like
  // entity.
  static const auto closure_kinds = {Kind::ImplicitClosure,
                                     Kind::ExplicitClosure};
  static const auto function_kinds = {Kind::ImplicitClosure,
                                      Kind::ExplicitClosure, Kind::Function,
                                      Kind::Constructor, Kind::Static};
  auto *closure_node = swift_demangle::GetFirstChildOfKind(node, closure_kinds);
  auto *parent_func_node =
      swift_demangle::GetFirstChildOfKind(closure_node, function_kinds);
  if (!parent_func_node)
    return "";
  swift_demangle::ReplaceChildWith(*node, *closure_node, *parent_func_node);

  ManglingErrorOr<std::string> mangled = swift::Demangle::mangleNode(node);
  if (!mangled.isSuccess())
    return "";

  if (parent_func_node->getKind() == Kind::Constructor)
    if (Module *module = func.CalculateSymbolContextModule().get())
      return HandleCtorAndAllocatorVariants(mangled.result(), *module, *node,
                                            *parent_func_node);
  return mangled.result();
}
} // namespace lldb_private
