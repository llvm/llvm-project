//===-- InstrumentationRuntimeMainThreadChecker.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InstrumentationRuntimeMainThreadChecker.h"

#include "Plugins/Process/Utility/HistoryThread.h"
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/Variable.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/InstrumentationRuntimeStopInfo.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/RegularExpression.h"
#ifdef LLDB_ENABLE_SWIFT
#include "Plugins/Process/Utility/HistoryThread.h"
#include "Plugins/TypeSystem/Swift/SwiftASTContext.h"
#include "swift/AST/ASTContext.h"
#include "swift/AST/NameLookup.h"
#include "swift/ClangImporter/ClangImporter.h"
#endif // LLDB_ENABLE_SWIFT

#include <memory>

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(InstrumentationRuntimeMainThreadChecker)

InstrumentationRuntimeMainThreadChecker::
    ~InstrumentationRuntimeMainThreadChecker() {
  Deactivate();
}

lldb::InstrumentationRuntimeSP
InstrumentationRuntimeMainThreadChecker::CreateInstance(
    const lldb::ProcessSP &process_sp) {
  return InstrumentationRuntimeSP(
      new InstrumentationRuntimeMainThreadChecker(process_sp));
}

void InstrumentationRuntimeMainThreadChecker::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(),
      "MainThreadChecker instrumentation runtime plugin.", CreateInstance,
      GetTypeStatic);
}

void InstrumentationRuntimeMainThreadChecker::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb_private::ConstString
InstrumentationRuntimeMainThreadChecker::GetPluginNameStatic() {
  return ConstString("MainThreadChecker");
}

lldb::InstrumentationRuntimeType
InstrumentationRuntimeMainThreadChecker::GetTypeStatic() {
  return eInstrumentationRuntimeTypeMainThreadChecker;
}

const RegularExpression &
InstrumentationRuntimeMainThreadChecker::GetPatternForRuntimeLibrary() {
  static RegularExpression regex(llvm::StringRef("libMainThreadChecker.dylib"));
  return regex;
}

bool InstrumentationRuntimeMainThreadChecker::CheckIfRuntimeIsValid(
    const lldb::ModuleSP module_sp) {
  static ConstString test_sym("__main_thread_checker_on_report");
  const Symbol *symbol =
      module_sp->FindFirstSymbolWithNameAndType(test_sym, lldb::eSymbolTypeAny);
  return symbol != nullptr;
}

#ifdef LLDB_ENABLE_SWIFT
static std::string TranslateObjCNameToSwiftName(std::string className,
                                                std::string selector,
                                                StackFrameSP swiftFrame) {
  if (className.empty() || selector.empty())
    return "";
  ModuleSP swiftModule = swiftFrame->GetFrameCodeAddress().GetModule();
  if (!swiftModule)
    return "";

  auto type_system_or_err = swiftModule->GetTypeSystemForLanguage(lldb::eLanguageTypeSwift);
  if (!type_system_or_err) {
    llvm::consumeError(type_system_or_err.takeError());
    return "";
  }

  auto *ts = llvm::dyn_cast_or_null<TypeSystemSwift>(&*type_system_or_err);
  if (!ts)
    return "";
  auto *ctx = ts->GetSwiftASTContext();
  if (!ctx)
    return "";
  swift::ClangImporter *imp = ctx->GetClangImporter();
  if (!imp)
    return "";

  size_t numArguments = llvm::StringRef(selector).count(':');
  llvm::SmallVector<llvm::StringRef, 4> parts;
  llvm::StringRef(selector).split(parts, ":", /*MaxSplit*/ -1,
      /*KeepEmpty*/ false);

  llvm::SmallVector<swift::Identifier, 2> selectorIdentifiers;
  for (size_t i = 0; i < parts.size(); i++) {
    selectorIdentifiers.push_back(ctx->GetIdentifier(parts[i]));
  }

  class MyConsumer : public swift::VisibleDeclConsumer {
  public:
    swift::ObjCSelector selectorToLookup;
    swift::DeclName result;

    MyConsumer(swift::ObjCSelector selector) : selectorToLookup(selector) {}

     void foundDecl(swift::ValueDecl *VD,
                           swift::DeclVisibilityKind Reason,
                           swift::DynamicLookupInfo) override{
      if (result)
        return; // Take the first result.
      swift::ClassDecl *cls = llvm::dyn_cast<swift::ClassDecl>(VD);
      if (!cls)
        return;
      auto funcs = cls->lookupDirect(selectorToLookup, true);
      if (funcs.size() == 0)
        return;

      // If the decl is actually an accessor, use the property name instead.
      swift::AbstractFunctionDecl *decl = funcs.front();
      if (auto accessor = llvm::dyn_cast<swift::AccessorDecl>(decl)) {
        result = accessor->getStorage()->getName();
        return;
      }

      result = decl->getName();
    }
  };

  MyConsumer consumer(swift::ObjCSelector(*ctx->GetASTContext(), numArguments,
                                          selectorIdentifiers));
  // FIXME(mracek): Switch to a new API that translates the Clang class name
  // to Swift class name, once this API exists. Now we assume they are the same.
  imp->lookupValue(ctx->GetIdentifier(className), consumer);

  if (!consumer.result)
    return "";
  llvm::SmallString<32> scratchSpace;
  return className + "." + consumer.result.getString(scratchSpace).str();
}
#endif // LLDB_ENABLE_SWIFT

StructuredData::ObjectSP
InstrumentationRuntimeMainThreadChecker::RetrieveReportData(
    ExecutionContextRef exe_ctx_ref) {
  ProcessSP process_sp = GetProcessSP();
  if (!process_sp)
    return StructuredData::ObjectSP();

  ThreadSP thread_sp = exe_ctx_ref.GetThreadSP();
  StackFrameSP frame_sp = thread_sp->GetSelectedFrame();
  ModuleSP runtime_module_sp = GetRuntimeModuleSP();
  Target &target = process_sp->GetTarget();

  if (!frame_sp)
    return StructuredData::ObjectSP();

  RegisterContextSP regctx_sp = frame_sp->GetRegisterContext();
  if (!regctx_sp)
    return StructuredData::ObjectSP();

  const RegisterInfo *reginfo = regctx_sp->GetRegisterInfoByName("arg1");
  if (!reginfo)
    return StructuredData::ObjectSP();

  uint64_t apiname_ptr = regctx_sp->ReadRegisterAsUnsigned(reginfo, 0);
  if (!apiname_ptr)
    return StructuredData::ObjectSP();

  std::string apiName = "";
  Status read_error;
  target.ReadCStringFromMemory(apiname_ptr, apiName, read_error);
  if (read_error.Fail())
    return StructuredData::ObjectSP();

  std::string className = "";
  std::string selector = "";
  if (apiName.substr(0, 2) == "-[") {
    size_t spacePos = apiName.find(' ');
    if (spacePos != std::string::npos) {
      className = apiName.substr(2, spacePos - 2);
      selector = apiName.substr(spacePos + 1, apiName.length() - spacePos - 2);
    }
  }

  // Gather the PCs of the user frames in the backtrace.
  StructuredData::Array *trace = new StructuredData::Array();
  auto trace_sp = StructuredData::ObjectSP(trace);
  StackFrameSP responsible_frame;
  for (unsigned I = 0; I < thread_sp->GetStackFrameCount(); ++I) {
    StackFrameSP frame = thread_sp->GetStackFrameAtIndex(I);
    Address addr = frame->GetFrameCodeAddressForSymbolication();
    if (addr.GetModule() == runtime_module_sp) // Skip PCs from the runtime.
      continue;

    // The first non-runtime frame is responsible for the bug.
    if (!responsible_frame)
      responsible_frame = frame;

    lldb::addr_t PC = addr.GetLoadAddress(&target);
    trace->AddItem(StructuredData::ObjectSP(new StructuredData::Integer(PC)));
  }

#ifdef LLDB_ENABLE_SWIFT
  if (responsible_frame) {
    if (responsible_frame->GetLanguage() == eLanguageTypeSwift) {
      std::string swiftApiName =
          TranslateObjCNameToSwiftName(className, selector, responsible_frame);
      if (swiftApiName != "")
        apiName = swiftApiName;
    }
  }
#endif // LLDB_ENABLE_SWIFT

  auto *d = new StructuredData::Dictionary();
  auto dict_sp = StructuredData::ObjectSP(d);
  d->AddStringItem("instrumentation_class", "MainThreadChecker");
  d->AddStringItem("api_name", apiName);
  d->AddStringItem("class_name", className);
  d->AddStringItem("selector", selector);
  d->AddStringItem("description",
                   apiName + " must be used from main thread only");
  d->AddIntegerItem("tid", thread_sp->GetIndexID());
  d->AddItem("trace", trace_sp);
  return dict_sp;
}

bool InstrumentationRuntimeMainThreadChecker::NotifyBreakpointHit(
    void *baton, StoppointCallbackContext *context, user_id_t break_id,
    user_id_t break_loc_id) {
  assert(baton && "null baton");
  if (!baton)
    return false; ///< false => resume execution.

  InstrumentationRuntimeMainThreadChecker *const instance =
      static_cast<InstrumentationRuntimeMainThreadChecker *>(baton);

  ProcessSP process_sp = instance->GetProcessSP();
  ThreadSP thread_sp = context->exe_ctx_ref.GetThreadSP();
  if (!process_sp || !thread_sp ||
      process_sp != context->exe_ctx_ref.GetProcessSP())
    return false;

  if (process_sp->GetModIDRef().IsLastResumeForUserExpression())
    return false;

  StructuredData::ObjectSP report =
      instance->RetrieveReportData(context->exe_ctx_ref);

  if (report) {
    std::string description = std::string(report->GetAsDictionary()
                                              ->GetValueForKey("description")
                                              ->GetAsString()
                                              ->GetValue());
    thread_sp->SetStopInfo(
        InstrumentationRuntimeStopInfo::CreateStopReasonWithInstrumentationData(
            *thread_sp, description, report));
    return true;
  }

  return false;
}

void InstrumentationRuntimeMainThreadChecker::Activate() {
  if (IsActive())
    return;

  ProcessSP process_sp = GetProcessSP();
  if (!process_sp)
    return;

  ModuleSP runtime_module_sp = GetRuntimeModuleSP();

  ConstString symbol_name("__main_thread_checker_on_report");
  const Symbol *symbol = runtime_module_sp->FindFirstSymbolWithNameAndType(
      symbol_name, eSymbolTypeCode);

  if (symbol == nullptr)
    return;

  if (!symbol->ValueIsAddress() || !symbol->GetAddressRef().IsValid())
    return;

  Target &target = process_sp->GetTarget();
  addr_t symbol_address = symbol->GetAddressRef().GetOpcodeLoadAddress(&target);

  if (symbol_address == LLDB_INVALID_ADDRESS)
    return;

  Breakpoint *breakpoint =
      process_sp->GetTarget()
          .CreateBreakpoint(symbol_address, /*internal=*/true,
                            /*hardware=*/false)
          .get();
  breakpoint->SetCallback(
      InstrumentationRuntimeMainThreadChecker::NotifyBreakpointHit, this, true);
  breakpoint->SetBreakpointKind("main-thread-checker-report");
  SetBreakpointID(breakpoint->GetID());

  SetActive(true);
}

void InstrumentationRuntimeMainThreadChecker::Deactivate() {
  SetActive(false);

  auto BID = GetBreakpointID();
  if (BID == LLDB_INVALID_BREAK_ID)
    return;

  if (ProcessSP process_sp = GetProcessSP()) {
    process_sp->GetTarget().RemoveBreakpointByID(BID);
    SetBreakpointID(LLDB_INVALID_BREAK_ID);
  }
}

lldb::ThreadCollectionSP
InstrumentationRuntimeMainThreadChecker::GetBacktracesFromExtendedStopInfo(
    StructuredData::ObjectSP info) {
  ThreadCollectionSP threads;
  threads = std::make_shared<ThreadCollection>();

  ProcessSP process_sp = GetProcessSP();

  if (info->GetObjectForDotSeparatedPath("instrumentation_class")
          ->GetStringValue() != "MainThreadChecker")
    return threads;

  std::vector<lldb::addr_t> PCs;
  auto trace = info->GetObjectForDotSeparatedPath("trace")->GetAsArray();
  trace->ForEach([&PCs](StructuredData::Object *PC) -> bool {
    PCs.push_back(PC->GetAsInteger()->GetValue());
    return true;
  });

  if (PCs.empty())
    return threads;

  StructuredData::ObjectSP thread_id_obj =
      info->GetObjectForDotSeparatedPath("tid");
  tid_t tid = thread_id_obj ? thread_id_obj->GetIntegerValue() : 0;

  // We gather symbolication addresses above, so no need for HistoryThread to
  // try to infer the call addresses.
  bool pcs_are_call_addresses = true;
  ThreadSP new_thread_sp = std::make_shared<HistoryThread>(
      *process_sp, tid, PCs, pcs_are_call_addresses);

  // Save this in the Process' ExtendedThreadList so a strong pointer retains
  // the object
  process_sp->GetExtendedThreadList().AddThread(new_thread_sp);
  threads->AddThread(new_thread_sp);

  return threads;
}
