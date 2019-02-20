//===-- SwiftRuntimeReporting.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SwiftRuntimeReporting.h"

#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/ClangASTContext.h"
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
#include "Plugins/Process/Utility/HistoryThread.h"
#include "swift/AST/ASTContext.h"
#include "swift/AST/NameLookup.h"
#include "swift/ClangImporter/ClangImporter.h"

using namespace lldb;
using namespace lldb_private;

SwiftRuntimeReporting::~SwiftRuntimeReporting() {
  Deactivate();
}

lldb::InstrumentationRuntimeSP
SwiftRuntimeReporting::CreateInstance(const lldb::ProcessSP &process_sp) {
  return InstrumentationRuntimeSP(new SwiftRuntimeReporting(process_sp));
}

void SwiftRuntimeReporting::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(),
      "SwiftRuntimeReporting instrumentation runtime plugin.", CreateInstance,
      GetTypeStatic);
}

void SwiftRuntimeReporting::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

lldb_private::ConstString SwiftRuntimeReporting::GetPluginNameStatic() {
  return ConstString("SwiftRuntimeReporting");
}

lldb::InstrumentationRuntimeType SwiftRuntimeReporting::GetTypeStatic() {
  return eInstrumentationRuntimeTypeSwiftRuntimeReporting;
}

const RegularExpression &
SwiftRuntimeReporting::GetPatternForRuntimeLibrary() {
  // TODO: Add support for Linux.
  static RegularExpression regex(llvm::StringRef("libswiftCore.dylib"));
  return regex;
}

bool SwiftRuntimeReporting::CheckIfRuntimeIsValid(
    const lldb::ModuleSP module_sp) {
  static ConstString test_sym("_swift_runtime_on_report");
  const Symbol *symbol =
      module_sp->FindFirstSymbolWithNameAndType(test_sym, lldb::eSymbolTypeAny);
  return symbol != nullptr;
}

static StructuredData::ArraySP ReadThreads(ProcessSP process_sp, addr_t addr) {
  StructuredData::ArraySP threads(new StructuredData::Array());

  int ptr_size = process_sp->GetAddressByteSize();
  Target &target = process_sp->GetTarget();

  Status read_error;

  uint64_t num_extra_threads = process_sp->ReadUnsignedIntegerFromMemory(addr, ptr_size, 0, read_error);
  if (num_extra_threads > 16) num_extra_threads = 16;
  addr_t threads_ptr = process_sp->ReadUnsignedIntegerFromMemory(addr + ptr_size, ptr_size, 0, read_error);
  for (size_t i = 0; i < num_extra_threads; i++) {
    StructuredData::ArraySP trace(new StructuredData::Array());
    int thread_struct_stride = 3 * ptr_size + 8;
    addr_t thread_ptr = threads_ptr + i * thread_struct_stride;
    std::string thread_description = "";
    addr_t description_ptr = process_sp->ReadUnsignedIntegerFromMemory(
        thread_ptr, ptr_size, 0, read_error);
    if (description_ptr)
      target.ReadCStringFromMemory(description_ptr, thread_description,
                                   read_error);
    // TODO
    // uint64_t thread_id =
    // process_sp->ReadUnsignedIntegerFromMemory(thread_ptr + ptr_size,
    // ptr_size, 0, read_error);
    uint64_t num_frames = process_sp->ReadUnsignedIntegerFromMemory(
        thread_ptr + 8 + ptr_size, ptr_size, 0, read_error);
    if (num_frames > 256) num_frames = 256;
    addr_t frames_ptr = process_sp->ReadUnsignedIntegerFromMemory(
        thread_ptr + 8 + 2 * ptr_size, ptr_size, 0, read_error);
    for (size_t j = 0; j < num_frames; j++) {
      addr_t frame = process_sp->ReadUnsignedIntegerFromMemory(
          frames_ptr + j * ptr_size, ptr_size, 0, read_error);
      trace->AddItem(
          StructuredData::ObjectSP(new StructuredData::Integer(frame)));
    }
    StructuredData::DictionarySP thread(new StructuredData::Dictionary());
    thread->AddItem("trace", StructuredData::ObjectSP(trace));
    thread->AddIntegerItem("tid", 0 /* FIXME, TODO */);
    thread->AddStringItem("description", thread_description);
    threads->AddItem(StructuredData::ObjectSP(thread));
  }

  return threads;
}

static StructuredData::ArraySP ReadFixits(ProcessSP process_sp, addr_t addr) {
  StructuredData::ArraySP fixits(new StructuredData::Array());

  int ptr_size = process_sp->GetAddressByteSize();
  Target &target = process_sp->GetTarget();

  Status read_error;
  uint64_t num_fixits = process_sp->ReadUnsignedIntegerFromMemory(addr, ptr_size, 0, read_error);
  if (num_fixits > 16) num_fixits = 16;
  addr_t fixits_ptr = process_sp->ReadUnsignedIntegerFromMemory(addr + ptr_size, ptr_size, 0, read_error);
  for (size_t i = 0; i < num_fixits; i++) {
    int fixit_struct_stride = 6 * ptr_size;
    addr_t fixit_ptr = fixits_ptr + i * fixit_struct_stride;

    std::string fixit_filename;
    addr_t description_ptr = process_sp->ReadUnsignedIntegerFromMemory(
        fixit_ptr, ptr_size, 0, read_error);
    if (description_ptr)
      target.ReadCStringFromMemory(description_ptr, fixit_filename,
                                   read_error);

    uint64_t start_line = process_sp->ReadUnsignedIntegerFromMemory(
        fixit_ptr + 1 * ptr_size, ptr_size, 0, read_error);
    uint64_t start_col = process_sp->ReadUnsignedIntegerFromMemory(
        fixit_ptr + 2 * ptr_size, ptr_size, 0, read_error);
    uint64_t end_line = process_sp->ReadUnsignedIntegerFromMemory(
        fixit_ptr + 3 * ptr_size, ptr_size, 0, read_error);
    uint64_t end_col = process_sp->ReadUnsignedIntegerFromMemory(
        fixit_ptr + 4 * ptr_size, ptr_size, 0, read_error);

    std::string fixit_replacement;
    addr_t replacement_ptr = process_sp->ReadUnsignedIntegerFromMemory(
        fixit_ptr + 5 * ptr_size, ptr_size, 0, read_error);
    if (replacement_ptr)
      target.ReadCStringFromMemory(replacement_ptr, fixit_replacement,
                                   read_error);

    StructuredData::DictionarySP fixit(new StructuredData::Dictionary());
    fixit->AddStringItem("filename", fixit_filename);
    fixit->AddIntegerItem("start_line", start_line);
    fixit->AddIntegerItem("start_col", start_col);
    fixit->AddIntegerItem("end_line", end_line);
    fixit->AddIntegerItem("end_col", end_col);
    fixit->AddStringItem("replacement", fixit_replacement);
    fixits->AddItem(fixit);
  }

  return fixits;
}

static StructuredData::ArraySP ReadNotes(ProcessSP process_sp, addr_t addr) {
  StructuredData::ArraySP notes(new StructuredData::Array());

  int ptr_size = process_sp->GetAddressByteSize();
  Target &target = process_sp->GetTarget();

  Status read_error;
  uint64_t num_notes = process_sp->ReadUnsignedIntegerFromMemory(addr, ptr_size, 0, read_error);
  if (num_notes > 16) num_notes = 16;
  addr_t fixits_ptr = process_sp->ReadUnsignedIntegerFromMemory(addr + ptr_size, ptr_size, 0, read_error);
  for (size_t i = 0; i < num_notes; i++) {
    int note_struct_stride = 3 * ptr_size;
    addr_t note_ptr = fixits_ptr + i * note_struct_stride;

    std::string note_description;
    addr_t description_ptr = process_sp->ReadUnsignedIntegerFromMemory(note_ptr, ptr_size, 0, read_error);
    if (description_ptr)
      target.ReadCStringFromMemory(description_ptr, note_description, read_error);

    auto fixits = ReadFixits(process_sp, note_ptr + ptr_size);

    StructuredData::DictionarySP note(new StructuredData::Dictionary());
    note->AddStringItem("description", note_description);
    note->AddItem("fixits", fixits);
    notes->AddItem(note);
  }

  return notes;
}

StructuredData::ObjectSP
SwiftRuntimeReporting::RetrieveReportData(ExecutionContextRef exe_ctx_ref) {
  ProcessSP process_sp = GetProcessSP();
  if (!process_sp)
    return StructuredData::ObjectSP();

  ThreadSP thread_sp = exe_ctx_ref.GetThreadSP();
  StackFrameSP frame_sp = thread_sp->GetSelectedFrame();
  ModuleSP runtime_module_sp = GetRuntimeModuleSP();
  Target &target = process_sp->GetTarget();

  const lldb::ABISP &abi = process_sp->GetABI();
  if (!abi)
    return StructuredData::ObjectSP();

  // Prepare the argument types: treat all of them as pointers
  ClangASTContext *clang_ast_context = target.GetScratchClangASTContext();
  ValueList args;
  Value input_value;
  input_value.SetCompilerType(
      clang_ast_context->GetBasicType(eBasicTypeVoid).GetPointerType());
  args.PushValue(input_value);
  args.PushValue(input_value);
  args.PushValue(input_value);

  if (!abi->GetArgumentValues(*thread_sp, args))
    return StructuredData::ObjectSP();

  bool is_fatal = (args.GetValueAtIndex(0)->GetScalar().UInt() & 0xff) == 1;
  addr_t message_ptr = args.GetValueAtIndex(1)->GetScalar().ULongLong();
  addr_t details_ptr = args.GetValueAtIndex(2)->GetScalar().ULongLong();

  std::string error_type = "";
  std::string current_stack_description = "";
  addr_t memory_address = 0;
  uint64_t frames_to_skip = 0;

  StructuredData::ArraySP extra_threads(new StructuredData::Array());
  StructuredData::ArraySP fixits(new StructuredData::Array());
  StructuredData::ArraySP notes(new StructuredData::Array());

  Status read_error;
  int ptr_size = process_sp->GetAddressByteSize();
  uint64_t version = process_sp->ReadUnsignedIntegerFromMemory(
      details_ptr, ptr_size, 0, read_error);
  if (version == 1 || version == 2) {
    addr_t error_type_ptr = process_sp->ReadUnsignedIntegerFromMemory(
        details_ptr + ptr_size, ptr_size, 0, read_error);
    if (error_type_ptr)
      target.ReadCStringFromMemory(error_type_ptr, error_type, read_error);

    addr_t current_stack_description_ptr =
        process_sp->ReadUnsignedIntegerFromMemory(details_ptr + 2 * ptr_size,
                                                  ptr_size, 0, read_error);
    if (current_stack_description_ptr)
      target.ReadCStringFromMemory(current_stack_description_ptr,
                                   current_stack_description, read_error);

    frames_to_skip = process_sp->ReadUnsignedIntegerFromMemory(
        details_ptr + 3 * ptr_size, ptr_size, 0, read_error);

    memory_address = process_sp->ReadUnsignedIntegerFromMemory(
        details_ptr + 4 * ptr_size, ptr_size, 0, read_error);

    extra_threads = ReadThreads(process_sp, details_ptr + 5 * ptr_size);

    if (version == 2) {
      fixits = ReadFixits(process_sp, details_ptr + 7 * ptr_size);
      notes = ReadNotes(process_sp, details_ptr + 9 * ptr_size);
    }
  }

  // Gather the PCs of the user frames in the backtrace.
  StructuredData::ArraySP trace(new StructuredData::Array());
  for (unsigned I = 0; I < thread_sp->GetStackFrameCount(); ++I) {
    StackFrameSP frame = thread_sp->GetStackFrameAtIndex(I);
    Address addr = frame->GetFrameCodeAddress();

    if (I < frames_to_skip + 1)
      continue;

    // Decrement return address above the 0th frame to get correct symbol/source
    // line.
    if (I != 0 && trace->GetSize() == 0) {
      addr.Slide(-1);
    }

    addr_t PC = addr.GetLoadAddress(&target);
    trace->AddItem(StructuredData::ObjectSP(new StructuredData::Integer(PC)));
  }

  StructuredData::ArraySP threads(new StructuredData::Array());

  StructuredData::DictionarySP thread(new StructuredData::Dictionary());
  thread->AddItem("trace", trace);
  thread->AddStringItem("description", current_stack_description);
  thread->AddIntegerItem("tid", thread_sp->GetIndexID());
  threads->AddItem(thread);
  for (size_t i = 0; i < extra_threads->GetSize(); i++) {
    threads->AddItem(extra_threads->GetItemAtIndex(i));
  }

  std::string message = "";
  target.ReadCStringFromMemory(message_ptr, message, read_error);
  if (read_error.Fail())
    return StructuredData::ObjectSP();

  // Trim the string.
  size_t first = message.find_first_not_of(" \t\n");
  size_t last = message.find_last_not_of(" \t\n");
  if (first != std::string::npos && last != std::string::npos &&
      first <= last) {
    message = message.substr(first, (last - first + 1));
  }

  StructuredData::DictionarySP d(new StructuredData::Dictionary());
  d->AddStringItem("instrumentation_class", "SwiftRuntimeReporting");
  d->AddStringItem("description", message);
  d->AddStringItem("issue_type", error_type);
  d->AddIntegerItem("memory_address", memory_address);
  d->AddBooleanItem("is_fatal", is_fatal);
  d->AddItem("threads", threads);
  d->AddItem("fixits", fixits);
  d->AddItem("notes", notes);
  return d;
}

bool SwiftRuntimeReporting::NotifyBreakpointHit(
    void *baton, StoppointCallbackContext *context, user_id_t break_id,
    user_id_t break_loc_id) {
  assert(baton && "null baton");
  if (!baton)
    return false; //< false => resume execution.

  SwiftRuntimeReporting *const instance =
      static_cast<SwiftRuntimeReporting *>(baton);

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
    std::string description = report->GetAsDictionary()
                                ->GetValueForKey("description")
                                ->GetAsString()
                                ->GetValue();
    thread_sp->SetStopInfo(
        InstrumentationRuntimeStopInfo::CreateStopReasonWithInstrumentationData(
            *thread_sp, description, report));
    return true;
  }

  return false;
}

void SwiftRuntimeReporting::Activate() {
  if (IsActive())
    return;

  ProcessSP process_sp = GetProcessSP();
  if (!process_sp)
    return;

  ModuleSP runtime_module_sp = GetRuntimeModuleSP();

  ConstString symbol_name("_swift_runtime_on_report");
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
  breakpoint->SetCallback(SwiftRuntimeReporting::NotifyBreakpointHit, this,
                          true);
  breakpoint->SetBreakpointKind("swift-language-runtime-report");
  SetBreakpointID(breakpoint->GetID());

  SetActive(true);
}

void SwiftRuntimeReporting::Deactivate() {
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
SwiftRuntimeReporting::GetBacktracesFromExtendedStopInfo(
    StructuredData::ObjectSP info) {
  ThreadCollectionSP result;
  result.reset(new ThreadCollection());
  
  ProcessSP process_sp = GetProcessSP();
  
  if (info->GetObjectForDotSeparatedPath("instrumentation_class")
      ->GetStringValue() != "SwiftRuntimeReporting")
    return result;

  auto threads = info->GetObjectForDotSeparatedPath("threads")->GetAsArray();
  threads->ForEach([process_sp,
                    result](StructuredData::Object *thread) -> bool {
    std::vector<lldb::addr_t> PCs;
    auto trace = thread->GetObjectForDotSeparatedPath("trace")->GetAsArray();
    trace->ForEach([&PCs](StructuredData::Object *PC) -> bool {
      PCs.push_back(PC->GetAsInteger()->GetValue());
      return true;
    });

    if (PCs.empty())
      return true;

    StructuredData::ObjectSP thread_id_obj =
        thread->GetObjectForDotSeparatedPath("tid");
    tid_t tid = thread_id_obj ? thread_id_obj->GetIntegerValue() : 0;

    uint32_t stop_id = 0;
    bool stop_id_is_valid = false;
    HistoryThread *history_thread =
        new HistoryThread(*process_sp, tid, PCs, stop_id, stop_id_is_valid);
    ThreadSP new_thread_sp(history_thread);

    StructuredData::ObjectSP description =
        thread->GetObjectForDotSeparatedPath("description");
    if (description)
      history_thread->SetName(description->GetStringValue().data());

    // Save this in the Process' ExtendedThreadList so a strong pointer
    // retains the object
    process_sp->GetExtendedThreadList().AddThread(new_thread_sp);
    result->AddThread(new_thread_sp);

    return true;
  });

  return result;
}
