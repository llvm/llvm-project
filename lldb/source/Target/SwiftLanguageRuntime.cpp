//===-- SwiftLanguageRuntime.cpp --------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2015 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See http://swift.org/LICENSE.txt for license information
// See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/SwiftLanguageRuntime.h"

#include <string.h>

#include "llvm/Support/raw_ostream.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"

#include "swift/ABI/MetadataValues.h"
#include "swift/ABI/System.h"
#include "swift/AST/ASTContext.h"
#include "swift/AST/Decl.h"
#include "swift/AST/Mangle.h"
#include "swift/AST/Module.h"
#include "swift/AST/Types.h"
#include "swift/Basic/Demangle.h"
#include "swift/Remote/MemoryReader.h"
#include "swift/RemoteAST/RemoteAST.h"

#include "lldb/Core/DataBuffer.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Mangled.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/UniqueCStringMap.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/DataFormatters/StringPrinter.h"
#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/DataFormatters/ValueObjectPrinter.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/OptionParser.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandObject.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/OptionValueBoolean.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/SwiftASTContext.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadPlanRunToAddress.h"
#include "lldb/Target/ThreadPlanStepOverRange.h"
#include "lldb/Target/ThreadPlanStepInRange.h"

#include "lldb/Utility/CleanUp.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/ProcessStructReader.h"
#include "lldb/Utility/StringLexer.h"

// FIXME: we should not need this
#include "Plugins/Language/Swift/SwiftFormatters.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
SwiftLanguageRuntime::~SwiftLanguageRuntime()
{
}

SwiftLanguageRuntime::SwiftLanguageRuntime (Process *process) :
    LanguageRuntime (process),
    m_negative_cache_mutex(),
    m_SwiftNativeNSErrorISA(),
    m_memory_reader_sp(),
    m_promises_map(),
    m_resolvers_map(),
    m_bridged_synthetics_map(),
    m_box_metadata_type()
{
    SetupSwiftError();
}

static llvm::Optional<lldb::addr_t>
FindSymbolForSwiftObject (Target& target,
                          const ConstString& object,
                          const SymbolType sym_type)
{
    llvm::Optional<lldb::addr_t> retval;
    
    SymbolContextList sc_list;
    if (target.GetImages().FindSymbolsWithNameAndType(object, sym_type, sc_list))
    {
        SymbolContext SwiftObject_Class;
        if (sc_list.GetSize() == 1 && sc_list.GetContextAtIndex(0, SwiftObject_Class))
        {
            if (SwiftObject_Class.symbol)
            {
                lldb::addr_t SwiftObject_class_addr = SwiftObject_Class.symbol->GetAddress().GetLoadAddress(&target);
                if (SwiftObject_class_addr && SwiftObject_class_addr != LLDB_INVALID_ADDRESS)
                    retval = SwiftObject_class_addr;
            }
        }
    }
    return retval;
}

AppleObjCRuntimeV2*
SwiftLanguageRuntime::GetObjCRuntime ()
{
    if (auto objc_runtime = GetProcess()->GetObjCLanguageRuntime())
    {
        if (objc_runtime->GetPluginName() == AppleObjCRuntimeV2::GetPluginNameStatic())
            return (AppleObjCRuntimeV2*)objc_runtime;
    }
    return nullptr;
}

void SwiftLanguageRuntime::SetupSwiftError ()
{
    Target& target(m_process->GetTarget());

    if (m_SwiftNativeNSErrorISA.hasValue())
        return;
    
    ConstString g_SwiftNativeNSError("_SwiftNativeNSError");
    
    m_SwiftNativeNSErrorISA = FindSymbolForSwiftObject(target, g_SwiftNativeNSError, eSymbolTypeObjCClass);
}

void
SwiftLanguageRuntime::ModulesDidLoad (const ModuleList &module_list)
{
}

static bool
GetObjectDescription_ResultVariable (Process *process,
                                     Stream& str,
                                     ValueObject &object)
{
    Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_DATAFORMATTERS));

    StreamString expr_string;
    expr_string.Printf("Swift._DebuggerSupport.stringForPrintObject(%s)", object.GetName().GetCString());
    
    if (log)
        log->Printf("[GetObjectDescription_ResultVariable] expression: %s", expr_string.GetData());
    
    ValueObjectSP result_sp;
    EvaluateExpressionOptions eval_options;
    eval_options.SetLanguage(lldb::eLanguageTypeSwift);
    eval_options.SetResultIsInternal(true);
    eval_options.SetGenerateDebugInfo(true);
    auto eval_result = process->GetTarget().EvaluateExpression(expr_string.GetData(), process->GetThreadList().GetSelectedThread()->GetSelectedFrame().get(), result_sp, eval_options);
    
    if (log)
    {
        switch (eval_result)
        {
            case eExpressionCompleted:
                log->Printf("[GetObjectDescription_ResultVariable] eExpressionCompleted");
                break;
            case eExpressionSetupError:
                log->Printf("[GetObjectDescription_ResultVariable] eExpressionSetupError");
                break;
            case eExpressionParseError:
                log->Printf("[GetObjectDescription_ResultVariable] eExpressionParseError");
                break;
            case eExpressionDiscarded:
                log->Printf("[GetObjectDescription_ResultVariable] eExpressionDiscarded");
                break;
            case eExpressionInterrupted:
                log->Printf("[GetObjectDescription_ResultVariable] eExpressionInterrupted");
                break;
            case eExpressionHitBreakpoint:
                log->Printf("[GetObjectDescription_ResultVariable] eExpressionHitBreakpoint");
                break;
            case eExpressionTimedOut:
                log->Printf("[GetObjectDescription_ResultVariable] eExpressionTimedOut");
                break;
            case eExpressionResultUnavailable:
                log->Printf("[GetObjectDescription_ResultVariable] eExpressionResultUnavailable");
                break;
            case eExpressionStoppedForDebug:
                log->Printf("[GetObjectDescription_ResultVariable] eExpressionStoppedForDebug");
                break;
        }
    }
    
    // sanitize the result of the expression before moving forward
    if (!result_sp)
    {
        if (log)
            log->Printf("[GetObjectDescription_ResultVariable] expression generated no result");
        return false;
    }
    if (result_sp->GetError().Fail())
    {
        if (log)
            log->Printf("[GetObjectDescription_ResultVariable] expression generated error: %s", result_sp->GetError().AsCString());
        return false;
    }
    if (false == result_sp->GetCompilerType().IsValid())
    {
        if (log)
            log->Printf("[GetObjectDescription_ResultVariable] expression generated invalid type");
        return false;
    }
    
    lldb_private::formatters::StringPrinter::ReadStringAndDumpToStreamOptions dump_options;
    dump_options.SetEscapeNonPrintables(false).SetQuote('\0').SetPrefixToken(nullptr);
    if (lldb_private::formatters::swift::String_SummaryProvider(*result_sp.get(), str, TypeSummaryOptions().SetLanguage(lldb::eLanguageTypeSwift).SetCapping(eTypeSummaryUncapped), dump_options))
    {
        if (log)
            log->Printf("[GetObjectDescription_ResultVariable] expression completed successfully");
        return true;
    }
    else
    {
        if (log)
            log->Printf("[GetObjectDescription_ResultVariable] expression generated invalid string data");
        return false;
    }
}

static bool
GetObjectDescription_ObjectReference (Process *process,
                                      Stream& str,
                                      ValueObject &object)
{
    Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_DATAFORMATTERS));
    
    StreamString expr_string;
    expr_string.Printf("Swift._DebuggerSupport.stringForPrintObject(Swift.unsafeBitCast(0x%" PRIx64 ", to: AnyObject.self))", object.GetValueAsUnsigned(0));

    if (log)
        log->Printf("[GetObjectDescription_ObjectReference] expression: %s", expr_string.GetData());

    ValueObjectSP result_sp;
    EvaluateExpressionOptions eval_options;
    eval_options.SetLanguage(lldb::eLanguageTypeSwift);
    eval_options.SetResultIsInternal(true);
    eval_options.SetGenerateDebugInfo(true);
    auto eval_result = process->GetTarget().EvaluateExpression(expr_string.GetData(), process->GetThreadList().GetSelectedThread()->GetSelectedFrame().get(), result_sp, eval_options);
    
    if (log)
    {
        switch (eval_result)
        {
            case eExpressionCompleted:
            log->Printf("[GetObjectDescription_ObjectReference] eExpressionCompleted");
            break;
            case eExpressionSetupError:
            log->Printf("[GetObjectDescription_ObjectReference] eExpressionSetupError");
            break;
            case eExpressionParseError:
            log->Printf("[GetObjectDescription_ObjectReference] eExpressionParseError");
            break;
            case eExpressionDiscarded:
            log->Printf("[GetObjectDescription_ObjectReference] eExpressionDiscarded");
            break;
            case eExpressionInterrupted:
            log->Printf("[GetObjectDescription_ObjectReference] eExpressionInterrupted");
            break;
            case eExpressionHitBreakpoint:
            log->Printf("[GetObjectDescription_ObjectReference] eExpressionHitBreakpoint");
            break;
            case eExpressionTimedOut:
            log->Printf("[GetObjectDescription_ObjectReference] eExpressionTimedOut");
            break;
            case eExpressionResultUnavailable:
            log->Printf("[GetObjectDescription_ObjectReference] eExpressionResultUnavailable");
            break;
            case eExpressionStoppedForDebug:
            log->Printf("[GetObjectDescription_ObjectReference] eExpressionStoppedForDebug");
            break;
        }
    }
    
    // sanitize the result of the expression before moving forward
    if (!result_sp)
    {
        if (log)
            log->Printf("[GetObjectDescription_ObjectReference] expression generated no result");
        return false;
    }
    if (result_sp->GetError().Fail())
    {
        if (log)
            log->Printf("[GetObjectDescription_ObjectReference] expression generated error: %s", result_sp->GetError().AsCString());
        return false;
    }
    if (false == result_sp->GetCompilerType().IsValid())
    {
        if (log)
            log->Printf("[GetObjectDescription_ObjectReference] expression generated invalid type");
        return false;
    }
    
    lldb_private::formatters::StringPrinter::ReadStringAndDumpToStreamOptions dump_options;
    dump_options.SetEscapeNonPrintables(false).SetQuote('\0').SetPrefixToken(nullptr);
    if (lldb_private::formatters::swift::String_SummaryProvider(*result_sp.get(), str, TypeSummaryOptions().SetLanguage(lldb::eLanguageTypeSwift).SetCapping(eTypeSummaryUncapped), dump_options))
    {
        if (log)
            log->Printf("[GetObjectDescription_ObjectReference] expression completed successfully");
        return true;
    }
    else
    {
        if (log)
            log->Printf("[GetObjectDescription_ObjectReference] expression generated invalid string data");
        return false;
    }
}

static bool
GetObjectDescription_ObjectCopy (Process *process,
                                 Stream& str,
                                 ValueObject &object)
{
    Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_DATAFORMATTERS));
    
    ValueObjectSP static_sp(object.GetStaticValue());
    
    CompilerType static_type(static_sp->GetCompilerType());
    
    Error error;
    
    lldb::addr_t copy_location = process->AllocateMemory(static_type.GetByteStride(), ePermissionsReadable|ePermissionsWritable, error);
    if (copy_location == LLDB_INVALID_ADDRESS)
    {
        if (log)
            log->Printf("[GetObjectDescription_ObjectCopy] copy_location invalid");
        return false;
    }
    lldb_utility::CleanUp<lldb::addr_t> cleanup( copy_location, [process] (lldb::addr_t value) { (void)process->DeallocateMemory(value); } );
    
    DataExtractor data_extractor;
    if (0 == static_sp->GetData(data_extractor, error))
    {
        if (log)
            log->Printf("[GetObjectDescription_ObjectCopy] data extraction failed");
        return false;
    }
    
    if (0 == process->WriteMemory(copy_location, data_extractor.GetDataStart(), data_extractor.GetByteSize(), error))
    {
        if (log)
            log->Printf("[GetObjectDescription_ObjectCopy] memory copy failed");
        return false;
    }
    
    StreamString expr_string;
    expr_string.Printf("Swift._DebuggerSupport.stringForPrintObject(Swift.UnsafePointer<%s>(bitPattern: 0x%" PRIx64 ")!.pointee)",static_type.GetTypeName().GetCString(),copy_location);
    
    if (log)
        log->Printf("[GetObjectDescription_ObjectCopy] expression: %s", expr_string.GetData());
    
    ValueObjectSP result_sp;
    EvaluateExpressionOptions eval_options;
    eval_options.SetLanguage(lldb::eLanguageTypeSwift);
    eval_options.SetResultIsInternal(true);
    eval_options.SetGenerateDebugInfo(true);
    auto eval_result = process->GetTarget().EvaluateExpression(expr_string.GetData(), process->GetThreadList().GetSelectedThread()->GetSelectedFrame().get(), result_sp, eval_options);
    
    if (log)
    {
        switch (eval_result)
        {
            case eExpressionCompleted:
                log->Printf("[GetObjectDescription_ObjectCopy] eExpressionCompleted");
                break;
            case eExpressionSetupError:
                log->Printf("[GetObjectDescription_ObjectCopy] eExpressionSetupError");
                break;
            case eExpressionParseError:
                log->Printf("[GetObjectDescription_ObjectCopy] eExpressionParseError");
                break;
            case eExpressionDiscarded:
                log->Printf("[GetObjectDescription_ObjectCopy] eExpressionDiscarded");
                break;
            case eExpressionInterrupted:
                log->Printf("[GetObjectDescription_ObjectCopy] eExpressionInterrupted");
                break;
            case eExpressionHitBreakpoint:
                log->Printf("[GetObjectDescription_ObjectCopy] eExpressionHitBreakpoint");
                break;
            case eExpressionTimedOut:
                log->Printf("[GetObjectDescription_ObjectCopy] eExpressionTimedOut");
                break;
            case eExpressionResultUnavailable:
                log->Printf("[GetObjectDescription_ObjectCopy] eExpressionResultUnavailable");
                break;
            case eExpressionStoppedForDebug:
                log->Printf("[GetObjectDescription_ObjectCopy] eExpressionStoppedForDebug");
                break;
        }
    }
    
    // sanitize the result of the expression before moving forward
    if (!result_sp)
    {
        if (log)
            log->Printf("[GetObjectDescription_ObjectCopy] expression generated no result");

        str.Printf("expression produced no result");
        return true;
    }
    if (result_sp->GetError().Fail())
    {
        if (log)
            log->Printf("[GetObjectDescription_ObjectCopy] expression generated error: %s", result_sp->GetError().AsCString());

        str.Printf("expression produced error: %s", result_sp->GetError().AsCString());
        return true;
    }
    if (false == result_sp->GetCompilerType().IsValid())
    {
        if (log)
            log->Printf("[GetObjectDescription_ObjectCopy] expression generated invalid type");

        str.Printf("expression produced invalid result type");
        return true;
    }
    
    lldb_private::formatters::StringPrinter::ReadStringAndDumpToStreamOptions dump_options;
    dump_options.SetEscapeNonPrintables(false).SetQuote('\0').SetPrefixToken(nullptr);
    if (lldb_private::formatters::swift::String_SummaryProvider(*result_sp.get(), str, TypeSummaryOptions().SetLanguage(lldb::eLanguageTypeSwift).SetCapping(eTypeSummaryUncapped), dump_options))
    {
        if (log)
            log->Printf("[GetObjectDescription_ObjectCopy] expression completed successfully");
    }
    else
    {
        if (log)
            log->Printf("[GetObjectDescription_ObjectCopy] expression generated invalid string data");
        
        str.Printf("expression produced unprintable string");
    }
    return true;
}

static bool
IsSwiftResultVariable (const ConstString &name)
{
    if (name)
    {
        llvm::StringRef name_sr(name.GetStringRef());
        if (name_sr.size() > 2 &&
            (name_sr.startswith("$R") || name_sr.startswith("$E")) &&
            ::isdigit(name_sr[2]))
            return true;
    }
    return false;
}

static bool
IsSwiftReferenceType (ValueObject &object)
{
    CompilerType object_type(object.GetCompilerType());
    if (llvm::dyn_cast_or_null<SwiftASTContext>(object_type.GetTypeSystem()))
    {
        Flags type_flags(object_type.GetTypeInfo());
        if (type_flags.AllSet(eTypeIsClass | eTypeHasValue | eTypeInstanceIsPointer))
            return true;
    }
    return false;
}

bool
SwiftLanguageRuntime::GetObjectDescription (Stream &str, ValueObject &object)
{
    if (object.IsUninitializedReference())
    {
        str.Printf("<uninitialized>");
        return true;
    }
    
    if (::IsSwiftResultVariable(object.GetName()))
    {
        // if this thing is a Swift expression result variable, it has two properties:
        // a) its name is something we can refer to in expressions for free
        // b) its type may be something we can't actually talk about in expressions
        // so, just use the result variable's name in the expression and be done with it
        StreamString probe_stream;
        if (GetObjectDescription_ResultVariable(m_process, probe_stream, object))
        {
            str.Printf("%s", probe_stream.GetData());
            return true;
        }
    }
    else if (::IsSwiftReferenceType(object))
    {
        // if this is a Swift class, it has two properties:
        // a) we do not need its type name, AnyObject is just as good
        // b) its value is something we can directly use to refer to it
        // so, just use the ValueObject's pointer-value and be done with it
        StreamString probe_stream;
        if (GetObjectDescription_ObjectReference(m_process, probe_stream, object))
        {
            str.Printf("%s", probe_stream.GetData());
            return true;
        }
    }

    // in general, don't try to use the name of the ValueObject as it might end up referring to the wrong thing
    return GetObjectDescription_ObjectCopy(m_process, str, object);
}

bool
SwiftLanguageRuntime::GetObjectDescription (Stream &str, Value &value, ExecutionContextScope *exe_scope)
{
    // This is only interesting to do with a ValueObject for Swift
    return false;
}

bool
SwiftLanguageRuntime::IsSwiftMangledName (const char *name)
{
    if (name && name[0] == '_' && name[1] == 'T')
        return true;
    else
        return false;
}

uint32_t
SwiftLanguageRuntime::FindEquivalentNames(ConstString type_name, std::vector<ConstString>& equivalents)
{
    return 0;
}

void
SwiftLanguageRuntime::MethodName::Clear()
{
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

static bool
StringHasAllOf (const llvm::StringRef &s,
                const char* which)
{
    for (const char *c = which;
         *c != 0;
         c++)
    {
        if (s.find(*c) == llvm::StringRef::npos)
            return false;
    }
    return true;
}

static bool
StringHasAnyOf (const llvm::StringRef &s,
                std::initializer_list<const char*> which,
                size_t &where)
{
    for (const char* item : which)
    {
        size_t where_item = s.find(item);
        if (where_item != llvm::StringRef::npos)
        {
            where = where_item;
            return true;
        }
    }
    where = llvm::StringRef::npos;
    return false;
}

bool
StringHasAnyOf (const llvm::StringRef &s,
                const char* which,
                size_t &where)
{
    for (const char *c = which;
         *c != 0;
         c++)
    {
        size_t where_item = s.find(*c);
        if (where_item != llvm::StringRef::npos)
        {
            where = where_item;
            return true;
        }
    }
    where = llvm::StringRef::npos;
    return false;
}

static bool
UnpackTerminatedSubstring (const llvm::StringRef &s,
                           const char start,
                           const char stop,
                           llvm::StringRef& dest)
{
    size_t pos_of_start = s.find(start);
    if (pos_of_start == llvm::StringRef::npos)
        return false;
    size_t pos_of_stop = s.rfind(stop);
    if (pos_of_stop == llvm::StringRef::npos)
        return false;
    size_t token_count = 1;
    size_t idx = pos_of_start+1;
    while (idx < s.size())
    {
        if (s[idx] == start)
            ++token_count;
        if (s[idx] == stop)
        {
            if (token_count == 1)
            {
                dest = s.slice(pos_of_start, idx+1);
                return true;
            }
        }
        idx++;
    }
    return false;
}

static bool
UnpackQualifiedName (const llvm::StringRef &s,
                     llvm::StringRef &decl,
                     llvm::StringRef &basename,
                     bool& was_operator)
{
    size_t pos_of_dot = s.rfind('.');
    if (pos_of_dot == llvm::StringRef::npos)
        return false;
    decl = s.substr(0,pos_of_dot);
    basename = s.substr(pos_of_dot+1);
    size_t idx_of_operator;
    was_operator = StringHasAnyOf(basename, {"@infix","@prefix","@postfix"}, idx_of_operator);
    if (was_operator)
        basename = basename.substr(0,idx_of_operator-1);
    return !decl.empty() && !basename.empty();
}

static bool
ParseLocalDeclName (const swift::Demangle::NodePointer &node,
                    StreamString &identifier,
                    swift::Demangle::Node::Kind &parent_kind,
                    swift::Demangle::Node::Kind &kind)
{
    swift::Demangle::Node::iterator end = node->end();
    for (swift::Demangle::Node::iterator pos = node->begin(); pos != end; ++pos)
    {
        swift::Demangle::NodePointer child = *pos;

        swift::Demangle::Node::Kind child_kind = child->getKind();
        switch (child_kind)
        {
        case swift::Demangle::Node::Kind::Number:
            break;

        default:
            if (child->hasText())
            {
                identifier.PutCString(child->getText().c_str());
                return true;
            }
            break;
        }
    }
    return false;
}

static bool
ParseFunction (const swift::Demangle::NodePointer &node,
               StreamString &identifier,
               swift::Demangle::Node::Kind &parent_kind,
               swift::Demangle::Node::Kind &kind)
{
    swift::Demangle::Node::iterator end = node->end();
    swift::Demangle::Node::iterator pos = node->begin();
    // First child is the function's scope
    parent_kind = (*pos)->getKind();
    ++pos;
    // Second child is either the type (no identifier)
    if (pos != end)
    {
        switch ((*pos)->getKind())
        {
            case swift::Demangle::Node::Kind::Type:
                break;

            case swift::Demangle::Node::Kind::LocalDeclName:
                if (ParseLocalDeclName (*pos, identifier, parent_kind, kind))
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
                    identifier.PutCString((*pos)->getText().c_str());
                return true;
        }
    }
    return false;
}

static bool
ParseGlobal (const swift::Demangle::NodePointer &node,
             StreamString &identifier,
             swift::Demangle::Node::Kind &parent_kind,
             swift::Demangle::Node::Kind &kind)
{
    swift::Demangle::Node::iterator end = node->end();
    for (swift::Demangle::Node::iterator pos = node->begin(); pos != end; ++pos)
    {
        swift::Demangle::NodePointer child = *pos;
        if (child)
        {
            kind = child->getKind();
            switch (child->getKind())
            {
                case swift::Demangle::Node::Kind::Allocator:
                    identifier.PutCString("__allocating_init");
                    ParseFunction (child, identifier, parent_kind, kind);
                    return true;

                case swift::Demangle::Node::Kind::Constructor:
                    identifier.PutCString("init");
                    ParseFunction (child, identifier, parent_kind, kind);
                    return true;

                case swift::Demangle::Node::Kind::Deallocator:
                    identifier.PutCString("__deallocating_deinit");
                    ParseFunction (child, identifier, parent_kind, kind);
                    return true;

                case swift::Demangle::Node::Kind::Destructor:
                    identifier.PutCString("deinit");
                    ParseFunction (child, identifier, parent_kind, kind);
                    return true;

                case swift::Demangle::Node::Kind::Getter:
                case swift::Demangle::Node::Kind::Setter:
                case swift::Demangle::Node::Kind::Function:
                    return ParseFunction (child, identifier, parent_kind, kind);

                    // Ignore these, they decorate a function at the same level, but don't contain any text
                case swift::Demangle::Node::Kind::ObjCAttribute:
                    break;

                default:
                    return false;
            }
        }
    }
    return false;
}

bool
SwiftLanguageRuntime::MethodName::ExtractFunctionBasenameFromMangled (const ConstString &mangled,
                                                                      ConstString &basename,
                                                                      bool &is_method)
{
    bool success = false;
    swift::Demangle::Node::Kind kind = swift::Demangle::Node::Kind::Global;
    swift::Demangle::Node::Kind parent_kind = swift::Demangle::Node::Kind::Global;
    if (mangled)
    {
        const char * mangled_cstr = mangled.GetCString();
        const size_t mangled_cstr_len = mangled.GetLength();

        if (mangled_cstr_len > 3)
        {
            llvm::StringRef mangled_ref(mangled_cstr, mangled_cstr_len);

            // Only demangle swift functions
            if (mangled_ref.startswith("_TF"))
            {
                swift::Demangle::NodePointer node = swift::Demangle::demangleSymbolAsNode(mangled_cstr, mangled_cstr_len);
                StreamString identifier;
                if (node)
                {
                    switch (node->getKind())
                    {
                        case swift::Demangle::Node::Kind::Global:
                            success = ParseGlobal (node, identifier, parent_kind, kind);
                            break;

                        default:
                            break;
                    }

                    if (!identifier.GetString().empty())
                    {
                        basename.SetCStringWithLength(identifier.GetString().c_str(), identifier.GetString().length());
                    }
                }
            }
        }
    }
    if (success)
    {
        switch (kind)
        {
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
                switch (parent_kind)
                {
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

void
SwiftLanguageRuntime::MethodName::Parse()
{
    if (!m_parsed && m_full)
    {
        //        ConstString mangled;
        //        m_full.GetMangledCounterpart(mangled);
        //        printf ("\n   parsing = '%s'\n", m_full.GetCString());
        //        if (mangled)
        //            printf ("   mangled = '%s'\n", mangled.GetCString());
        m_parse_error = false;
        m_parsed = true;
        llvm::StringRef full (m_full.GetCString());
        bool was_operator = false;
        
        if (full.find("::") != llvm::StringRef::npos)
        {
            // :: is not an allowed operator in Swift (func ::(...) { fails to compile)
            // but it's a very legitimate token in C++ - as a defense, reject anything
            // with a :: in it as invalid Swift
            m_parse_error = true;
            return;
        }
        
        if (StringHasAllOf(full,".:()"))
        {
            const size_t open_paren = full.find(" (");
            llvm::StringRef funcname = full.substr(0, open_paren);
            UnpackQualifiedName(funcname,
                                m_context,
                                m_basename,
                                was_operator);
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

            const size_t idx_of_colon = full.find(':', open_paren == llvm::StringRef::npos ? 0 : open_paren);
            full = full.substr(idx_of_colon+2);
            if (full.empty())
                return;
            if (full[0] == '<')
            {
                if (UnpackTerminatedSubstring(full, '<', '>', m_template_args))
                {
                    full = full.substr(m_template_args.size());
                }
                else
                {
                    m_parse_error = true;
                    return;
                }
            }
            if (full.empty())
                return;
            if (full[0] == '(')
            {
                if (UnpackTerminatedSubstring(full, '(', ')', m_metatype_ref))
                {
                    full = full.substr(m_template_args.size());
                    if (full[0] == '<')
                    {
                        if (UnpackTerminatedSubstring(full, '<', '>', m_template_args))
                        {
                            full = full.substr(m_template_args.size());
                        }
                        else
                        {
                            m_parse_error = true;
                            return;
                        }
                    }
                }
                else
                {
                    m_parse_error = true;
                    return;
                }
            }
            if (full.empty())
                return;
            if (full[0] == '(')
            {
                if (UnpackTerminatedSubstring(full, '(', ')',  m_arguments))
                {
                    full = full.substr(m_template_args.size());
                }
                else
                {
                    m_parse_error = true;
                    return;
                }
            }
            if (full.empty())
                return;
            size_t idx_of_ret = full.find("->");
            if (idx_of_ret == llvm::StringRef::npos)
            {
                full = full.substr(idx_of_ret);
                if (full.empty())
                {
                    m_parse_error = true;
                    return;
                }
                if (full[0] == ' ')
                    full = full.substr(1);
                m_return_type = full;
            }
        }
        else if (full.find('.') != llvm::StringRef::npos)
        {
            // this is probably just a full name (module.type.func)
            UnpackQualifiedName(full,
                                m_context,
                                m_basename,
                                was_operator);
            if (was_operator)
                m_type = eTypeOperator;
            else
                m_type = eTypeUnknownMethod;
        }
        else
        {
            // this is most probably just a basename
            m_basename = full;
            m_type = eTypeUnknownMethod;
        }
    }
}

llvm::StringRef
SwiftLanguageRuntime::MethodName::GetBasename ()
{
    if (!m_parsed)
        Parse();
    return m_basename;
}

llvm::StringRef
SwiftLanguageRuntime::MethodName::GetContext ()
{
    if (!m_parsed)
        Parse();
    return m_context;
}

llvm::StringRef
SwiftLanguageRuntime::MethodName::GetArguments ()
{
    if (!m_parsed)
        Parse();
    return m_arguments;
}

llvm::StringRef
SwiftLanguageRuntime::MethodName::GetQualifiers ()
{
    if (!m_parsed)
        Parse();
    return m_qualifiers;
}

llvm::StringRef
SwiftLanguageRuntime::MethodName::GetMetatypeReference ()
{
    if (!m_parsed)
        Parse();
    return m_qualifiers;
}

llvm::StringRef
SwiftLanguageRuntime::MethodName::GetTemplateArguments ()
{
    if (!m_parsed)
        Parse();
    return m_template_args;
}

llvm::StringRef
SwiftLanguageRuntime::MethodName::GetReturnType ()
{
    if (!m_parsed)
        Parse();
    return m_return_type;
}

const CompilerType&
SwiftLanguageRuntime::GetBoxMetadataType ()
{
    if (m_box_metadata_type.IsValid())
        return m_box_metadata_type;

    static ConstString g_type_name("__lldb_autogen_boxmetadata");
    const bool is_packed = false;
    if (ClangASTContext *ast_ctx = GetProcess()->GetTarget().GetScratchClangASTContext())
    {
        CompilerType voidstar = ast_ctx->GetBasicType(lldb::eBasicTypeVoid).GetPointerType();
        CompilerType uint32 = ClangASTContext::GetIntTypeFromBitSize(ast_ctx->getASTContext(), 32, false);
            
        m_box_metadata_type = ast_ctx->GetOrCreateStructForIdentifier(g_type_name, {
            {"kind", voidstar},
            {"offset", uint32}
        }, is_packed);
    }
    
    return m_box_metadata_type;
}

std::shared_ptr<swift::remote::MemoryReader>
SwiftLanguageRuntime::GetMemoryReader ()
{
    class MemoryReader : public swift::remote::MemoryReader
    {
    public:
        MemoryReader (Process* p,
                      size_t max_read_amount = 50*1024) :
        m_process(p)
        {
            lldbassert(m_process && "MemoryReader requires a valid Process");
            m_max_read_amount = max_read_amount;
        }
        
        virtual
        ~MemoryReader () = default;
        
        uint8_t
        getPointerSize() override
        {
            return m_process->GetAddressByteSize();
        }
        
        uint8_t
        getSizeSize() override
        {
            return getPointerSize(); // FIXME: sizeof(size_t)
        }
        
        swift::remote::RemoteAddress
        getSymbolAddress(const std::string &name) override
        {
            Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES));

            if (name.empty())
                return swift::remote::RemoteAddress(nullptr);
            
            if (log)
                log->Printf("[MemoryReader] asked to retrieve address of symbol %s", name.c_str());
            
            ConstString name_cs(name.c_str(),name.size());
            SymbolContextList sc_list;
            if (m_process->GetTarget().GetImages().FindSymbolsWithNameAndType(name_cs, lldb::eSymbolTypeAny, sc_list))
            {
                SymbolContext sym_ctx;
                if (sc_list.GetSize() == 1 && sc_list.GetContextAtIndex(0, sym_ctx))
                {
                    if (sym_ctx.symbol)
                    {
                        auto load_addr = sym_ctx.symbol->GetLoadAddress(&m_process->GetTarget());
                        if (log)
                            log->Printf("[MemoryReader] symbol resolved to 0x%" PRIx64, load_addr);
                        return swift::remote::RemoteAddress(load_addr);
                    }
                }
            }

            if (log)
                log->Printf("[MemoryReader] symbol resolution failed");
            return swift::remote::RemoteAddress(nullptr);
        }
        
        bool
        readBytes(swift::remote::RemoteAddress address, uint8_t *dest, uint64_t size) override
        {
            Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES));

            if (log)
                log->Printf("[MemoryReader] asked to read %" PRIu64 " bytes at address 0x%" PRIx64,
                            size,
                            address.getAddressData());
             
            if (size > m_max_read_amount)
            {
                if (log)
                    log->Printf("[MemoryReader] memory read exceeds maximum allowed size");
                return false;
            }

            Target &target(m_process->GetTarget());
            Address addr(address.getAddressData());
            Error error;
            if (size > target.ReadMemory(addr, true, dest, size, error))
            {
                if (log)
                    log->Printf("[MemoryReader] memory read returned fewer bytes than asked for");
                return false;
            }
            if (error.Fail())
            {
                if (log)
                    log->Printf("[MemoryReader] memory read returned error: %s", error.AsCString());
                return false;
            }
            
            if (log)
            {
                StreamString stream;
                for (uint64_t i = 0;
                     i < size;
                     i++)
                {
                    stream.PutHex8(dest[i]);
                    stream.PutChar(' ');
                }
                log->Printf("[MemoryReader] memory read returned data: %s", stream.GetData());
            }
            
            return true;
        }
        
        bool
        readString(swift::remote::RemoteAddress address, std::string &dest) override
        {
            Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES));
            
            if (log)
                log->Printf("[MemoryReader] asked to read string data at address 0x%" PRIx64, address.getAddressData());

            std::vector<char> storage(m_max_read_amount, 0);
            Target &target(m_process->GetTarget());
            Address addr(address.getAddressData());
            Error error;
            target.ReadCStringFromMemory(addr,
                                         &storage[0],
                                         storage.size(),
                                         error);
            if (error.Success())
            {
                dest.assign(&storage[0]);
                if (log)
                    log->Printf("[MemoryReader] memory read returned data: %s", dest.c_str());
                return true;
            }
            else
            {
                if (log)
                    log->Printf("[MemoryReader] memory read returned error: %s", error.AsCString());
                return false;
            }
        }
        
    private:
        Process* m_process;
        size_t m_max_read_amount;
    };
    
    if (!m_memory_reader_sp)
        m_memory_reader_sp.reset(new MemoryReader(GetProcess()));
    
    return m_memory_reader_sp;
}

SwiftASTContext*
SwiftLanguageRuntime::GetScratchSwiftASTContext ()
{
    Error error;
    return m_process->GetTarget().GetScratchSwiftASTContext(error);
}

SwiftLanguageRuntime::MemberVariableOffsetResolver::MemberVariableOffsetResolver(swift::ASTContext *ast_ctx,
                                                                                 SwiftLanguageRuntime *runtime,
                                                                                 swift::TypeBase *type) :
m_swift_ast(ast_ctx),
m_swift_runtime(runtime),
m_offsets()
{
    lldbassert(m_swift_ast && "MemberVariableOffsetResolver requires a swift::ASTContext");
    lldbassert(m_swift_runtime && "MemberVariableOffsetResolver requires a SwiftLanguageRuntime");
    lldbassert(type && "MemberVariableOffsetResolver requires a swift::Type");
    m_swift_type = type;
    m_remote_ast.reset(new swift::remoteAST::RemoteASTContext(*ast_ctx, m_swift_runtime->GetMemoryReader()));
}

llvm::Optional<uint64_t>
SwiftLanguageRuntime::MemberVariableOffsetResolver::ResolveOffset (ValueObject *valobj,
                                                                   ConstString ivar_name,
                                                                   Error* error)
{
    if (error)
        error->Clear();
    
    Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES));
    
    if (log)
        log->Printf("[MemberVariableOffsetResolver] asked to resolve offset for ivar %s", ivar_name.AsCString());
    
    auto iter = m_offsets.find(ivar_name.AsCString()), end = m_offsets.end();
    if (iter != end)
        return iter->second;
    
    auto optmeta = swift::remote::RemoteAddress(nullptr);

    const swift::TypeKind type_kind = m_swift_type->getKind();
    switch (type_kind)
    {
        case swift::TypeKind::Class:
        case swift::TypeKind::BoundGenericClass:
        {
            if (log)
                log->Printf("[MemberVariableOffsetResolver] type is a class - trying to get metadata for valueobject %s", (valobj ? valobj->GetName().AsCString() : "<null>"));
            // retrieve the metadata for class types as this is where we get the maximum benefit
            if (valobj)
            {
                lldb::addr_t value = valobj->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
                if (value == 0 || value == LLDB_INVALID_ADDRESS)
                    break;
                Error error;
                lldb::addr_t meta_ptr = m_swift_runtime->GetProcess()->ReadPointerFromMemory(value, error);
                if (error.Fail() || meta_ptr == 0 || meta_ptr == LLDB_INVALID_ADDRESS)
                    break;
                if (auto objc_runtime = m_swift_runtime->GetObjCRuntime())
                {
                    if (objc_runtime->GetRuntimeVersion() == ObjCLanguageRuntime::ObjCRuntimeVersions::eAppleObjC_V2)
                    {
                        meta_ptr = ((AppleObjCRuntimeV2*)objc_runtime)->GetPointerISA(meta_ptr);
                    }
                }
                optmeta = swift::remote::RemoteAddress(meta_ptr);
            }
            if (log)
                log->Printf("[MemberVariableOffsetResolver] optmeta = 0x%" PRIx64, optmeta.getAddressData());
        }
            break;
        default:
        {
            if (log)
                log->Printf("[MemberVariableOffsetResolver] type is not a class - no metadata needed");
        }
            break;
    }
    
    
    swift::remoteAST::Result<uint64_t> result = m_remote_ast->getOffsetOfMember(m_swift_type, optmeta, ivar_name.GetStringRef());
    if (result)
    {
        if (log)
            log->Printf("[MemberVariableOffsetResolver] offset discovered = %" PRIu64, (uint64_t)result.getValue());
        m_offsets.emplace(ivar_name.AsCString(), result.getValue());
        return result.getValue();
    }
    else
    {
        const auto& failure = result.getFailure();
        if (error)
            error->SetErrorStringWithFormat("error in resolving type offset: %s", failure.render().c_str());
        if (log)
            log->Printf("[MemberVariableOffsetResolver] failure: %s", failure.render().c_str());
        return llvm::Optional<uint64_t>();
    }
}

SwiftLanguageRuntime::MetadataPromise::MetadataPromise (swift::ASTContext *ast_ctx,
                                                        SwiftLanguageRuntime *runtime,
                                                        lldb::addr_t location) :
m_swift_ast(ast_ctx),
m_swift_runtime(runtime),
m_metadata_location(location)
{
    lldbassert(m_swift_ast && "MetadataPromise requires a swift::ASTContext");
    lldbassert(m_swift_runtime && "MetadataPromise requires a SwiftLanguageRuntime");
    m_remote_ast.reset(new swift::remoteAST::RemoteASTContext(*ast_ctx, m_swift_runtime->GetMemoryReader()));
}

CompilerType
SwiftLanguageRuntime::MetadataPromise::FulfillTypePromise (Error *error)
{
    if (error)
        error->Clear();
    
    Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES));
    
    if (log)
        log->Printf("[MetadataPromise] asked to fulfill type promise at location 0x%" PRIx64, m_metadata_location);
    
    if (m_compiler_type.hasValue())
        return m_compiler_type.getValue();
    
    swift::remoteAST::Result<swift::Type> result = m_remote_ast->getTypeForRemoteTypeMetadata(swift::remote::RemoteAddress(m_metadata_location));
    
    if (result)
    {
        m_compiler_type = CompilerType(m_swift_ast, result.getValue().getPointer());
        if (log)
            log->Printf("[MetadataPromise] result is type %s", m_compiler_type->GetTypeName().AsCString());
        return m_compiler_type.getValue();
    }
    else
    {
        const auto& failure = result.getFailure();
        if (error)
            error->SetErrorStringWithFormat("error in resolving type: %s", failure.render().c_str());
        if (log)
            log->Printf("[MetadataPromise] failure: %s", failure.render().c_str());
        return (m_compiler_type = CompilerType()).getValue();
    }
}

llvm::Optional<swift::MetadataKind>
SwiftLanguageRuntime::MetadataPromise::FulfillKindPromise (Error *error)
{
    if (error)
        error->Clear();

    Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_TYPES));
    
    if (log)
        log->Printf("[MetadataPromise] asked to fulfill kind promise at location 0x%" PRIx64, m_metadata_location);
    
    if (m_metadata_kind.hasValue())
        return m_metadata_kind;
    
    swift::remoteAST::Result<swift::MetadataKind> result = m_remote_ast->getKindForRemoteTypeMetadata(swift::remote::RemoteAddress(m_metadata_location));
    
    if (result)
    {
        m_metadata_kind = result.getValue();
        if (log)
            log->Printf("[MetadataPromise] result is kind %u", result.getValue());
        return m_metadata_kind;
    }
    else
    {
        const auto& failure = result.getFailure();
        if (error)
            error->SetErrorStringWithFormat("error in resolving type: %s", failure.render().c_str());
        if (log)
            log->Printf("[MetadataPromise] failure: %s", failure.render().c_str());
        return m_metadata_kind;
    }
}

bool
SwiftLanguageRuntime::MetadataPromise::IsStaticallyDetermined ()
{
    if (llvm::Optional<swift::MetadataKind> kind_promise = FulfillKindPromise())
    {
        switch (kind_promise.getValue())
        {
            case swift::MetadataKind::Class:
            case swift::MetadataKind::Existential:
            case swift::MetadataKind::ObjCClassWrapper:
                return false;
            default:
                return true;
        }
    }
    
    return true;
}

static inline swift::Type
GetSwiftType (const CompilerType& type)
{
    return swift::Type(reinterpret_cast<swift::TypeBase*>(type.GetOpaqueQualType()));
}

SwiftLanguageRuntime::MetadataPromiseSP
SwiftLanguageRuntime::GetMetadataPromise (lldb::addr_t addr,
                                          SwiftASTContext* swift_ast_ctx)
{
    if (!swift_ast_ctx)
        swift_ast_ctx = GetScratchSwiftASTContext();
    
    if (!swift_ast_ctx || swift_ast_ctx->HasFatalErrors())
        return nullptr;

    if (addr == 0 || addr == LLDB_INVALID_ADDRESS)
        return nullptr;
    
    if (auto objc_runtime = GetObjCRuntime())
    {
        if (objc_runtime->GetRuntimeVersion() == ObjCLanguageRuntime::ObjCRuntimeVersions::eAppleObjC_V2)
        {
            addr = ((AppleObjCRuntimeV2*)objc_runtime)->GetPointerISA(addr);
        }
    }

    typename decltype(m_promises_map)::key_type key{swift_ast_ctx->GetASTContext(),addr};
    
    auto iter = m_promises_map.find(key), end = m_promises_map.end();
    if (iter != end)
        return iter->second;
    
    MetadataPromiseSP promise_sp(new MetadataPromise(std::get<0>(key),
                                                     this,
                                                     std::get<1>(key)));
    m_promises_map.emplace(key, promise_sp);
    return promise_sp;
}

SwiftLanguageRuntime::MemberVariableOffsetResolverSP
SwiftLanguageRuntime::GetMemberVariableOffsetResolver (CompilerType compiler_type)
{
    if (!compiler_type.IsValid())
        return nullptr;
    
    SwiftASTContext *swift_ast_ctx = llvm::dyn_cast_or_null<SwiftASTContext>(compiler_type.GetTypeSystem());
    if (!swift_ast_ctx || swift_ast_ctx->HasFatalErrors())
        return nullptr;
    
    swift::TypeBase *swift_type = reinterpret_cast<swift::TypeBase*>(compiler_type.GetCanonicalType().GetOpaqueQualType());

    typename decltype(m_resolvers_map)::key_type key{swift_ast_ctx->GetASTContext(),swift_type};
    
    auto iter = m_resolvers_map.find(key), end = m_resolvers_map.end();
    if (iter != end)
        return iter->second;
    
    MemberVariableOffsetResolverSP resolver_sp(new MemberVariableOffsetResolver(std::get<0>(key),
                                                                                this,
                                                                                std::get<1>(key)));
    m_resolvers_map.emplace(key, resolver_sp);
    return resolver_sp;
}

static size_t
BaseClassDepth (ValueObject& in_value)
{
    ValueObject* ptr = &in_value;
    size_t depth = 0;
    while (ptr->IsBaseClass())
    {
        depth++;
        ptr = ptr->GetParent();
    }
    return depth;
}

bool
SwiftLanguageRuntime::GetDynamicTypeAndAddress_Class (ValueObject &in_value,
                                                      lldb::DynamicValueType use_dynamic,
                                                      TypeAndOrName &class_type_or_name,
                                                      Address &address)
{
    CompilerType value_type(in_value.GetCompilerType());
    
    AddressType address_type;
    lldb::addr_t class_metadata_ptr = in_value.GetPointerValue(&address_type);
    if (auto objc_runtime = GetObjCRuntime())
    {
        if (objc_runtime->IsTaggedPointer(class_metadata_ptr))
        {
            Value::ValueType value_type;
            return objc_runtime->GetDynamicTypeAndAddress(in_value, use_dynamic, class_type_or_name, address, value_type, /* allow_swift = */ true);
        }
    }
    if (class_metadata_ptr == LLDB_INVALID_ADDRESS || class_metadata_ptr == 0)
        return false;
    address.SetRawAddress(class_metadata_ptr);

    size_t base_depth = BaseClassDepth(in_value);
    
    lldb::addr_t class_instance_location;
    if (in_value.IsBaseClass())
        class_instance_location = in_value.GetPointerValue();
    else
        class_instance_location = in_value.GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
    if (class_instance_location == LLDB_INVALID_ADDRESS)
        return false;
    Error error;
    lldb::addr_t class_metadata_location = m_process->ReadPointerFromMemory(class_instance_location, error);
    if (error.Fail() || class_metadata_location == 0 || class_metadata_location == LLDB_INVALID_ADDRESS)
        return false;

    SwiftASTContext *swift_ast_ctx = llvm::dyn_cast_or_null<SwiftASTContext>(in_value.GetCompilerType().GetTypeSystem());
    
    MetadataPromiseSP promise_sp(GetMetadataPromise(class_metadata_location,swift_ast_ctx));
    if (!promise_sp)
        return false;
    
    CompilerType class_type(promise_sp->FulfillTypePromise());
    if (!class_type)
        return false;
    
    while (base_depth > 0)
    {
        class_type = class_type.GetDirectBaseClassAtIndex(0, nullptr);
        assert(class_type && "failed to get base class");
        base_depth--;
    }
    
    class_type_or_name.SetCompilerType(class_type);
    
    if (error.Fail())
        return false;
    
    return class_type_or_name.GetCompilerType().IsValid();
}

SwiftLanguageRuntime::SwiftErrorDescriptor::SwiftErrorDescriptor() :
m_kind(Kind::eNotAnError)
{
}

bool
SwiftLanguageRuntime::IsValidErrorValue (ValueObject& in_value,
                                         SwiftErrorDescriptor *out_error_descriptor)
{
    // see GetDynamicTypeAndAddress_ErrorType for details
    
    CompilerType var_type(in_value.GetStaticValue()->GetCompilerType());
    SwiftASTContext::ProtocolInfo protocol_info;
    if (!SwiftASTContext::GetProtocolTypeInfo(var_type,
                                              protocol_info))
        return false;
    if (!protocol_info.m_is_errortype)
        return false;

    static ConstString g_instance_type_child_name("instance_type");
    ValueObjectSP instance_type_sp(in_value.GetStaticValue()->GetChildMemberWithName(g_instance_type_child_name, true));
    if (!instance_type_sp)
        return false;
    lldb::addr_t metadata_location = instance_type_sp->GetValueAsUnsigned(0);
    if (metadata_location == 0 || metadata_location == LLDB_INVALID_ADDRESS)
        return false;
    
    SetupSwiftError();
    if (m_SwiftNativeNSErrorISA.hasValue())
    {
        if (auto objc_runtime = GetObjCRuntime())
        {
            if (auto descriptor = objc_runtime->GetClassDescriptor(*instance_type_sp))
            {
                if (descriptor->GetISA() != m_SwiftNativeNSErrorISA.getValue())
                {
                    // not a _SwiftNativeNSError - but statically typed as ErrorType
                    // return true here
                    if (out_error_descriptor)
                    {
                        *out_error_descriptor = SwiftErrorDescriptor();
                        out_error_descriptor->m_kind = SwiftErrorDescriptor::Kind::eBridged;
                        out_error_descriptor->m_bridged.instance_ptr_value = instance_type_sp->GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
                    }
                    return true;
                }
            }
        }
    }

    if (GetObjCRuntime())
    {
        // this is a swift native error but it can be bridged to ObjC
        // so it needs to be layout compatible

        size_t ptr_size = m_process->GetAddressByteSize();
        size_t metadata_offset = ptr_size + 4 + (ptr_size == 8 ? 4 : 0); // CFRuntimeBase
        metadata_offset += ptr_size + ptr_size + ptr_size; // CFIndex + 2*CFRef
        
        metadata_location += metadata_offset;
        Error error;
        lldb::addr_t metadata_ptr_value = m_process->ReadPointerFromMemory(metadata_location, error);
        if (metadata_ptr_value == 0 || metadata_ptr_value == LLDB_INVALID_ADDRESS || error.Fail())
            return false;
        
        if (out_error_descriptor)
        {
            *out_error_descriptor = SwiftErrorDescriptor();
            out_error_descriptor->m_kind = SwiftErrorDescriptor::Kind::eSwiftBridgeableNative;
            out_error_descriptor->m_bridgeable_native.metadata_location = metadata_location;
            out_error_descriptor->m_bridgeable_native.metadata_ptr_value = metadata_ptr_value;
        }
    }
    else
    {
        // this is a swift native error and it has no way to be bridged to ObjC
        // so it adopts a more compact layout

        Error error;

        size_t ptr_size = m_process->GetAddressByteSize();
        size_t metadata_offset = 2 * ptr_size;
        metadata_location += metadata_offset;
        lldb::addr_t metadata_ptr_value = m_process->ReadPointerFromMemory(metadata_location, error);
        if (metadata_ptr_value == 0 || metadata_ptr_value == LLDB_INVALID_ADDRESS || error.Fail())
            return false;

        lldb::addr_t witness_table_location = metadata_location + ptr_size;
        lldb::addr_t witness_table_ptr_value = m_process->ReadPointerFromMemory(witness_table_location, error);
        if (witness_table_ptr_value == 0 || witness_table_ptr_value == LLDB_INVALID_ADDRESS || error.Fail())
            return false;

        lldb::addr_t payload_location = witness_table_location + ptr_size;

        if (out_error_descriptor)
        {
            *out_error_descriptor = SwiftErrorDescriptor();
            out_error_descriptor->m_kind = SwiftErrorDescriptor::Kind::eSwiftPureNative;
            out_error_descriptor->m_pure_native.metadata_location = metadata_ptr_value;
            out_error_descriptor->m_pure_native.witness_table_location = witness_table_ptr_value;
            out_error_descriptor->m_pure_native.payload_ptr = payload_location;
        }
    }

    return true;
}

bool
SwiftLanguageRuntime::GetDynamicTypeAndAddress_ErrorType (ValueObject &in_value,
                                                          lldb::DynamicValueType use_dynamic,
                                                          TypeAndOrName &class_type_or_name,
                                                          Address &address)
{
    // layout of error type
    // pointer to -------> SwiftError {
                            // --------------
                            // CFRuntimeBase
                            // CFIndex
                            // CFStringRef
                            // CFDictionaryRef
                            // --------------
                            // Metadata
                            // WitnessTable
                            // --------------
                            // tail allocated actual object data *
                            // }
    // * for a struct, it's the inline data
    // * for a class, it's the inline pointer-to-the-data (aka, the swift class instance)
    SwiftErrorDescriptor error_descriptor;
    if (!IsValidErrorValue(in_value,
                          &error_descriptor))
        return false;

    Error error;
    CompilerType var_type(in_value.GetStaticValue()->GetCompilerType());
    size_t ptr_size = m_process->GetAddressByteSize();
    SwiftASTContext *swift_ast_ctx = llvm::dyn_cast_or_null<SwiftASTContext>(var_type.GetTypeSystem());
    if (!swift_ast_ctx)
        return false;

    switch (error_descriptor.m_kind)
    {
        case SwiftErrorDescriptor::Kind::eNotAnError:
            return false;
        case SwiftErrorDescriptor::Kind::eSwiftBridgeableNative:
        {
            MetadataPromiseSP promise_sp(GetMetadataPromise(error_descriptor.m_bridgeable_native.metadata_ptr_value,swift_ast_ctx));
            if (!promise_sp)
                return false;
            error_descriptor.m_bridgeable_native.metadata_location += 2*ptr_size;
            if (!promise_sp->IsStaticallyDetermined())
            {
                // figure out the actual dynamic type via the metadata at the "isa" pointer
                error_descriptor.m_bridgeable_native.metadata_location = m_process->ReadPointerFromMemory(error_descriptor.m_bridgeable_native.metadata_location, error);
                if (error_descriptor.m_bridgeable_native.metadata_location == 0 || error_descriptor.m_bridgeable_native.metadata_location == LLDB_INVALID_ADDRESS || error.Fail())
                    return false;
                error_descriptor.m_bridgeable_native.metadata_ptr_value = m_process->ReadPointerFromMemory(error_descriptor.m_bridgeable_native.metadata_location, error);
                if (error_descriptor.m_bridgeable_native.metadata_ptr_value == 0 || error_descriptor.m_bridgeable_native.metadata_ptr_value == LLDB_INVALID_ADDRESS || error.Fail())
                    return false;
                promise_sp = GetMetadataPromise(error_descriptor.m_bridgeable_native.metadata_ptr_value,swift_ast_ctx);
                if (!promise_sp || !promise_sp->FulfillTypePromise())
                {
                    // this could still be a random ObjC object
                    if (auto objc_runtime = GetObjCRuntime())
                    {
                        DataExtractor extractor(&error_descriptor.m_bridgeable_native.metadata_location,
                                                sizeof(error_descriptor.m_bridgeable_native.metadata_location),
                                                GetProcess()->GetByteOrder(),
                                                GetProcess()->GetAddressByteSize());
                        ExecutionContext exe_ctx(GetProcess());
                        auto scratch_ast = GetProcess()->GetTarget().GetScratchClangASTContext();
                        if (scratch_ast)
                        {
                            auto valobj_sp = ValueObject::CreateValueObjectFromData(in_value.GetName().AsCString(), extractor, exe_ctx, scratch_ast->GetBasicType(eBasicTypeObjCID));
                            if (valobj_sp)
                            {
                                Value::ValueType value_type;
                                if (objc_runtime->GetDynamicTypeAndAddress(*valobj_sp, use_dynamic, class_type_or_name, address, value_type))
                                {
                                    address.SetLoadAddress(error_descriptor.m_bridgeable_native.metadata_location,
                                                           &GetProcess()->GetTarget());
                                    if (!class_type_or_name.GetCompilerType().IsPointerType())
                                    {
                                        // the language runtimes do not return pointer-to-types when doing dynamic type resolution
                                        // what usually happens is that the static type has pointer-like traits that ValueObjectDynamic
                                        // then preserves in the dynamic value - since the static type here is a Swift protocol object
                                        // the dynamic type won't know to pointerize. But we truly need an ObjCObjectPointer here or else
                                        // type printing WILL be confused. Hence, make the pointer type ourselves if we didn't get one already
                                        class_type_or_name.SetCompilerType(class_type_or_name.GetCompilerType().GetPointerType());
                                    }
                                    return true;
                                }
                            }
                        }
                    }
                    
                    return false;
                }
            }

            if (!promise_sp)
                return false;
            address.SetLoadAddress(error_descriptor.m_bridgeable_native.metadata_location, &m_process->GetTarget());
            CompilerType metadata_type(promise_sp->FulfillTypePromise());
            if (metadata_type.IsValid() && error.Success())
            {
                class_type_or_name.SetCompilerType(metadata_type);
                return true;
            }
        }
            break;
        case SwiftErrorDescriptor::Kind::eBridged:
        {
            if (error_descriptor.m_bridged.instance_ptr_value != 0 &&
                error_descriptor.m_bridged.instance_ptr_value != LLDB_INVALID_ADDRESS)
            {
                Error error_type_lookup_error;
                if (CompilerType error_type = swift_ast_ctx->GetNSErrorType(error_type_lookup_error))
                {
                    class_type_or_name.SetCompilerType(error_type);
                    address.SetRawAddress(error_descriptor.m_bridged.instance_ptr_value);
                    return true;
                }
            }
        }
            break;
        case SwiftErrorDescriptor::Kind::eSwiftPureNative:
        {
            Error error;
            if (MetadataPromiseSP promise_sp = GetMetadataPromise(error_descriptor.m_pure_native.metadata_location, swift_ast_ctx))
            {
                if (promise_sp->IsStaticallyDetermined())
                {
                    if (CompilerType compiler_type = promise_sp->FulfillTypePromise())
                    {
                        class_type_or_name.SetCompilerType(compiler_type);
                        address.SetRawAddress(error_descriptor.m_pure_native.payload_ptr);
                        return true;
                    }
                }
                else
                {
                    error_descriptor.m_pure_native.metadata_location = m_process->ReadPointerFromMemory(error_descriptor.m_pure_native.payload_ptr, error);
                    if (error_descriptor.m_pure_native.metadata_location == 0 ||
                        error_descriptor.m_pure_native.metadata_location == LLDB_INVALID_ADDRESS ||
                        error.Fail())
                        return false;
                    error_descriptor.m_pure_native.payload_ptr = error_descriptor.m_pure_native.metadata_location;
                    error_descriptor.m_pure_native.metadata_location = m_process->ReadPointerFromMemory(error_descriptor.m_pure_native.payload_ptr, error);
                    if (MetadataPromiseSP promise_sp = GetMetadataPromise(error_descriptor.m_pure_native.metadata_location, swift_ast_ctx))
                    {
                        if (CompilerType compiler_type = promise_sp->FulfillTypePromise())
                        {
                            class_type_or_name.SetCompilerType(compiler_type);
                            address.SetRawAddress(error_descriptor.m_pure_native.payload_ptr);
                            return true;
                        }
                    }
                }
            }
        }
            break;
    }

    return false;
}

bool
SwiftLanguageRuntime::GetDynamicTypeAndAddress_Protocol (ValueObject &in_value,
                                                         lldb::DynamicValueType use_dynamic,
                                                         TypeAndOrName &class_type_or_name,
                                                         Address &address)
{
    CompilerType var_type(in_value.GetCompilerType());
    SwiftASTContext::ProtocolInfo protocol_info;
    if (!SwiftASTContext::GetProtocolTypeInfo(var_type,
                                              protocol_info))
        return false;

    if (protocol_info.m_is_errortype)
        return GetDynamicTypeAndAddress_ErrorType(in_value,
                                                  use_dynamic,
                                                  class_type_or_name,
                                                  address);

    MetadataPromiseSP promise_sp;
    static ConstString g_instance_type_child_name("instance_type");
    ValueObjectSP instance_type_sp(in_value.GetStaticValue()->GetChildMemberWithName(g_instance_type_child_name, true));
    if (!instance_type_sp)
        return false;
    ValueObjectSP payload0_sp(in_value.GetStaticValue()->GetChildAtIndex(0, true));
    if (!payload0_sp)
        return false;
    // @objc protocols are automatically class-only, and there is no static/dynamic to deal with
    bool is_class = protocol_info.m_is_objc || protocol_info.m_is_class_only || protocol_info.m_is_anyobject;
    if (!is_class)
    {
        promise_sp = GetMetadataPromise(instance_type_sp->GetValueAsUnsigned(0));
        if (!promise_sp)
            return false;
        if (promise_sp->FulfillKindPromise().hasValue() &&
            promise_sp->FulfillKindPromise().getValue() == swift::MetadataKind::Class)
            is_class = true;
    }
    if (is_class)
    {
        if (GetDynamicTypeAndAddress_Class(*payload0_sp, use_dynamic, class_type_or_name, address))
            return true;
        
        // only for @objc protocols, try to fallback to the ObjC runtime as a source of type information
        // this is not exactly a great solution and we need to be careful with how we use the results of this
        // computation, but assuming some care, at least data formatters will work
        if (!protocol_info.m_is_objc)
            return false;
        auto objc_runtime = GetObjCRuntime();
        if (!objc_runtime)
            return false;
        auto descriptor_sp = objc_runtime->GetClassDescriptor(*payload0_sp);
        if (!descriptor_sp)
            return false;
        std::vector<clang::NamedDecl *> decls;
        objc_runtime->GetDeclVendor()->FindDecls(descriptor_sp->GetClassName(), true, 1, decls);
        if (decls.size() == 0)
            return false;
        CompilerType type = ClangASTContext::GetTypeForDecl(decls[0]);
        if (!type.IsValid())
            return false;
        
        lldb::addr_t class_metadata_ptr = payload0_sp->GetAddressOf();
        if (class_metadata_ptr == LLDB_INVALID_ADDRESS || class_metadata_ptr == 0)
            return false;
        address.SetRawAddress(class_metadata_ptr);
        
        class_type_or_name.SetCompilerType(type.GetPointerType());
        return class_type_or_name.GetCompilerType().IsValid();
    }
    
    if (promise_sp->FulfillKindPromise().hasValue() &&
        (promise_sp->FulfillKindPromise().getValue() == swift::MetadataKind::Struct ||
         promise_sp->FulfillKindPromise().getValue() == swift::MetadataKind::Enum ||
         promise_sp->FulfillKindPromise().getValue() == swift::MetadataKind::Tuple))
    {
        Error error;
        class_type_or_name.SetCompilerType(promise_sp->FulfillTypePromise());
        if (error.Fail())
            return false;
        // the choices made here affect dynamic type resolution
        // for an inline protocol object, e.g.:
        // (P) $R1 = {
        //     (Builtin.RawPointer) payload_data_0 = 0x0000000000000001
        //     (Builtin.RawPointer) payload_data_1 = 0x0000000000000002
        //     (Builtin.RawPointer) payload_data_2 = 0x0000000000000000
        //     (Builtin.RawPointer) instance_type = 0x000000010054c2f8
        //     (Builtin.RawPointer) protocol_witness_0 = 0x000000010054c100
        // }
        // pick &payload_data_0
        // for a pointed-to protocol object, e.g.:
        // (Q) $R2 = {
        //     (Builtin.RawPointer) payload_data_0 = 0x00000001006079b0
        //     (Builtin.RawPointer) payload_data_1 = 0x0000000000000000
        //     (Builtin.RawPointer) payload_data_2 = 0x0000000000000000
        //     (Builtin.RawPointer) instance_type = 0x000000010054c648
        //     (Builtin.RawPointer) protocol_witness_0 = 0x000000010054c7b0
        // }
        // pick the value of payload_data_0
        switch (SwiftASTContext::GetAllocationStrategy(class_type_or_name.GetCompilerType()))
        {
            case SwiftASTContext::TypeAllocationStrategy::eInline:
                // FIXME: we should not have to do this - but we are getting confused w.r.t.
                // frozen-dried vs. live versions of objects, so hack around it for now
                if (in_value.GetValue().GetValueAddressType() == eAddressTypeHost)
                    address.SetRawAddress(in_value.GetValue().GetScalar().ULongLong());
                else
                    address.SetRawAddress(in_value.GetAddressOf());
                return true;
            case SwiftASTContext::TypeAllocationStrategy::ePointer:
                address.SetRawAddress(payload0_sp->GetValueAsUnsigned(0));
                return true;
            default:
                // TODO we don't know how to deal with the dynamic case quite yet
                return false;
        }
    }
    return false;
}

bool
SwiftLanguageRuntime::GetDynamicTypeAndAddress_Promise (ValueObject &in_value,
                                                        MetadataPromiseSP promise_sp,
                                                        lldb::DynamicValueType use_dynamic,
                                                        TypeAndOrName &class_type_or_name,
                                                        Address &address)
{
    if (!promise_sp)
        return false;
    
    CompilerType var_type(in_value.GetCompilerType());
    Error error;
    
    if (!promise_sp->FulfillKindPromise())
        return false;
    
    switch (promise_sp->FulfillKindPromise().getValue())
    {
        case swift::MetadataKind::Class:
        {
            CompilerType dyn_type(promise_sp->FulfillTypePromise());
            if (!dyn_type.IsValid())
                return false;
            class_type_or_name.SetCompilerType(dyn_type);
            lldb::addr_t val_ptr_addr = in_value.GetPointerValue();
            val_ptr_addr = GetProcess()->ReadPointerFromMemory(val_ptr_addr, error);
            address.SetLoadAddress(val_ptr_addr, &m_process->GetTarget());
            return true;
        }
            break;
        case swift::MetadataKind::Struct:
        case swift::MetadataKind::Tuple:
        {
            CompilerType dyn_type(promise_sp->FulfillTypePromise());
            if (!dyn_type.IsValid())
                return false;
            class_type_or_name.SetCompilerType(dyn_type);
            lldb::addr_t val_ptr_addr = in_value.GetPointerValue();
            address.SetLoadAddress(val_ptr_addr, &m_process->GetTarget());
            return true;
        }
            break;
        case swift::MetadataKind::Enum:
        {
            CompilerType dyn_type(promise_sp->FulfillTypePromise());
            if (!dyn_type.IsValid())
                return false;
            class_type_or_name.SetCompilerType(dyn_type);
            lldb::addr_t val_ptr_addr = in_value.GetPointerValue();
            {
                // FIXME: this is a question that should be asked of the metadata
                // but currently the enum nominal type descriptor doesn't know to answer it
                uint32_t num_payload_cases = 0;
                uint32_t num_nopayload_cases = 0;
                if (SwiftASTContext::GetEnumTypeInfo(dyn_type, num_payload_cases, num_nopayload_cases) &&
                    num_payload_cases == 1 &&
                    num_nopayload_cases == 1)
                    val_ptr_addr = GetProcess()->ReadPointerFromMemory(val_ptr_addr, error);
            }
            address.SetLoadAddress(val_ptr_addr, &m_process->GetTarget());
            return true;
        }
            break;
        case swift::MetadataKind::Existential:
        {
            SwiftASTContext *swift_ast_ctx = llvm::dyn_cast_or_null<SwiftASTContext>(var_type.GetTypeSystem());

            CompilerType protocol_type(promise_sp->FulfillTypePromise());
            if (swift_ast_ctx->IsErrorType(protocol_type))
            {
                if (swift_ast_ctx)
                {
                    Error error;
                    // the offset
                    size_t ptr_size = m_process->GetAddressByteSize();
                    size_t metadata_offset = ptr_size + 4 + (ptr_size == 8 ? 4 : 0);
                    metadata_offset += ptr_size + ptr_size + ptr_size;
                    lldb::addr_t archetype_ptr_value = in_value.GetValueAsUnsigned(0);
                    lldb::addr_t base_errortype_ptr = m_process->ReadPointerFromMemory(archetype_ptr_value, error);
                    lldb::addr_t static_metadata_ptrptr = base_errortype_ptr + metadata_offset;
                    lldb::addr_t static_metadata_ptr = m_process->ReadPointerFromMemory(static_metadata_ptrptr, error);
                    MetadataPromiseSP promise_sp(GetMetadataPromise(static_metadata_ptr, swift_ast_ctx));
                    if (promise_sp)
                    {
                        lldb::addr_t load_addr = static_metadata_ptrptr + 2 * ptr_size;
                        if (promise_sp->FulfillKindPromise() &&
                            promise_sp->FulfillKindPromise().getValue() == swift::MetadataKind::Class)
                        {
                            load_addr = m_process->ReadPointerFromMemory(load_addr, error);
                            lldb::addr_t dynamic_metadata_location = m_process->ReadPointerFromMemory(load_addr, error);
                            promise_sp = GetMetadataPromise(dynamic_metadata_location, swift_ast_ctx);
                        }
                        CompilerType clang_type(promise_sp->FulfillTypePromise());
                        if (clang_type.IsValid() && load_addr != 0 && load_addr != LLDB_INVALID_ADDRESS)
                        {
                            class_type_or_name.SetCompilerType(clang_type);
                            address.SetLoadAddress(load_addr, &m_process->GetTarget());
                            return true;
                        }
                    }
                }
            }
            else
            {
                Error error;
                lldb::addr_t ptr_to_instance_type = in_value.GetValueAsUnsigned(0) + (3 * m_process->GetAddressByteSize());
                lldb::addr_t metadata_of_impl_addr = m_process->ReadPointerFromMemory(ptr_to_instance_type, error);
                if (error.Fail() || metadata_of_impl_addr == 0 || metadata_of_impl_addr == LLDB_INVALID_ADDRESS)
                    return false;
                MetadataPromiseSP promise_of_impl_sp(GetMetadataPromise(metadata_of_impl_addr, swift_ast_ctx));
                if (GetDynamicTypeAndAddress_Promise(in_value, promise_of_impl_sp, use_dynamic, class_type_or_name, address))
                {
                    lldb::addr_t load_addr = in_value.GetValueAsUnsigned(0);
                    if (promise_of_impl_sp->FulfillKindPromise() &&
                        promise_of_impl_sp->FulfillKindPromise().getValue() == swift::MetadataKind::Class)
                    {
                        load_addr = m_process->ReadPointerFromMemory(load_addr, error);
                        if (error.Fail() || load_addr == 0 || load_addr == LLDB_INVALID_ADDRESS)
                            return false;
                    }
                    else if (promise_of_impl_sp->FulfillKindPromise() &&
                             (promise_of_impl_sp->FulfillKindPromise().getValue() == swift::MetadataKind::Enum ||
                              promise_of_impl_sp->FulfillKindPromise().getValue() == swift::MetadataKind::Struct))
                    {}
                    else
                        lldbassert(false && "class, enum and struct are the only protocol implementor types I know about");
                    address.SetLoadAddress(load_addr, &m_process->GetTarget());
                    return true;
                }
            }
        }
            break;
        default:
            break;
    }
    
    return false;
}

SwiftLanguageRuntime::MetadataPromiseSP
SwiftLanguageRuntime::GetPromiseForTypeNameAndFrame (const char* type_name,
                                                     StackFrame* frame)
{
    if (!frame || !type_name || !type_name[0])
        return nullptr;
    
    SwiftASTContext *swift_ast_ctx = nullptr;
    const SymbolContext& sc(frame->GetSymbolContext(eSymbolContextFunction));
    if (sc.function)
        swift_ast_ctx = llvm::dyn_cast_or_null<SwiftASTContext>(sc.function->GetCompilerType().GetTypeSystem());

    StreamString type_metadata_ptr_var_name;
    type_metadata_ptr_var_name.Printf("$swift.type.%s",type_name);
    VariableList* var_list = frame->GetVariableList(false);
    if (!var_list)
        return nullptr;
    
    VariableSP var_sp(var_list->FindVariable(ConstString(type_metadata_ptr_var_name.GetData())));
    if (!var_sp)
        return nullptr;
    
    ValueObjectSP metadata_ptr_var_sp(frame->GetValueObjectForFrameVariable(var_sp, lldb::eNoDynamicValues));
    if (!metadata_ptr_var_sp || metadata_ptr_var_sp->UpdateValueIfNeeded() == false)
        return nullptr;
    
    lldb::addr_t metadata_location(metadata_ptr_var_sp->GetValueAsUnsigned(0));
    if (metadata_location == 0 || metadata_location == LLDB_INVALID_ADDRESS)
        return nullptr;
    
    return GetMetadataPromise(metadata_location, swift_ast_ctx);
}

CompilerType
SwiftLanguageRuntime::DoArchetypeBindingForType (StackFrame& stack_frame,
                                                 CompilerType base_type,
                                                 SwiftASTContext *ast_context)
{
    if (base_type.GetTypeInfo() & lldb::eTypeIsSwift)
    {
        if (!ast_context)
            ast_context = llvm::dyn_cast_or_null<SwiftASTContext>(base_type.GetTypeSystem());

        if (ast_context)
        {
            swift::Type target_swift_type(GetSwiftType(base_type));

            target_swift_type = target_swift_type.transform([this, &stack_frame, ast_context] (swift::Type candidate_type) -> swift::Type {
                if (swift::ArchetypeType *candidate_archetype = llvm::dyn_cast_or_null<swift::ArchetypeType>(candidate_type.getPointer()))
                {
                    llvm::StringRef candidate_name = candidate_archetype->getName().str();
                    
                    CompilerType concrete_type = this->GetConcreteType(&stack_frame, ConstString(candidate_name));
                    Error import_error;
                    CompilerType target_concrete_type = ast_context->ImportType(concrete_type, import_error);
                    
                    if (target_concrete_type.IsValid())
                        return swift::Type(GetSwiftType(target_concrete_type));
                    else
                        return candidate_type;
                }
                else
                    return candidate_type;
            });
            
            return CompilerType(ast_context->GetASTContext(), target_swift_type.getPointer());
        }
    }
    return base_type;
}

bool
SwiftLanguageRuntime::GetDynamicTypeAndAddress_Archetype (ValueObject &in_value,
                                                          lldb::DynamicValueType use_dynamic,
                                                          TypeAndOrName &class_type_or_name,
                                                          Address &address)
{
    const char* type_name(in_value.GetTypeName().GetCString());
    StackFrame* frame(in_value.GetFrameSP().get());
    MetadataPromiseSP promise_sp(GetPromiseForTypeNameAndFrame(type_name,frame));
    if (!promise_sp)
        return false;
    if (!GetDynamicTypeAndAddress_Promise(in_value, promise_sp, use_dynamic, class_type_or_name, address))
        return false;
    if (promise_sp->FulfillKindPromise() &&
        promise_sp->FulfillKindPromise().getValue() == swift::MetadataKind::Class)
    {
        // when an archetype represents a class, it will represent the static type of the class
        // but the dynamic type might be different
        Error error;
        lldb::addr_t addr_of_meta = address.GetLoadAddress(&m_process->GetTarget());
        addr_of_meta = m_process->ReadPointerFromMemory(addr_of_meta, error);
        if (addr_of_meta == LLDB_INVALID_ADDRESS || addr_of_meta == 0 || error.Fail())
            return true; // my gut says we should fail here, but we seemed to be on a good track before..
        MetadataPromiseSP actual_type_promise(GetMetadataPromise(addr_of_meta));
        if (actual_type_promise && actual_type_promise.get() != promise_sp.get())
        {
            CompilerType static_type(class_type_or_name.GetCompilerType());
            class_type_or_name.SetCompilerType(actual_type_promise->FulfillTypePromise());
            if (error.Fail() || class_type_or_name.GetCompilerType().IsValid() == false)
                class_type_or_name.SetCompilerType(static_type);
        }
    }
    return true;
}

bool
SwiftLanguageRuntime::GetDynamicTypeAndAddress_Tuple (ValueObject &in_value,
                                                      lldb::DynamicValueType use_dynamic,
                                                      TypeAndOrName &class_type_or_name,
                                                      Address &address)
{
    std::vector<CompilerType> dyn_types;
    
    for (size_t idx = 0;
         idx < in_value.GetNumChildren();
         idx++)
    {
        ValueObjectSP child_sp(in_value.GetChildAtIndex(idx, true));
        TypeAndOrName type_and_or_name;
        Address address;
        Value::ValueType value_type;
        if (GetDynamicTypeAndAddress(*child_sp.get(), use_dynamic, type_and_or_name, address, value_type) == false)
            dyn_types.push_back(child_sp->GetCompilerType());
        else
            dyn_types.push_back(type_and_or_name.GetCompilerType());
    }
    
    SwiftASTContext *swift_ast_ctx = llvm::dyn_cast_or_null<SwiftASTContext>(in_value.GetCompilerType().GetTypeSystem());
    
    CompilerType dyn_tuple_type(swift_ast_ctx->CreateTupleType(dyn_types));
    
    class_type_or_name.SetCompilerType(dyn_tuple_type);
    lldb::addr_t tuple_address = in_value.GetPointerValue();
    Error error;
    tuple_address = m_process->ReadPointerFromMemory(tuple_address, error);
    if (error.Fail() || tuple_address == 0 || tuple_address == LLDB_INVALID_ADDRESS)
        return false;
    
    address.SetLoadAddress(tuple_address, in_value.GetTargetSP().get());
    
    return true;
}

bool
SwiftLanguageRuntime::GetDynamicTypeAndAddress_Struct (ValueObject &in_value,
                                                       lldb::DynamicValueType use_dynamic,
                                                       TypeAndOrName &class_type_or_name,
                                                       Address &address)
{
    // struct can't inherit from each other, but they can be generic, in which case
    // we need to turn MyStruct<U> into MyStruct<$swift.type.U>
    std::vector<CompilerType> generic_args;
    
    lldb_private::StackFrame* frame = in_value.GetExecutionContextRef().GetFrameSP().get();
    
    if (!frame)
        return false;
    
    // this will be a BoundGenericStruct, bound to archetypes
    CompilerType struct_type(in_value.GetCompilerType());
    
    SwiftASTContext *swift_ast_ctx = llvm::dyn_cast_or_null<SwiftASTContext>(struct_type.GetTypeSystem());
    
    size_t num_type_args = struct_type.GetNumTemplateArguments();
    
    for (size_t i = 0; i < num_type_args; i++)
    {
        lldb::TemplateArgumentKind kind;
        CompilerType type_arg;
        type_arg = struct_type.GetTemplateArgument(i, kind);
        if (kind != lldb::eTemplateArgumentKindType || type_arg.IsValid() == false)
            return false;
        CompilerType resolved_type_arg(DoArchetypeBindingForType(*frame, type_arg, swift_ast_ctx));
        if (!resolved_type_arg)
            return false;
        generic_args.push_back(resolved_type_arg);
    }
    
    class_type_or_name.SetCompilerType(swift_ast_ctx->BindGenericType(struct_type, generic_args, true));
    
    lldb::addr_t struct_address = in_value.GetPointerValue();
    if (0 == struct_address || LLDB_INVALID_ADDRESS == struct_address)
        struct_address = in_value.GetAddressOf(true, nullptr);
    if (0 == struct_address || LLDB_INVALID_ADDRESS == struct_address)
    {
        if (false == SwiftASTContext::IsPossibleZeroSizeType(class_type_or_name.GetCompilerType()))
            return false;
    }
    
    address.SetLoadAddress(struct_address, in_value.GetTargetSP().get());
    return true;
}

bool
SwiftLanguageRuntime::GetDynamicTypeAndAddress_Enum (ValueObject &in_value,
                                                     lldb::DynamicValueType use_dynamic,
                                                     TypeAndOrName &class_type_or_name,
                                                     Address &address)
{
    // enums can't inherit from each other, but they can be generic, in which case
    // we need to turn MyEnum<U> into MyEnum<$swift.type.U>
    std::vector<CompilerType> generic_args;
    
    lldb_private::StackFrame* frame = in_value.GetExecutionContextRef().GetFrameSP().get();
    
    if (!frame)
        return false;
    
    // this will be a BoundGenericEnum, bound to archetypes
    CompilerType enum_type(in_value.GetCompilerType());
    
    SwiftASTContext *swift_ast_ctx = llvm::dyn_cast_or_null<SwiftASTContext>(enum_type.GetTypeSystem());
    
    size_t num_type_args = enum_type.GetNumTemplateArguments();
    for (size_t i = 0; i < num_type_args; i++)
    {
        lldb::TemplateArgumentKind kind;
        CompilerType type_arg;
        type_arg = enum_type.GetTemplateArgument(i, kind);
        if (kind != lldb::eTemplateArgumentKindType || type_arg.IsValid() == false)
            return false;
        CompilerType resolved_type_arg(DoArchetypeBindingForType(*frame, type_arg, swift_ast_ctx));
        if (!resolved_type_arg)
            return false;
        generic_args.push_back(resolved_type_arg);
    }
    
    class_type_or_name.SetCompilerType(swift_ast_ctx->BindGenericType(enum_type, generic_args, true));

    lldb::addr_t enum_address = in_value.GetPointerValue();
    if (0 == enum_address || LLDB_INVALID_ADDRESS == enum_address)
        enum_address = in_value.GetAddressOf(true, nullptr);
    if (0 == enum_address || LLDB_INVALID_ADDRESS == enum_address)
    {
        if (false == SwiftASTContext::IsPossibleZeroSizeType(class_type_or_name.GetCompilerType()))
            return false;
    }

    address.SetLoadAddress(enum_address, in_value.GetTargetSP().get());
    return true;
}

bool
SwiftLanguageRuntime::GetDynamicTypeAndAddress_IndirectEnumCase (ValueObject &in_value,
                                                                 lldb::DynamicValueType use_dynamic,
                                                                 TypeAndOrName &class_type_or_name,
                                                                 Address &address)
{
    static ConstString g_offset("offset");
    
    DataExtractor data;
    Error error;
    if (in_value.GetParent() &&
        in_value.GetParent()->GetData(data, error) &&
        error.Success())
    {
        bool has_payload;
        bool is_indirect;
        CompilerType payload_type;
        if (SwiftASTContext::GetSelectedEnumCase(in_value.GetParent()->GetCompilerType(),
                                                 data,
                                                 nullptr,
                                                 &has_payload,
                                                 &payload_type,
                                                 &is_indirect))
        {
            if (has_payload && is_indirect && payload_type)
                class_type_or_name.SetCompilerType(payload_type);
            lldb::addr_t box_addr = in_value.GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
            if (box_addr != LLDB_INVALID_ADDRESS)
            {
                box_addr = MaskMaybeBridgedPointer(box_addr);
                lldb::addr_t box_location = m_process->ReadPointerFromMemory(box_addr, error);
                if (box_location != LLDB_INVALID_ADDRESS)
                {
                    box_location = MaskMaybeBridgedPointer(box_location);
                    ProcessStructReader reader(m_process, box_location, GetBoxMetadataType());
                    uint32_t offset = reader.GetField<uint32_t>(g_offset);
                    lldb::addr_t box_value = box_addr + offset;
                    
                    // try to read one byte at the box value
                    m_process->ReadUnsignedIntegerFromMemory(box_value, 1, 0, error);
                    if (error.Fail()) // and if that fails, then we're off in no man's land
                        return false;
                    
                    Flags type_info(payload_type.GetTypeInfo());
                    if (type_info.AllSet(eTypeIsSwift | eTypeIsClass))
                    {
                        lldb::addr_t old_box_value = box_value;
                        box_value = m_process->ReadPointerFromMemory(box_value, error);
                        if (box_value != LLDB_INVALID_ADDRESS)
                        {
                            DataExtractor data(&box_value, m_process->GetAddressByteSize(), m_process->GetByteOrder(), m_process->GetAddressByteSize());
                            ValueObjectSP valobj_sp(ValueObject::CreateValueObjectFromData("_", data, *m_process, payload_type));
                            if (valobj_sp)
                            {
                                Value::ValueType value_type;
                                if (GetDynamicTypeAndAddress(*valobj_sp, use_dynamic, class_type_or_name, address, value_type))
                                {
                                    address.SetRawAddress(old_box_value);
                                    return true;
                                }
                            }
                        }
                    }
                    else if (type_info.AllSet(eTypeIsSwift | eTypeIsProtocol))
                    {
                        SwiftASTContext::ProtocolInfo protocol_info;
                        if (SwiftASTContext::GetProtocolTypeInfo(payload_type,
                                                                 protocol_info))
                        {
                            auto ptr_size = m_process->GetAddressByteSize();
                            std::vector<uint8_t> buffer(ptr_size * protocol_info.m_num_storage_words,
                                                        0);
                            for (uint32_t idx = 0;
                                 idx < protocol_info.m_num_storage_words;
                                 idx++)
                            {
                                lldb::addr_t word = m_process->ReadUnsignedIntegerFromMemory(box_value+idx*ptr_size, ptr_size, 0, error);
                                if (error.Fail())
                                    return false;
                                memcpy(&buffer[idx*ptr_size], &word, ptr_size);
                            }
                            DataExtractor data(&buffer[0], buffer.size(), m_process->GetByteOrder(), m_process->GetAddressByteSize());
                            ValueObjectSP valobj_sp(ValueObject::CreateValueObjectFromData("_", data, *m_process, payload_type));
                            if (valobj_sp)
                            {
                                Value::ValueType value_type;
                                if (GetDynamicTypeAndAddress(*valobj_sp, use_dynamic, class_type_or_name, address, value_type))
                                {
                                    address.SetRawAddress(box_value);
                                    return true;
                                }
                            }
                        }
                    }
                    else
                    {
                        // this is most likely a statically known type
                        address.SetLoadAddress(box_value, &m_process->GetTarget());
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

// Dynamic type resolution tends to want to generate scalar data - but there are caveats
// Per original comment here
// "Our address is the location of the dynamic type stored in memory.  It isn't a load address,
//  because we aren't pointing to the LOCATION that stores the pointer to us, we're pointing to us..."
// See inlined comments for exceptions to this general rule.
Value::ValueType
SwiftLanguageRuntime::GetValueType (Value::ValueType static_value_type,
                                    const CompilerType& static_type,
                                    const CompilerType& dynamic_type,
                                    bool is_indirect_enum_case)
{
    Flags static_type_flags(static_type.GetTypeInfo());
    Flags dynamic_type_flags(dynamic_type.GetTypeInfo());
    
    if (dynamic_type_flags.AllSet(eTypeIsSwift))
    {
        // for a protocol object where does the dynamic data live if the target object is a struct? (for a class, it's easy)
        if (static_type_flags.AllSet(eTypeIsSwift | eTypeIsProtocol) &&
            dynamic_type_flags.AnySet(eTypeIsStructUnion | eTypeIsEnumeration))
        {
            SwiftASTContext *swift_ast_ctx = llvm::dyn_cast_or_null<SwiftASTContext>(static_type.GetTypeSystem());
            
            if (swift_ast_ctx && swift_ast_ctx->IsErrorType(static_type))
            {
                // ErrorType values are always a pointer
                return Value::eValueTypeLoadAddress;
            }
            
            switch (SwiftASTContext::GetAllocationStrategy(dynamic_type))
            {
                case SwiftASTContext::TypeAllocationStrategy::eDynamic:
                case SwiftASTContext::TypeAllocationStrategy::eUnknown:
                    break;
                case SwiftASTContext::TypeAllocationStrategy::eInline: // inline data; same as the static data
                    return static_value_type;
                case SwiftASTContext::TypeAllocationStrategy::ePointer: // pointed-to; in the target
                    return Value::eValueTypeLoadAddress;
            }
        }
        if (static_type_flags.AllSet(eTypeIsSwift | eTypeIsArchetype))
        {
            // if I am handling a non-pointer Swift type obtained from an archetype, then the runtime vends the location
            // of the object, not the object per se (since the object is not a pointer itself, this is way easier to achieve)
            // hence, it's a load address, not a scalar containing a pointer as for ObjC classes
            if (dynamic_type_flags.AllClear(eTypeIsPointer | eTypeIsReference | eTypeInstanceIsPointer))
                return Value::eValueTypeLoadAddress;
        }

        if (static_type_flags.AllSet(eTypeIsSwift | eTypeIsPointer) && static_type_flags.AllClear(eTypeIsArchetype))
        {
            if (is_indirect_enum_case || static_type_flags.AllClear(eTypeIsBuiltIn))
                return Value::eValueTypeLoadAddress;
        }
    }
    
    if (static_type_flags.AllSet(eTypeIsSwift) &&
        dynamic_type_flags.AllSet(eTypeIsSwift) &&
        dynamic_type_flags.AllClear(eTypeIsPointer | eTypeInstanceIsPointer))
        return static_value_type;
    else
        return Value::eValueTypeScalar;
}

static bool
IsIndirectEnumCase (ValueObject& valobj)
{
    return (valobj.GetLanguageFlags() & SwiftASTContext::LanguageFlags::eIsIndirectEnumCase) == SwiftASTContext::LanguageFlags::eIsIndirectEnumCase;
}

bool
SwiftLanguageRuntime::GetDynamicTypeAndAddress (ValueObject &in_value,
                                                lldb::DynamicValueType use_dynamic,
                                                TypeAndOrName &class_type_or_name,
                                                Address &address,
                                                Value::ValueType &value_type)
{
    class_type_or_name.Clear();

    if (SwiftASTContext *swift_ast = llvm::dyn_cast_or_null<SwiftASTContext>(in_value.GetCompilerType().GetTypeSystem()))
    {
        if (swift_ast->HasFatalErrors() || !swift_ast->GetClangImporter())
        {
            return false;
        }
    }
    else
    {
        return false;
    }
    
    if (use_dynamic == lldb::eNoDynamicValues || !CouldHaveDynamicValue(in_value))
        return false;
    
    bool success = false;
    const bool is_indirect_enum_case = IsIndirectEnumCase(in_value);
    
    if (is_indirect_enum_case)
        success = GetDynamicTypeAndAddress_IndirectEnumCase(in_value,use_dynamic,class_type_or_name,address);
    else
    {
        CompilerType var_type(in_value.GetCompilerType());
        Flags type_info(var_type.GetTypeInfo());
        if (type_info.AnySet(eTypeIsSwift))
        {
            if (type_info.AnySet(eTypeIsClass))
                success = GetDynamicTypeAndAddress_Class(in_value,use_dynamic,class_type_or_name,address);
            else if (type_info.AnySet(eTypeIsEnumeration))
                success = GetDynamicTypeAndAddress_Enum(in_value,use_dynamic,class_type_or_name,address);
            else if (type_info.AnySet(eTypeIsProtocol))
                success = GetDynamicTypeAndAddress_Protocol(in_value,use_dynamic,class_type_or_name,address);
            else if (type_info.AnySet(eTypeIsArchetype))
                success = GetDynamicTypeAndAddress_Archetype(in_value,use_dynamic,class_type_or_name,address);
            else if (type_info.AnySet(eTypeIsTuple))
                success = GetDynamicTypeAndAddress_Tuple(in_value,use_dynamic,class_type_or_name,address);
            else if (type_info.AnySet(eTypeIsStructUnion))
                success = GetDynamicTypeAndAddress_Struct(in_value,use_dynamic,class_type_or_name,address);
            else if (type_info.AllSet(eTypeIsBuiltIn | eTypeIsPointer | eTypeHasValue))
                success = GetDynamicTypeAndAddress_Class(in_value,use_dynamic,class_type_or_name,address);
        }
    }
    
    if (success)
    {
        value_type = GetValueType(in_value.GetValue().GetValueType(),
                                  in_value.GetCompilerType(),
                                  class_type_or_name.GetCompilerType(),
                                  is_indirect_enum_case);
    }
    return success;
}

TypeAndOrName
SwiftLanguageRuntime::FixUpDynamicType(const TypeAndOrName& type_and_or_name,
                                       ValueObject& static_value)
{
    TypeAndOrName ret(type_and_or_name);
    bool should_be_made_into_ref = false;
    bool should_be_made_into_ptr = false;
    Flags type_flags(static_value.GetCompilerType().GetTypeInfo());
    Flags type_andor_name_flags(type_and_or_name.GetCompilerType().GetTypeInfo());
    
    // if the static type is a pointer or reference, so should the dynamic type
    // caveat: if the static type is a Swift class instance, the dynamic type
    // could either be a Swift type (no need to change anything), or an ObjC type
    // in which case it needs to be made into a pointer
    if (type_flags.AnySet(eTypeIsPointer))
        should_be_made_into_ptr = (type_flags.AllClear(eTypeIsArchetype | eTypeIsBuiltIn) && !IsIndirectEnumCase(static_value));
    else if (type_flags.AnySet(eTypeInstanceIsPointer))
        should_be_made_into_ptr = !type_andor_name_flags.AllSet(eTypeIsSwift);
    else if (type_flags.AnySet(eTypeIsReference))
        should_be_made_into_ref = true;
    else if (type_flags.AllSet(eTypeIsSwift | eTypeIsProtocol))
        should_be_made_into_ptr = type_and_or_name.GetCompilerType().IsRuntimeGeneratedType() && !type_and_or_name.GetCompilerType().IsPointerType();
    
    if (type_and_or_name.HasType())
    {
        // The type will always be the type of the dynamic object.  If our parent's type was a pointer,
        // then our type should be a pointer to the type of the dynamic object.  If a reference, then the original type
        // should be okay...
        CompilerType orig_type = type_and_or_name.GetCompilerType();
        CompilerType corrected_type = orig_type;
        if (should_be_made_into_ptr)
            corrected_type = orig_type.GetPointerType ();
        else if (should_be_made_into_ref)
            corrected_type = orig_type.GetLValueReferenceType ();
        ret.SetCompilerType(corrected_type);
    }
    else /*if (m_dynamic_type_info.HasName())*/
    {
        // If we are here we need to adjust our dynamic type name to include the correct & or * symbol
        std::string corrected_name (type_and_or_name.GetName().GetCString());
        if (should_be_made_into_ptr)
            corrected_name.append(" *");
        else if (should_be_made_into_ref)
            corrected_name.append(" &");
        // the parent type should be a correctly pointer'ed or referenc'ed type
        ret.SetCompilerType(static_value.GetCompilerType());
        ret.SetName(corrected_name.c_str());
    }
    return ret;
}

bool
SwiftLanguageRuntime::IsRuntimeSupportValue (ValueObject& valobj)
{
    static llvm::StringRef g_dollar_swift_type("$swift.type.");
    ConstString valobj_name(valobj.GetName());
    llvm::StringRef valobj_name_sr(valobj_name.GetStringRef());
    if (valobj_name_sr.startswith(g_dollar_swift_type))
        return true;
    static llvm::StringRef g_globalinit("globalinit_");
    static ConstString g_builtin_word("Builtin.Word");
    static ConstString g__argc("_argc");
    static ConstString g__unsafeArgv("_unsafeArgv");
    static ConstString g_dollar_error("$error");
    static ConstString g_tmp_closure ("$tmpClosure");

    ConstString valobj_type_name(valobj.GetTypeName());
    if (valobj_name_sr.startswith(g_globalinit) && valobj_type_name == g_builtin_word)
        return true;
    if (valobj_name == g__argc
       || valobj_name == g__unsafeArgv
       || valobj_name == g_dollar_error
       || valobj_name == g_tmp_closure)
        return true;
    return false;
}

bool
SwiftLanguageRuntime::CouldHaveDynamicValue (ValueObject &in_value)
{
    //if (in_value.IsDynamic())
    //    return false;
    if (IsIndirectEnumCase(in_value))
        return true;
    CompilerType var_type(in_value.GetCompilerType());
    Flags var_type_flags(var_type.GetTypeInfo());
    if (var_type_flags.AllSet(eTypeIsSwift | eTypeInstanceIsPointer))
    {
        // Swift class instances are actually pointers, but base class instances
        // are inlined at offset 0 in the class data. If we just let base classes
        // be dynamic, it would cause an infinite recursion. So we would usually disable it
        // But if the base class is a generic type we still need to bind it, and that is
        // a good job for dynamic types to perform
        if (in_value.IsBaseClass())
        {
            CompilerType base_type(in_value.GetCompilerType());
            if (SwiftASTContext::IsFullyRealized(base_type))
                return false;
        }
        return true;
    }
    return var_type.IsPossibleDynamicType(nullptr, false, false, true);
}

CompilerType
SwiftLanguageRuntime::GetConcreteType (ExecutionContextScope *exe_scope, ConstString abstract_type_name)
{
    if (!exe_scope)
        return CompilerType();

    StackFrame* frame(exe_scope->CalculateStackFrame().get());
    if (!frame)
        return CompilerType();

    MetadataPromiseSP promise_sp(GetPromiseForTypeNameAndFrame(abstract_type_name.GetCString(), frame));
    if (!promise_sp)
        return CompilerType();
    
    return promise_sp->FulfillTypePromise();
}

bool
SwiftLanguageRuntime::GetTargetOfPartialApply (CompileUnit &cu, ConstString &apply_name, SymbolContext &sc)
{
    ModuleSP module = cu.GetModule();
    SymbolContextList sc_list;
    const char *apply_name_str = apply_name.AsCString();
    if (apply_name_str[0] == '_'
        && apply_name_str[1] == 'T'
        && apply_name_str[2] == 'P'
        && apply_name_str[3] == 'A'
        && apply_name_str[4] == '_')
    {
        ConstString apply_target(apply_name_str + 5);
        size_t num_symbols = module->FindFunctions(apply_target, NULL, eFunctionNameTypeFull, true, false, false, sc_list);
        if (num_symbols == 0)
            return false;
        
        size_t num_found = 0;
        for (size_t i = 0; i < num_symbols; i++)
        {
            SymbolContext tmp_sc;
            if (sc_list.GetContextAtIndex(i, tmp_sc))
            {
                if (tmp_sc.comp_unit && tmp_sc.comp_unit == &cu)
                {
                    sc = tmp_sc;
                    num_found++;
                }
                else if (cu.GetModule() == tmp_sc.module_sp)
                {
                    sc = tmp_sc;
                    num_found++;
                }
            }
        }
        if (num_found == 1)
            return true;
        else
        {
            sc.Clear(false);
            return false;
        }
    }
    else
    {
        return false;
    }
}

bool SwiftLanguageRuntime::IsSymbolARuntimeThunk(const Symbol &symbol)
{
    
    const char *symbol_name = symbol.GetMangled().GetMangledName().AsCString();
    if (symbol_name)
    {
        if (symbol_name[0] == '_'
            && symbol_name[1] == 'T')
        {
            char char_2 = symbol_name[2];
            if (char_2 == 'T')
            {
                char char_3 = symbol_name[3];
                if (char_3 == 'o')
                    return true;
                else if (char_3 == 'W' || char_3 == 'R' || char_3 == 'r')
                    return true;
            }
            else if (char_2 == 'P' && symbol_name[3] == 'A' && symbol_name[4] == '_')
            {
                return true;
            }
        }
        return false;
    }
    else
        return false;
}

lldb::ThreadPlanSP
SwiftLanguageRuntime::GetStepThroughTrampolinePlan (Thread &thread, bool stop_others)
{
    // Here are the trampolines we have at present.
    // 1) The thunks from protocol invocations to the call in the actual object implementing the protocol.
    // 2) Thunks for going from Swift ObjC classes to their actual method invocations
    // 3) Thunks that retain captured objects in closure invocations.
    
    ThreadPlanSP new_thread_plan_sp;
    
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    StackFrameSP stack_sp = thread.GetStackFrameAtIndex(0);
    if (stack_sp)
    {
        SymbolContext sc = stack_sp->GetSymbolContext (eSymbolContextEverything);
        Symbol *symbol = sc.symbol;
        
        // Note, I don't really need to consult IsSymbolARuntimeThunk here, but it is fast to do and
        // keeps this list and the one in IsSymbolARuntimeThunk in sync.
        if (symbol && IsSymbolARuntimeThunk(*symbol))
        {
            // Only do this if you are at the beginning of the thunk function:
            lldb::addr_t cur_addr = thread.GetRegisterContext()->GetPC();
            lldb::addr_t symbol_addr = symbol->GetAddress().GetLoadAddress (&thread.GetProcess()->GetTarget());
            
            if (symbol_addr != cur_addr)
                return new_thread_plan_sp;
            
            Address target_address;
            ConstString symbol_mangled_name = symbol->GetMangled().GetMangledName();
            const char *symbol_name = symbol_mangled_name.AsCString();
            if (symbol_name && symbol_name[0] == '_'
                && symbol_name[1] == 'T')
            {
                if (symbol_name[2] == 'T'
                    && symbol_name[3] == 'o')
                {
                    // Thunks of type 2 always have mangled names of the form _TTo<Target Mangled Name Less _T>
                    // so we can just find that function and run to it:
                    std::string target_name(symbol_name+4);
                    target_name.insert (0, "_T");
                    ModuleList modules = thread.GetProcess()->GetTarget().GetImages();
                    SymbolContextList sc_list;
                    modules.FindFunctionSymbols (ConstString (target_name), eFunctionNameTypeFull, sc_list);
                    if (sc_list.GetSize() == 1)
                    {
                        SymbolContext sc;
                        sc_list.GetContextAtIndex(0, sc);
                        
                        if (sc.symbol)
                            target_address = sc.symbol->GetAddress();
                    }
                }
                else if (symbol_name[2] == 'T'
                         && symbol_name[3] == 'W')
                {
                    // The TTW symbols encode the protocol conformance requirements and it is possible to go to
                    // the AST and get it to replay the logic that it used to determine what to dispatch to.
                    // But that ties us too closely to the logic of the compiler, and these thunks are quite
                    // simple, they just do a little retaining, and then call the correct function.
                    // So for simplicity's sake, I'm just going to get the base name of the function
                    // this protocol thunk is preparing to call, then step into through the thunk, stopping if I end up
                    // in a frame with that function name.
                    
                    swift::Demangle::NodePointer demangled_nodes = swift::Demangle::demangleSymbolAsNode(symbol_name, symbol_mangled_name.GetLength());

                    // Now find the ProtocolWitness node in the demangled result.
                    
                    swift::Demangle::NodePointer witness_node = demangled_nodes;
                    bool found_witness_node = false;
                    while (witness_node)
                    {
                        if (witness_node->getKind() == swift::Demangle::Node::Kind::ProtocolWitness)
                        {
                            found_witness_node = true;
                            break;
                        }
                        witness_node = witness_node->getFirstChild();
                    }
                    if (!found_witness_node)
                    {
                        if (log)
                            log->Printf ("Stepped into witness thunk \"%s\" but could not find the ProtocolWitness node in the demangled nodes.",
                                         symbol_name);
                        return new_thread_plan_sp;
                    }
                    
                    size_t num_children = witness_node->getNumChildren();
                    if (num_children < 2)
                    {
                        if (log)
                            log->Printf ("Stepped into witness thunk \"%s\" but the ProtocolWitness node doesn't have enough nodes.",
                                         symbol_name);
                        return new_thread_plan_sp;
                    }
                    
                    swift::Demangle::NodePointer function_node = witness_node->getChild(1);
                    if (function_node == nullptr || function_node->getKind() != swift::Demangle::Node::Kind::Function)
                    {
                        if (log)
                            log->Printf ("Stepped into witness thunk \"%s\" but could not find the function in the ProtocolWitness node.",
                                         symbol_name);
                        return new_thread_plan_sp;
                    }
                    
                    // Okay, now find the name of this function.
                    num_children = function_node->getNumChildren();
                    swift::Demangle::NodePointer name_node(nullptr);
                    for (size_t i = 0; i < num_children; i++)
                    {
                        if (function_node->getChild(i)->getKind() == swift::Demangle::Node::Kind::Identifier)
                        {
                            name_node = function_node->getChild(i);
                            break;
                        }
                    }
                    
                    if (!name_node)
                    {
                        if (log)
                            log->Printf ("Stepped into witness thunk \"%s\" but could not find the Function name in the function node.",
                                         symbol_name);
                        return new_thread_plan_sp;
                    }
                    
                    std::string function_name(name_node->getText());
                    if (function_name.empty())
                    {
                        if (log)
                            log->Printf ("Stepped into witness thunk \"%s\" but the Function name was empty.",
                                         symbol_name);
                        return new_thread_plan_sp;
                    }
                    
                    // We have to get the address range of the thunk symbol, and make a "step through range stepping in"
                    AddressRange sym_addr_range (sc.symbol->GetAddress(), sc.symbol->GetByteSize());
                    new_thread_plan_sp.reset(new ThreadPlanStepInRange(thread,
                                                                       sym_addr_range,
                                                                       sc,
                                                                       function_name.c_str(),
                                                                       eOnlyDuringStepping,
                                                                       eLazyBoolNo,
                                                                       eLazyBoolNo));
                    return new_thread_plan_sp;
                    
                }
                else if (symbol_name[2] == 'T'
                         &&  (symbol_name[3] == 'R' || symbol_name[3] == 'r'))
                {
                    // The TTR and TTr symbols are "reabstraction thunks.  I have no idea how to figure out what
                    // their targets are other than to just step through them in the hopes I get somewhere interesting.
                    if (log)
                        log->Printf ("Stepping through reabstraction thunk: %s", symbol_name);
                    AddressRange sym_addr_range (sc.symbol->GetAddress(), sc.symbol->GetByteSize());
                    new_thread_plan_sp.reset(new ThreadPlanStepInRange(thread,
                                                                       sym_addr_range,
                                                                       sc,
                                                                       nullptr,
                                                                       eOnlyDuringStepping,
                                                                       eLazyBoolNo,
                                                                       eLazyBoolNo));
                    return new_thread_plan_sp;
                }
                else if (symbol_name[2] == 'P'
                         && symbol_name[3] == 'A'
                         && symbol_name[4] == '_')
                {
                    // This is a closure thunk.  The actual closure call will follow this call, but the function to
                    // be called is encoded in the name of the thunk.
                    if (sc.comp_unit)
                    {
                        SymbolContext target_ctx;
                        if (GetTargetOfPartialApply (*sc.comp_unit, symbol_mangled_name, target_ctx))
                        {
                            if (target_ctx.symbol)
                                target_address = target_ctx.symbol->GetAddress();
                            else if (target_ctx.function)
                                target_address = target_ctx.function->GetAddressRange().GetBaseAddress();
                        }
                    }
                }
                else
                {
                    // See if we can step through allocating init constructors:
                    ConstString demangled_name = symbol->GetMangled().GetDemangledName(lldb::eLanguageTypeSwift);
                    const char *demangled_str = demangled_name.AsCString();
                    if (strstr ("__allocating_init", demangled_str) != NULL)
                    {
                        if (log)
                            log->Printf ("Stepping through __allocating_init for symbol: %s", symbol_name);
                        
                        AddressRange sym_addr_range (sc.symbol->GetAddress(), sc.symbol->GetByteSize());
                        new_thread_plan_sp.reset(new ThreadPlanStepInRange(thread,
                                                                           sym_addr_range,
                                                                           sc,
                                                                           nullptr,
                                                                           eOnlyDuringStepping,
                                                                           eLazyBoolNo,
                                                                           eLazyBoolNo));
                        return new_thread_plan_sp;
                    }
                }
                if (target_address.IsValid())
                {
                    new_thread_plan_sp.reset(new ThreadPlanRunToAddress(thread, target_address, stop_others));
                }
            }
        }
    }
    return new_thread_plan_sp;
}

void
SwiftLanguageRuntime::FindFunctionPointersInCall (
        StackFrame &frame,
        std::vector<Address> &addresses,
        bool debug_only,
        bool resolve_thunks)
{
    // Extract the mangled name from the stack frame, and realize the function type in the Target's SwiftASTContext.
    // Then walk the arguments looking for function pointers.  If we find one in the FIRST argument, we can fetch
    // the pointer value and return that.
    // FIXME: when we can ask swift/llvm for the location of function arguments, then we can do this for all the
    // function pointer arguments we find.
    
    SymbolContext sc = frame.GetSymbolContext(eSymbolContextSymbol);
    if (sc.symbol)
    {
        Mangled mangled_name = sc.symbol->GetMangled();
        if (mangled_name.GuessLanguage() == lldb::eLanguageTypeSwift)
        {
            Error error;
            Target &target = frame.GetThread()->GetProcess()->GetTarget();
            SwiftASTContext *swift_ast = target.GetScratchSwiftASTContext(error);
            if (swift_ast)
            {
                CompilerType function_type = swift_ast->GetTypeFromMangledTypename (mangled_name.GetMangledName().AsCString(), error);
                if (error.Success())
                {
                    if (function_type.IsFunctionType())
                    {
                        // FIXME: For now we only check the first argument since we don't know how to find the values
                        // of arguments further in the argument list.
                        // int num_arguments = function_type.GetFunctionArgumentCount();
                        // for (int i = 0; i < num_arguments; i++)
                        
                        for (int i = 0; i < 1; i++)
                        {
                            CompilerType argument_type = function_type.GetFunctionArgumentTypeAtIndex(i);
                            if (argument_type.IsFunctionPointerType())
                            {
                                // We found a function pointer argument.  Try to track down its value.  This is a hack
                                // for now, we really should ask swift/llvm how to find the argument(s) given the
                                // Swift decl for this function, and then look those up in the frame.
                                
                                ABISP abi_sp(frame.GetThread()->GetProcess()->GetABI());
                                ValueList argument_values;
                                Value input_value;
                                CompilerType clang_void_ptr_type = target.GetScratchClangASTContext()->GetBasicType(eBasicTypeVoid).GetPointerType();

                                input_value.SetValueType (Value::eValueTypeScalar);
                                input_value.SetCompilerType (clang_void_ptr_type);
                                argument_values.PushValue(input_value);
                                
                                bool success = abi_sp->GetArgumentValues (*(frame.GetThread().get()), argument_values);
                                if (success)
                                {
                                    // Now get a pointer value from the zeroth argument.
                                    Error error;
                                    DataExtractor data;
                                    ExecutionContext exe_ctx;
                                    frame.CalculateExecutionContext(exe_ctx);
                                    error = argument_values.GetValueAtIndex(0)->GetValueAsData (&exe_ctx, 
                                                                                                data, 
                                                                                                0,
                                                                                                NULL);
                                    lldb::offset_t offset = 0;
                                    lldb::addr_t fn_ptr_addr = data.GetPointer(&offset);
                                    Address fn_ptr_address;
                                    fn_ptr_address.SetLoadAddress (fn_ptr_addr, &target);
                                    // Now check to see if this has debug info:
                                    bool add_it = true;
                                    
                                    if (resolve_thunks)
                                    {
                                        SymbolContext sc;
                                        fn_ptr_address.CalculateSymbolContext(&sc, eSymbolContextEverything);
                                        if (sc.comp_unit && sc.symbol)
                                        {
                                            ConstString symbol_name = sc.symbol->GetMangled().GetMangledName();
                                            if (symbol_name)
                                            {
                                                SymbolContext target_context;
                                                if (GetTargetOfPartialApply(*sc.comp_unit, symbol_name, target_context))
                                                {
                                                    if (target_context.symbol)
                                                        fn_ptr_address = target_context.symbol->GetAddress();
                                                    else if (target_context.function)
                                                        fn_ptr_address = target_context.function->GetAddressRange().GetBaseAddress();
                                                }
                                            }
                                        }
                                    }
                                    
                                    if (debug_only)
                                    {
                                        LineEntry line_entry;
                                        fn_ptr_address.CalculateSymbolContextLineEntry(line_entry);
                                        if (!line_entry.IsValid())
                                            add_it = false;

                                    }
                                    if (add_it)
                                        addresses.push_back (fn_ptr_address);

                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------
// Exception breakpoint Precondition class for Swift:
//------------------------------------------------------------------
void
SwiftLanguageRuntime::SwiftExceptionPrecondition::AddTypeName(const char *class_name)
{
    m_type_names.insert(class_name);
}

void
SwiftLanguageRuntime::SwiftExceptionPrecondition::AddEnumSpec(const char *enum_name, const char *element_name)
{
    std::unordered_map<std::string, std::vector<std::string> >::value_type new_value (enum_name, std::vector<std::string>());
    auto result = m_enum_spec.emplace (new_value);
    result.first->second.push_back(element_name);
}

SwiftLanguageRuntime::SwiftExceptionPrecondition::SwiftExceptionPrecondition()
{
}

ValueObjectSP
SwiftLanguageRuntime::CalculateErrorValueObjectAtAddress (lldb::addr_t addr, ConstString name, bool persistent)
{
    ValueObjectSP error_valobj_sp;

    Error error;
    SwiftASTContext *ast_context = m_process->GetTarget().GetScratchSwiftASTContext(error);
    if (!ast_context || error.Fail())
        return error_valobj_sp;

    CompilerType swift_error_proto_type = ast_context->GetErrorType();
    Value addr_value;

    error_valobj_sp = ValueObjectConstResult::Create (m_process,
                                                      swift_error_proto_type,
                                                      name,
                                                      addr,
                                                      eAddressTypeLoad,
                                                      m_process->GetAddressByteSize());

    if (error_valobj_sp && error_valobj_sp->GetError().Success())
    {
        error_valobj_sp = error_valobj_sp->GetQualifiedRepresentationIfAvailable(lldb::eDynamicCanRunTarget, true);
        if (!IsValidErrorValue(*(error_valobj_sp.get())))
        {
            error_valobj_sp.reset();
        }
    }

    if (persistent && error_valobj_sp)
    {
        PersistentExpressionState *persistent_state = m_process->GetTarget().GetPersistentExpressionStateForLanguage(eLanguageTypeSwift);
                    
        ConstString persistent_variable_name (persistent_state->GetNextPersistentVariableName(true));

        lldb::ValueObjectSP const_valobj_sp;
        
        // Check in case our value is already a constant value
        if (error_valobj_sp->GetIsConstant())
        {
            const_valobj_sp = error_valobj_sp;
            const_valobj_sp->SetName (persistent_variable_name);
        }
        else
            const_valobj_sp = error_valobj_sp->CreateConstantValue (persistent_variable_name);

        lldb::ValueObjectSP live_valobj_sp = error_valobj_sp;
        
        error_valobj_sp = const_valobj_sp;

        ExpressionVariableSP clang_expr_variable_sp(persistent_state->CreatePersistentVariable(error_valobj_sp));
        clang_expr_variable_sp->m_live_sp = live_valobj_sp;
        clang_expr_variable_sp->m_flags |= ClangExpressionVariable::EVIsProgramReference;

        error_valobj_sp = clang_expr_variable_sp->GetValueObject();
    }
    return error_valobj_sp;
}

ValueObjectSP
SwiftLanguageRuntime::CalculateErrorValueFromFirstArgument(StackFrameSP frame_sp, ConstString variable_name)
{
    ProcessSP process_sp(frame_sp->GetThread()->GetProcess());
    ABISP abi_sp(process_sp->GetABI());
    ValueList argument_values;
    Value input_value;
    Error error;
    Target *target = frame_sp->CalculateTarget().get();
    ValueObjectSP error_valobj_sp;

    ClangASTContext *clang_ast_context = target->GetScratchClangASTContext();
    CompilerType clang_void_ptr_type = clang_ast_context->GetBasicType(eBasicTypeVoid).GetPointerType();

    input_value.SetValueType (Value::eValueTypeScalar);
    input_value.SetCompilerType (clang_void_ptr_type);
    argument_values.PushValue(input_value);

    bool success = abi_sp->GetArgumentValues (*(frame_sp->GetThread().get()), argument_values);
    if (success)
    {
        ExecutionContext exe_ctx;
        frame_sp->CalculateExecutionContext(exe_ctx);
        DataExtractor data;

        SwiftASTContext *ast_context = target->GetScratchSwiftASTContext(error);
        if (!ast_context || error.Fail())
            return error_valobj_sp;

        CompilerType swift_error_proto_type = ast_context->GetErrorType();
        if (swift_error_proto_type.IsValid())
        {
            Value *arg0 = argument_values.GetValueAtIndex(0);
            Error extract_error = arg0->GetValueAsData (&exe_ctx,
                                                        data,
                                                        0,
                                                        nullptr);
            if (extract_error.Success())
            {
                error_valobj_sp = ValueObjectConstResult::Create (frame_sp.get(),
                                                                                swift_error_proto_type,
                                                                                variable_name,
                                                                                data);
                if (error_valobj_sp->GetError().Fail())
                {
                    // If we couldn't make the error ValueObject, then we will always stop.
                    // FIXME: Some logging here would be good.
                    return error_valobj_sp;
                }

                error_valobj_sp = error_valobj_sp->GetQualifiedRepresentationIfAvailable(lldb::eDynamicCanRunTarget, true);
            }
        }
    }
    return error_valobj_sp;
}

void
SwiftLanguageRuntime::RegisterGlobalError(Target &target, ConstString name, lldb::addr_t addr)
{
    Error ast_context_error;
    SwiftASTContext *ast_context = target.GetScratchSwiftASTContext(ast_context_error);
    
    if (ast_context_error.Success() && ast_context && !ast_context->HasFatalErrors())
    {
        SwiftPersistentExpressionState *persistent_state = llvm::cast<SwiftPersistentExpressionState>(target.GetPersistentExpressionStateForLanguage(lldb::eLanguageTypeSwift));
        
        std::string module_name = "$__lldb_module_for_";
        module_name.append(&name.GetCString()[1]);
        
        Error module_creation_error;
        swift::ModuleDecl *module_decl = ast_context->CreateModule(ConstString(module_name), module_creation_error);
        
        if (module_creation_error.Success() && module_decl)
        {
            const bool is_static = false;
            const bool is_let = true;
            
            swift::VarDecl *var_decl = new (*ast_context->GetASTContext()) swift::VarDecl(is_static,
                                                                                          is_let,
                                                                                          swift::SourceLoc(),
                                                                                          ast_context->GetIdentifier(name.GetCString()),
                                                                                          GetSwiftType(ast_context->GetErrorType()),
                                                                                          module_decl);
            var_decl->setDebuggerVar(true);
            
            persistent_state->RegisterSwiftPersistentDecl(var_decl);
            
            ConstString mangled_name;
            
            {
                swift::Mangle::Mangler mangler;
                
                mangler.mangleGlobalVariableFull(var_decl);
                mangled_name = ConstString(mangler.finalize().c_str());
            }
            
            lldb::addr_t symbol_addr;
            
            {
                ProcessSP process_sp(target.GetProcessSP());
                Error alloc_error;
                
                symbol_addr = process_sp->AllocateMemory(process_sp->GetAddressByteSize(), lldb::ePermissionsWritable | lldb::ePermissionsReadable, alloc_error);
                
                if (alloc_error.Success() && symbol_addr != LLDB_INVALID_ADDRESS)
                {
                    Error write_error;
                    process_sp->WritePointerToMemory(symbol_addr, addr, write_error);
                    
                    if (write_error.Success())
                    {
                        persistent_state->RegisterSymbol(mangled_name, symbol_addr);
                    }
                }
            }
        }
    }
}

bool
SwiftLanguageRuntime::SwiftExceptionPrecondition::EvaluatePrecondition(StoppointCallbackContext &context)
{
    if (!m_type_names.empty())
    {
        StackFrameSP frame_sp = context.exe_ctx_ref.GetFrameSP();
        if (!frame_sp)
            return true;

        ValueObjectSP error_valobj_sp = CalculateErrorValueFromFirstArgument(frame_sp, ConstString("__swift_error_var"));
        if (!error_valobj_sp || error_valobj_sp->GetError().Fail())
            return true;

        // This shouldn't fail, since at worst it will return me the object I just successfully got.
        std::string full_error_name (error_valobj_sp->GetCompilerType().GetTypeName().AsCString());
        size_t last_dot_pos = full_error_name.rfind('.');
        std::string type_name_base;
        if (last_dot_pos == std::string::npos)
            type_name_base = full_error_name;
        else
        {
            if (last_dot_pos + 1 <= full_error_name.size())
                type_name_base = full_error_name.substr(last_dot_pos + 1, full_error_name.size());
        }
        
        // The type name will be the module and then the type.  If the match name has a dot, we require a complete
        // match against the type, if the type name has no dot, we match it against the base.
        
        for (std::string name : m_type_names)
        {
            if (name.rfind('.') != std::string::npos)
            {
                if (name == full_error_name)
                    return true;
            }
            else
            {
                if (name == type_name_base)
                    return true;
            }
        }
        return false;

    }
    return true;
}

void
SwiftLanguageRuntime::SwiftExceptionPrecondition::GetDescription(Stream &stream, lldb::DescriptionLevel level)
{
    if (level == eDescriptionLevelFull
        || level == eDescriptionLevelVerbose)
    {
        if (m_type_names.size() > 0)
        {
            stream.Printf("\nType Filters:");
            for (std::string name : m_type_names)
            {
                stream.Printf(" %s", name.c_str());
            }
            stream.Printf("\n");
        }
    }

}

Error
SwiftLanguageRuntime::SwiftExceptionPrecondition::ConfigurePrecondition(Args &args)
{
    Error error;
    std::vector<std::string> object_typenames;
    args.GetOptionValuesAsStrings("exception-typename", object_typenames);
    for (auto type_name : object_typenames)
        AddTypeName(type_name.c_str());
    return error;
}

void
SwiftLanguageRuntime::AddToLibraryNegativeCache (const char *library_name)
{
    std::lock_guard<std::mutex> locker(m_negative_cache_mutex);
    m_library_negative_cache.insert(library_name);
}

bool
SwiftLanguageRuntime::IsInLibraryNegativeCache (const char *library_name)
{
    std::lock_guard<std::mutex> locker(m_negative_cache_mutex);
    return m_library_negative_cache.count(library_name) == 1;
}

lldb::addr_t
SwiftLanguageRuntime::MaskMaybeBridgedPointer (lldb::addr_t addr,
                                               lldb::addr_t *masked_bits)
{
    if (!m_process)
        return addr;
    const ArchSpec& arch_spec(m_process->GetTarget().GetArchitecture());
    ArchSpec::Core core_kind = arch_spec.GetCore();
    bool is_arm = false;
    bool is_intel = false;
    bool is_32 = false;
    bool is_64 = false;
    if (core_kind >= ArchSpec::Core::kCore_arm_first &&
        core_kind <= ArchSpec::Core::kCore_arm_last)
    {
        is_arm = true;
    }
    else if (core_kind >= ArchSpec::Core::kCore_x86_64_first &&
             core_kind <= ArchSpec::Core::kCore_x86_64_last)
    {
        is_intel = true;
    }
    else if (core_kind >= ArchSpec::Core::kCore_x86_32_first &&
             core_kind <= ArchSpec::Core::kCore_x86_32_last)
    {
        is_intel = true;
    }
    else
    {
        // this is a really random CPU core to be running on - just get out fast
        return addr;
    }
    
    switch (arch_spec.GetAddressByteSize())
    {
        case 4:
            is_32 = true;
            break;
        case 8:
            is_64 = true;
            break;
        default:
            // this is a really random pointer size to be running on - just get out fast
            return addr;
    }

    lldb::addr_t mask = 0;
    
    if (is_arm && is_64)
        mask = SWIFT_ABI_ARM64_SWIFT_SPARE_BITS_MASK;
    
    if (is_arm && is_32)
        mask = SWIFT_ABI_ARM_SWIFT_SPARE_BITS_MASK;

    if (is_intel && is_64)
        mask = SWIFT_ABI_X86_64_SWIFT_SPARE_BITS_MASK;
    
    if (is_intel && is_32)
        mask = SWIFT_ABI_I386_SWIFT_SPARE_BITS_MASK;
    
    if (masked_bits)
        *masked_bits = addr & mask;
    return addr & ~mask;
}

lldb::addr_t
SwiftLanguageRuntime::MaybeMaskNonTrivialReferencePointer (lldb::addr_t addr)
{
    if (auto objc_runtime = GetObjCRuntime())
    {
        // tagged pointers don't perform any masking
        if (objc_runtime->IsTaggedPointer(addr))
            return addr;
    }
    
    if (!m_process)
        return addr;
    const ArchSpec& arch_spec(m_process->GetTarget().GetArchitecture());
    ArchSpec::Core core_kind = arch_spec.GetCore();
    bool is_arm = false;
    bool is_intel = false;
    bool is_32 = false;
    bool is_64 = false;
    if (core_kind >= ArchSpec::Core::kCore_arm_first &&
        core_kind <= ArchSpec::Core::kCore_arm_last)
    {
        is_arm = true;
    }
    else if (core_kind >= ArchSpec::Core::kCore_x86_64_first &&
             core_kind <= ArchSpec::Core::kCore_x86_64_last)
    {
        is_intel = true;
    }
    else if (core_kind >= ArchSpec::Core::kCore_x86_32_first &&
             core_kind <= ArchSpec::Core::kCore_x86_32_last)
    {
        is_intel = true;
    }
    else
    {
        // this is a really random CPU core to be running on - just get out fast
        return addr;
    }
    
    switch (arch_spec.GetAddressByteSize())
    {
        case 4:
            is_32 = true;
            break;
        case 8:
            is_64 = true;
            break;
        default:
            // this is a really random pointer size to be running on - just get out fast
            return addr;
    }
    
    lldb::addr_t mask = 0;
    
    if (is_arm && is_64)
        mask = SWIFT_ABI_ARM64_OBJC_NUM_RESERVED_LOW_BITS;
    else if (is_intel && is_64)
        mask = SWIFT_ABI_X86_64_OBJC_NUM_RESERVED_LOW_BITS;
    else
        mask = SWIFT_ABI_DEFAULT_OBJC_NUM_RESERVED_LOW_BITS;

    mask = (1<<mask) | (1<<(mask+1));
    
    return addr & ~mask;
}


ConstString
SwiftLanguageRuntime::GetErrorBackstopName ()
{
    return ConstString("swift_errorInMain");
}

ConstString
SwiftLanguageRuntime::GetStandardLibraryBaseName()
{
    static ConstString g_swiftCore("swiftCore");
    return g_swiftCore;
}

ConstString
SwiftLanguageRuntime::GetStandardLibraryName ()
{
    PlatformSP platform_sp(m_process->GetTarget().GetPlatform());
    if (platform_sp)
        return platform_sp->GetFullNameForDylib(GetStandardLibraryBaseName());
    return GetStandardLibraryBaseName();
}

bool
SwiftLanguageRuntime::GetReferenceCounts (ValueObject& valobj, size_t &strong, size_t &weak)
{
    CompilerType compiler_type(valobj.GetCompilerType());
    Flags type_flags(compiler_type.GetTypeInfo());
    if (llvm::isa<SwiftASTContext>(compiler_type.GetTypeSystem()) &&
        type_flags.AllSet(eTypeInstanceIsPointer))
    {
        lldb::addr_t ptr_value = valobj.GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
        if (ptr_value == LLDB_INVALID_ADDRESS)
            return false;
        ptr_value += GetProcess()->GetAddressByteSize();
        Error error;
        strong = GetProcess()->ReadUnsignedIntegerFromMemory(ptr_value, 4, 0, error) >> 2;
        if (error.Fail())
            return false;
        weak = GetProcess()->ReadUnsignedIntegerFromMemory(ptr_value + 4, 4, 0, error) >> 2;
        if (error.Fail())
            return false;
        return true;
    }
    return false;
}

class ProjectionSyntheticChildren : public SyntheticChildren
{
public:
    struct FieldProjection
    {
        ConstString name;
        CompilerType type;
        int32_t byte_offset;
        
        FieldProjection (CompilerType parent_type,
                         ExecutionContext *exe_ctx,
                         size_t idx)
        {
            const bool transparent_pointers = false;
            const bool omit_empty_base_classes = true;
            const bool ignore_array_bounds = false;
            bool child_is_base_class = false;
            bool child_is_deref_of_parent = false;
            std::string child_name;

            uint32_t child_byte_size;
            uint32_t child_bitfield_bit_size;
            uint32_t child_bitfield_bit_offset;
            uint64_t language_flags;

            type = parent_type.GetChildCompilerTypeAtIndex(exe_ctx,
                                                           idx,
                                                           transparent_pointers,
                                                           omit_empty_base_classes,
                                                           ignore_array_bounds,
                                                           child_name,
                                                           child_byte_size,
                                                           byte_offset,
                                                           child_bitfield_bit_size,
                                                           child_bitfield_bit_offset,
                                                           child_is_base_class,
                                                           child_is_deref_of_parent,
                                                           nullptr,
                                                           language_flags);
            
            if (child_is_base_class)
                type.Clear(); // invalidate - base classes are dealt with outside of the projection
            else
                name.SetCStringWithLength(child_name.c_str(), child_name.size());
        }
        
        bool
        IsValid ()
        {
            return !name.IsEmpty() && type.IsValid();
        }
        
        explicit operator bool()
        {
            return IsValid();
        }
    };
    
    struct TypeProjection
    {
        std::vector<FieldProjection> field_projections;
        ConstString type_name;
    };
    
    typedef std::unique_ptr<TypeProjection> TypeProjectionUP;

    bool
    IsScripted ()
    {
        return false;
    }
    
    std::string
    GetDescription ()
    {
        return "projection synthetic children";
    }

    ProjectionSyntheticChildren(const Flags& flags,
                                TypeProjectionUP &&projection) :
    SyntheticChildren(flags),
    m_projection(std::move(projection))
    {
    }
    
protected:
    TypeProjectionUP m_projection;
    
    class ProjectionFrontEndProvider : public SyntheticChildrenFrontEnd
    {
    public:
        ProjectionFrontEndProvider (ValueObject &backend,
                                    TypeProjectionUP &projection) :
        SyntheticChildrenFrontEnd(backend),
        m_num_bases(0),
        m_projection(projection.get())
        {
            lldbassert(m_projection && "need a valid projection");
            CompilerType type(backend.GetCompilerType());
            m_num_bases = type.GetNumDirectBaseClasses();
        }
        
        size_t
        CalculateNumChildren () override
        {
            return m_projection->field_projections.size() + m_num_bases;
        }
        
        lldb::ValueObjectSP
        GetChildAtIndex (size_t idx) override
        {
            if (idx < m_num_bases)
            {
                if (ValueObjectSP base_object_sp = m_backend.GetChildAtIndex(idx, true))
                {
                    CompilerType base_type(base_object_sp->GetCompilerType());
                    ConstString base_type_name(base_type.GetTypeName());
                    if (base_type_name.IsEmpty() ||
                        !base_type_name.GetStringRef().startswith("_TtC"))
                        return base_object_sp;
                    base_object_sp = m_backend.GetSyntheticBase(0,
                                                                base_type,
                                                                true,
                                                                Mangled(base_type_name,true).GetDemangledName(lldb::eLanguageTypeSwift));
                    return base_object_sp;
                }
                else
                    return nullptr;
            }
            idx -= m_num_bases;
            if (idx < m_projection->field_projections.size())
            {
                auto& projection(m_projection->field_projections.at(idx));
                return m_backend.GetSyntheticChildAtOffset(projection.byte_offset, projection.type, true, projection.name);
            }
            return nullptr;
        }
        
        size_t
        GetIndexOfChildWithName (const ConstString &name) override
        {
            for (size_t idx = 0;
                 idx < m_projection->field_projections.size();
                 idx++)
            {
                if (m_projection->field_projections.at(idx).name == name)
                    return idx;
            }
            return UINT32_MAX;
        }
        
        bool
        Update () override
        {
            return false;
        }
        
        bool
        MightHaveChildren () override
        {
            return true;
        }
        
        ConstString
        GetSyntheticTypeName ()  override
        {
            return m_projection->type_name;
        }
        
    private:
        size_t m_num_bases;
        TypeProjectionUP::element_type *m_projection;
    };
    
public:
    SyntheticChildrenFrontEnd::AutoPointer
    GetFrontEnd (ValueObject &backend)
    {
        return SyntheticChildrenFrontEnd::AutoPointer(new ProjectionFrontEndProvider(backend, m_projection));
    }
};

lldb::SyntheticChildrenSP
SwiftLanguageRuntime::GetBridgedSyntheticChildProvider (ValueObject& valobj)
{
    const char *type_name(valobj.GetCompilerType().GetTypeName().AsCString());
    
    if (type_name && *type_name)
    {
        auto iter = m_bridged_synthetics_map.find(type_name), end = m_bridged_synthetics_map.end();
        if (iter != end)
            return iter->second;
    }

    ProjectionSyntheticChildren::TypeProjectionUP type_projection(new ProjectionSyntheticChildren::TypeProjectionUP::element_type());

    if (SwiftASTContext *swift_ast_ctx = GetScratchSwiftASTContext())
    {
        Error error;
        CompilerType swift_type = swift_ast_ctx->GetTypeFromMangledTypename(type_name, error);

        if (swift_type.IsValid())
        {
            ExecutionContext exe_ctx(GetProcess());
            bool any_projected = false;
            for (size_t idx = 0;
                 idx < swift_type.GetNumChildren(true);
                 idx++)
            {
                // if a projection fails, keep going - we have offsets here, so it should be OK to skip some members
                if (auto projection = ProjectionSyntheticChildren::FieldProjection(swift_type,
                                                                                   &exe_ctx,
                                                                                   idx))
                {
                    any_projected = true;
                    type_projection->field_projections.push_back(projection);
                }
            }
            
            if (any_projected)
            {
                type_projection->type_name = swift_type.GetDisplayTypeName();
                SyntheticChildrenSP synth_sp = SyntheticChildrenSP(new ProjectionSyntheticChildren(SyntheticChildren::Flags(),
                                                                                                   std::move(type_projection)));
                return (m_bridged_synthetics_map[type_name] = synth_sp);
            }
        }
    }

    return nullptr;
}

lldb::addr_t
SwiftLanguageRuntime::GetErrorReturnLocationForFrame(lldb::StackFrameSP frame_sp)
{
    lldb::addr_t return_addr = LLDB_INVALID_ADDRESS;

    if (!frame_sp)
        return return_addr;

    ConstString error_location_name ("$error");
    VariableListSP variables_sp = frame_sp->GetInScopeVariableList(false);
    VariableSP error_loc_var_sp = variables_sp->FindVariable(error_location_name, eValueTypeVariableArgument);
    if (error_loc_var_sp)
    {
        ValueObjectSP error_loc_val_sp = frame_sp->GetValueObjectForFrameVariable(error_loc_var_sp, eNoDynamicValues);
        if (error_loc_val_sp && error_loc_val_sp->GetError().Success())
            return_addr = error_loc_val_sp->GetAddressOf();
    }

    return return_addr;
}

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
LanguageRuntime *
SwiftLanguageRuntime::CreateInstance (Process *process, lldb::LanguageType language)
{
    if (language == eLanguageTypeSwift)
        return new SwiftLanguageRuntime (process);
    else
        return NULL;
}

lldb::BreakpointResolverSP
SwiftLanguageRuntime::CreateExceptionResolver (Breakpoint *bkpt, bool catch_bp, bool throw_bp)
{
    BreakpointResolverSP resolver_sp;
    
    if (throw_bp)
        resolver_sp.reset (new BreakpointResolverName (bkpt,
                                                       "swift_willThrow",
                                                       eFunctionNameTypeBase,
                                                       eLanguageTypeUnknown,
                                                       Breakpoint::Exact,
                                                       0,
                                                       eLazyBoolNo));
    // FIXME: We don't do catch breakpoints for ObjC yet.
    // Should there be some way for the runtime to specify what it can do in this regard?
    return resolver_sp;
}

static const char *
SwiftDemangleNodeKindToCString(const swift::Demangle::Node::Kind node_kind)
{
#define NODE(e) case swift::Demangle::Node::Kind::e: return #e;
    
    switch (node_kind)
    {
#include "swift/Basic/DemangleNodes.def"
    }
    return "swift::Demangle::Node::Kind::???";
#undef NODE
}

class CommandObjectSwift_Demangle : public CommandObjectParsed
{
public:
    
    CommandObjectSwift_Demangle (CommandInterpreter &interpreter) :
    CommandObjectParsed (interpreter,
                         "demangle",
                         "Demangle a Swift mangled name",
                         "language swift demangle"),
    m_options()
    {
    }
    
    ~CommandObjectSwift_Demangle ()
    {
    }
    
    virtual
    Options *
    GetOptions ()
    {
        return &m_options;
    }
    
    class CommandOptions : public Options
    {
    public:
        
        CommandOptions() :
        Options(),
        m_expand(false,false)
        {
            OptionParsingStarting (nullptr);
        }
        
        virtual
        ~CommandOptions ()
        {
        }
        
        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg,
                        ExecutionContext *execution_context)
        {
            Error error;
            const int short_option = m_getopt_table[option_idx].val;
            switch (short_option)
            {
                case 'e':
                    m_expand.SetCurrentValue(true);
                    break;
                    
                default:
                    error.SetErrorStringWithFormat ("invalid short option character '%c'", short_option);
                    break;
            }
            
            return error;
        }
        
        void
        OptionParsingStarting (ExecutionContext *execution_context)
        {
            m_expand.Clear();
        }
        
        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }
        
        // Options table: Required for subclasses of Options.
        
        static OptionDefinition g_option_table[];
        OptionValueBoolean m_expand;
    };
    
protected:
    void
    PrintNode (swift::Demangle::NodePointer node_ptr,
               Stream& stream,
               int depth = 0)
    {
        if (!node_ptr)
            return;
        
        std::string indent(2*depth, ' ');
        
        stream.Printf("%s", indent.c_str());
        
        stream.Printf("kind=%s", SwiftDemangleNodeKindToCString(node_ptr->getKind()));
        if (node_ptr->hasText())
            stream.Printf(", text=\"%s\"", node_ptr->getText().c_str());
        if (node_ptr->hasIndex())
            stream.Printf(", index=%" PRIu64, node_ptr->getIndex());
        
        stream.Printf("\n");
        
        for (auto&& child : *node_ptr)
        {
            PrintNode(child, stream, depth + 1);
        }
    }
    
    bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        for (size_t i = 0;
             i < command.GetArgumentCount();
             i++)
        {
            const char* arg = command.GetArgumentAtIndex(i);
            if (arg && *arg)
            {
                auto node_ptr = swift::Demangle::demangleSymbolAsNode(arg, strlen(arg));
                if (node_ptr)
                {
                    if (m_options.m_expand)
                    {
                        PrintNode(node_ptr,
                                  result.GetOutputStream());
                    }
                    result.GetOutputStream().Printf("%s ---> %s\n",
                                                    arg,
                                                    swift::Demangle::nodeToString(node_ptr).c_str());
                }
            }
        }
        result.SetStatus(lldb::eReturnStatusSuccessFinishResult);
        return true;
    }

    CommandOptions m_options;
};

OptionDefinition
CommandObjectSwift_Demangle::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_1, false, "expand", 'e', OptionParser::eNoArgument, NULL, NULL, 0, eArgTypeNone, "Whether LLDB should print the demangled tree"},
    { 0, false, NULL, 0, 0, NULL, NULL, 0, eArgTypeNone, NULL }
};

class CommandObjectSwift_RefCount : public CommandObjectRaw
{
public:
    
    CommandObjectSwift_RefCount (CommandInterpreter &interpreter) :
    CommandObjectRaw (interpreter,
                      "refcount",
                      "Inspect the reference count data for a Swift object",
                      "language swift refcount",
                      eCommandProcessMustBePaused | eCommandRequiresFrame)
    {
    }
    
    ~CommandObjectSwift_RefCount ()
    {
    }
    
    virtual
    Options *
    GetOptions ()
    {
        return nullptr;
    }
    
protected:
    bool
    DoExecute (const char *command, CommandReturnObject &result)
    {
        ExecutionContext exe_ctx (m_interpreter.GetExecutionContext());
        StackFrameSP frame_sp(exe_ctx.GetFrameSP());
        EvaluateExpressionOptions options;
        options.SetLanguage(lldb::eLanguageTypeSwift);
        options.SetResultIsInternal(true);
        options.SetUseDynamic();
        ValueObjectSP result_valobj_sp;
        if (exe_ctx.GetTargetSP()->EvaluateExpression(command, frame_sp.get(), result_valobj_sp) == eExpressionCompleted)
        {
            if (result_valobj_sp)
            {
                if (result_valobj_sp->GetError().Fail())
                {
                    result.SetStatus(lldb::eReturnStatusFailed);
                    result.AppendError(result_valobj_sp->GetError().AsCString());
                    return false;
                }
                result_valobj_sp = result_valobj_sp->GetQualifiedRepresentationIfAvailable(lldb::eDynamicCanRunTarget, true);
                CompilerType result_type(result_valobj_sp->GetCompilerType());
                if (result_type.GetTypeInfo() & lldb::eTypeInstanceIsPointer)
                {
                    size_t strong=0,weak=0;
                    if (!exe_ctx.GetProcessSP()->GetSwiftLanguageRuntime()->GetReferenceCounts(*result_valobj_sp.get(), strong, weak))
                    {
                        result.AppendError("refcount not available");
                        result.SetStatus(lldb::eReturnStatusFailed);
                        return false;
                    }
                    else
                    {
                        result.AppendMessageWithFormat("refcount data: (strong = %zu, weak = %zu)\n", strong, weak);
                        result.SetStatus(lldb::eReturnStatusSuccessFinishResult);
                        return true;
                    }
                }
                else
                {
                    result.AppendError("refcount only available for class types");
                    result.SetStatus(lldb::eReturnStatusFailed);
                    return false;
                }
            }
        }
        result.SetStatus(lldb::eReturnStatusFailed);
        if (result_valobj_sp && result_valobj_sp->GetError().Fail())
            result.AppendError(result_valobj_sp->GetError().AsCString());
        return false;
    }
};

class CommandObjectMultiwordSwift : public CommandObjectMultiword
{
public:
    
    CommandObjectMultiwordSwift (CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "swift",
                            "A set of commands for operating on the Swift Language Runtime.",
                            "swift <subcommand> [<subcommand-options>]")
    {
        LoadSubCommand ("demangle",   CommandObjectSP (new CommandObjectSwift_Demangle (interpreter)));
        LoadSubCommand ("refcount",   CommandObjectSP (new CommandObjectSwift_RefCount (interpreter)));
    }
    
    virtual
    ~CommandObjectMultiwordSwift ()
    {
    }
};

void
SwiftLanguageRuntime::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   "Language runtime for the Swift language",
                                   CreateInstance,
                                   [] (CommandInterpreter& interpreter) -> lldb::CommandObjectSP {
                                       return CommandObjectSP(new CommandObjectMultiwordSwift(interpreter));
                                   });
}

void
SwiftLanguageRuntime::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}

lldb_private::ConstString
SwiftLanguageRuntime::GetPluginNameStatic()
{
    static ConstString g_name("swift");
    return g_name;
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
lldb_private::ConstString
SwiftLanguageRuntime::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
SwiftLanguageRuntime::GetPluginVersion()
{
    return 1;
}

