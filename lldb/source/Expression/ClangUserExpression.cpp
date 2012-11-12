//===-- ClangUserExpression.cpp ---------------------------------*- C++ -*-===//
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
#  include <sys/types.h>
#endif

// C++ Includes
#include <cstdlib>
#include <string>
#include <map>

#include "lldb/Core/ConstString.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Expression/ASTResultSynthesizer.h"
#include "lldb/Expression/ClangExpressionDeclMap.h"
#include "lldb/Expression/ClangExpressionParser.h"
#include "lldb/Expression/ClangFunction.h"
#include "lldb/Expression/ClangUserExpression.h"
#include "lldb/Expression/ExpressionSourceCode.h"
#include "lldb/Host/Host.h"
#include "lldb/Symbol/Block.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/ClangExternalASTSourceCommon.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/ThreadPlanCallUserExpression.h"

#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"

using namespace lldb_private;

ClangUserExpression::ClangUserExpression (const char *expr,
                                          const char *expr_prefix,
                                          lldb::LanguageType language,
                                          ResultType desired_type) :
    ClangExpression (),
    m_expr_text (expr),
    m_expr_prefix (expr_prefix ? expr_prefix : ""),
    m_language (language),
    m_transformed_text (),
    m_desired_type (desired_type),
    m_enforce_valid_object (false),
    m_cplusplus (false),
    m_objectivec (false),
    m_static_method(false),
    m_needs_object_ptr (false),
    m_const_object (false),
    m_target (NULL),
    m_evaluated_statically (false),
    m_const_result ()
{
    switch (m_language)
    {
    case lldb::eLanguageTypeC_plus_plus:
        m_allow_cxx = true;
        break;
    case lldb::eLanguageTypeObjC:
        m_allow_objc = true;
        break;
    case lldb::eLanguageTypeObjC_plus_plus:
    default:
        m_allow_cxx = true;
        m_allow_objc = true;
        break;
    }
}

ClangUserExpression::~ClangUserExpression ()
{
}

clang::ASTConsumer *
ClangUserExpression::ASTTransformer (clang::ASTConsumer *passthrough)
{    
    ClangASTContext *clang_ast_context = m_target->GetScratchClangASTContext();
    
    if (!clang_ast_context)
        return NULL;
    
    if (!m_result_synthesizer.get())
        m_result_synthesizer.reset(new ASTResultSynthesizer(passthrough,
                                                            *m_target));
    
    return m_result_synthesizer.get();
}

void
ClangUserExpression::ScanContext(ExecutionContext &exe_ctx, Error &err)
{
    m_target = exe_ctx.GetTargetPtr();
    
    if (!(m_allow_cxx || m_allow_objc))
        return;
    
    StackFrame *frame = exe_ctx.GetFramePtr();
    if (frame == NULL)
        return;
    
    SymbolContext sym_ctx = frame->GetSymbolContext(lldb::eSymbolContextFunction | lldb::eSymbolContextBlock);
    
    if (!sym_ctx.function)
        return;
    
    // Find the block that defines the function represented by "sym_ctx"
    Block *function_block = sym_ctx.GetFunctionBlock();
    
    if (!function_block)
        return;

    clang::DeclContext *decl_context = function_block->GetClangDeclContext();

    if (!decl_context)
        return;
            
    if (clang::CXXMethodDecl *method_decl = llvm::dyn_cast<clang::CXXMethodDecl>(decl_context))
    {
        if (m_allow_cxx && method_decl->isInstance())
        {
            if (m_enforce_valid_object)
            {
                lldb::VariableListSP variable_list_sp (function_block->GetBlockVariableList (true));
                
                const char *thisErrorString = "Stopped in a C++ method, but 'this' isn't available; pretending we are in a generic context";
                
                if (!variable_list_sp)
                {
                    err.SetErrorString(thisErrorString);
                    return;
                }
                
                lldb::VariableSP this_var_sp (variable_list_sp->FindVariable(ConstString("this")));
                
                if (!this_var_sp ||
                    !this_var_sp->IsInScope(frame) || 
                    !this_var_sp->LocationIsValidForFrame (frame))
                {
                    err.SetErrorString(thisErrorString);
                    return;
                }
            }
            
            m_cplusplus = true;
            m_needs_object_ptr = true;
        }
    }
    else if (clang::ObjCMethodDecl *method_decl = llvm::dyn_cast<clang::ObjCMethodDecl>(decl_context))
    {        
        if (m_allow_objc)
        {
            if (m_enforce_valid_object)
            {
                lldb::VariableListSP variable_list_sp (function_block->GetBlockVariableList (true));
                
                const char *selfErrorString = "Stopped in an Objective-C method, but 'self' isn't available; pretending we are in a generic context";
                
                if (!variable_list_sp)
                {
                    err.SetErrorString(selfErrorString);
                    return;
                }
                
                lldb::VariableSP self_variable_sp = variable_list_sp->FindVariable(ConstString("self"));
                
                if (!self_variable_sp || 
                    !self_variable_sp->IsInScope(frame) || 
                    !self_variable_sp->LocationIsValidForFrame (frame))
                {
                    err.SetErrorString(selfErrorString);
                    return;
                }
            }
            
            m_objectivec = true;
            m_needs_object_ptr = true;
            
            if (!method_decl->isInstanceMethod())
                m_static_method = true;
        }
    }
    else if (clang::FunctionDecl *function_decl = llvm::dyn_cast<clang::FunctionDecl>(decl_context))
    {
        // We might also have a function that said in the debug information that it captured an
        // object pointer.  The best way to deal with getting to the ivars at present it by pretending
        // that this is a method of a class in whatever runtime the debug info says the object pointer
        // belongs to.  Do that here.
        
        ClangASTMetadata *metadata = ClangASTContext::GetMetadata (&decl_context->getParentASTContext(), (uintptr_t) function_decl);
        if (metadata && metadata->HasObjectPtr())
        {
            lldb::LanguageType language = metadata->GetObjectPtrLanguage();
            if (language == lldb::eLanguageTypeC_plus_plus)
            {
                m_cplusplus = true;
                m_needs_object_ptr = true;
            }
            else if (language == lldb::eLanguageTypeObjC)
            {
                m_objectivec = true;
                m_needs_object_ptr = true;
            }
        }
    }
}

// This is a really nasty hack, meant to fix Objective-C expressions of the form
// (int)[myArray count].  Right now, because the type information for count is
// not available, [myArray count] returns id, which can't be directly cast to
// int without causing a clang error.
static void
ApplyObjcCastHack(std::string &expr)
{
#define OBJC_CAST_HACK_FROM "(int)["
#define OBJC_CAST_HACK_TO   "(int)(long long)["

    size_t from_offset;
    
    while ((from_offset = expr.find(OBJC_CAST_HACK_FROM)) != expr.npos)
        expr.replace(from_offset, sizeof(OBJC_CAST_HACK_FROM) - 1, OBJC_CAST_HACK_TO);

#undef OBJC_CAST_HACK_TO
#undef OBJC_CAST_HACK_FROM
}

// Another hack, meant to allow use of unichar despite it not being available in
// the type information.  Although we could special-case it in type lookup,
// hopefully we'll figure out a way to #include the same environment as is
// present in the original source file rather than try to hack specific type
// definitions in as needed.
static void
ApplyUnicharHack(std::string &expr)
{
#define UNICHAR_HACK_FROM "unichar"
#define UNICHAR_HACK_TO   "unsigned short"
    
    size_t from_offset;
    
    while ((from_offset = expr.find(UNICHAR_HACK_FROM)) != expr.npos)
        expr.replace(from_offset, sizeof(UNICHAR_HACK_FROM) - 1, UNICHAR_HACK_TO);
    
#undef UNICHAR_HACK_TO
#undef UNICHAR_HACK_FROM
}

bool
ClangUserExpression::Parse (Stream &error_stream, 
                            ExecutionContext &exe_ctx,
                            lldb_private::ExecutionPolicy execution_policy,
                            bool keep_result_in_memory)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    Error err;
 
    InstallContext(exe_ctx);
    
    ScanContext(exe_ctx, err);
    
    if (!err.Success())
    {
        error_stream.Printf("warning: %s\n", err.AsCString());
    }
    
    StreamString m_transformed_stream;
    
    ////////////////////////////////////
    // Generate the expression
    //
    
    ApplyObjcCastHack(m_expr_text);
    //ApplyUnicharHack(m_expr_text);

    std::auto_ptr <ExpressionSourceCode> source_code (ExpressionSourceCode::CreateWrapped(m_expr_prefix.c_str(), m_expr_text.c_str()));
    
    lldb::LanguageType lang_type;
    
    if (m_cplusplus)
        lang_type = lldb::eLanguageTypeC_plus_plus;
    else if(m_objectivec)
        lang_type = lldb::eLanguageTypeObjC;
    else
        lang_type = lldb::eLanguageTypeC;
    
    if (!source_code->GetText(m_transformed_text, lang_type, m_const_object, m_static_method))
    {
        error_stream.PutCString ("error: couldn't construct expression body");
        return false;
    }
    
    if (log)
        log->Printf("Parsing the following code:\n%s", m_transformed_text.c_str());
    
    ////////////////////////////////////
    // Set up the target and compiler
    //
    
    Target *target = exe_ctx.GetTargetPtr();
    
    if (!target)
    {
        error_stream.PutCString ("error: invalid target\n");
        return false;
    }
    
    //////////////////////////
    // Parse the expression
    //
        
    m_expr_decl_map.reset(new ClangExpressionDeclMap(keep_result_in_memory, exe_ctx));
    
    if (!m_expr_decl_map->WillParse(exe_ctx))
    {
        error_stream.PutCString ("error: current process state is unsuitable for expression parsing\n");
        return false;
    }
    
    Process *process = exe_ctx.GetProcessPtr();
    ExecutionContextScope *exe_scope = process;
    
    if (!exe_scope)
        exe_scope = exe_ctx.GetTargetPtr();
    
    ClangExpressionParser parser(exe_scope, *this);
    
    unsigned num_errors = parser.Parse (error_stream);
    
    if (num_errors)
    {
        error_stream.Printf ("error: %d errors parsing expression\n", num_errors);
        
        m_expr_decl_map->DidParse();
        
        return false;
    }
    
    //////////////////////////////////////////////////////////////////////////////////////////
    // Prepare the output of the parser for execution, evaluating it statically if possible
    //
        
    if (execution_policy != eExecutionPolicyNever && process)
        m_data_allocator.reset(new ProcessDataAllocator(*process));
    
    Error jit_error = parser.PrepareForExecution (m_jit_alloc,
                                                  m_jit_start_addr,
                                                  m_jit_end_addr,
                                                  exe_ctx,
                                                  m_data_allocator.get(),
                                                  m_evaluated_statically,
                                                  m_const_result,
                                                  execution_policy);
    
    if (log && m_data_allocator.get())
    {
        StreamString dump_string;
        m_data_allocator->Dump(dump_string);
        
        log->Printf("Data buffer contents:\n%s", dump_string.GetString().c_str());
    }
        
    if (jit_error.Success())
    {
        if (process && m_jit_alloc != LLDB_INVALID_ADDRESS)
            m_jit_process_wp = lldb::ProcessWP(process->shared_from_this());
        return true;
    }
    else
    {
        const char *error_cstr = jit_error.AsCString();
        if (error_cstr && error_cstr[0])
            error_stream.Printf ("error: %s\n", error_cstr);
        else
            error_stream.Printf ("error: expression can't be interpreted or run\n");
        return false;
    }
}

bool
ClangUserExpression::PrepareToExecuteJITExpression (Stream &error_stream,
                                                    ExecutionContext &exe_ctx,
                                                    lldb::addr_t &struct_address,
                                                    lldb::addr_t &object_ptr,
                                                    lldb::addr_t &cmd_ptr)
{
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));

    lldb::TargetSP target;
    lldb::ProcessSP process;
    lldb::StackFrameSP frame;
    
    if (!LockAndCheckContext(exe_ctx,
                             target,
                             process, 
                             frame))
    {
        error_stream.Printf("The context has changed before we could JIT the expression!");
        return false;
    }
    
    if (m_jit_start_addr != LLDB_INVALID_ADDRESS)
    {
        Error materialize_error;
        
        if (m_needs_object_ptr)
        {
            ConstString object_name;
            
            if (m_cplusplus)
            {
                object_name.SetCString("this");
            }
            else if (m_objectivec)
            {
                object_name.SetCString("self");
            }
            else
            {
                error_stream.Printf("Need object pointer but don't know the language\n");
                return false;
            }
            
            if (!(m_expr_decl_map->GetObjectPointer(object_ptr, object_name, materialize_error)))
            {
                error_stream.Printf("warning: couldn't get required object pointer (substituting NULL): %s\n", materialize_error.AsCString());
                object_ptr = 0;
            }
            
            if (m_objectivec)
            {
                ConstString cmd_name("_cmd");
                
                if (!(m_expr_decl_map->GetObjectPointer(cmd_ptr, cmd_name, materialize_error, true)))
                {
                    error_stream.Printf("warning: couldn't get object pointer (substituting NULL): %s\n", materialize_error.AsCString());
                    cmd_ptr = 0;
                }
            }
        }
                
        if (!m_expr_decl_map->Materialize(struct_address, materialize_error))
        {
            error_stream.Printf("Couldn't materialize struct: %s\n", materialize_error.AsCString());
            return false;
        }

#if 0
		// jingham: look here
        StreamFile logfile ("/tmp/exprs.txt", "a");
        logfile.Printf("0x%16.16llx: thread = 0x%4.4x, expr = '%s'\n", m_jit_start_addr, exe_ctx.thread ? exe_ctx.thread->GetID() : -1, m_expr_text.c_str());
#endif
        
        if (log)
        {
            log->Printf("-- [ClangUserExpression::PrepareToExecuteJITExpression] Materializing for execution --");
            
            log->Printf("  Function address  : 0x%llx", (uint64_t)m_jit_start_addr);
            
            if (m_needs_object_ptr)
                log->Printf("  Object pointer    : 0x%llx", (uint64_t)object_ptr);
            
            log->Printf("  Structure address : 0x%llx", (uint64_t)struct_address);
                    
            StreamString args;
            
            Error dump_error;
            
            if (struct_address)
            {
                if (!m_expr_decl_map->DumpMaterializedStruct(args, dump_error))
                {
                    log->Printf("  Couldn't extract variable values : %s", dump_error.AsCString("unknown error"));
                }
                else
                {
                    log->Printf("  Structure contents:\n%s", args.GetData());
                }
            }
        }
    }
    return true;
}

ThreadPlan *
ClangUserExpression::GetThreadPlanToExecuteJITExpression (Stream &error_stream,
                                                          ExecutionContext &exe_ctx)
{
    lldb::addr_t struct_address;
            
    lldb::addr_t object_ptr = 0;
    lldb::addr_t cmd_ptr = 0;
    
    PrepareToExecuteJITExpression (error_stream, exe_ctx, struct_address, object_ptr, cmd_ptr);
    
    // FIXME: This should really return a ThreadPlanCallUserExpression, in order to make sure that we don't release the
    // ClangUserExpression resources before the thread plan finishes execution in the target.  But because we are 
    // forcing unwind_on_error to be true here, in practical terms that can't happen.
    
    return ClangFunction::GetThreadPlanToCallFunction (exe_ctx, 
                                                       m_jit_start_addr, 
                                                       struct_address, 
                                                       error_stream,
                                                       true,
                                                       true, 
                                                       (m_needs_object_ptr ? &object_ptr : NULL),
                                                       (m_needs_object_ptr && m_objectivec) ? &cmd_ptr : NULL);
}

bool
ClangUserExpression::FinalizeJITExecution (Stream &error_stream,
                                           ExecutionContext &exe_ctx,
                                           lldb::ClangExpressionVariableSP &result,
                                           lldb::addr_t function_stack_pointer)
{
    Error expr_error;
    
    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    
    if (log)
    {
        log->Printf("-- [ClangUserExpression::FinalizeJITExecution] Dematerializing after execution --");
    
        StreamString args;
        
        Error dump_error;
        
        if (!m_expr_decl_map->DumpMaterializedStruct(args, dump_error))
        {
            log->Printf("  Couldn't extract variable values : %s", dump_error.AsCString("unknown error"));
        }
        else
        {
            log->Printf("  Structure contents:\n%s", args.GetData());
        }
    }
    
    lldb::addr_t function_stack_bottom = function_stack_pointer - Host::GetPageSize();
    
        
    if (!m_expr_decl_map->Dematerialize(result, function_stack_pointer, function_stack_bottom, expr_error))
    {
        error_stream.Printf ("Couldn't dematerialize struct : %s\n", expr_error.AsCString("unknown error"));
        return false;
    }
    
    if (result)
        result->TransferAddress();
    
    return true;
}        

ExecutionResults
ClangUserExpression::Execute (Stream &error_stream,
                              ExecutionContext &exe_ctx,
                              bool discard_on_error,
                              ClangUserExpression::ClangUserExpressionSP &shared_ptr_to_me,
                              lldb::ClangExpressionVariableSP &result,
                              bool run_others,
                              uint32_t timeout_usec)
{
    // The expression log is quite verbose, and if you're just tracking the execution of the
    // expression, it's quite convenient to have these logs come out with the STEP log as well.
    lldb::LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_EXPRESSIONS | LIBLLDB_LOG_STEP));

    if (m_jit_start_addr != LLDB_INVALID_ADDRESS)
    {
        lldb::addr_t struct_address = LLDB_INVALID_ADDRESS;
                
        lldb::addr_t object_ptr = 0;
        lldb::addr_t cmd_ptr = 0;
        
        if (!PrepareToExecuteJITExpression (error_stream, exe_ctx, struct_address, object_ptr, cmd_ptr))
        {
            error_stream.Printf("Errored out in %s, couldn't PrepareToExecuteJITExpression", __FUNCTION__);
            return eExecutionSetupError;
        }
        
        const bool stop_others = true;
        const bool try_all_threads = true;
        
        Address wrapper_address (m_jit_start_addr);
        lldb::ThreadPlanSP call_plan_sp(new ThreadPlanCallUserExpression (exe_ctx.GetThreadRef(), 
                                                                          wrapper_address, 
                                                                          struct_address, 
                                                                          stop_others, 
                                                                          discard_on_error, 
                                                                          (m_needs_object_ptr ? &object_ptr : NULL),
                                                                          ((m_needs_object_ptr && m_objectivec) ? &cmd_ptr : NULL),
                                                                          shared_ptr_to_me));
        
        if (!call_plan_sp || !call_plan_sp->ValidatePlan (NULL))
            return eExecutionSetupError;
        
        lldb::addr_t function_stack_pointer = static_cast<ThreadPlanCallFunction *>(call_plan_sp.get())->GetFunctionStackPointer();
    
        call_plan_sp->SetPrivate(true);
    
        if (log)
            log->Printf("-- [ClangUserExpression::Execute] Execution of expression begins --");
        
        if (exe_ctx.GetProcessPtr())
            exe_ctx.GetProcessPtr()->SetRunningUserExpression(true);
            
        ExecutionResults execution_result = exe_ctx.GetProcessRef().RunThreadPlan (exe_ctx, 
                                                                                   call_plan_sp, 
                                                                                   stop_others, 
                                                                                   try_all_threads, 
                                                                                   discard_on_error,
                                                                                   timeout_usec, 
                                                                                   error_stream);
        
        if (exe_ctx.GetProcessPtr())
            exe_ctx.GetProcessPtr()->SetRunningUserExpression(false);
            
        if (log)
            log->Printf("-- [ClangUserExpression::Execute] Execution of expression completed --");

        if (execution_result == eExecutionInterrupted)
        {
            const char *error_desc = NULL;
            
            if (call_plan_sp)
            {
                lldb::StopInfoSP real_stop_info_sp = call_plan_sp->GetRealStopInfo();
                if (real_stop_info_sp)
                    error_desc = real_stop_info_sp->GetDescription();
            }
            if (error_desc)
                error_stream.Printf ("Execution was interrupted, reason: %s.", error_desc);
            else
                error_stream.Printf ("Execution was interrupted.");
                
            if (discard_on_error)
                error_stream.Printf ("\nThe process has been returned to the state before execution.");
            else
                error_stream.Printf ("\nThe process has been left at the point where it was interrupted.");

            return execution_result;
        }
        else if (execution_result != eExecutionCompleted)
        {
            error_stream.Printf ("Couldn't execute function; result was %s\n", Process::ExecutionResultAsCString (execution_result));
            return execution_result;
        }
        
        if  (FinalizeJITExecution (error_stream, exe_ctx, result, function_stack_pointer))
            return eExecutionCompleted;
        else
        {
            error_stream.Printf("Errored out in %s: Couldn't FinalizeJITExpression", __FUNCTION__);
            return eExecutionSetupError;
        }
    }
    else
    {
        error_stream.Printf("Expression can't be run, because there is no JIT compiled function");
        return eExecutionSetupError;
    }
}

ExecutionResults
ClangUserExpression::Evaluate (ExecutionContext &exe_ctx,
                               lldb_private::ExecutionPolicy execution_policy,
                               lldb::LanguageType language,
                               ResultType desired_type,
                               bool discard_on_error,
                               const char *expr_cstr,
                               const char *expr_prefix,
                               lldb::ValueObjectSP &result_valobj_sp,
                               bool run_others,
                               uint32_t timeout_usec)
{
    Error error;
    return EvaluateWithError (exe_ctx,
                              execution_policy,
                              language,
                              desired_type,
                              discard_on_error,
                              expr_cstr,
                              expr_prefix,
                              result_valobj_sp,
                              error,
                              run_others,
                              timeout_usec);
}

ExecutionResults
ClangUserExpression::EvaluateWithError (ExecutionContext &exe_ctx,
                                        lldb_private::ExecutionPolicy execution_policy,
                                        lldb::LanguageType language,
                                        ResultType desired_type,
                                        bool discard_on_error,
                                        const char *expr_cstr,
                                        const char *expr_prefix,
                                        lldb::ValueObjectSP &result_valobj_sp,
                                        Error &error,
                                        bool run_others,
                                        uint32_t timeout_usec)
{
    lldb::LogSP log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_EXPRESSIONS | LIBLLDB_LOG_STEP));

    ExecutionResults execution_results = eExecutionSetupError;
    
    Process *process = exe_ctx.GetProcessPtr();

    if (process == NULL || process->GetState() != lldb::eStateStopped)
    {
        if (execution_policy == eExecutionPolicyAlways)
        {
            if (log)
                log->Printf("== [ClangUserExpression::Evaluate] Expression may not run, but is not constant ==");
            
            error.SetErrorString ("expression needed to run but couldn't");
            
            return execution_results;
        }
    }
    
    if (process == NULL || !process->CanJIT())
        execution_policy = eExecutionPolicyNever;
    
    ClangUserExpressionSP user_expression_sp (new ClangUserExpression (expr_cstr, expr_prefix, language, desired_type));

    StreamString error_stream;
        
    if (log)
        log->Printf("== [ClangUserExpression::Evaluate] Parsing expression %s ==", expr_cstr);
    
    const bool keep_expression_in_memory = true;
    
    if (!user_expression_sp->Parse (error_stream, exe_ctx, execution_policy, keep_expression_in_memory))
    {
        if (error_stream.GetString().empty())
            error.SetErrorString ("expression failed to parse, unknown error");
        else
            error.SetErrorString (error_stream.GetString().c_str());
    }
    else
    {
        lldb::ClangExpressionVariableSP expr_result;

        if (user_expression_sp->EvaluatedStatically())
        {
            if (log)
                log->Printf("== [ClangUserExpression::Evaluate] Expression evaluated as a constant ==");
            
            if (user_expression_sp->m_const_result)
                result_valobj_sp = user_expression_sp->m_const_result->GetValueObject();
            else
                error.SetError(ClangUserExpression::kNoResult, lldb::eErrorTypeGeneric);
            
            execution_results = eExecutionCompleted;
        }
        else if (execution_policy == eExecutionPolicyNever)
        {
            if (log)
                log->Printf("== [ClangUserExpression::Evaluate] Expression may not run, but is not constant ==");
            
            if (error_stream.GetString().empty())
                error.SetErrorString ("expression needed to run but couldn't");
        }
        else
        {    
            error_stream.GetString().clear();
            
            if (log)
                log->Printf("== [ClangUserExpression::Evaluate] Executing expression ==");

            execution_results = user_expression_sp->Execute (error_stream, 
                                                             exe_ctx, 
                                                             discard_on_error,
                                                             user_expression_sp, 
                                                             expr_result,
                                                             run_others,
                                                             timeout_usec);
            
            if (execution_results != eExecutionCompleted)
            {
                if (log)
                    log->Printf("== [ClangUserExpression::Evaluate] Execution completed abnormally ==");
                
                if (error_stream.GetString().empty())
                    error.SetErrorString ("expression failed to execute, unknown error");
                else
                    error.SetErrorString (error_stream.GetString().c_str());
            }
            else 
            {
                if (expr_result)
                {
                    result_valobj_sp = expr_result->GetValueObject();
                    
                    if (log)
                        log->Printf("== [ClangUserExpression::Evaluate] Execution completed normally with result %s ==", result_valobj_sp->GetValueAsCString());
                }
                else
                {
                    if (log)
                        log->Printf("== [ClangUserExpression::Evaluate] Execution completed normally with no result ==");
                    
                    error.SetError(ClangUserExpression::kNoResult, lldb::eErrorTypeGeneric);
                }
            }
        }
    }
    
    if (result_valobj_sp.get() == NULL)
        result_valobj_sp = ValueObjectConstResult::Create (NULL, error);

    return execution_results;
}
