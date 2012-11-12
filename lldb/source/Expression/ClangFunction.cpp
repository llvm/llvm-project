//===-- ClangFunction.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


// C Includes
// C++ Includes
// Other libraries and framework includes
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/Module.h"

// Project includes
#include "lldb/Expression/ASTStructExtractor.h"
#include "lldb/Expression/ClangExpressionParser.h"
#include "lldb/Expression/ClangFunction.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/State.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectList.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/ThreadPlanCallFunction.h"
#include "lldb/Core/Log.h"

using namespace lldb_private;

//----------------------------------------------------------------------
// ClangFunction constructor
//----------------------------------------------------------------------
ClangFunction::ClangFunction 
(
    ExecutionContextScope &exe_scope,
    ClangASTContext *ast_context, 
    void *return_qualtype, 
    const Address& functionAddress, 
    const ValueList &arg_value_list
) :
    m_function_ptr (NULL),
    m_function_addr (functionAddress),
    m_function_return_qual_type(return_qualtype),
    m_clang_ast_context (ast_context),
    m_wrapper_function_name ("__lldb_caller_function"),
    m_wrapper_struct_name ("__lldb_caller_struct"),
    m_wrapper_args_addrs (),
    m_arg_values (arg_value_list),
    m_compiled (false),
    m_JITted (false)
{
    m_jit_process_wp = lldb::ProcessWP(exe_scope.CalculateProcess());
    // Can't make a ClangFunction without a process.
    assert (m_jit_process_wp.lock());
}

ClangFunction::ClangFunction
(
    ExecutionContextScope &exe_scope,
    Function &function, 
    ClangASTContext *ast_context, 
    const ValueList &arg_value_list
) :
    m_function_ptr (&function),
    m_function_addr (),
    m_function_return_qual_type (),
    m_clang_ast_context (ast_context),
    m_wrapper_function_name ("__lldb_function_caller"),
    m_wrapper_struct_name ("__lldb_caller_struct"),
    m_wrapper_args_addrs (),
    m_arg_values (arg_value_list),
    m_compiled (false),
    m_JITted (false)
{
    m_jit_process_wp = lldb::ProcessWP(exe_scope.CalculateProcess());
    // Can't make a ClangFunction without a process.
    assert (m_jit_process_wp.lock());

    m_function_addr = m_function_ptr->GetAddressRange().GetBaseAddress();
    m_function_return_qual_type = m_function_ptr->GetReturnClangType();
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ClangFunction::~ClangFunction()
{
}

unsigned
ClangFunction::CompileFunction (Stream &errors)
{
    if (m_compiled)
        return 0;
    
    // FIXME: How does clang tell us there's no return value?  We need to handle that case.
    unsigned num_errors = 0;
    
    std::string return_type_str (ClangASTType::GetTypeNameForOpaqueQualType (m_clang_ast_context->getASTContext(),
                                                                             m_function_return_qual_type));
    
    // Cons up the function we're going to wrap our call in, then compile it...
    // We declare the function "extern "C"" because the compiler might be in C++
    // mode which would mangle the name and then we couldn't find it again...
    m_wrapper_function_text.clear();
    m_wrapper_function_text.append ("extern \"C\" void ");
    m_wrapper_function_text.append (m_wrapper_function_name);
    m_wrapper_function_text.append (" (void *input)\n{\n    struct ");
    m_wrapper_function_text.append (m_wrapper_struct_name);
    m_wrapper_function_text.append (" \n  {\n");
    m_wrapper_function_text.append ("    ");
    m_wrapper_function_text.append (return_type_str);
    m_wrapper_function_text.append (" (*fn_ptr) (");

    // Get the number of arguments.  If we have a function type and it is prototyped,
    // trust that, otherwise use the values we were given.

    // FIXME: This will need to be extended to handle Variadic functions.  We'll need
    // to pull the defined arguments out of the function, then add the types from the
    // arguments list for the variable arguments.

    uint32_t num_args = UINT32_MAX;
    bool trust_function = false;
    // GetArgumentCount returns -1 for an unprototyped function.
    if (m_function_ptr)
    {
        int num_func_args = m_function_ptr->GetArgumentCount();
        if (num_func_args >= 0)
            trust_function = true;
        else
            num_args = num_func_args;
    }

    if (num_args == UINT32_MAX)
        num_args = m_arg_values.GetSize();

    std::string args_buffer;  // This one stores the definition of all the args in "struct caller".
    std::string args_list_buffer;  // This one stores the argument list called from the structure.
    for (size_t i = 0; i < num_args; i++)
    {
        std::string type_name;

        if (trust_function)
        {
            lldb::clang_type_t arg_clang_type = m_function_ptr->GetArgumentTypeAtIndex(i);
            type_name = ClangASTType::GetTypeNameForOpaqueQualType (m_clang_ast_context->getASTContext(),
                                                                    arg_clang_type);
        }
        else
        {
            Value *arg_value = m_arg_values.GetValueAtIndex(i);
            lldb::clang_type_t clang_qual_type = arg_value->GetClangType ();
            if (clang_qual_type != NULL)
            {
                type_name = ClangASTType::GetTypeNameForOpaqueQualType (m_clang_ast_context->getASTContext(),
                                                                        clang_qual_type);
            }
            else
            {   
                errors.Printf("Could not determine type of input value %lu.", i);
                return 1;
            }
        }

        m_wrapper_function_text.append (type_name);
        if (i < num_args - 1)
            m_wrapper_function_text.append (", ");

        char arg_buf[32];
        args_buffer.append ("    ");
        args_buffer.append (type_name);
        snprintf(arg_buf, 31, "arg_%llu", (uint64_t)i);
        args_buffer.push_back (' ');
        args_buffer.append (arg_buf);
        args_buffer.append (";\n");

        args_list_buffer.append ("__lldb_fn_data->");
        args_list_buffer.append (arg_buf);
        if (i < num_args - 1)
            args_list_buffer.append (", ");

    }
    m_wrapper_function_text.append (");\n"); // Close off the function calling prototype.

    m_wrapper_function_text.append (args_buffer);

    m_wrapper_function_text.append ("    ");
    m_wrapper_function_text.append (return_type_str);
    m_wrapper_function_text.append (" return_value;");
    m_wrapper_function_text.append ("\n  };\n  struct ");
    m_wrapper_function_text.append (m_wrapper_struct_name);
    m_wrapper_function_text.append ("* __lldb_fn_data = (struct ");
    m_wrapper_function_text.append (m_wrapper_struct_name);
    m_wrapper_function_text.append (" *) input;\n");

    m_wrapper_function_text.append ("  __lldb_fn_data->return_value = __lldb_fn_data->fn_ptr (");
    m_wrapper_function_text.append (args_list_buffer);
    m_wrapper_function_text.append (");\n}\n");

    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_EXPRESSIONS));
    if (log)
        log->Printf ("Expression: \n\n%s\n\n", m_wrapper_function_text.c_str());
        
    // Okay, now compile this expression
    
    lldb::ProcessSP jit_process_sp(m_jit_process_wp.lock());
    if (jit_process_sp)
    {
        m_parser.reset(new ClangExpressionParser(jit_process_sp.get(), *this));
        
        num_errors = m_parser->Parse (errors);
    }
    else
    {
        errors.Printf("no process - unable to inject function");
        num_errors = 1;
    }
    
    m_compiled = (num_errors == 0);
    
    if (!m_compiled)
        return num_errors;

    return num_errors;
}

bool
ClangFunction::WriteFunctionWrapper (ExecutionContext &exe_ctx, Stream &errors)
{
    Process *process = exe_ctx.GetProcessPtr();

    if (!process)
        return false;
    
    lldb::ProcessSP jit_process_sp(m_jit_process_wp.lock());
    
    if (process != jit_process_sp.get())
        return false;
    
    if (!m_compiled)
        return false;

    if (m_JITted)
        return true;
    
    lldb::ClangExpressionVariableSP const_result;
    
    bool evaluated_statically = false; // should stay that way
    
    Error jit_error (m_parser->PrepareForExecution (m_jit_alloc,
                                                    m_jit_start_addr,
                                                    m_jit_end_addr,
                                                    exe_ctx, 
                                                    NULL,
                                                    evaluated_statically,
                                                    const_result,
                                                    eExecutionPolicyAlways));
    
    if (!jit_error.Success())
        return false;
    if (process && m_jit_alloc != LLDB_INVALID_ADDRESS)
        m_jit_process_wp = lldb::ProcessWP(process->shared_from_this());

    return true;
}

bool
ClangFunction::WriteFunctionArguments (ExecutionContext &exe_ctx, lldb::addr_t &args_addr_ref, Stream &errors)
{
    return WriteFunctionArguments(exe_ctx, args_addr_ref, m_function_addr, m_arg_values, errors);    
}

// FIXME: Assure that the ValueList we were passed in is consistent with the one that defined this function.

bool
ClangFunction::WriteFunctionArguments (ExecutionContext &exe_ctx, 
                                       lldb::addr_t &args_addr_ref, 
                                       Address function_address, 
                                       ValueList &arg_values, 
                                       Stream &errors)
{
    // All the information to reconstruct the struct is provided by the
    // StructExtractor.
    if (!m_struct_valid)
    {
        errors.Printf("Argument information was not correctly parsed, so the function cannot be called.");
        return false;
    }
        
    Error error;
    using namespace clang;
    ExecutionResults return_value = eExecutionSetupError;

    Process *process = exe_ctx.GetProcessPtr();

    if (process == NULL)
        return return_value;

    lldb::ProcessSP jit_process_sp(m_jit_process_wp.lock());
    
    if (process != jit_process_sp.get())
        return false;
                
    if (args_addr_ref == LLDB_INVALID_ADDRESS)
    {
        args_addr_ref = process->AllocateMemory(m_struct_size, lldb::ePermissionsReadable|lldb::ePermissionsWritable, error);
        if (args_addr_ref == LLDB_INVALID_ADDRESS)
            return false;
        m_wrapper_args_addrs.push_back (args_addr_ref);
    } 
    else 
    {
        // Make sure this is an address that we've already handed out.
        if (find (m_wrapper_args_addrs.begin(), m_wrapper_args_addrs.end(), args_addr_ref) == m_wrapper_args_addrs.end())
        {
            return false;
        }
    }

    // TODO: verify fun_addr needs to be a callable address
    Scalar fun_addr (function_address.GetCallableLoadAddress(exe_ctx.GetTargetPtr()));
    int first_offset = m_member_offsets[0];
    process->WriteScalarToMemory(args_addr_ref + first_offset, fun_addr, process->GetAddressByteSize(), error);

    // FIXME: We will need to extend this for Variadic functions.

    Error value_error;
    
    size_t num_args = arg_values.GetSize();
    if (num_args != m_arg_values.GetSize())
    {
        errors.Printf ("Wrong number of arguments - was: %lu should be: %lu", num_args, m_arg_values.GetSize());
        return false;
    }
    
    for (size_t i = 0; i < num_args; i++)
    {
        // FIXME: We should sanity check sizes.

        int offset = m_member_offsets[i+1]; // Clang sizes are in bytes.
        Value *arg_value = arg_values.GetValueAtIndex(i);
        
        // FIXME: For now just do scalars:
        
        // Special case: if it's a pointer, don't do anything (the ABI supports passing cstrings)
        
        if (arg_value->GetValueType() == Value::eValueTypeHostAddress &&
            arg_value->GetContextType() == Value::eContextTypeClangType &&
            ClangASTContext::IsPointerType(arg_value->GetClangType()))
            continue;
        
        const Scalar &arg_scalar = arg_value->ResolveValue(&exe_ctx, m_clang_ast_context->getASTContext());

        if (!process->WriteScalarToMemory(args_addr_ref + offset, arg_scalar, arg_scalar.GetByteSize(), error))
            return false;
    }

    return true;
}

bool
ClangFunction::InsertFunction (ExecutionContext &exe_ctx, lldb::addr_t &args_addr_ref, Stream &errors)
{
    using namespace clang;
    
    if (CompileFunction(errors) != 0)
        return false;
    if (!WriteFunctionWrapper(exe_ctx, errors))
        return false;
    if (!WriteFunctionArguments(exe_ctx, args_addr_ref, errors))
        return false;

    lldb::LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    if (log)
        log->Printf ("Call Address: 0x%llx Struct Address: 0x%llx.\n", m_jit_start_addr, args_addr_ref);
        
    return true;
}

ThreadPlan *
ClangFunction::GetThreadPlanToCallFunction (ExecutionContext &exe_ctx, 
                                            lldb::addr_t func_addr, 
                                            lldb::addr_t &args_addr, 
                                            Stream &errors, 
                                            bool stop_others, 
                                            bool discard_on_error, 
                                            lldb::addr_t *this_arg,
                                            lldb::addr_t *cmd_arg)
{
    // FIXME: Use the errors Stream for better error reporting.
    Thread *thread = exe_ctx.GetThreadPtr();
    if (thread == NULL)
    {
        errors.Printf("Can't call a function without a valid thread.");
        return NULL;
    }

    // Okay, now run the function:

    Address wrapper_address (func_addr);
    ThreadPlan *new_plan = new ThreadPlanCallFunction (*thread, 
                                                       wrapper_address,
                                                       ClangASTType(),
                                                       args_addr,
                                                       stop_others, 
                                                       discard_on_error,
                                                       this_arg,
                                                       cmd_arg);
    new_plan->SetIsMasterPlan(true);
    new_plan->SetOkayToDiscard (false);
    return new_plan;
}

bool
ClangFunction::FetchFunctionResults (ExecutionContext &exe_ctx, lldb::addr_t args_addr, Value &ret_value)
{
    // Read the return value - it is the last field in the struct:
    // FIXME: How does clang tell us there's no return value?  We need to handle that case.
    // FIXME: Create our ThreadPlanCallFunction with the return ClangASTType, and then use GetReturnValueObject
    // to fetch the value.  That way we can fetch any values we need.
    Process *process = exe_ctx.GetProcessPtr();
    
    if (process == NULL)
        return false;

    lldb::ProcessSP jit_process_sp(m_jit_process_wp.lock());
    
    if (process != jit_process_sp.get())
        return false;
                
    Error error;
    ret_value.GetScalar() = process->ReadUnsignedIntegerFromMemory (args_addr + m_return_offset, m_return_size, 0, error);

    if (error.Fail())
        return false;

    ret_value.SetContext (Value::eContextTypeClangType, m_function_return_qual_type);
    ret_value.SetValueType(Value::eValueTypeScalar);
    return true;
}

void
ClangFunction::DeallocateFunctionResults (ExecutionContext &exe_ctx, lldb::addr_t args_addr)
{
    std::list<lldb::addr_t>::iterator pos;
    pos = std::find(m_wrapper_args_addrs.begin(), m_wrapper_args_addrs.end(), args_addr);
    if (pos != m_wrapper_args_addrs.end())
        m_wrapper_args_addrs.erase(pos);
    
    exe_ctx.GetProcessRef().DeallocateMemory(args_addr);
}

ExecutionResults
ClangFunction::ExecuteFunction(ExecutionContext &exe_ctx, Stream &errors, Value &results)
{
    return ExecuteFunction (exe_ctx, errors, 1000, true, results);
}

ExecutionResults
ClangFunction::ExecuteFunction(ExecutionContext &exe_ctx, Stream &errors, bool stop_others, Value &results)
{
    const bool try_all_threads = false;
    const bool discard_on_error = true;
    return ExecuteFunction (exe_ctx, NULL, errors, stop_others, 0UL, try_all_threads, discard_on_error, results);
}

ExecutionResults
ClangFunction::ExecuteFunction(
        ExecutionContext &exe_ctx, 
        Stream &errors, 
        uint32_t timeout_usec, 
        bool try_all_threads, 
        Value &results)
{
    const bool stop_others = true;
    const bool discard_on_error = true;
    return ExecuteFunction (exe_ctx, NULL, errors, stop_others, timeout_usec,
                            try_all_threads, discard_on_error, results);
}

// This is the static function
ExecutionResults 
ClangFunction::ExecuteFunction (
        ExecutionContext &exe_ctx, 
        lldb::addr_t function_address, 
        lldb::addr_t &void_arg,
        bool stop_others,
        bool try_all_threads,
        bool discard_on_error,
        uint32_t timeout_usec,
        Stream &errors,
        lldb::addr_t *this_arg)
{
    lldb::ThreadPlanSP call_plan_sp (ClangFunction::GetThreadPlanToCallFunction (exe_ctx, 
                                                                                 function_address, 
                                                                                 void_arg, 
                                                                                 errors, 
                                                                                 stop_others, 
                                                                                 discard_on_error, 
                                                                                 this_arg));
    if (!call_plan_sp)
        return eExecutionSetupError;
    
    call_plan_sp->SetPrivate(true);
    
    // <rdar://problem/12027563> we need to make sure we record the fact that we are running an expression here
    // otherwise this fact will fail to be recorded when fetching an Objective-C object description
    if (exe_ctx.GetProcessPtr())
        exe_ctx.GetProcessPtr()->SetRunningUserExpression(true);
    
    ExecutionResults results = exe_ctx.GetProcessRef().RunThreadPlan (exe_ctx, call_plan_sp,
                                                                      stop_others, 
                                                                      try_all_threads, 
                                                                      discard_on_error,
                                                                      timeout_usec,
                                                                      errors);
    
    if (exe_ctx.GetProcessPtr())
        exe_ctx.GetProcessPtr()->SetRunningUserExpression(false);
    
    return results;
}

ExecutionResults
ClangFunction::ExecuteFunction(
        ExecutionContext &exe_ctx, 
        lldb::addr_t *args_addr_ptr, 
        Stream &errors, 
        bool stop_others, 
        uint32_t timeout_usec, 
        bool try_all_threads,
        bool discard_on_error, 
        Value &results)
{
    using namespace clang;
    ExecutionResults return_value = eExecutionSetupError;
    
    lldb::addr_t args_addr;
    
    if (args_addr_ptr != NULL)
        args_addr = *args_addr_ptr;
    else
        args_addr = LLDB_INVALID_ADDRESS;
        
    if (CompileFunction(errors) != 0)
        return eExecutionSetupError;
    
    if (args_addr == LLDB_INVALID_ADDRESS)
    {
        if (!InsertFunction(exe_ctx, args_addr, errors))
            return eExecutionSetupError;
    }
    
    return_value = ClangFunction::ExecuteFunction (exe_ctx, 
                                                   m_jit_start_addr, 
                                                   args_addr, 
                                                   stop_others, 
                                                   try_all_threads, 
                                                   discard_on_error, 
                                                   timeout_usec, 
                                                   errors);

    if (args_addr_ptr != NULL)
        *args_addr_ptr = args_addr;
    
    if (return_value != eExecutionCompleted)
        return return_value;

    FetchFunctionResults(exe_ctx, args_addr, results);
    
    if (args_addr_ptr == NULL)
        DeallocateFunctionResults(exe_ctx, args_addr);
        
    return eExecutionCompleted;
}

clang::ASTConsumer *
ClangFunction::ASTTransformer (clang::ASTConsumer *passthrough)
{
    return new ASTStructExtractor(passthrough, m_wrapper_struct_name.c_str(), *this);
}
