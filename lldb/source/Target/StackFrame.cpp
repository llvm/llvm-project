//===-- StackFrame.cpp ------------------------------------------*- C++ -*-===//
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
// Project includes
#include "lldb/Target/StackFrame.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Disassembler.h"
#include "lldb/Core/FormatEntity.h"
#include "lldb/Core/Mangled.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Core/ValueObjectVariable.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/SymbolContextScope.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/VariableList.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;

// The first bits in the flags are reserved for the SymbolContext::Scope bits
// so we know if we have tried to look up information in our internal symbol
// context (m_sc) already.
#define RESOLVED_FRAME_CODE_ADDR (uint32_t(eSymbolContextEverything + 1))
#define RESOLVED_FRAME_ID_SYMBOL_SCOPE (RESOLVED_FRAME_CODE_ADDR << 1)
#define GOT_FRAME_BASE (RESOLVED_FRAME_ID_SYMBOL_SCOPE << 1)
#define RESOLVED_VARIABLES (GOT_FRAME_BASE << 1)
#define RESOLVED_GLOBAL_VARIABLES (RESOLVED_VARIABLES << 1)

StackFrame::StackFrame(const ThreadSP &thread_sp, user_id_t frame_idx,
                       user_id_t unwind_frame_index, addr_t cfa,
                       bool cfa_is_valid, addr_t pc, uint32_t stop_id,
                       bool stop_id_is_valid, bool is_history_frame,
                       const SymbolContext *sc_ptr)
    : m_thread_wp(thread_sp), m_frame_index(frame_idx),
      m_concrete_frame_index(unwind_frame_index), m_reg_context_sp(),
      m_id(pc, cfa, nullptr), m_frame_code_addr(pc), m_sc(), m_flags(),
      m_frame_base(), m_frame_base_error(), m_cfa_is_valid(cfa_is_valid),
      m_stop_id(stop_id), m_stop_id_is_valid(stop_id_is_valid),
      m_is_history_frame(is_history_frame), m_variable_list_sp(),
      m_variable_list_value_objects(), m_disassembly(),
      m_mutex(Mutex::eMutexTypeRecursive) {
  // If we don't have a CFA value, use the frame index for our StackID so that
  // recursive
  // functions properly aren't confused with one another on a history stack.
  if (m_is_history_frame && !m_cfa_is_valid) {
    m_id.SetCFA(m_frame_index);
  }

  if (sc_ptr != nullptr) {
    m_sc = *sc_ptr;
    m_flags.Set(m_sc.GetResolvedMask());
  }
}

StackFrame::StackFrame(const ThreadSP &thread_sp, user_id_t frame_idx,
                       user_id_t unwind_frame_index,
                       const RegisterContextSP &reg_context_sp, addr_t cfa,
                       addr_t pc, const SymbolContext *sc_ptr)
    : m_thread_wp(thread_sp), m_frame_index(frame_idx),
      m_concrete_frame_index(unwind_frame_index),
      m_reg_context_sp(reg_context_sp), m_id(pc, cfa, nullptr),
      m_frame_code_addr(pc), m_sc(), m_flags(), m_frame_base(),
      m_frame_base_error(), m_cfa_is_valid(true), m_stop_id(0),
      m_stop_id_is_valid(false), m_is_history_frame(false),
      m_variable_list_sp(), m_variable_list_value_objects(), m_disassembly(),
      m_mutex(Mutex::eMutexTypeRecursive) {
  if (sc_ptr != nullptr) {
    m_sc = *sc_ptr;
    m_flags.Set(m_sc.GetResolvedMask());
  }

  if (reg_context_sp && !m_sc.target_sp) {
    m_sc.target_sp = reg_context_sp->CalculateTarget();
    if (m_sc.target_sp)
      m_flags.Set(eSymbolContextTarget);
  }
}

StackFrame::StackFrame(const ThreadSP &thread_sp, user_id_t frame_idx,
                       user_id_t unwind_frame_index,
                       const RegisterContextSP &reg_context_sp, addr_t cfa,
                       const Address &pc_addr, const SymbolContext *sc_ptr)
    : m_thread_wp(thread_sp), m_frame_index(frame_idx),
      m_concrete_frame_index(unwind_frame_index),
      m_reg_context_sp(reg_context_sp),
      m_id(pc_addr.GetLoadAddress(thread_sp->CalculateTarget().get()), cfa,
           nullptr),
      m_frame_code_addr(pc_addr), m_sc(), m_flags(), m_frame_base(),
      m_frame_base_error(), m_cfa_is_valid(true), m_stop_id(0),
      m_stop_id_is_valid(false), m_is_history_frame(false),
      m_variable_list_sp(), m_variable_list_value_objects(), m_disassembly(),
      m_mutex(Mutex::eMutexTypeRecursive) {
  if (sc_ptr != nullptr) {
    m_sc = *sc_ptr;
    m_flags.Set(m_sc.GetResolvedMask());
  }

  if (!m_sc.target_sp && reg_context_sp) {
    m_sc.target_sp = reg_context_sp->CalculateTarget();
    if (m_sc.target_sp)
      m_flags.Set(eSymbolContextTarget);
  }

  ModuleSP pc_module_sp(pc_addr.GetModule());
  if (!m_sc.module_sp || m_sc.module_sp != pc_module_sp) {
    if (pc_module_sp) {
      m_sc.module_sp = pc_module_sp;
      m_flags.Set(eSymbolContextModule);
    } else {
      m_sc.module_sp.reset();
    }
  }
}

StackFrame::~StackFrame() = default;

StackID &StackFrame::GetStackID() {
  Mutex::Locker locker(m_mutex);
  // Make sure we have resolved the StackID object's symbol context scope if
  // we already haven't looked it up.

  if (m_flags.IsClear(RESOLVED_FRAME_ID_SYMBOL_SCOPE)) {
    if (m_id.GetSymbolContextScope()) {
      // We already have a symbol context scope, we just don't have our
      // flag bit set.
      m_flags.Set(RESOLVED_FRAME_ID_SYMBOL_SCOPE);
    } else {
      // Calculate the frame block and use this for the stack ID symbol
      // context scope if we have one.
      SymbolContextScope *scope = GetFrameBlock();
      if (scope == nullptr) {
        // We don't have a block, so use the symbol
        if (m_flags.IsClear(eSymbolContextSymbol))
          GetSymbolContext(eSymbolContextSymbol);

        // It is ok if m_sc.symbol is nullptr here
        scope = m_sc.symbol;
      }
      // Set the symbol context scope (the accessor will set the
      // RESOLVED_FRAME_ID_SYMBOL_SCOPE bit in m_flags).
      SetSymbolContextScope(scope);
    }
  }
  return m_id;
}

uint32_t StackFrame::GetFrameIndex() const {
  ThreadSP thread_sp = GetThread();
  if (thread_sp)
    return thread_sp->GetStackFrameList()->GetVisibleStackFrameIndex(
        m_frame_index);
  else
    return m_frame_index;
}

void StackFrame::SetSymbolContextScope(SymbolContextScope *symbol_scope) {
  Mutex::Locker locker(m_mutex);
  m_flags.Set(RESOLVED_FRAME_ID_SYMBOL_SCOPE);
  m_id.SetSymbolContextScope(symbol_scope);
}

const Address &StackFrame::GetFrameCodeAddress() {
  Mutex::Locker locker(m_mutex);
  if (m_flags.IsClear(RESOLVED_FRAME_CODE_ADDR) &&
      !m_frame_code_addr.IsSectionOffset()) {
    m_flags.Set(RESOLVED_FRAME_CODE_ADDR);

    // Resolve the PC into a temporary address because if ResolveLoadAddress
    // fails to resolve the address, it will clear the address object...
    ThreadSP thread_sp(GetThread());
    if (thread_sp) {
      TargetSP target_sp(thread_sp->CalculateTarget());
      if (target_sp) {
        if (m_frame_code_addr.SetOpcodeLoadAddress(
                m_frame_code_addr.GetOffset(), target_sp.get(),
                eAddressClassCode)) {
          ModuleSP module_sp(m_frame_code_addr.GetModule());
          if (module_sp) {
            m_sc.module_sp = module_sp;
            m_flags.Set(eSymbolContextModule);
          }
        }
      }
    }
  }
  return m_frame_code_addr;
}

bool StackFrame::ChangePC(addr_t pc) {
  Mutex::Locker locker(m_mutex);
  // We can't change the pc value of a history stack frame - it is immutable.
  if (m_is_history_frame)
    return false;
  m_frame_code_addr.SetRawAddress(pc);
  m_sc.Clear(false);
  m_flags.Reset(0);
  ThreadSP thread_sp(GetThread());
  if (thread_sp)
    thread_sp->ClearStackFrames();
  return true;
}

const char *StackFrame::Disassemble() {
  Mutex::Locker locker(m_mutex);
  if (m_disassembly.GetSize() == 0) {
    ExecutionContext exe_ctx(shared_from_this());
    Target *target = exe_ctx.GetTargetPtr();
    if (target) {
      const char *plugin_name = nullptr;
      const char *flavor = nullptr;
      Disassembler::Disassemble(target->GetDebugger(),
                                target->GetArchitecture(), plugin_name, flavor,
                                exe_ctx, 0, 0, 0, m_disassembly);
    }
    if (m_disassembly.GetSize() == 0)
      return nullptr;
  }
  return m_disassembly.GetData();
}

Block *StackFrame::GetFrameBlock() {
  if (m_sc.block == nullptr && m_flags.IsClear(eSymbolContextBlock))
    GetSymbolContext(eSymbolContextBlock);

  if (m_sc.block) {
    Block *inline_block = m_sc.block->GetContainingInlinedBlock();
    if (inline_block) {
      // Use the block with the inlined function info
      // as the frame block we want this frame to have only the variables
      // for the inlined function and its non-inlined block child blocks.
      return inline_block;
    } else {
      // This block is not contained withing any inlined function blocks
      // with so we want to use the top most function block.
      return &m_sc.function->GetBlock(false);
    }
  }
  return nullptr;
}

//----------------------------------------------------------------------
// Get the symbol context if we already haven't done so by resolving the
// PC address as much as possible. This way when we pass around a
// StackFrame object, everyone will have as much information as
// possible and no one will ever have to look things up manually.
//----------------------------------------------------------------------
const SymbolContext &StackFrame::GetSymbolContext(uint32_t resolve_scope) {
  Mutex::Locker locker(m_mutex);
  // Copy our internal symbol context into "sc".
  if ((m_flags.Get() & resolve_scope) != resolve_scope) {
    uint32_t resolved = 0;

    // If the target was requested add that:
    if (!m_sc.target_sp) {
      m_sc.target_sp = CalculateTarget();
      if (m_sc.target_sp)
        resolved |= eSymbolContextTarget;
    }

    // Resolve our PC to section offset if we haven't already done so
    // and if we don't have a module. The resolved address section will
    // contain the module to which it belongs
    if (!m_sc.module_sp && m_flags.IsClear(RESOLVED_FRAME_CODE_ADDR))
      GetFrameCodeAddress();

    // If this is not frame zero, then we need to subtract 1 from the PC
    // value when doing address lookups since the PC will be on the
    // instruction following the function call instruction...

    Address lookup_addr(GetFrameCodeAddress());
    if (m_frame_index > 0 && lookup_addr.IsValid()) {
      addr_t offset = lookup_addr.GetOffset();
      if (offset > 0) {
        lookup_addr.SetOffset(offset - 1);

      } else {
        // lookup_addr is the start of a section.  We need
        // do the math on the actual load address and re-compute
        // the section.  We're working with a 'noreturn' function
        // at the end of a section.
        ThreadSP thread_sp(GetThread());
        if (thread_sp) {
          TargetSP target_sp(thread_sp->CalculateTarget());
          if (target_sp) {
            addr_t addr_minus_one =
                lookup_addr.GetLoadAddress(target_sp.get()) - 1;
            lookup_addr.SetLoadAddress(addr_minus_one, target_sp.get());
          } else {
            lookup_addr.SetOffset(offset - 1);
          }
        }
      }
    }

    if (m_sc.module_sp) {
      // We have something in our stack frame symbol context, lets check
      // if we haven't already tried to lookup one of those things. If we
      // haven't then we will do the query.

      uint32_t actual_resolve_scope = 0;

      if (resolve_scope & eSymbolContextCompUnit) {
        if (m_flags.IsClear(eSymbolContextCompUnit)) {
          if (m_sc.comp_unit)
            resolved |= eSymbolContextCompUnit;
          else
            actual_resolve_scope |= eSymbolContextCompUnit;
        }
      }

      if (resolve_scope & eSymbolContextFunction) {
        if (m_flags.IsClear(eSymbolContextFunction)) {
          if (m_sc.function)
            resolved |= eSymbolContextFunction;
          else
            actual_resolve_scope |= eSymbolContextFunction;
        }
      }

      if (resolve_scope & eSymbolContextBlock) {
        if (m_flags.IsClear(eSymbolContextBlock)) {
          if (m_sc.block)
            resolved |= eSymbolContextBlock;
          else
            actual_resolve_scope |= eSymbolContextBlock;
        }
      }

      if (resolve_scope & eSymbolContextSymbol) {
        if (m_flags.IsClear(eSymbolContextSymbol)) {
          if (m_sc.symbol)
            resolved |= eSymbolContextSymbol;
          else
            actual_resolve_scope |= eSymbolContextSymbol;
        }
      }

      if (resolve_scope & eSymbolContextLineEntry) {
        if (m_flags.IsClear(eSymbolContextLineEntry)) {
          if (m_sc.line_entry.IsValid())
            resolved |= eSymbolContextLineEntry;
          else
            actual_resolve_scope |= eSymbolContextLineEntry;
        }
      }

      if (actual_resolve_scope) {
        // We might be resolving less information than what is already
        // in our current symbol context so resolve into a temporary
        // symbol context "sc" so we don't clear out data we have
        // already found in "m_sc"
        SymbolContext sc;
        // Set flags that indicate what we have tried to resolve
        resolved |= m_sc.module_sp->ResolveSymbolContextForAddress(
            lookup_addr, actual_resolve_scope, sc);
        // Only replace what we didn't already have as we may have
        // information for an inlined function scope that won't match
        // what a standard lookup by address would match
        if ((resolved & eSymbolContextCompUnit) && m_sc.comp_unit == nullptr)
          m_sc.comp_unit = sc.comp_unit;
        if ((resolved & eSymbolContextFunction) && m_sc.function == nullptr)
          m_sc.function = sc.function;
        if ((resolved & eSymbolContextBlock) && m_sc.block == nullptr)
          m_sc.block = sc.block;
        if ((resolved & eSymbolContextSymbol) && m_sc.symbol == nullptr)
          m_sc.symbol = sc.symbol;
        if ((resolved & eSymbolContextLineEntry) &&
            !m_sc.line_entry.IsValid()) {
          m_sc.line_entry = sc.line_entry;
          if (m_sc.target_sp) {
            // Be sure to apply and file remappings to our file and line
            // entries when handing out a line entry
            FileSpec new_file_spec;
            if (m_sc.target_sp->GetSourcePathMap().FindFile(
                    m_sc.line_entry.file, new_file_spec))
              m_sc.line_entry.file = new_file_spec;
          }
        }
      }
    } else {
      // If we don't have a module, then we can't have the compile unit,
      // function, block, line entry or symbol, so we can safely call
      // ResolveSymbolContextForAddress with our symbol context member m_sc.
      if (m_sc.target_sp) {
        resolved |= m_sc.target_sp->GetImages().ResolveSymbolContextForAddress(
            lookup_addr, resolve_scope, m_sc);
      }
    }

    // Update our internal flags so we remember what we have tried to locate so
    // we don't have to keep trying when more calls to this function are made.
    // We might have dug up more information that was requested (for example
    // if we were asked to only get the block, we will have gotten the
    // compile unit, and function) so set any additional bits that we resolved
    m_flags.Set(resolve_scope | resolved);
  }

  // Return the symbol context with everything that was possible to resolve
  // resolved.
  return m_sc;
}

VariableList *StackFrame::GetVariableList(bool get_file_globals) {
  Mutex::Locker locker(m_mutex);
  if (m_flags.IsClear(RESOLVED_VARIABLES)) {
    m_flags.Set(RESOLVED_VARIABLES);

    Block *frame_block = GetFrameBlock();

    if (frame_block) {
      const bool get_child_variables = true;
      const bool can_create = true;
      const bool stop_if_child_block_is_inlined_function = true;
      m_variable_list_sp.reset(new VariableList());
      frame_block->AppendBlockVariables(can_create, get_child_variables,
                                        stop_if_child_block_is_inlined_function,
                                        [this](Variable *v) { return true; },
                                        m_variable_list_sp.get());
    }
  }

  if (m_flags.IsClear(RESOLVED_GLOBAL_VARIABLES) && get_file_globals) {
    m_flags.Set(RESOLVED_GLOBAL_VARIABLES);

    if (m_flags.IsClear(eSymbolContextCompUnit))
      GetSymbolContext(eSymbolContextCompUnit);

    if (m_sc.comp_unit) {
      VariableListSP global_variable_list_sp(
          m_sc.comp_unit->GetVariableList(true));
      if (m_variable_list_sp)
        m_variable_list_sp->AddVariables(global_variable_list_sp.get());
      else
        m_variable_list_sp = global_variable_list_sp;
    }
  }

  return m_variable_list_sp.get();
}

VariableListSP
StackFrame::GetInScopeVariableList(bool get_file_globals,
                                   bool must_have_valid_location) {
  Mutex::Locker locker(m_mutex);
  // We can't fetch variable information for a history stack frame.
  if (m_is_history_frame)
    return VariableListSP();

  VariableListSP var_list_sp(new VariableList);
  GetSymbolContext(eSymbolContextCompUnit | eSymbolContextBlock);

  if (m_sc.block) {
    const bool can_create = true;
    const bool get_parent_variables = true;
    const bool stop_if_block_is_inlined_function = true;
    m_sc.block->AppendVariables(
        can_create, get_parent_variables, stop_if_block_is_inlined_function,
        [this, must_have_valid_location](Variable *v) {
          return v->IsInScope(this) && (!must_have_valid_location ||
                                        v->LocationIsValidForFrame(this));
        },
        var_list_sp.get());
  }

  if (m_sc.comp_unit && get_file_globals) {
    VariableListSP global_variable_list_sp(
        m_sc.comp_unit->GetVariableList(true));
    if (global_variable_list_sp)
      var_list_sp->AddVariables(global_variable_list_sp.get());
  }

  return var_list_sp;
}

ValueObjectSP StackFrame::GetValueForVariableExpressionPath(
    const char *var_expr_cstr, DynamicValueType use_dynamic, uint32_t options,
    VariableSP &var_sp, Error &error) {
  // We can't fetch variable information for a history stack frame.
  if (m_is_history_frame)
    return ValueObjectSP();

  if (var_expr_cstr && var_expr_cstr[0]) {
    const bool check_ptr_vs_member =
        (options & eExpressionPathOptionCheckPtrVsMember) != 0;
    const bool no_fragile_ivar =
        (options & eExpressionPathOptionsNoFragileObjcIvar) != 0;
    const bool no_synth_child =
        (options & eExpressionPathOptionsNoSyntheticChildren) != 0;
    // const bool no_synth_array = (options &
    // eExpressionPathOptionsNoSyntheticArrayRange) != 0;
    error.Clear();
    bool deref = false;
    bool address_of = false;
    ValueObjectSP valobj_sp;
    const bool get_file_globals = true;
    // When looking up a variable for an expression, we need only consider the
    // variables that are in scope.
    VariableListSP var_list_sp(GetInScopeVariableList(get_file_globals));
    VariableList *variable_list = var_list_sp.get();

    if (variable_list) {
      // If first character is a '*', then show pointer contents
      const char *var_expr = var_expr_cstr;
      if (var_expr[0] == '*') {
        deref = true;
        var_expr++; // Skip the '*'
      } else if (var_expr[0] == '&') {
        address_of = true;
        var_expr++; // Skip the '&'
      }

      std::string var_path(var_expr);
      size_t separator_idx = var_path.find_first_of(".-[=+~|&^%#@!/?,<>{}");
      StreamString var_expr_path_strm;

      ConstString name_const_string;
      if (separator_idx == std::string::npos)
        name_const_string.SetCString(var_path.c_str());
      else
        name_const_string.SetCStringWithLength(var_path.c_str(), separator_idx);

      var_sp = variable_list->FindVariable(name_const_string, false);

      bool synthetically_added_instance_object = false;

      if (var_sp) {
        var_path.erase(0, name_const_string.GetLength());
      }

      if (!var_sp && (options & eExpressionPathOptionsAllowDirectIVarAccess)) {
        // Check for direct ivars access which helps us with implicit
        // access to ivars with the "this->" or "self->"
        GetSymbolContext(eSymbolContextFunction | eSymbolContextBlock);
        lldb::LanguageType method_language = eLanguageTypeUnknown;
        bool is_instance_method = false;
        ConstString method_object_name;
        if (m_sc.GetFunctionMethodInfo(method_language, is_instance_method,
                                       method_object_name)) {
          if (is_instance_method && method_object_name) {
            var_sp = variable_list->FindVariable(method_object_name);
            if (var_sp) {
              separator_idx = 0;
              var_path.insert(0, "->");
              synthetically_added_instance_object = true;
            }
          }
        }
      }

      if (!var_sp && (options & eExpressionPathOptionsInspectAnonymousUnions)) {
        // Check if any anonymous unions are there which contain a variable with
        // the name we need
        for (size_t i = 0; i < variable_list->GetSize(); i++) {
          if (VariableSP variable_sp = variable_list->GetVariableAtIndex(i)) {
            if (variable_sp->GetName().IsEmpty()) {
              if (Type *var_type = variable_sp->GetType()) {
                if (var_type->GetForwardCompilerType().IsAnonymousType()) {
                  valobj_sp =
                      GetValueObjectForFrameVariable(variable_sp, use_dynamic);
                  if (!valobj_sp)
                    return valobj_sp;
                  valobj_sp = valobj_sp->GetChildMemberWithName(
                      name_const_string, true);
                  if (valobj_sp)
                    break;
                }
              }
            }
          }
        }
      }

      if (var_sp && !valobj_sp) {
        valobj_sp = GetValueObjectForFrameVariable(var_sp, use_dynamic);
        if (!valobj_sp)
          return valobj_sp;
      }
      if (valobj_sp) {
        // We are dumping at least one child
        while (separator_idx != std::string::npos) {
          // Calculate the next separator index ahead of time
          ValueObjectSP child_valobj_sp;
          const char separator_type = var_path[0];
          switch (separator_type) {
          case '-':
            if (var_path.size() >= 2 && var_path[1] != '>')
              return ValueObjectSP();

            if (no_fragile_ivar) {
              // Make sure we aren't trying to deref an objective
              // C ivar if this is not allowed
              const uint32_t pointer_type_flags =
                  valobj_sp->GetCompilerType().GetTypeInfo(nullptr);
              if ((pointer_type_flags & eTypeIsObjC) &&
                  (pointer_type_flags & eTypeIsPointer)) {
                // This was an objective C object pointer and
                // it was requested we skip any fragile ivars
                // so return nothing here
                return ValueObjectSP();
              }
            }
            var_path.erase(0, 1); // Remove the '-'
            LLVM_FALLTHROUGH;
          case '.': {
            const bool expr_is_ptr = var_path[0] == '>';

            var_path.erase(0, 1); // Remove the '.' or '>'
            separator_idx = var_path.find_first_of(".-[");
            ConstString child_name;
            if (separator_idx == std::string::npos)
              child_name.SetCString(var_path.c_str());
            else
              child_name.SetCStringWithLength(var_path.c_str(), separator_idx);

            if (check_ptr_vs_member) {
              // We either have a pointer type and need to verify
              // valobj_sp is a pointer, or we have a member of a
              // class/union/struct being accessed with the . syntax
              // and need to verify we don't have a pointer.
              const bool actual_is_ptr = valobj_sp->IsPointerType();

              if (actual_is_ptr != expr_is_ptr) {
                // Incorrect use of "." with a pointer, or "->" with
                // a class/union/struct instance or reference.
                valobj_sp->GetExpressionPath(var_expr_path_strm, false);
                if (actual_is_ptr)
                  error.SetErrorStringWithFormat(
                      "\"%s\" is a pointer and . was used to attempt to access "
                      "\"%s\". Did you mean \"%s->%s\"?",
                      var_expr_path_strm.GetString().c_str(),
                      child_name.GetCString(),
                      var_expr_path_strm.GetString().c_str(), var_path.c_str());
                else
                  error.SetErrorStringWithFormat(
                      "\"%s\" is not a pointer and -> was used to attempt to "
                      "access \"%s\". Did you mean \"%s.%s\"?",
                      var_expr_path_strm.GetString().c_str(),
                      child_name.GetCString(),
                      var_expr_path_strm.GetString().c_str(), var_path.c_str());
                return ValueObjectSP();
              }
            }
            child_valobj_sp =
                valobj_sp->GetChildMemberWithName(child_name, true);
            if (!child_valobj_sp) {
              if (!no_synth_child) {
                child_valobj_sp = valobj_sp->GetSyntheticValue();
                if (child_valobj_sp)
                  child_valobj_sp =
                      child_valobj_sp->GetChildMemberWithName(child_name, true);
              }

              if (no_synth_child || !child_valobj_sp) {
                // No child member with name "child_name"
                if (synthetically_added_instance_object) {
                  // We added a "this->" or "self->" to the beginning of the
                  // expression
                  // and this is the first pointer ivar access, so just return
                  // the normal
                  // error
                  error.SetErrorStringWithFormat(
                      "no variable or instance variable named '%s' found in "
                      "this frame",
                      name_const_string.GetCString());
                } else {
                  valobj_sp->GetExpressionPath(var_expr_path_strm, false);
                  if (child_name) {
                    error.SetErrorStringWithFormat(
                        "\"%s\" is not a member of \"(%s) %s\"",
                        child_name.GetCString(),
                        valobj_sp->GetTypeName().AsCString("<invalid type>"),
                        var_expr_path_strm.GetString().c_str());
                  } else {
                    error.SetErrorStringWithFormat(
                        "incomplete expression path after \"%s\" in \"%s\"",
                        var_expr_path_strm.GetString().c_str(), var_expr_cstr);
                  }
                }
                return ValueObjectSP();
              }
            }
            synthetically_added_instance_object = false;
            // Remove the child name from the path
            var_path.erase(0, child_name.GetLength());
            if (use_dynamic != eNoDynamicValues) {
              ValueObjectSP dynamic_value_sp(
                  child_valobj_sp->GetDynamicValue(use_dynamic));
              if (dynamic_value_sp)
                child_valobj_sp = dynamic_value_sp;
            }
          } break;

          case '[':
            // Array member access, or treating pointer as an array
            if (var_path.size() > 2) // Need at least two brackets and a number
            {
              char *end = nullptr;
              long child_index = ::strtol(&var_path[1], &end, 0);
              if (end && *end == ']' &&
                  *(end - 1) != '[') // this code forces an error in the case of
                                     // arr[]. as bitfield[] is not a good
                                     // syntax we're good to go
              {
                if (valobj_sp->GetCompilerType().IsPointerToScalarType() &&
                    deref) {
                  // what we have is *ptr[low]. the most similar C++ syntax is
                  // to deref ptr
                  // and extract bit low out of it. reading array item low
                  // would be done by saying ptr[low], without a deref * sign
                  Error error;
                  ValueObjectSP temp(valobj_sp->Dereference(error));
                  if (error.Fail()) {
                    valobj_sp->GetExpressionPath(var_expr_path_strm, false);
                    error.SetErrorStringWithFormat(
                        "could not dereference \"(%s) %s\"",
                        valobj_sp->GetTypeName().AsCString("<invalid type>"),
                        var_expr_path_strm.GetString().c_str());
                    return ValueObjectSP();
                  }
                  valobj_sp = temp;
                  deref = false;
                } else if (valobj_sp->GetCompilerType().IsArrayOfScalarType() &&
                           deref) {
                  // what we have is *arr[low]. the most similar C++ syntax is
                  // to get arr[0]
                  // (an operation that is equivalent to deref-ing arr)
                  // and extract bit low out of it. reading array item low
                  // would be done by saying arr[low], without a deref * sign
                  Error error;
                  ValueObjectSP temp(valobj_sp->GetChildAtIndex(0, true));
                  if (error.Fail()) {
                    valobj_sp->GetExpressionPath(var_expr_path_strm, false);
                    error.SetErrorStringWithFormat(
                        "could not get item 0 for \"(%s) %s\"",
                        valobj_sp->GetTypeName().AsCString("<invalid type>"),
                        var_expr_path_strm.GetString().c_str());
                    return ValueObjectSP();
                  }
                  valobj_sp = temp;
                  deref = false;
                }

                bool is_incomplete_array = false;
                if (valobj_sp->IsPointerType()) {
                  bool is_objc_pointer = true;

                  if (valobj_sp->GetCompilerType().GetMinimumLanguage() !=
                      eLanguageTypeObjC)
                    is_objc_pointer = false;
                  else if (!valobj_sp->GetCompilerType().IsPointerType())
                    is_objc_pointer = false;

                  if (no_synth_child && is_objc_pointer) {
                    error.SetErrorStringWithFormat(
                        "\"(%s) %s\" is an Objective-C pointer, and cannot be "
                        "subscripted",
                        valobj_sp->GetTypeName().AsCString("<invalid type>"),
                        var_expr_path_strm.GetString().c_str());

                    return ValueObjectSP();
                  } else if (is_objc_pointer) {
                    // dereferencing ObjC variables is not valid.. so let's try
                    // and recur to synthetic children
                    ValueObjectSP synthetic = valobj_sp->GetSyntheticValue();
                    if (!synthetic                 /* no synthetic */
                        || synthetic == valobj_sp) /* synthetic is the same as
                                                      the original object */
                    {
                      valobj_sp->GetExpressionPath(var_expr_path_strm, false);
                      error.SetErrorStringWithFormat(
                          "\"(%s) %s\" is not an array type",
                          valobj_sp->GetTypeName().AsCString("<invalid type>"),
                          var_expr_path_strm.GetString().c_str());
                    } else if (
                        static_cast<uint32_t>(child_index) >=
                        synthetic
                            ->GetNumChildren() /* synthetic does not have that many values */) {
                      valobj_sp->GetExpressionPath(var_expr_path_strm, false);
                      error.SetErrorStringWithFormat(
                          "array index %ld is not valid for \"(%s) %s\"",
                          child_index,
                          valobj_sp->GetTypeName().AsCString("<invalid type>"),
                          var_expr_path_strm.GetString().c_str());
                    } else {
                      child_valobj_sp =
                          synthetic->GetChildAtIndex(child_index, true);
                      if (!child_valobj_sp) {
                        valobj_sp->GetExpressionPath(var_expr_path_strm, false);
                        error.SetErrorStringWithFormat(
                            "array index %ld is not valid for \"(%s) %s\"",
                            child_index, valobj_sp->GetTypeName().AsCString(
                                             "<invalid type>"),
                            var_expr_path_strm.GetString().c_str());
                      }
                    }
                  } else {
                    child_valobj_sp =
                        valobj_sp->GetSyntheticArrayMember(child_index, true);
                    if (!child_valobj_sp) {
                      valobj_sp->GetExpressionPath(var_expr_path_strm, false);
                      error.SetErrorStringWithFormat(
                          "failed to use pointer as array for index %ld for "
                          "\"(%s) %s\"",
                          child_index,
                          valobj_sp->GetTypeName().AsCString("<invalid type>"),
                          var_expr_path_strm.GetString().c_str());
                    }
                  }
                } else if (valobj_sp->GetCompilerType().IsArrayType(
                               nullptr, nullptr, &is_incomplete_array)) {
                  // Pass false to dynamic_value here so we can tell the
                  // difference between
                  // no dynamic value and no member of this type...
                  child_valobj_sp =
                      valobj_sp->GetChildAtIndex(child_index, true);
                  if (!child_valobj_sp &&
                      (is_incomplete_array || !no_synth_child))
                    child_valobj_sp =
                        valobj_sp->GetSyntheticArrayMember(child_index, true);

                  if (!child_valobj_sp) {
                    valobj_sp->GetExpressionPath(var_expr_path_strm, false);
                    error.SetErrorStringWithFormat(
                        "array index %ld is not valid for \"(%s) %s\"",
                        child_index,
                        valobj_sp->GetTypeName().AsCString("<invalid type>"),
                        var_expr_path_strm.GetString().c_str());
                  }
                } else if (valobj_sp->GetCompilerType().IsScalarType()) {
                  // this is a bitfield asking to display just one bit
                  child_valobj_sp = valobj_sp->GetSyntheticBitFieldChild(
                      child_index, child_index, true);
                  if (!child_valobj_sp) {
                    valobj_sp->GetExpressionPath(var_expr_path_strm, false);
                    error.SetErrorStringWithFormat(
                        "bitfield range %ld-%ld is not valid for \"(%s) %s\"",
                        child_index, child_index,
                        valobj_sp->GetTypeName().AsCString("<invalid type>"),
                        var_expr_path_strm.GetString().c_str());
                  }
                } else {
                  ValueObjectSP synthetic = valobj_sp->GetSyntheticValue();
                  if (no_synth_child /* synthetic is forbidden */ ||
                      !synthetic /* no synthetic */
                      ||
                      synthetic == valobj_sp) /* synthetic is the same as the
                                                 original object */
                  {
                    if (valobj_sp->GetCompilerType().IsArrayType(
                            NULL, NULL, &is_incomplete_array)) {
                      // Pass false to dynamic_value here so we can tell the
                      // difference between
                      // no dynamic value and no member of this type...
                      child_valobj_sp =
                          valobj_sp->GetChildAtIndex(child_index, true);
                      if (!child_valobj_sp &&
                          (is_incomplete_array || no_synth_child == false))
                        child_valobj_sp = valobj_sp->GetSyntheticArrayMember(
                            child_index, true);

                      if (!child_valobj_sp) {
                        valobj_sp->GetExpressionPath(var_expr_path_strm, false);
                        error.SetErrorStringWithFormat(
                            "array index %ld is not valid for \"(%s) %s\"",
                            child_index, valobj_sp->GetTypeName().AsCString(
                                             "<invalid type>"),
                            var_expr_path_strm.GetString().c_str());
                      }
                    } else {
                      valobj_sp->GetExpressionPath(var_expr_path_strm, false);
                      error.SetErrorStringWithFormat(
                          "\"(%s) %s\" is not an array type",
                          valobj_sp->GetTypeName().AsCString("<invalid type>"),
                          var_expr_path_strm.GetString().c_str());
                    }
                  } else if (
                      static_cast<uint32_t>(child_index) >=
                      synthetic
                          ->GetNumChildren() /* synthetic does not have that many values */) {
                    valobj_sp->GetExpressionPath(var_expr_path_strm, false);
                    error.SetErrorStringWithFormat(
                        "array index %ld is not valid for \"(%s) %s\"",
                        child_index,
                        valobj_sp->GetTypeName().AsCString("<invalid type>"),
                        var_expr_path_strm.GetString().c_str());
                  } else {
                    child_valobj_sp =
                        synthetic->GetChildAtIndex(child_index, true);
                    if (!child_valobj_sp) {
                      valobj_sp->GetExpressionPath(var_expr_path_strm, false);
                      error.SetErrorStringWithFormat(
                          "array index %ld is not valid for \"(%s) %s\"",
                          child_index,
                          valobj_sp->GetTypeName().AsCString("<invalid type>"),
                          var_expr_path_strm.GetString().c_str());
                    }
                  }
                }

                if (!child_valobj_sp) {
                  // Invalid array index...
                  return ValueObjectSP();
                }

                // Erase the array member specification '[%i]' where
                // %i is the array index
                var_path.erase(0, (end - var_path.c_str()) + 1);
                separator_idx = var_path.find_first_of(".-[");
                if (use_dynamic != eNoDynamicValues) {
                  ValueObjectSP dynamic_value_sp(
                      child_valobj_sp->GetDynamicValue(use_dynamic));
                  if (dynamic_value_sp)
                    child_valobj_sp = dynamic_value_sp;
                }
                // Break out early from the switch since we were
                // able to find the child member
                break;
              } else if (end && *end == '-') {
                // this is most probably a BitField, let's take a look
                char *real_end = nullptr;
                long final_index = ::strtol(end + 1, &real_end, 0);
                bool expand_bitfield = true;
                if (real_end && *real_end == ']') {
                  // if the format given is [high-low], swap range
                  if (child_index > final_index) {
                    long temp = child_index;
                    child_index = final_index;
                    final_index = temp;
                  }

                  if (valobj_sp->GetCompilerType().IsPointerToScalarType() &&
                      deref) {
                    // what we have is *ptr[low-high]. the most similar C++
                    // syntax is to deref ptr
                    // and extract bits low thru high out of it. reading array
                    // items low thru high
                    // would be done by saying ptr[low-high], without a deref *
                    // sign
                    Error error;
                    ValueObjectSP temp(valobj_sp->Dereference(error));
                    if (error.Fail()) {
                      valobj_sp->GetExpressionPath(var_expr_path_strm, false);
                      error.SetErrorStringWithFormat(
                          "could not dereference \"(%s) %s\"",
                          valobj_sp->GetTypeName().AsCString("<invalid type>"),
                          var_expr_path_strm.GetString().c_str());
                      return ValueObjectSP();
                    }
                    valobj_sp = temp;
                    deref = false;
                  } else if (valobj_sp->GetCompilerType()
                                 .IsArrayOfScalarType() &&
                             deref) {
                    // what we have is *arr[low-high]. the most similar C++
                    // syntax is to get arr[0]
                    // (an operation that is equivalent to deref-ing arr)
                    // and extract bits low thru high out of it. reading array
                    // items low thru high
                    // would be done by saying arr[low-high], without a deref *
                    // sign
                    Error error;
                    ValueObjectSP temp(valobj_sp->GetChildAtIndex(0, true));
                    if (error.Fail()) {
                      valobj_sp->GetExpressionPath(var_expr_path_strm, false);
                      error.SetErrorStringWithFormat(
                          "could not get item 0 for \"(%s) %s\"",
                          valobj_sp->GetTypeName().AsCString("<invalid type>"),
                          var_expr_path_strm.GetString().c_str());
                      return ValueObjectSP();
                    }
                    valobj_sp = temp;
                    deref = false;
                  }
                  /*else if (valobj_sp->IsArrayType() ||
                  valobj_sp->IsPointerType())
                  {
                      child_valobj_sp =
                  valobj_sp->GetSyntheticArrayRangeChild(child_index,
                  final_index, true);
                      expand_bitfield = false;
                      if (!child_valobj_sp)
                      {
                          valobj_sp->GetExpressionPath (var_expr_path_strm,
                  false);
                          error.SetErrorStringWithFormat ("array range %i-%i is
                  not valid for \"(%s) %s\"",
                                                          child_index,
                  final_index,
                                                          valobj_sp->GetTypeName().AsCString("<invalid
                  type>"),
                                                          var_expr_path_strm.GetString().c_str());
                      }
                  }*/

                  if (expand_bitfield) {
                    child_valobj_sp = valobj_sp->GetSyntheticBitFieldChild(
                        child_index, final_index, true);
                    if (!child_valobj_sp) {
                      valobj_sp->GetExpressionPath(var_expr_path_strm, false);
                      error.SetErrorStringWithFormat(
                          "bitfield range %ld-%ld is not valid for \"(%s) %s\"",
                          child_index, final_index,
                          valobj_sp->GetTypeName().AsCString("<invalid type>"),
                          var_expr_path_strm.GetString().c_str());
                    }
                  }
                }

                if (!child_valobj_sp) {
                  // Invalid bitfield range...
                  return ValueObjectSP();
                }

                // Erase the bitfield member specification '[%i-%i]' where
                // %i is the index
                var_path.erase(0, (real_end - var_path.c_str()) + 1);
                separator_idx = var_path.find_first_of(".-[");
                if (use_dynamic != eNoDynamicValues) {
                  ValueObjectSP dynamic_value_sp(
                      child_valobj_sp->GetDynamicValue(use_dynamic));
                  if (dynamic_value_sp)
                    child_valobj_sp = dynamic_value_sp;
                }
                // Break out early from the switch since we were
                // able to find the child member
                break;
              }
            } else {
              error.SetErrorStringWithFormat(
                  "invalid square bracket encountered after \"%s\" in \"%s\"",
                  var_expr_path_strm.GetString().c_str(), var_path.c_str());
            }
            return ValueObjectSP();

          default:
            // Failure...
            {
              valobj_sp->GetExpressionPath(var_expr_path_strm, false);
              error.SetErrorStringWithFormat(
                  "unexpected char '%c' encountered after \"%s\" in \"%s\"",
                  separator_type, var_expr_path_strm.GetString().c_str(),
                  var_path.c_str());

              return ValueObjectSP();
            }
          }

          if (child_valobj_sp)
            valobj_sp = child_valobj_sp;

          if (var_path.empty())
            break;
        }
        if (valobj_sp) {
          const CompilerType &clang_type(valobj_sp->GetCompilerType());
          Flags clang_type_flags(clang_type.GetTypeInfo());
          if ((deref || address_of) && clang_type_flags.AllSet(eTypeIsSwift)) {
            error.SetErrorStringWithFormat(
                "cannot evaluate %s on a Swift variable", deref ? "*" : "&");
            return ValueObjectSP();
          }
          if (deref) {
            ValueObjectSP deref_valobj_sp(valobj_sp->Dereference(error));
            valobj_sp = deref_valobj_sp;
          } else if (address_of) {
            ValueObjectSP address_of_valobj_sp(valobj_sp->AddressOf(error));
            valobj_sp = address_of_valobj_sp;
          }
        }
        return valobj_sp;
      } else {
        error.SetErrorStringWithFormat(
            "no variable named '%s' found in this frame",
            name_const_string.GetCString());
      }
    }
  } else {
    error.SetErrorStringWithFormat("invalid variable path '%s'", var_expr_cstr);
  }
  return ValueObjectSP();
}

bool StackFrame::GetFrameBaseValue(Scalar &frame_base, Error *error_ptr) {
  Mutex::Locker locker(m_mutex);
  if (!m_cfa_is_valid) {
    m_frame_base_error.SetErrorString(
        "No frame base available for this historical stack frame.");
    return false;
  }

  if (m_flags.IsClear(GOT_FRAME_BASE)) {
    if (m_sc.function) {
      m_frame_base.Clear();
      m_frame_base_error.Clear();

      m_flags.Set(GOT_FRAME_BASE);
      ExecutionContext exe_ctx(shared_from_this());
      Value expr_value;
      addr_t loclist_base_addr = LLDB_INVALID_ADDRESS;
      if (m_sc.function->GetFrameBaseExpression().IsLocationList())
        loclist_base_addr =
            m_sc.function->GetAddressRange().GetBaseAddress().GetLoadAddress(
                exe_ctx.GetTargetPtr());

      if (m_sc.function->GetFrameBaseExpression().Evaluate(
              &exe_ctx, nullptr, nullptr, nullptr, loclist_base_addr, nullptr,
              nullptr, expr_value, &m_frame_base_error) == false) {
        // We should really have an error if evaluate returns, but in case
        // we don't, lets set the error to something at least.
        if (m_frame_base_error.Success())
          m_frame_base_error.SetErrorString(
              "Evaluation of the frame base expression failed.");
      } else {
        m_frame_base = expr_value.ResolveValue(&exe_ctx);
      }
    } else {
      m_frame_base_error.SetErrorString("No function in symbol context.");
    }
  }

  if (m_frame_base_error.Success())
    frame_base = m_frame_base;

  if (error_ptr)
    *error_ptr = m_frame_base_error;
  return m_frame_base_error.Success();
}

RegisterContextSP StackFrame::GetRegisterContext() {
  Mutex::Locker locker(m_mutex);
  if (!m_reg_context_sp) {
    ThreadSP thread_sp(GetThread());
    if (thread_sp)
      m_reg_context_sp = thread_sp->CreateRegisterContextForFrame(this);
  }
  return m_reg_context_sp;
}

bool StackFrame::HasDebugInformation() {
  GetSymbolContext(eSymbolContextLineEntry);
  return m_sc.line_entry.IsValid();
}

ValueObjectSP
StackFrame::GetValueObjectForFrameVariable(const VariableSP &variable_sp,
                                           DynamicValueType use_dynamic) {
  Mutex::Locker locker(m_mutex);
  ValueObjectSP valobj_sp;
  if (m_is_history_frame) {
    return valobj_sp;
  }
  VariableList *var_list = GetVariableList(true);
  if (var_list) {
    // Make sure the variable is a frame variable
    const uint32_t var_idx = var_list->FindIndexForVariable(variable_sp.get());
    const uint32_t num_variables = var_list->GetSize();
    if (var_idx < num_variables) {
      valobj_sp = m_variable_list_value_objects.GetValueObjectAtIndex(var_idx);
      if (!valobj_sp) {
        if (m_variable_list_value_objects.GetSize() < num_variables)
          m_variable_list_value_objects.Resize(num_variables);
        valobj_sp = ValueObjectVariable::Create(this, variable_sp);
        m_variable_list_value_objects.SetValueObjectAtIndex(var_idx, valobj_sp);
      }
    }
  }
  if (use_dynamic != eNoDynamicValues && valobj_sp) {
    ValueObjectSP dynamic_sp = valobj_sp->GetDynamicValue(use_dynamic);
    if (dynamic_sp)
      return dynamic_sp;
  }
  return valobj_sp;
}

ValueObjectSP StackFrame::TrackGlobalVariable(const VariableSP &variable_sp,
                                              DynamicValueType use_dynamic) {
  Mutex::Locker locker(m_mutex);
  if (m_is_history_frame)
    return ValueObjectSP();

  // Check to make sure we aren't already tracking this variable?
  ValueObjectSP valobj_sp(
      GetValueObjectForFrameVariable(variable_sp, use_dynamic));
  if (!valobj_sp) {
    // We aren't already tracking this global
    VariableList *var_list = GetVariableList(true);
    // If this frame has no variables, create a new list
    if (var_list == nullptr)
      m_variable_list_sp.reset(new VariableList());

    // Add the global/static variable to this frame
    m_variable_list_sp->AddVariable(variable_sp);

    // Now make a value object for it so we can track its changes
    valobj_sp = GetValueObjectForFrameVariable(variable_sp, use_dynamic);
  }
  return valobj_sp;
}

bool StackFrame::IsInlined() {
  if (m_sc.block == nullptr)
    GetSymbolContext(eSymbolContextBlock);
  if (m_sc.block)
    return m_sc.block->GetContainingInlinedBlock() != nullptr;
  return false;
}

lldb::LanguageType StackFrame::GetLanguage() {
  CompileUnit *cu = GetSymbolContext(eSymbolContextCompUnit).comp_unit;
  if (cu)
    return cu->GetLanguage();
  return lldb::eLanguageTypeUnknown;
}

lldb::LanguageType StackFrame::GuessLanguage() {
  LanguageType lang_type = GetLanguage();

  if (lang_type == eLanguageTypeUnknown) {
    Function *f = GetSymbolContext(eSymbolContextFunction).function;
    if (f) {
      lang_type = f->GetMangled().GuessLanguage();
    }
  }

  return lang_type;
}

TargetSP StackFrame::CalculateTarget() {
  TargetSP target_sp;
  ThreadSP thread_sp(GetThread());
  if (thread_sp) {
    ProcessSP process_sp(thread_sp->CalculateProcess());
    if (process_sp)
      target_sp = process_sp->CalculateTarget();
  }
  return target_sp;
}

ProcessSP StackFrame::CalculateProcess() {
  ProcessSP process_sp;
  ThreadSP thread_sp(GetThread());
  if (thread_sp)
    process_sp = thread_sp->CalculateProcess();
  return process_sp;
}

ThreadSP StackFrame::CalculateThread() { return GetThread(); }

StackFrameSP StackFrame::CalculateStackFrame() { return shared_from_this(); }

void StackFrame::CalculateExecutionContext(ExecutionContext &exe_ctx) {
  exe_ctx.SetContext(shared_from_this());
}

void StackFrame::DumpUsingSettingsFormat(Stream *strm,
                                         const char *frame_marker) {
  if (strm == nullptr)
    return;

  GetSymbolContext(eSymbolContextEverything);
  ExecutionContext exe_ctx(shared_from_this());
  StreamString s;

  if (frame_marker)
    s.PutCString(frame_marker);

  const FormatEntity::Entry *frame_format = nullptr;
  Target *target = exe_ctx.GetTargetPtr();
  if (target)
    frame_format = target->GetDebugger().GetFrameFormat();
  if (frame_format && FormatEntity::Format(*frame_format, s, &m_sc, &exe_ctx,
                                           nullptr, nullptr, false, false)) {
    strm->Write(s.GetData(), s.GetSize());
  } else {
    Dump(strm, true, false);
    strm->EOL();
  }
}

void StackFrame::Dump(Stream *strm, bool show_frame_index,
                      bool show_fullpaths) {
  if (strm == nullptr)
    return;

  if (show_frame_index)
    strm->Printf("frame #%u: ", m_frame_index);
  ExecutionContext exe_ctx(shared_from_this());
  Target *target = exe_ctx.GetTargetPtr();
  strm->Printf("0x%0*" PRIx64 " ",
               target ? (target->GetArchitecture().GetAddressByteSize() * 2)
                      : 16,
               GetFrameCodeAddress().GetLoadAddress(target));
  GetSymbolContext(eSymbolContextEverything);
  const bool show_module = true;
  const bool show_inline = true;
  const bool show_function_arguments = true;
  const bool show_function_name = true;
  m_sc.DumpStopContext(strm, exe_ctx.GetBestExecutionContextScope(),
                       GetFrameCodeAddress(), show_fullpaths, show_module,
                       show_inline, show_function_arguments,
                       show_function_name);
}

void StackFrame::UpdateCurrentFrameFromPreviousFrame(StackFrame &prev_frame) {
  Mutex::Locker locker(m_mutex);
  assert(GetStackID() ==
         prev_frame.GetStackID()); // TODO: remove this after some testing
  m_variable_list_sp = prev_frame.m_variable_list_sp;
  m_variable_list_value_objects.Swap(prev_frame.m_variable_list_value_objects);
  if (!m_disassembly.GetString().empty())
    m_disassembly.GetString().swap(m_disassembly.GetString());
}

void StackFrame::UpdatePreviousFrameFromCurrentFrame(StackFrame &curr_frame) {
  Mutex::Locker locker(m_mutex);
  assert(GetStackID() ==
         curr_frame.GetStackID());     // TODO: remove this after some testing
  m_id.SetPC(curr_frame.m_id.GetPC()); // Update the Stack ID PC value
  assert(GetThread() == curr_frame.GetThread());
  m_frame_index = curr_frame.m_frame_index;
  m_concrete_frame_index = curr_frame.m_concrete_frame_index;
  m_reg_context_sp = curr_frame.m_reg_context_sp;
  m_frame_code_addr = curr_frame.m_frame_code_addr;
  assert(!m_sc.target_sp || !curr_frame.m_sc.target_sp ||
         m_sc.target_sp.get() == curr_frame.m_sc.target_sp.get());
  assert(!m_sc.module_sp || !curr_frame.m_sc.module_sp ||
         m_sc.module_sp.get() == curr_frame.m_sc.module_sp.get());
  assert(m_sc.comp_unit == nullptr || curr_frame.m_sc.comp_unit == nullptr ||
         m_sc.comp_unit == curr_frame.m_sc.comp_unit);
  assert(m_sc.function == nullptr || curr_frame.m_sc.function == nullptr ||
         m_sc.function == curr_frame.m_sc.function);
  m_sc = curr_frame.m_sc;
  m_flags.Clear(GOT_FRAME_BASE | eSymbolContextEverything);
  m_flags.Set(m_sc.GetResolvedMask());
  m_frame_base.Clear();
  m_frame_base_error.Clear();
}

bool StackFrame::HasCachedData() const {
  if (m_variable_list_sp)
    return true;
  if (m_variable_list_value_objects.GetSize() > 0)
    return true;
  if (!m_disassembly.GetString().empty())
    return true;
  return false;
}

bool StackFrame::GetStatus(Stream &strm, bool show_frame_info, bool show_source,
                           const char *frame_marker) {

  if (show_frame_info) {
    strm.Indent();
    DumpUsingSettingsFormat(&strm, frame_marker);
  }

  if (show_source) {
    ExecutionContext exe_ctx(shared_from_this());
    bool have_source = false, have_debuginfo = false;
    Debugger::StopDisassemblyType disasm_display =
        Debugger::eStopDisassemblyTypeNever;
    Target *target = exe_ctx.GetTargetPtr();
    if (target) {
      Debugger &debugger = target->GetDebugger();
      const uint32_t source_lines_before =
          debugger.GetStopSourceLineCount(true);
      const uint32_t source_lines_after =
          debugger.GetStopSourceLineCount(false);
      disasm_display = debugger.GetStopDisassemblyDisplay();

      GetSymbolContext(eSymbolContextCompUnit | eSymbolContextLineEntry);
      if (m_sc.comp_unit && m_sc.line_entry.IsValid()) {
        have_debuginfo = true;
        if (source_lines_before > 0 || source_lines_after > 0) {
          size_t num_lines =
              target->GetSourceManager().DisplaySourceLinesWithLineNumbers(
                  m_sc.line_entry.file, m_sc.line_entry.line,
                  source_lines_before, source_lines_after, "->", &strm);
          if (num_lines != 0)
            have_source = true;
          // TODO: Give here a one time warning if source file is missing.
        }
      }
      switch (disasm_display) {
      case Debugger::eStopDisassemblyTypeNever:
        break;

      case Debugger::eStopDisassemblyTypeNoDebugInfo:
        if (have_debuginfo)
          break;
        LLVM_FALLTHROUGH;

      case Debugger::eStopDisassemblyTypeNoSource:
        if (have_source)
          break;
        LLVM_FALLTHROUGH;

      case Debugger::eStopDisassemblyTypeAlways:
        if (target) {
          const uint32_t disasm_lines = debugger.GetDisassemblyLineCount();
          if (disasm_lines > 0) {
            const ArchSpec &target_arch = target->GetArchitecture();
            AddressRange pc_range;
            pc_range.GetBaseAddress() = GetFrameCodeAddress();
            pc_range.SetByteSize(disasm_lines *
                                 target_arch.GetMaximumOpcodeByteSize());
            const char *plugin_name = nullptr;
            const char *flavor = nullptr;
            Disassembler::Disassemble(target->GetDebugger(), target_arch,
                                      plugin_name, flavor, exe_ctx, pc_range,
                                      disasm_lines, 0,
                                      Disassembler::eOptionMarkPCAddress, strm);
          }
        }
        break;
      }
    }
  }
  return true;
}
