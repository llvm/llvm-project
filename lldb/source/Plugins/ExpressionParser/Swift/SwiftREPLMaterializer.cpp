//===-- SwiftREPLMaterializer.cpp -------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "SwiftREPLMaterializer.h"
#include "SwiftASTManipulator.h"
#include "SwiftPersistentExpressionState.h"

#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"
#include "lldb/Core/DumpDataExtractor.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Expression/IRMemoryMap.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Log.h"

#include "swift/Demangling/Demangle.h"

using namespace lldb_private;

static llvm::StringRef
GetNameOfDemangledVariable(swift::Demangle::NodePointer node_pointer) {
  if (!node_pointer ||
      node_pointer->getKind() != swift::Demangle::Node::Kind::Global)
    return llvm::StringRef();

  swift::Demangle::NodePointer variable_pointer =
      node_pointer->getFirstChild();

  if (!variable_pointer ||
      variable_pointer->getKind() != swift::Demangle::Node::Kind::Variable)
    return llvm::StringRef();

  for (swift::Demangle::NodePointer child : *variable_pointer) {
    if (child &&
        child->getKind() == swift::Demangle::Node::Kind::Identifier &&
        child->hasText()) {
      return child->getText();
    }
  }
  return llvm::StringRef();
}

/// Dereference global resilient values that are store in fixed-size
/// buffers, if the runtime says it's necessary.
static lldb::addr_t FixupResilientGlobal(lldb::addr_t var_addr,
                                         CompilerType compiler_type,
                                         IRExecutionUnit &execution_unit,
                                         lldb::ProcessSP process_sp,
                                         Status &error) {
  if (process_sp)
    if (auto *runtime = SwiftLanguageRuntime::Get(process_sp)) {
      if (!runtime->IsStoredInlineInBuffer(compiler_type)) {
        if (var_addr != LLDB_INVALID_ADDRESS) {
          size_t ptr_size = process_sp->GetAddressByteSize();
          llvm::SmallVector<uint8_t, 8> bytes;
          bytes.reserve(ptr_size);
          execution_unit.ReadMemory(bytes.data(), var_addr, ptr_size, error);
          if (error.Success())
            memcpy(&var_addr, bytes.data(), sizeof(var_addr));
        }
      }
    }
  return var_addr;
}

class EntityREPLResultVariable : public Materializer::Entity {
public:
  EntityREPLResultVariable(const CompilerType &type,
                           swift::ValueDecl *swift_decl,
                           SwiftREPLMaterializer *parent,
                           Materializer::PersistentVariableDelegate *delegate)
      : Entity(), m_type(type), m_parent(parent), m_swift_decl(swift_decl),
        m_temporary_allocation(LLDB_INVALID_ADDRESS),
        m_temporary_allocation_size(0), m_delegate(delegate) {
    // Hard-coding to maximum size of a pointer since all results are
    // materialized by reference
    m_size = 8;
    m_alignment = 8;
  }

  void Materialize(lldb::StackFrameSP &frame_sp, IRMemoryMap &map,
                   lldb::addr_t process_address, Status &err) override {
    // no action required
  }

  void MakeREPLResult(IRExecutionUnit &execution_unit, Status &err,
                      const IRExecutionUnit::JittedGlobalVariable *variable) {
    err.Clear();

    ExecutionContextScope *exe_scope =
        execution_unit.GetBestExecutionContextScope();

    if (!exe_scope) {
      err.SetErrorString("Couldn't dematerialize a result variable: invalid "
                         "execution context scope");
      return;
    }

    lldb::TargetSP target_sp = exe_scope->CalculateTarget();

    if (!target_sp) {
      err.SetErrorString("Couldn't dematerialize a result variable: no target");
      return;
    }

    lldb::LanguageType lang =
        (m_type.GetMinimumLanguage() == lldb::eLanguageTypeSwift)
            ? lldb::eLanguageTypeSwift
            : lldb::eLanguageTypeObjC_plus_plus;

    PersistentExpressionState *persistent_state =
        target_sp->GetPersistentExpressionStateForLanguage(lang);

    if (!persistent_state) {
      err.SetErrorString("Couldn't dematerialize a result variable: language "
                         "doesn't have persistent state");
      return;
    }

    ConstString name = m_delegate
                           ? m_delegate->GetName()
                           : persistent_state->GetNextPersistentVariableName();

    lldb::ExpressionVariableSP ret;

    ret = persistent_state
              ->CreatePersistentVariable(exe_scope, name, m_type,
                                         execution_unit.GetByteOrder(),
                                         execution_unit.GetAddressByteSize())
              ->shared_from_this();

    if (!ret) {
      err.SetErrorStringWithFormat("couldn't dematerialize a result variable: "
                                   "failed to make persistent variable %s",
                                   name.AsCString());
      return;
    }

    lldb::ProcessSP process_sp =
        execution_unit.GetBestExecutionContextScope()->CalculateProcess();

    ret->m_live_sp = ValueObjectConstResult::Create(
        exe_scope, m_type, name,
        variable ? variable->m_remote_addr : LLDB_INVALID_ADDRESS,
        eAddressTypeLoad, execution_unit.GetAddressByteSize());

    ret->ValueUpdated();

    if (variable) {
      const size_t pvar_byte_size = ret->GetByteSize().value_or(0);
      uint8_t *pvar_data = ret->GetValueBytes();

      Status read_error;
      // Handle resilient globals in fixed-size buffers.
      lldb::addr_t var_addr = variable->m_remote_addr;
      if (auto ast_ctx = m_type.GetTypeSystem()
                             .dyn_cast_or_null<SwiftASTContextForExpressions>())
        if (!ast_ctx->IsFixedSize(m_type))
          var_addr = FixupResilientGlobal(var_addr, m_type, execution_unit,
                                          process_sp, read_error);

      execution_unit.ReadMemory(pvar_data, var_addr, pvar_byte_size,
                                read_error);

      if (!read_error.Success()) {
        err.SetErrorString("Couldn't dematerialize a result variable: couldn't "
                           "read its memory");
        return;
      }
    }

    if (m_delegate) {
      m_delegate->DidDematerialize(ret);
    }

    // Register the variable with the persistent decls under the assumed,
    // just-generated name so it can be reused.

    if (m_swift_decl) {
      llvm::cast<SwiftPersistentExpressionState>(persistent_state)
          ->RegisterSwiftPersistentDeclAlias(
              {SwiftASTContext::GetSwiftASTContext(
                   &m_swift_decl->getASTContext()),
               m_swift_decl},
              name.GetStringRef());
    }

    return;
  }

  void Dematerialize(lldb::StackFrameSP &frame_sp, IRMemoryMap &map,
                     lldb::addr_t process_address, lldb::addr_t frame_top,
                     lldb::addr_t frame_bottom, Status &err) override {
    IRExecutionUnit *execution_unit =
        llvm::cast<SwiftREPLMaterializer>(m_parent)->GetExecutionUnit();

    if (!execution_unit) {
      return;
    }

    swift::Demangle::Context demangle_ctx;
    llvm::StringRef result_name = SwiftASTManipulator::GetResultName();

    for (const IRExecutionUnit::JittedGlobalVariable &variable :
         execution_unit->GetJittedGlobalVariables()) {
      auto *node_pointer = SwiftLanguageRuntime::DemangleSymbolAsNode(
          variable.m_name.GetStringRef(), demangle_ctx);

      llvm::StringRef variable_name = GetNameOfDemangledVariable(node_pointer);
      if (variable_name == result_name) {
        MakeREPLResult(*execution_unit, err, &variable);
        return;
      }

      demangle_ctx.clear();
    }

    std::optional<uint64_t> size =
        m_type.GetByteSize(execution_unit->GetBestExecutionContextScope());
    if (size && *size == 0) {
      MakeREPLResult(*execution_unit, err, nullptr);
      return;
    }

    err.SetErrorToGenericError();
    err.SetErrorStringWithFormat(
        "Couldn't dematerialize result: corresponding symbol wasn't found");
  }

  void DumpToLog(IRMemoryMap &map, lldb::addr_t process_address,
                 Log *log) override {
    StreamString dump_stream;

    const lldb::addr_t load_addr = process_address + m_offset;

    dump_stream.Printf("0x%" PRIx64 ": EntityResultVariable\n", load_addr);

    Status err;

    lldb::addr_t ptr = LLDB_INVALID_ADDRESS;

    {
      dump_stream.Printf("Pointer:\n");

      DataBufferHeap data(m_size, 0);

      map.ReadMemory(data.GetBytes(), load_addr, m_size, err);

      if (!err.Success()) {
        dump_stream.Printf("  <could not be read>\n");
      } else {
        DataExtractor extractor(data.GetBytes(), data.GetByteSize(),
                                map.GetByteOrder(), map.GetAddressByteSize());

        DumpHexBytes(&dump_stream, data.GetBytes(),
                               data.GetByteSize(), 16, load_addr);

        lldb::offset_t offset;

        ptr = extractor.GetAddress(&offset);

        dump_stream.PutChar('\n');
      }
    }

    if (m_temporary_allocation == LLDB_INVALID_ADDRESS) {
      dump_stream.Printf("Points to process memory:\n");
    } else {
      dump_stream.Printf("Temporary allocation:\n");
    }

    if (ptr == LLDB_INVALID_ADDRESS) {
      dump_stream.Printf("  <could not be be found>\n");
    } else {
      DataBufferHeap data(m_temporary_allocation_size, 0);

      map.ReadMemory(data.GetBytes(), m_temporary_allocation,
                     m_temporary_allocation_size, err);

      if (!err.Success()) {
        dump_stream.Printf("  <could not be read>\n");
      } else {
        DumpHexBytes(&dump_stream, data.GetBytes(),
                               data.GetByteSize(), 16, m_temporary_allocation);

        dump_stream.PutChar('\n');
      }
    }

    log->PutCString(dump_stream.GetData());
  }

  void Wipe(IRMemoryMap &map, lldb::addr_t process_address) override {
    m_temporary_allocation = LLDB_INVALID_ADDRESS;
    m_temporary_allocation_size = 0;
  }

private:
  CompilerType m_type;

  SwiftREPLMaterializer *m_parent;
  swift::ValueDecl *m_swift_decl; // only used for the REPL; nullptr otherwise

  lldb::addr_t m_temporary_allocation;
  size_t m_temporary_allocation_size;

  Materializer::PersistentVariableDelegate *m_delegate;
};

uint32_t SwiftREPLMaterializer::AddREPLResultVariable(
    const CompilerType &type, swift::ValueDecl *decl,
    PersistentVariableDelegate *delegate, Status &err) {
  EntityVector::iterator iter = m_entities.insert(m_entities.end(), EntityUP());

  iter->reset(new EntityREPLResultVariable(type, decl, this, delegate));
  uint32_t ret = AddStructMember(**iter);
  (*iter)->SetOffset(ret);

  return ret;
}

class EntityREPLPersistentVariable : public Materializer::Entity {
public:
  EntityREPLPersistentVariable(
      lldb::ExpressionVariableSP &persistent_variable_sp,
      SwiftREPLMaterializer *parent,
      Materializer::PersistentVariableDelegate *)
      : Entity(), m_persistent_variable_sp(persistent_variable_sp),
        m_parent(parent) {
    // Hard-coding to maximum size of a pointer since persistent variables are
    // materialized by reference
    m_size = 8;
    m_alignment = 8;
  }

  void Materialize(lldb::StackFrameSP &frame_sp, IRMemoryMap &map,
                   lldb::addr_t process_address, Status &err) override {
    // no action required
  }

  void Dematerialize(lldb::StackFrameSP &frame_sp, IRMemoryMap &map,
                     lldb::addr_t process_address, lldb::addr_t frame_top,
                     lldb::addr_t frame_bottom, Status &err) override {
    if (llvm::cast<SwiftExpressionVariable>(m_persistent_variable_sp.get())
            ->GetIsComputed())
      return;

    IRExecutionUnit *execution_unit = m_parent->GetExecutionUnit();

    if (!execution_unit) {
      return;
    }

    swift::Demangle::Context demangle_ctx;

    for (const IRExecutionUnit::JittedGlobalVariable &variable :
         execution_unit->GetJittedGlobalVariables()) {
      // e.g.
      // kind=Global
      //   kind=Variable
      //     kind=Module, text="lldb_expr_0"
      //     kind=Identifier, text="a"

      auto *node_pointer = SwiftLanguageRuntime::DemangleSymbolAsNode(
          variable.m_name.GetStringRef(), demangle_ctx);

      llvm::StringRef last_component = GetNameOfDemangledVariable(node_pointer);

      if (last_component.empty())
        continue;

      if (m_persistent_variable_sp->GetName().GetStringRef() ==
              last_component) {
        ExecutionContextScope *exe_scope =
            execution_unit->GetBestExecutionContextScope();

        if (!exe_scope) {
          err.SetErrorString("Couldn't dematerialize a persistent variable: "
                             "invalid execution context scope");
          return;
        }

        CompilerType compiler_type =
            m_persistent_variable_sp->GetCompilerType();

        m_persistent_variable_sp->m_live_sp = ValueObjectConstResult::Create(
            exe_scope, compiler_type, m_persistent_variable_sp->GetName(),
            variable.m_remote_addr, eAddressTypeLoad,
            execution_unit->GetAddressByteSize());

        // Read the contents of the spare memory area

        m_persistent_variable_sp->ValueUpdated();

        Status read_error;
        lldb::addr_t var_addr = variable.m_remote_addr;

        // Handle resilient globals in fixed-size buffers.
        if (Flags(m_persistent_variable_sp->m_flags)
            .Test(ExpressionVariable::EVIsSwiftFixedBuffer))
          var_addr =
              FixupResilientGlobal(var_addr, compiler_type, *execution_unit,
                                   exe_scope->CalculateProcess(), read_error);

        // FIXME: This may not work if the value is not bitwise-takable.
        execution_unit->ReadMemory(
            m_persistent_variable_sp->GetValueBytes(), var_addr,
            m_persistent_variable_sp->GetByteSize().value_or(0), read_error);

        if (!read_error.Success()) {
          err.SetErrorStringWithFormat(
              "couldn't read the contents of %s from memory: %s",
              m_persistent_variable_sp->GetName().GetCString(),
              read_error.AsCString());
          return;
        }

        m_persistent_variable_sp->m_flags &=
            ~ExpressionVariable::EVNeedsFreezeDry;

        return;
      }
      demangle_ctx.clear();
    }

    err.SetErrorToGenericError();
    err.SetErrorStringWithFormat(
        "Couldn't dematerialize %s: corresponding symbol wasn't found",
        m_persistent_variable_sp->GetName().GetCString());
  }

  void DumpToLog(IRMemoryMap &map, lldb::addr_t process_address,
                 Log *log) override {
    StreamString dump_stream;

    Status err;

    const lldb::addr_t load_addr = process_address + m_offset;

    dump_stream.Printf("0x%" PRIx64 ": EntityPersistentVariable (%s)\n",
                       load_addr,
                       m_persistent_variable_sp->GetName().AsCString());

    {
      dump_stream.Printf("Pointer:\n");

      DataBufferHeap data(m_size, 0);

      map.ReadMemory(data.GetBytes(), load_addr, m_size, err);

      if (!err.Success()) {
        dump_stream.Printf("  <could not be read>\n");
      } else {
        DumpHexBytes(&dump_stream, data.GetBytes(), data.GetByteSize(), 16,
                     load_addr);

        dump_stream.PutChar('\n');
      }
    }

    {
      dump_stream.Printf("Target:\n");

      lldb::addr_t target_address = LLDB_INVALID_ADDRESS;

      map.ReadPointerFromMemory(&target_address, load_addr, err);

      if (!err.Success()) {
        dump_stream.Printf("  <could not be read>\n");
      } else {
        DataBufferHeap data(m_persistent_variable_sp->GetByteSize().value_or(0),
                            0);

        map.ReadMemory(data.GetBytes(), target_address,
                       m_persistent_variable_sp->GetByteSize().value_or(0),
                       err);

        if (!err.Success()) {
          dump_stream.Printf("  <could not be read>\n");
        } else {
          DumpHexBytes(&dump_stream, data.GetBytes(), data.GetByteSize(), 16,
                       target_address);

          dump_stream.PutChar('\n');
        }
      }
    }

    log->PutCString(dump_stream.GetData());
  }

  void Wipe(IRMemoryMap &map, lldb::addr_t process_address) override {}

private:
  lldb::ExpressionVariableSP m_persistent_variable_sp;
  SwiftREPLMaterializer *m_parent;
};

uint32_t SwiftREPLMaterializer::AddPersistentVariable(
    lldb::ExpressionVariableSP &persistent_variable_sp,
    PersistentVariableDelegate *delegate, Status &err) {
  EntityVector::iterator iter = m_entities.insert(m_entities.end(), EntityUP());
  iter->reset(
      new EntityREPLPersistentVariable(persistent_variable_sp, this, delegate));
  uint32_t ret = AddStructMember(**iter);
  (*iter)->SetOffset(ret);
  return ret;
}
