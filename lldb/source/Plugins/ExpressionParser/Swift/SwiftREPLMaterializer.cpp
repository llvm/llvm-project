//===-- SwiftREPLMaterializer.cpp -------------------------------*- C++ -*-===//
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

#include "SwiftREPLMaterializer.h"
#include "SwiftASTManipulator.h"

#include "lldb/Core/Log.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Expression/IRExecutionUnit.h"
#include "lldb/Expression/IRMemoryMap.h"
#include "lldb/Target/Target.h"

#include "swift/Basic/Demangle.h"

using namespace lldb_private;

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
                   lldb::addr_t process_address, Error &err) {
    // no action required
  }

  void MakeREPLResult(IRExecutionUnit &execution_unit, Error &err,
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
    }

    ConstString name =
        m_delegate ? m_delegate->GetName()
                   : persistent_state->GetNextPersistentVariableName(false);

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
      const size_t pvar_byte_size = ret->GetByteSize();
      uint8_t *pvar_data = ret->GetValueBytes();

      Error read_error;

      execution_unit.ReadMemory(pvar_data, variable->m_remote_addr,
                                pvar_byte_size, read_error);

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
          ->RegisterSwiftPersistentDeclAlias(m_swift_decl, name);
    }

    return;
  }

  void Dematerialize(lldb::StackFrameSP &frame_sp, IRMemoryMap &map,
                     lldb::addr_t process_address, lldb::addr_t frame_top,
                     lldb::addr_t frame_bottom, Error &err) {
    IRExecutionUnit *execution_unit =
        llvm::cast<SwiftREPLMaterializer>(m_parent)->GetExecutionUnit();

    if (!execution_unit) {
      return;
    }

    for (const IRExecutionUnit::JittedGlobalVariable &variable :
         execution_unit->GetJittedGlobalVariables()) {
      if (strstr(variable.m_name.GetCString(),
                 SwiftASTManipulator::GetResultName())) {
        MakeREPLResult(*execution_unit, err, &variable);
        return;
      }
    }

    if (SwiftASTContext::IsPossibleZeroSizeType(m_type)) {
      MakeREPLResult(*execution_unit, err, nullptr);
      return;
    }

    err.SetErrorToGenericError();
    err.SetErrorStringWithFormat(
        "Couldn't dematerialize result: corresponding symbol wasn't found");
  }

  void DumpToLog(IRMemoryMap &map, lldb::addr_t process_address, Log *log) {
    StreamString dump_stream;

    const lldb::addr_t load_addr = process_address + m_offset;

    dump_stream.Printf("0x%" PRIx64 ": EntityResultVariable\n", load_addr);

    Error err;

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

        extractor.DumpHexBytes(&dump_stream, data.GetBytes(),
                               data.GetByteSize(), 16, load_addr);

        lldb::offset_t offset;

        ptr = extractor.GetPointer(&offset);

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
        DataExtractor extractor(data.GetBytes(), data.GetByteSize(),
                                map.GetByteOrder(), map.GetAddressByteSize());

        extractor.DumpHexBytes(&dump_stream, data.GetBytes(),
                               data.GetByteSize(), 16, m_temporary_allocation);

        dump_stream.PutChar('\n');
      }
    }

    log->PutCString(dump_stream.GetData());
  }

  void Wipe(IRMemoryMap &map, lldb::addr_t process_address) {
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
    PersistentVariableDelegate *delegate, Error &err) {
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
      Materializer::PersistentVariableDelegate *delegate)
      : Entity(), m_persistent_variable_sp(persistent_variable_sp),
        m_parent(parent), m_delegate(delegate) {
    // Hard-coding to maximum size of a pointer since persistent variables are
    // materialized by reference
    m_size = 8;
    m_alignment = 8;
  }

  void Materialize(lldb::StackFrameSP &frame_sp, IRMemoryMap &map,
                   lldb::addr_t process_address, Error &err) {
    // no action required
  }

  void Dematerialize(lldb::StackFrameSP &frame_sp, IRMemoryMap &map,
                     lldb::addr_t process_address, lldb::addr_t frame_top,
                     lldb::addr_t frame_bottom, Error &err) {
    if (llvm::cast<SwiftExpressionVariable>(m_persistent_variable_sp.get())
            ->GetIsComputed())
      return;

    IRExecutionUnit *execution_unit = m_parent->GetExecutionUnit();

    if (!execution_unit) {
      return;
    }

    for (const IRExecutionUnit::JittedGlobalVariable &variable :
         execution_unit->GetJittedGlobalVariables()) {
      // e.g.
      // kind=Global
      //   kind=Variable
      //     kind=Module, text="lldb_expr_0"
      //     kind=Identifier, text="a"

      swift::Demangle::NodePointer node_pointer =
          swift::Demangle::demangleSymbolAsNode(variable.m_name.GetCString(),
                                                variable.m_name.GetLength());

      if (!node_pointer ||
          node_pointer->getKind() != swift::Demangle::Node::Kind::Global)
        continue;

      swift::Demangle::NodePointer variable_pointer =
          node_pointer->getFirstChild();

      if (!variable_pointer ||
          variable_pointer->getKind() != swift::Demangle::Node::Kind::Variable)
        continue;

      llvm::StringRef last_component;

      for (swift::Demangle::NodePointer child : *variable_pointer) {
        if (child &&
            child->getKind() == swift::Demangle::Node::Kind::Identifier &&
            child->hasText()) {
          last_component = child->getText();
          break;
        }
      }

      if (last_component.empty())
        continue;

      if (m_persistent_variable_sp->GetName().GetStringRef().equals(
              last_component)) {
        ExecutionContextScope *exe_scope =
            execution_unit->GetBestExecutionContextScope();

        if (!exe_scope) {
          err.SetErrorString("Couldn't dematerialize a persistent variable: "
                             "invalid execution context scope");
          return;
        }

        m_persistent_variable_sp->m_live_sp = ValueObjectConstResult::Create(
            exe_scope, m_persistent_variable_sp->GetCompilerType(),
            m_persistent_variable_sp->GetName(), variable.m_remote_addr,
            eAddressTypeLoad, execution_unit->GetAddressByteSize());

        // Read the contents of the spare memory area

        m_persistent_variable_sp->ValueUpdated();

        Error read_error;

        execution_unit->ReadMemory(
            m_persistent_variable_sp->GetValueBytes(), variable.m_remote_addr,
            m_persistent_variable_sp->GetByteSize(), read_error);

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
    }

    err.SetErrorToGenericError();
    err.SetErrorStringWithFormat(
        "Couldn't dematerialize %s: corresponding symbol wasn't found",
        m_persistent_variable_sp->GetName().GetCString());
  }

  void DumpToLog(IRMemoryMap &map, lldb::addr_t process_address, Log *log) {
    StreamString dump_stream;

    Error err;

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
        DataExtractor extractor(data.GetBytes(), data.GetByteSize(),
                                map.GetByteOrder(), map.GetAddressByteSize());

        extractor.DumpHexBytes(&dump_stream, data.GetBytes(),
                               data.GetByteSize(), 16, load_addr);

        dump_stream.PutChar('\n');
      }
    }

    {
      dump_stream.Printf("Target:\n");

      lldb::addr_t target_address;

      map.ReadPointerFromMemory(&target_address, load_addr, err);

      if (!err.Success()) {
        dump_stream.Printf("  <could not be read>\n");
      } else {
        DataBufferHeap data(m_persistent_variable_sp->GetByteSize(), 0);

        map.ReadMemory(data.GetBytes(), target_address,
                       m_persistent_variable_sp->GetByteSize(), err);

        if (!err.Success()) {
          dump_stream.Printf("  <could not be read>\n");
        } else {
          DataExtractor extractor(data.GetBytes(), data.GetByteSize(),
                                  map.GetByteOrder(), map.GetAddressByteSize());

          extractor.DumpHexBytes(&dump_stream, data.GetBytes(),
                                 data.GetByteSize(), 16, target_address);

          dump_stream.PutChar('\n');
        }
      }
    }

    log->PutCString(dump_stream.GetData());
  }

  void Wipe(IRMemoryMap &map, lldb::addr_t process_address) {}

private:
  lldb::ExpressionVariableSP m_persistent_variable_sp;
  SwiftREPLMaterializer *m_parent;
  Materializer::PersistentVariableDelegate *m_delegate;
};

uint32_t SwiftREPLMaterializer::AddPersistentVariable(
    lldb::ExpressionVariableSP &persistent_variable_sp,
    PersistentVariableDelegate *delegate, Error &err) {
  EntityVector::iterator iter = m_entities.insert(m_entities.end(), EntityUP());
  iter->reset(
      new EntityREPLPersistentVariable(persistent_variable_sp, this, delegate));
  uint32_t ret = AddStructMember(**iter);
  (*iter)->SetOffset(ret);
  return ret;
}
