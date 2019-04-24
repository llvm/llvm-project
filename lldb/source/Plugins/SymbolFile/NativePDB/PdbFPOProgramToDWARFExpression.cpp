//===-- PDBFPOProgramToDWARFExpression.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PdbFPOProgramToDWARFExpression.h"
#include "CodeViewRegisterMapping.h"

#include "lldb/Core/StreamBuffer.h"
#include "lldb/Core/dwarf.h"
#include "lldb/Symbol/PostfixExpression.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/Stream.h"
#include "llvm/ADT/DenseMap.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/EnumTables.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::postfix;

namespace {

class FPOProgramASTVisitorMergeDependent : public Visitor<> {
public:
  void Visit(BinaryOpNode &binary, Node *&) override {
    Dispatch(binary.Left());
    Dispatch(binary.Right());
  }

  void Visit(UnaryOpNode &unary, Node *&) override {
    Dispatch(unary.Operand());
  }

  void Visit(RegisterNode &, Node *&) override {}
  void Visit(IntegerNode &, Node *&) override {}
  void Visit(SymbolNode &symbol, Node *&ref) override;

  static void
  Merge(const llvm::DenseMap<llvm::StringRef, Node *> &dependent_programs,
        Node *&ast) {
    FPOProgramASTVisitorMergeDependent(dependent_programs).Dispatch(ast);
  }

private:
  FPOProgramASTVisitorMergeDependent(
      const llvm::DenseMap<llvm::StringRef, Node *> &dependent_programs)
      : m_dependent_programs(dependent_programs) {}

  const llvm::DenseMap<llvm::StringRef, Node *> &m_dependent_programs;
};

void FPOProgramASTVisitorMergeDependent::Visit(SymbolNode &symbol, Node *&ref) {
  auto it = m_dependent_programs.find(symbol.GetName());
  if (it == m_dependent_programs.end())
    return;

  ref = it->second;
  Dispatch(ref);
}

class FPOProgramASTVisitorResolveRegisterRefs : public Visitor<bool> {
public:
  static bool
  Resolve(const llvm::DenseMap<llvm::StringRef, Node *> &dependent_programs,
          llvm::Triple::ArchType arch_type, llvm::BumpPtrAllocator &alloc,
          Node *&ast) {
    return FPOProgramASTVisitorResolveRegisterRefs(dependent_programs,
                                                   arch_type, alloc)
        .Dispatch(ast);
  }

  bool Visit(BinaryOpNode &binary, Node *&) override {
    return Dispatch(binary.Left()) && Dispatch(binary.Right());
  }

  bool Visit(UnaryOpNode &unary, Node *&) override {
    return Dispatch(unary.Operand());
  }

  bool Visit(RegisterNode &, Node *&) override { return true; }

  bool Visit(IntegerNode &, Node *&) override { return true; }

  bool Visit(SymbolNode &symbol, Node *&ref) override;

private:
  FPOProgramASTVisitorResolveRegisterRefs(
      const llvm::DenseMap<llvm::StringRef, Node *> &dependent_programs,
      llvm::Triple::ArchType arch_type, llvm::BumpPtrAllocator &alloc)
      : m_dependent_programs(dependent_programs), m_arch_type(arch_type),
        m_alloc(alloc) {}

  const llvm::DenseMap<llvm::StringRef, Node *> &m_dependent_programs;
  llvm::Triple::ArchType m_arch_type;
  llvm::BumpPtrAllocator &m_alloc;
};

static uint32_t ResolveLLDBRegisterNum(llvm::StringRef reg_name, llvm::Triple::ArchType arch_type) {
  // lookup register name to get lldb register number
  llvm::ArrayRef<llvm::EnumEntry<uint16_t>> register_names =
      llvm::codeview::getRegisterNames();
  auto it = llvm::find_if(
      register_names,
      [&reg_name](const llvm::EnumEntry<uint16_t> &register_entry) {
        return reg_name.compare_lower(register_entry.Name) == 0;
      });

  if (it == register_names.end())
    return LLDB_INVALID_REGNUM;

  auto reg_id = static_cast<llvm::codeview::RegisterId>(it->Value);
  return npdb::GetLLDBRegisterNumber(arch_type, reg_id);
}

bool FPOProgramASTVisitorResolveRegisterRefs::Visit(SymbolNode &symbol,
                                                    Node *&ref) {
  // Look up register reference as lvalue in preceding assignments.
  auto it = m_dependent_programs.find(symbol.GetName());
  if (it != m_dependent_programs.end()) {
    // Dependent programs are handled elsewhere.
    return true;
  }

  uint32_t reg_num =
      ResolveLLDBRegisterNum(symbol.GetName().drop_front(1), m_arch_type);

  if (reg_num == LLDB_INVALID_REGNUM)
    return false;

  ref = MakeNode<RegisterNode>(m_alloc, reg_num);
  return true;
}

class FPOProgramASTVisitorDWARFCodegen : public Visitor<> {
public:
  static void Emit(Stream &stream, Node *&ast) {
    FPOProgramASTVisitorDWARFCodegen(stream).Dispatch(ast);
  }

  void Visit(RegisterNode &reg, Node *&);
  void Visit(BinaryOpNode &binary, Node *&);
  void Visit(UnaryOpNode &unary, Node *&);
  void Visit(SymbolNode &symbol, Node *&) {
    llvm_unreachable("Symbols should have been resolved by now!");
  }
  void Visit(IntegerNode &integer, Node *&);

private:
  FPOProgramASTVisitorDWARFCodegen(Stream &stream) : m_out_stream(stream) {}

  Stream &m_out_stream;
};

void FPOProgramASTVisitorDWARFCodegen::Visit(RegisterNode &reg, Node *&) {
  uint32_t reg_num = reg.GetRegNum();
  lldbassert(reg_num != LLDB_INVALID_REGNUM);

  if (reg_num > 31) {
    m_out_stream.PutHex8(DW_OP_bregx);
    m_out_stream.PutULEB128(reg_num);
  } else
    m_out_stream.PutHex8(DW_OP_breg0 + reg_num);

  m_out_stream.PutSLEB128(0);
}

void FPOProgramASTVisitorDWARFCodegen::Visit(IntegerNode &integer, Node *&) {
  uint32_t value = integer.GetValue();
  m_out_stream.PutHex8(DW_OP_constu);
  m_out_stream.PutULEB128(value);
}

void FPOProgramASTVisitorDWARFCodegen::Visit(BinaryOpNode &binary, Node *&) {
  Dispatch(binary.Left());
  Dispatch(binary.Right());

  switch (binary.GetOpType()) {
  case BinaryOpNode::Plus:
    m_out_stream.PutHex8(DW_OP_plus);
    // NOTE: can be optimized by using DW_OP_plus_uconst opcpode
    //       if right child node is constant value
    break;
  case BinaryOpNode::Minus:
    m_out_stream.PutHex8(DW_OP_minus);
    break;
  case BinaryOpNode::Align:
    // emit align operator a @ b as
    // a & ~(b - 1)
    // NOTE: implicitly assuming that b is power of 2
    m_out_stream.PutHex8(DW_OP_lit1);
    m_out_stream.PutHex8(DW_OP_minus);
    m_out_stream.PutHex8(DW_OP_not);

    m_out_stream.PutHex8(DW_OP_and);
    break;
  }
}

void FPOProgramASTVisitorDWARFCodegen::Visit(UnaryOpNode &unary, Node *&) {
  Dispatch(unary.Operand());

  switch (unary.GetOpType()) {
  case UnaryOpNode::Deref:
    m_out_stream.PutHex8(DW_OP_deref);
    break;
  }
}

} // namespace

static bool ParseFPOSingleAssignmentProgram(llvm::StringRef program,
                                            llvm::BumpPtrAllocator &alloc,
                                            llvm::StringRef &register_name,
                                            Node *&ast) {
  // lvalue of assignment is always first token
  // rvalue program goes next
  std::tie(register_name, program) = getToken(program);
  if (register_name.empty())
    return false;

  ast = Parse(program, alloc);
  return ast != nullptr;
}

static Node *ParseFPOProgram(llvm::StringRef program,
                             llvm::StringRef register_name,
                             llvm::Triple::ArchType arch_type,
                             llvm::BumpPtrAllocator &alloc) {
  llvm::DenseMap<llvm::StringRef, Node *> dependent_programs;

  size_t cur = 0;
  while (true) {
    size_t assign_index = program.find('=', cur);
    if (assign_index == llvm::StringRef::npos) {
      llvm::StringRef tail = program.slice(cur, llvm::StringRef::npos);
      if (!tail.trim().empty()) {
        // missing assign operator
        return nullptr;
      }
      break;
    }
    llvm::StringRef assignment_program = program.slice(cur, assign_index);

    llvm::StringRef lvalue_name;
    Node *rvalue_ast = nullptr;
    if (!ParseFPOSingleAssignmentProgram(assignment_program, alloc, lvalue_name,
                                         rvalue_ast)) {
      return nullptr;
    }

    lldbassert(rvalue_ast);

    // check & resolve assignment program
    if (!FPOProgramASTVisitorResolveRegisterRefs::Resolve(
            dependent_programs, arch_type, alloc, rvalue_ast))
      return nullptr;

    if (lvalue_name == register_name) {
      // found target assignment program - no need to parse further

      // emplace valid dependent subtrees to make target assignment independent
      // from predecessors
      FPOProgramASTVisitorMergeDependent::Merge(dependent_programs, rvalue_ast);

      return rvalue_ast;
    }

    dependent_programs[lvalue_name] = rvalue_ast;
    cur = assign_index + 1;
  }

  return nullptr;
}

bool lldb_private::npdb::TranslateFPOProgramToDWARFExpression(
    llvm::StringRef program, llvm::StringRef register_name,
    llvm::Triple::ArchType arch_type, Stream &stream) {
  llvm::BumpPtrAllocator node_alloc;
  Node *target_program =
      ParseFPOProgram(program, register_name, arch_type, node_alloc);
  if (target_program == nullptr) {
    return false;
  }

  FPOProgramASTVisitorDWARFCodegen::Emit(stream, target_program);
  return true;
}
