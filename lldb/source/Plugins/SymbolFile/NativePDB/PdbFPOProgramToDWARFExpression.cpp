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
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/Stream.h"
#include "llvm/ADT/DenseMap.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/EnumTables.h"

using namespace lldb;
using namespace lldb_private;

namespace {

class FPOProgramNode;
class FPOProgramASTVisitor;

class NodeAllocator {
public:
  template <typename T, typename... Args> T *makeNode(Args &&... args) {
    void *new_node_mem = m_alloc.Allocate(sizeof(T), alignof(T));
    return new (new_node_mem) T(std::forward<Args>(args)...);
  }

private:
  llvm::BumpPtrAllocator m_alloc;
};

class FPOProgramNode {
public:
  enum Kind {
    Symbol,
    Register,
    IntegerLiteral,
    BinaryOp,
    UnaryOp,
  };

protected:
  FPOProgramNode(Kind kind) : m_token_kind(kind) {}

public:
  virtual ~FPOProgramNode() = default;
  virtual void Accept(FPOProgramASTVisitor *visitor) = 0;

  Kind GetKind() const { return m_token_kind; }

private:
  Kind m_token_kind;
};

class FPOProgramNodeSymbol: public FPOProgramNode {
public:
  FPOProgramNodeSymbol(llvm::StringRef name)
      : FPOProgramNode(Symbol), m_name(name) {}

  void Accept(FPOProgramASTVisitor *visitor) override;

  llvm::StringRef GetName() const { return m_name; }

private:
  llvm::StringRef m_name;
};

class FPOProgramNodeRegisterRef : public FPOProgramNode {
public:
  FPOProgramNodeRegisterRef(uint32_t lldb_reg_num)
      : FPOProgramNode(Register), m_lldb_reg_num(lldb_reg_num) {}

  void Accept(FPOProgramASTVisitor *visitor) override;

  uint32_t GetLLDBRegNum() const { return m_lldb_reg_num; }

private:
  uint32_t m_lldb_reg_num;
};

class FPOProgramNodeIntegerLiteral : public FPOProgramNode {
public:
  FPOProgramNodeIntegerLiteral(uint32_t value)
      : FPOProgramNode(IntegerLiteral), m_value(value) {}

  void Accept(FPOProgramASTVisitor *visitor) override;

  uint32_t GetValue() const { return m_value; }

private:
  uint32_t m_value;
};

class FPOProgramNodeBinaryOp : public FPOProgramNode {
public:
  enum OpType {
    Plus,
    Minus,
    Align,
  };

  FPOProgramNodeBinaryOp(OpType op_type, FPOProgramNode *left,
                         FPOProgramNode *right)
      : FPOProgramNode(BinaryOp), m_op_type(op_type), m_left(left),
        m_right(right) {}

  void Accept(FPOProgramASTVisitor *visitor) override;

  OpType GetOpType() const { return m_op_type; }

  const FPOProgramNode *Left() const { return m_left; }
  FPOProgramNode *&Left() { return m_left; }

  const FPOProgramNode *Right() const { return m_right; }
  FPOProgramNode *&Right() { return m_right; }

private:
  OpType m_op_type;
  FPOProgramNode *m_left;
  FPOProgramNode *m_right;
};

class FPOProgramNodeUnaryOp : public FPOProgramNode {
public:
  enum OpType {
    Deref,
  };

  FPOProgramNodeUnaryOp(OpType op_type, FPOProgramNode *operand)
      : FPOProgramNode(UnaryOp), m_op_type(op_type), m_operand(operand) {}

  void Accept(FPOProgramASTVisitor *visitor) override;

  OpType GetOpType() const { return m_op_type; }

  const FPOProgramNode *Operand() const { return m_operand; }
  FPOProgramNode *&Operand() { return m_operand; }

private:
  OpType m_op_type;
  FPOProgramNode *m_operand;
};

class FPOProgramASTVisitor {
public:
  virtual ~FPOProgramASTVisitor() = default;

  virtual void Visit(FPOProgramNodeSymbol *node) {}
  virtual void Visit(FPOProgramNodeRegisterRef *node) {}
  virtual void Visit(FPOProgramNodeIntegerLiteral *node) {}
  virtual void Visit(FPOProgramNodeBinaryOp *node) {}
  virtual void Visit(FPOProgramNodeUnaryOp *node) {}
};

void FPOProgramNodeSymbol::Accept(FPOProgramASTVisitor *visitor) {
  visitor->Visit(this);
}

void FPOProgramNodeRegisterRef::Accept(FPOProgramASTVisitor *visitor) {
  visitor->Visit(this);
}

void FPOProgramNodeIntegerLiteral::Accept(FPOProgramASTVisitor *visitor) {
  visitor->Visit(this);
}

void FPOProgramNodeBinaryOp::Accept(FPOProgramASTVisitor *visitor) {
  visitor->Visit(this);
}

void FPOProgramNodeUnaryOp::Accept(FPOProgramASTVisitor *visitor) {
  visitor->Visit(this);
}

class FPOProgramASTVisitorMergeDependent : public FPOProgramASTVisitor {
public:
  FPOProgramASTVisitorMergeDependent(
      const llvm::DenseMap<llvm::StringRef, FPOProgramNode *>
          &dependent_programs)
      : m_dependent_programs(dependent_programs) {}

  void Merge(FPOProgramNode *&node_ref);

private:
  void Visit(FPOProgramNodeRegisterRef *node) override {}
  void Visit(FPOProgramNodeIntegerLiteral *node) override {}
  void Visit(FPOProgramNodeBinaryOp *node) override;
  void Visit(FPOProgramNodeUnaryOp *node) override;

  void TryReplace(FPOProgramNode *&node_ref) const;

private:
  const llvm::DenseMap<llvm::StringRef, FPOProgramNode *> &m_dependent_programs;
};

void FPOProgramASTVisitorMergeDependent::Merge(FPOProgramNode *&node_ref) {
  TryReplace(node_ref);
  node_ref->Accept(this);
}

void FPOProgramASTVisitorMergeDependent::Visit(FPOProgramNodeBinaryOp *node) {
  Merge(node->Left());
  Merge(node->Right());
}
void FPOProgramASTVisitorMergeDependent::Visit(FPOProgramNodeUnaryOp *node) {
  Merge(node->Operand());
}

void FPOProgramASTVisitorMergeDependent::TryReplace(
    FPOProgramNode *&node_ref) const {

  while (node_ref->GetKind() == FPOProgramNode::Symbol) {
    auto *node_symbol_ref = static_cast<FPOProgramNodeSymbol *>(node_ref);

    auto it = m_dependent_programs.find(node_symbol_ref->GetName());
    if (it == m_dependent_programs.end()) {
      break;
    }

    node_ref = it->second;
  }
}

class FPOProgramASTVisitorResolveRegisterRefs : public FPOProgramASTVisitor {
public:
  FPOProgramASTVisitorResolveRegisterRefs(
      const llvm::DenseMap<llvm::StringRef, FPOProgramNode *>
          &dependent_programs,
      llvm::Triple::ArchType arch_type, NodeAllocator &alloc)
      : m_dependent_programs(dependent_programs), m_arch_type(arch_type),
        m_alloc(alloc) {}

  bool Resolve(FPOProgramNode *&program);

private:
  void Visit(FPOProgramNodeBinaryOp *node) override;
  void Visit(FPOProgramNodeUnaryOp *node) override;

  bool TryReplace(FPOProgramNode *&node_ref);

  const llvm::DenseMap<llvm::StringRef, FPOProgramNode *> &m_dependent_programs;
  llvm::Triple::ArchType m_arch_type;
  NodeAllocator &m_alloc;
  bool m_no_error_flag = true;
};

bool FPOProgramASTVisitorResolveRegisterRefs::Resolve(FPOProgramNode *&program) {
  if (!TryReplace(program))
    return false;
  program->Accept(this);
  return m_no_error_flag;
}

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

bool FPOProgramASTVisitorResolveRegisterRefs::TryReplace(
    FPOProgramNode *&node_ref) {
  if (node_ref->GetKind() != FPOProgramNode::Symbol)
    return true;

  auto *symbol = static_cast<FPOProgramNodeSymbol *>(node_ref);

  // Look up register reference as lvalue in preceding assignments.
  auto it = m_dependent_programs.find(symbol->GetName());
  if (it != m_dependent_programs.end()) {
    // Dependent programs are handled elsewhere.
    return true;
  }

  uint32_t reg_num =
      ResolveLLDBRegisterNum(symbol->GetName().drop_front(1), m_arch_type);

  if (reg_num == LLDB_INVALID_REGNUM)
    return false;

  node_ref = m_alloc.makeNode<FPOProgramNodeRegisterRef>(reg_num);
  return true;
}

void FPOProgramASTVisitorResolveRegisterRefs::Visit(
    FPOProgramNodeBinaryOp *node) {
  m_no_error_flag = Resolve(node->Left()) && Resolve(node->Right());
}

void FPOProgramASTVisitorResolveRegisterRefs::Visit(
    FPOProgramNodeUnaryOp *node) {
  m_no_error_flag = Resolve(node->Operand());
}

class FPOProgramASTVisitorDWARFCodegen : public FPOProgramASTVisitor {
public:
  FPOProgramASTVisitorDWARFCodegen(Stream &stream) : m_out_stream(stream) {}

  void Emit(FPOProgramNode *program);

private:
  void Visit(FPOProgramNodeRegisterRef *node) override;
  void Visit(FPOProgramNodeIntegerLiteral *node) override;
  void Visit(FPOProgramNodeBinaryOp *node) override;
  void Visit(FPOProgramNodeUnaryOp *node) override;

private:
  Stream &m_out_stream;
};

void FPOProgramASTVisitorDWARFCodegen::Emit(FPOProgramNode *program) {
  program->Accept(this);
}

void FPOProgramASTVisitorDWARFCodegen::Visit(FPOProgramNodeRegisterRef *node) {

  uint32_t reg_num = node->GetLLDBRegNum();
  lldbassert(reg_num != LLDB_INVALID_REGNUM);

  if (reg_num > 31) {
    m_out_stream.PutHex8(DW_OP_bregx);
    m_out_stream.PutULEB128(reg_num);
  } else
    m_out_stream.PutHex8(DW_OP_breg0 + reg_num);

  m_out_stream.PutSLEB128(0);
}

void FPOProgramASTVisitorDWARFCodegen::Visit(
    FPOProgramNodeIntegerLiteral *node) {
  uint32_t value = node->GetValue();
  m_out_stream.PutHex8(DW_OP_constu);
  m_out_stream.PutULEB128(value);
}

void FPOProgramASTVisitorDWARFCodegen::Visit(FPOProgramNodeBinaryOp *node) {

  Emit(node->Left());
  Emit(node->Right());

  switch (node->GetOpType()) {
  case FPOProgramNodeBinaryOp::Plus:
    m_out_stream.PutHex8(DW_OP_plus);
    // NOTE: can be optimized by using DW_OP_plus_uconst opcpode
    //       if right child node is constant value
    break;
  case FPOProgramNodeBinaryOp::Minus:
    m_out_stream.PutHex8(DW_OP_minus);
    break;
  case FPOProgramNodeBinaryOp::Align:
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

void FPOProgramASTVisitorDWARFCodegen::Visit(FPOProgramNodeUnaryOp *node) {
  Emit(node->Operand());

  switch (node->GetOpType()) {
  case FPOProgramNodeUnaryOp::Deref:
    m_out_stream.PutHex8(DW_OP_deref);
    break;
  }
}

} // namespace

static bool ParseFPOSingleAssignmentProgram(llvm::StringRef program,
                                            NodeAllocator &alloc,
                                            llvm::StringRef &register_name,
                                            FPOProgramNode *&ast) {
  llvm::SmallVector<llvm::StringRef, 16> tokens;
  llvm::SplitString(program, tokens, " ");

  if (tokens.empty())
    return false;

  llvm::SmallVector<FPOProgramNode *, 4> eval_stack;

  llvm::DenseMap<llvm::StringRef, FPOProgramNodeBinaryOp::OpType> ops_binary = {
      {"+", FPOProgramNodeBinaryOp::Plus},
      {"-", FPOProgramNodeBinaryOp::Minus},
      {"@", FPOProgramNodeBinaryOp::Align},
  };

  llvm::DenseMap<llvm::StringRef, FPOProgramNodeUnaryOp::OpType> ops_unary = {
      {"^", FPOProgramNodeUnaryOp::Deref},
  };

  constexpr llvm::StringLiteral ra_search_keyword = ".raSearch";

  // lvalue of assignment is always first token
  // rvalue program goes next
  for (size_t i = 1; i < tokens.size(); ++i) {
    llvm::StringRef cur = tokens[i];

    auto ops_binary_it = ops_binary.find(cur);
    if (ops_binary_it != ops_binary.end()) {
      // token is binary operator
      if (eval_stack.size() < 2) {
        return false;
      }
      FPOProgramNode *right = eval_stack.pop_back_val();
      FPOProgramNode *left = eval_stack.pop_back_val();
      FPOProgramNode *node = alloc.makeNode<FPOProgramNodeBinaryOp>(
          ops_binary_it->second, left, right);
      eval_stack.push_back(node);
      continue;
    }

    auto ops_unary_it = ops_unary.find(cur);
    if (ops_unary_it != ops_unary.end()) {
      // token is unary operator
      if (eval_stack.empty()) {
        return false;
      }
      FPOProgramNode *operand = eval_stack.pop_back_val();
      FPOProgramNode *node =
          alloc.makeNode<FPOProgramNodeUnaryOp>(ops_unary_it->second, operand);
      eval_stack.push_back(node);
      continue;
    }

    if (cur.startswith("$")) {
      eval_stack.push_back(alloc.makeNode<FPOProgramNodeSymbol>(cur));
      continue;
    }

    if (cur == ra_search_keyword) {
      // TODO: .raSearch is unsupported
      return false;
    }

    uint32_t value;
    if (!cur.getAsInteger(10, value)) {
      // token is integer literal
      eval_stack.push_back(alloc.makeNode<FPOProgramNodeIntegerLiteral>(value));
      continue;
    }

    // unexpected token
    return false;
  }

  if (eval_stack.size() != 1) {
    return false;
  }

  register_name = tokens[0];
  ast = eval_stack.pop_back_val();

  return true;
}

static FPOProgramNode *ParseFPOProgram(llvm::StringRef program,
                                       llvm::StringRef register_name,
                                       llvm::Triple::ArchType arch_type,
                                       NodeAllocator &alloc) {
  llvm::DenseMap<llvm::StringRef, FPOProgramNode *> dependent_programs;

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
    FPOProgramNode *rvalue_ast = nullptr;
    if (!ParseFPOSingleAssignmentProgram(assignment_program, alloc, lvalue_name,
                                         rvalue_ast)) {
      return nullptr;
    }

    lldbassert(rvalue_ast);

    // check & resolve assignment program
    FPOProgramASTVisitorResolveRegisterRefs resolver(dependent_programs,
                                                     arch_type, alloc);
    if (!resolver.Resolve(rvalue_ast)) {
      return nullptr;
    }

    if (lvalue_name == register_name) {
      // found target assignment program - no need to parse further

      // emplace valid dependent subtrees to make target assignment independent
      // from predecessors
      FPOProgramASTVisitorMergeDependent merger(dependent_programs);
      merger.Merge(rvalue_ast);

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
  NodeAllocator node_alloc;
  FPOProgramNode *target_program =
      ParseFPOProgram(program, register_name, arch_type, node_alloc);
  if (target_program == nullptr) {
    return false;
  }

  FPOProgramASTVisitorDWARFCodegen codegen(stream);
  codegen.Emit(target_program);
  return true;
}
