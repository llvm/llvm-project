//
// Created by tanmay on 6/8/22.
//

#ifndef LLVM_COMPUTATIONGRAPH_H
#define LLVM_COMPUTATIONGRAPH_H

namespace llvm {

/// Base node type of the Computation Graph
/// Data Members:
/// NodeID: An integer to uniquely identify this node.
/// InstructionList: Nodes can be associated with one or more intructions in the
/// LLVM IR.
///
/// Kind: This graph can represent the following types of nodes:
/// 1. Number: Represents constants or literals
/// 2. Variable: Represents the registers with no specific value like inputs
/// to the functions or output of the function.
/// 3. UnaryInstruction: Represents unary instructions
/// 4. BinaryInstruction: Represents the binary instructions
/// 5. Unknown: Node not fitting any specific category
///
/// Depth: The depth of the node in the computation graph
///
/// UseList: List of pointers to nodes that use the result of this node/are
/// dependent on the this node.

class CGNode {
public:
  enum class NodeKind {
    Number,
    Variable,
    UnaryInstruction,
    BinaryInstruction,
    Unknown
  };

  inline const std::string toString(NodeKind K) const {
    switch (K)
    {
    case NodeKind::Number:   return "Number";
    case NodeKind::Variable:   return "Variable";
    case NodeKind::UnaryInstruction: return "UnaryInstruction";
    case NodeKind::BinaryInstruction: return "BinaryInstruction";
    case NodeKind::Unknown: return "Unknown";
    default: return "[Unclassified NodeKind]";
    }
  }

  CGNode() = delete;
  CGNode(const int NID,
         const NodeKind K,
         SmallVector<Instruction*> I) : NodeID(NID),
                                         InstructionList(I),
                                         Kind(K)  {}
  CGNode(const CGNode &N) = default;
  CGNode(CGNode &&N) : NodeID(N.NodeID),
                       InstructionList(N.InstructionList),
                       Kind(N.Kind) {}
  virtual ~CGNode() = 0;

  CGNode &operator=(const CGNode &N) {
    InstructionList = N.InstructionList;
    Kind = N.Kind;
    Depth = N.Depth;
    UseList = N.UseList;
    return *this;
  }

  CGNode &operator=(CGNode &&N) {
    NodeID = N.NodeID;
    InstructionList = std::move(N.InstructionList);
    Kind = N.Kind;
    Depth = N.Depth;
    UseList = std::move(N.UseList);
    return *this;
  }

  virtual operator std::string() const {
    std::string ReturnString = "NodeID: " + std::to_string(NodeID) + "\n";
    ReturnString += "Instructions:\n";
    for(Instruction* I : InstructionList) {
      std::string StringStream;
      raw_string_ostream InstructionListString(StringStream);
      InstructionListString << *I << "\n";
      ReturnString += InstructionListString.str() + "\n";
    }
    ReturnString += "Kind: " + toString(Kind) + "\n";
    ReturnString += "Depth: " + std::to_string(Depth) + "\n";
    ReturnString += "Users: [";
    for(CGNode* User : UseList) {
      ReturnString += std::to_string(User->NodeID) + ", ";
    }
    ReturnString += "]";

    return ReturnString;
  }

  int getNodeID() const {
    return NodeID;
  }

  SmallVector<Instruction*> &getInstructions() {
    assert(!InstructionList.empty() && "Instruction List is empty");
    return InstructionList;
  }

  NodeKind getKind() const { return Kind; }
  int getDepth() const { return Depth; }

  SmallVector<CGNode*> &getUseList() {
    return (SmallVector<CGNode*>&) UseList;
  }

protected:
  void setKind(NodeKind K) { Kind = K; }
  void setDepth(int D) { Depth = D; }

private:
  /// Append the list of instructions in \p Input to this node.
  void appendInstructions(const SmallVector<Instruction*> &Instructions) {
    llvm::append_range(InstructionList, Instructions);
  }
  void appendInstructions(CGNode &Input) {
    appendInstructions(Input.getInstructions());
  }
  void appendUse(const SmallVector<CGNode*> &Uses) {
    llvm::append_range(UseList, Uses);
  }

  int NodeID;
  SmallVector<Instruction*> InstructionList;
  NodeKind Kind;
  int Depth;
  SmallVector<CGNode*> UseList;
};

///// Subclass of CGNode representing the Number values in computation graph.
///// Data Members:
///// Number: templated type
/////     A number of any numerical type - integer, float, double
//template <typename NumType> class NumberCGNode : public CGNode {
//public:
//  NumberCGNode() = delete;
//  NumberCGNode(const int NID,
//               SmallVector<Instruction*> I,
//               NumType N) : CGNode(NID,
//                                   NodeKind::Number, I),
//                            Number(N) {}
//  NumberCGNode(const NumberCGNode &N) = delete;
//  NumberCGNode(NumberCGNode &&N) : CGNode(std::move(N)) {}
//  ~NumberCGNode() = default;
//
//  NumberCGNode &operator=(const NumberCGNode &N) {
//    CGNode::operator=(N);
//    Number = N.Number;
//    return *this;
//  }
//
//  NumberCGNode &operator=(NumberCGNode &&N) {
//    CGNode::operator=(std::move(N));
//    Number = N.Number;
//    return *this;
//  }
//
//  NumType getNumber() const { return Number; }
//
//  operator std::string () const {
//    std::string ReturnString = CGNode::operator std::string() + "\n";
//    ReturnString += "Number: " + std::to_string(Number);
//        return ReturnString;
//  }
//
//protected:
//  void setNumber(NumType N) { Number = N; }
//
//private:
//  NumType Number;
//};
//
///// Subclass of CGNode representing the Variable node of the computation graph.
///// Data Members:
///// VariableName: str
/////     Name of the variable/register
//class VariableCGNode : public CGNode {
//public:
//  VariableCGNode(const int NID,
//                 SmallVector<Instruction*> I,
//                 std::string V) : CGNode(NID,
//                                         NodeKind::Variable,
//                                         I),
//                                  VariableName(V) {}
//  VariableCGNode(const VariableCGNode &N) = delete;
//  VariableCGNode(VariableCGNode &&N) : CGNode(std::move(N)) {}
//  ~VariableCGNode() = default;
//
//  VariableCGNode &operator=(const VariableCGNode &N) {
//    CGNode::operator=(N);
//    VariableName = N.VariableName;
//    return *this;
//  }
//
//  VariableCGNode &operator=(VariableCGNode &&N) {
//    CGNode::operator=(std::move(N));
//    VariableName = N.VariableName;
//    return *this;
//  }
//
//  std::string getVariableName() const { return VariableName; }
//
//  operator std::string () const override {
//    std::string ReturnString = CGNode::operator std::string() + "\n";
//    ReturnString += "Variable: " + VariableName;
//    return ReturnString;
//  }
//
//protected:
//  void setVariableName(std::string V) { VariableName = V; }
//
//private:
//  std::string VariableName;
//};


/// Subclass of CGNode representing a Unary Instruction in LLVM.
/// Data Members:
/// Operand: CGNode*
///     Pointer to the operand of this unary instruction
/// Operation: string
///     String representing the operation performed by the associated Unary
///     Instruction
class UnaryInstructionCGNode : public CGNode {
public:
  enum OperationKind {
    Minus,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Log,
    Exp
  };

  inline const std::string toString(OperationKind K) const {
    switch (K)
    {
    case OperationKind::Minus:   return "Minus";
    default: return "[Unclassified OperationKind]";
    }
  }

  UnaryInstructionCGNode(const int NID,
                 SmallVector<Instruction*> I,
                         CGNode* O,
                         OperationKind Op) : CGNode(NID,
                                                    NodeKind::Variable,
                                                    I),
                                             Operand(O),
                                             Operation(Op) {}
  UnaryInstructionCGNode(const VariableCGNode &N) = delete;
  UnaryInstructionCGNode(VariableCGNode &&N) : CGNode(std::move(N)) {}
  ~UnaryInstructionCGNode() = default;

  UnaryInstructionCGNode &operator=(const UnaryInstructionCGNode &N) {
    CGNode::operator=(N);
    Operand = &*N.Operand;
    Operation = N.Operation;
    return *this;
  }

  UnaryInstructionCGNode &operator=(UnaryInstructionCGNode &&N) {
    CGNode::operator=(std::move(N));
    Operand = std::move(N.Operand);
    Operation = N.Operation;
    return *this;
  }

  CGNode* getOperand() const { return &*Operand; }
  OperationKind getOperation() const { return Operation; }


  operator std::string () const override {
    std::string ReturnString = CGNode::operator std::string() + "\n";
    ReturnString += "Operand: " +
                    std::to_string(Operand->getNodeID()) + "\n";
    ReturnString += "Operation: " + toString(Operation);
    return ReturnString;
  }

protected:
  void setOperand(CGNode* N) { Operand = N; }

private:
  CGNode* Operand;
  OperationKind Operation;
};

/// Subclass of CGNode representing a Binary Instruction in LLVM.
/// Data Members:
/// LeftOperand: CGNode*
///     Pointer to the left operand of this binary instruction
/// RightOperand: CGNode*
///     Pointer to the right operand of this binary instruction
/// Operation: string
///     String representing the operation performed by the associated Binary
///     Instruction
class BinaryInstructionCGNode : public CGNode {
public:
  enum OperationKind {
    Plus,
    Minus,
    Multiply,
    Divide
  };

  inline const std::string toString(OperationKind K) const {
    switch (K)
    {
    case OperationKind::Minus:   return "Minus";
    default: return "[Unclassified OperationKind]";
    }
  }

  BinaryInstructionCGNode(const int NID, SmallVector<Instruction *> I,
                          CGNode *LO, CGNode *RO, OperationKind Op)
      : CGNode(NID,
                                                    NodeKind::Variable,
                                                    I),
                                              LeftOperand(LO),
                                              RightOperand(RO),
                                              Operation(Op) {}
  BinaryInstructionCGNode(const VariableCGNode &N) = delete;
  BinaryInstructionCGNode(VariableCGNode &&N) : CGNode(std::move(N)) {}
  ~BinaryInstructionCGNode() = default;

  BinaryInstructionCGNode &operator=(const BinaryInstructionCGNode &N) {
    CGNode::operator=(N);
    LeftOperand = &*N.LeftOperand;
    RightOperand = &*N.RightOperand;
    Operation = N.Operation;
    return *this;
  }

  BinaryInstructionCGNode &operator=(BinaryInstructionCGNode &&N) {
    CGNode::operator=(std::move(N));
    LeftOperand = std::move(N.LeftOperand);
    RightOperand = std::move(N.RightOperand);
    Operation = N.Operation;
    return *this;
  }

  std::pair<CGNode*, CGNode*> getOperands() const {
      return std::make_pair(&*LeftOperand, &*RightOperand);
  }
  OperationKind getOperation() const { return Operation; }


  operator std::string () const override {
    std::string ReturnString = CGNode::operator std::string() + "\n";
    ReturnString += "Left Operand: " +
                    std::to_string(LeftOperand->getNodeID()) + "\n";
    ReturnString += "Right Operand: " +
                    std::to_string(RightOperand->getNodeID()) + "\n";
    ReturnString += "Operation: " + toString(Operation);
    return ReturnString;
  }

protected:

private:
  CGNode* LeftOperand;
  CGNode* RightOperand;
  OperationKind Operation;
};

/// Subclass of CGNode representing the Number node of the computation graph.
/// Data Members:
/// Number: templated type
///     A number of any numerical type - integer, float, double
class ComputationGraph {
public:
  static int NodeNumber;


};

} // namespace llvm

#endif // LLVM_COMPUTATIONGRAPH_H
