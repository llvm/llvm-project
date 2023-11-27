//===- SourceExpressionAnalysis.cpp - Mapping Source Expression
//---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the mapping between LLVM Value and Source level
// expression, by utilizing the debug intrinsics.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/SourceExpressionAnalysis.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include <unordered_map>
using namespace llvm;

#define DEBUG_TYPE "source_expr"

// This function translates LLVM opcodes to source-level expressions using DWARF
// operation encodings. It takes an LLVM opcode as input and returns the
// corresponding symbol as a string. If the opcode is supported,
// the function returns the appropriate symbol, such as "+",
// "-", "*", "/", "<<", ">>", "&", "|", "^", or "%". If the opcode is not
// supported, the function returns "unknown".
std::string
LoadStoreSourceExpression::getExpressionFromOpcode(unsigned Opcode) {
  // Map LLVM opcodes to source-level expressions
  switch (Opcode) {
  case Instruction::Add:
  case Instruction::FAdd:
    return "+";
  case Instruction::Sub:
  case Instruction::FSub:
    return "-";
  case Instruction::Mul:
  case Instruction::FMul:
    return "*";
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::FDiv:
    return "/";
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::FRem:
    return "%";
  case Instruction::Shl:
    return "<<";
  case Instruction::LShr:
  case Instruction::AShr:
    return ">>";
  case Instruction::And:
    return "&";
  case Instruction::Or:
    return "|";
  case Instruction::Xor:
    return "^";
  default:
    return "unknown";
  }
}

// Function to remove the '&' character from a string
static const std::string removeAmpersand(StringRef AddrStr) {
  std::string Result = AddrStr.str();

  size_t Found = Result.find('&');
  if (Found != std::string::npos) {
    Result.erase(Found, 1);
  }
  return Result;
}

// Process the debug metadata for the given stored value. This function
// retrieves the corresponding debug values (DbgValueInst) and debug declare
// instructions (DbgDeclareInst) associated with the stored value. If a
// DbgDeclareInst is found, the associated DILocalVariable is retrieved and
// returned. If a DbgValueInst is found, the associated DILocalVariable is
// retrieved and the source expression is stored in the 'sourceExpressionsMap'
// for the stored value. This function is used to extract debug information for
// the source expressions.
//
// @param StoredValue The stored value to process.
// @return The DILocalVariable associated with the stored value, or nullptr if
// no debug metadata is found.
DILocalVariable *
LoadStoreSourceExpression::processDbgMetadata(Value *StoredValue) {
  if (StoredValue->isUsedByMetadata()) {
    // Find the corresponding DbgValues and DbgDeclareInsts
    SmallVector<DbgValueInst *, 8> DbgValues;
    findDbgValues(DbgValues, StoredValue);

    TinyPtrVector<DbgDeclareInst *> DbgDeclareInsts =
        FindDbgDeclareUses(StoredValue);

    if (!DbgDeclareInsts.empty()) {
      // Handle the case where DbgDeclareInst is found
      DbgDeclareInst *DbgDeclare = DbgDeclareInsts[0];
      DILocalVariable *LocalVar = DbgDeclare->getVariable();
      SourceExpressionsMap[StoredValue] = LocalVar->getName().str();
      return LocalVar;
    } else if (!DbgValues.empty()) {
      // Handle the case where DbgValueInst is found
      DbgValueInst *DbgValue = DbgValues[0];
      DILocalVariable *LocalVar = DbgValue->getVariable();
      SourceExpressionsMap[StoredValue] = LocalVar->getName().str();
      return LocalVar;
    }
  }

  return nullptr;
}

// Get the source-level expression for an LLVM value.
// @param Operand The LLVM value to generate the source-level expression for.
std::string LoadStoreSourceExpression::getSourceExpression(Value *Operand) {

  if (SourceExpressionsMap.count(Operand))
    return SourceExpressionsMap[Operand];

  if (GetElementPtrInst *GepInstruction =
          dyn_cast<GetElementPtrInst>(Operand)) {
    return getSourceExpressionForGetElementPtr(GepInstruction);
  } else if (BinaryOperator *BinaryOp = dyn_cast<BinaryOperator>(Operand)) {
    return getSourceExpressionForBinaryOperator(BinaryOp, Operand);
  } else if (SExtInst *SextInstruction = dyn_cast<SExtInst>(Operand)) {
    return getSourceExpressionForSExtInst(SextInstruction);
  } else {
    // Check if the operand has debug metadata associated with it
    if (!isa<ConstantInt>(Operand)) {
      DILocalVariable *LocalVar = processDbgMetadata(Operand);
      if (LocalVar) {
        SourceExpressionsMap[Operand] = LocalVar->getName().str();
        return SourceExpressionsMap[Operand];
      }
    }
  }

  // If no specific case matches, return the name of the operand or its
  // representation
  return Operand->getNameOrAsOperand();
}

// Get the type tag from the given DIType
// Returns:
//   0: If the DIType is null or the type tag is unknown or unsupported
//   DW_TAG_base_type, DW_TAG_pointer_type, DW_TAG_const_type, etc.: The type
//   tag
static uint16_t getTypeTag(DIType *TypeToBeProcessed) {
  if (!TypeToBeProcessed)
    return 0;

  if (auto *BasicType = dyn_cast<DIBasicType>(TypeToBeProcessed)) {
    return BasicType->getTag();
  } else if (auto *DerivedType = dyn_cast<DIDerivedType>(TypeToBeProcessed)) {
    return DerivedType->getTag();
  } else if (auto *CompositeType =
                 dyn_cast<DICompositeType>(TypeToBeProcessed)) {
    return CompositeType->getTag();
  }

  // Return 0 for unknown or unsupported type tags
  return 0;
}

// Get the source-level expression for a GetElementPtr instruction.
// @param GepInstruction The GetElementPtr instruction.
// @return The source-level expression for the address computation.
std::string LoadStoreSourceExpression::getSourceExpressionForGetElementPtr(
    GetElementPtrInst *GepInstruction) {
  // GetElementPtr instruction - construct source expression for address
  // computation
  Value *BasePointer = GepInstruction->getOperand(0);
  Value *Offset = GepInstruction->getOperand(GepInstruction->getNumIndices());
  // auto *type = GepInstruction->getSourceElementType();

  int OffsetVal = INT_MIN;
  if (ConstantInt *OffsetConstant = dyn_cast<ConstantInt>(Offset)) {
    // Retrieve the value of the constant integer as an integer
    OffsetVal = OffsetConstant->getSExtValue();
  }

  DILocalVariable *LocalVar = processDbgMetadata(BasePointer);
  DIType *Type = LocalVar ? LocalVar->getType() : nullptr;

  std::string BasePointerName = getSourceExpression(BasePointer);
  std::string OffsetName = getSourceExpression(Offset);

  SmallString<32> Expression;
  raw_svector_ostream OS(Expression);

  uint16_t Tag = getTypeTag(Type);
  auto *SourceElementType = GepInstruction->getSourceElementType();

  // If the source element type is a struct or an array of structs, set the
  // source expression as "unknown"
  if (Tag == dwarf::DW_TAG_structure_type ||
      SourceElementType->getTypeID() == Type::TypeID::StructTyID ||
      (SourceElementType->getTypeID() == Type::TypeID::ArrayTyID &&
       SourceElementType->getArrayElementType()->getTypeID() ==
           Type::TypeID::StructTyID) ||
      SourceExpressionsMap[BasePointer] == "unknown") {
    SourceExpressionsMap[GepInstruction] = "unknown";
    return "unknown";
  } else if (Tag == dwarf::DW_TAG_array_type ||
             isa<PointerType>(BasePointer->getType())) {
    if (BasePointerName.find('[') == std::string::npos) {
      // Construct the source expression for the address computation with
      // square brackets
      OS << "&" << BasePointerName << "[" << OffsetName << "]";
    } else if (BasePointerName.find('[') != std::string::npos) {
      // If basePointerName already contains square brackets, combine it
      // with offsetName directly
      OS << BasePointerName << "[" << OffsetName << "]";
    } else if (BasePointerName.find('[') != std::string::npos &&
               OffsetVal != INT_MIN) {
      // If basePointerName already contains square brackets, combine it
      // with offsetName directly
      OS << BasePointerName << " + " << OffsetVal;
    }
  }
  SourceExpressionsMap[GepInstruction] = Expression.str().str();

  // Return the constructed source expression
  return Expression.str().str();
}

// Get the source-level expression for a binary operator instruction.
// @param BinaryOp The binary operator instruction.
// @param Operand The operand associated with the instruction.
// @return The source-level expression for the binary operation.
std::string LoadStoreSourceExpression::getSourceExpressionForBinaryOperator(
    BinaryOperator *BinaryOp, Value *Operand) {
  // Binary operator - build source expression using two operands
  Value *Operand1 = BinaryOp->getOperand(0);
  Value *Operand2 = BinaryOp->getOperand(1);

  // dbgs() << Operand2->getNameOrAsOperand() << " " <<
  // Operand1->getNameOrAsOperand() << "\n"; dbgs() << *Operand;
  std::string Name1 = getSourceExpression(Operand1);
  std::string Name2 = getSourceExpression(Operand2);
  std::string Opcode = BinaryOp->getOpcodeName();

  SmallString<32> Expression;
  raw_svector_ostream OS(Expression);

  OS << "(" << Name1 << " " << getExpressionFromOpcode(BinaryOp->getOpcode())
     << " " << Name2 << ")";

  SourceExpressionsMap[Operand] = Expression.str().str();
  // Return the constructed source expression
  return Expression.str().str();
}

// Helper function to get the type name as a string
static std::string getTypeNameAsString(Type *TypeToBeProcessed) {
  std::string TypeName;
  raw_string_ostream TypeNameStream(TypeName);
  TypeToBeProcessed->print(TypeNameStream);
  return TypeNameStream.str();
}

// Get the source-level expression for a sign extension instruction.
// @param SextInstruction The sign extension instruction.
// @return The source-level expression for the operand.
std::string LoadStoreSourceExpression::getSourceExpressionForSExtInst(
    SExtInst *SextInstruction) {
  // Signed Extension instruction - return the source expression for its operand

  Value *OperandVal = SextInstruction->getOperand(0);
  std::string OperandName = getSourceExpression(OperandVal);

  // Get the target type name for the signed extension
  std::string TargetType = getTypeNameAsString(SextInstruction->getType());

  // Construct the source expression with the casting operation
  std::string SourceExpression = "(" + TargetType + ")" + OperandName;

  // Update the source expression map
  SourceExpressionsMap[OperandVal] = SourceExpression;

  // Return the source expression
  return SourceExpression;
}

// Process the StoreInst and generate the source expression for the stored
// value. This function takes a StoreInst pointer and processes the associated
// metadata to retrieve the variable name. It then constructs the source
// expressions for both the pointer operand and the value operand. If the
// operands are instructions, it calls the appropriate function to get their
// source expressions. Otherwise, it constructs the source expressions directly
// for non-instruction operands. The resulting source expressions are stored in
// the sourceExpressionsMap.
void LoadStoreSourceExpression::processStoreInst(StoreInst *I) {
  Value *PointerOperand = I->getPointerOperand();
  Value *ValueOperand = I->getValueOperand();

  std::string PointerExpression, ValueExpression;

  PointerExpression = getSourceExpression(PointerOperand);
  ValueExpression = getSourceExpression(ValueOperand);
  // Store the source expressions for both operands in the SourceExpressionsMap
  SourceExpressionsMap[PointerOperand] = PointerExpression;
  SourceExpressionsMap[ValueOperand] = ValueExpression;
}

// Process the LoadInst and generate the source expressions for the loaded value
// and its corresponding store instruction (if applicable).
void LoadStoreSourceExpression::processLoadInst(LoadInst *I) {
  SmallVector<std::string> SourceExpressions;

  Value *Val = I->getPointerOperand();

  // Check if the pointer operand of the LoadInst is an instruction
  if (isa<Instruction>(Val)) {
    std::string Expression;
    auto It = SourceExpressionsMap.find(Val);
    if (It != SourceExpressionsMap.end()) {
      Expression = It->second;
    } else {
      Expression = getSourceExpression(Val);
    }

    // Map the LoadInst to its source expression in the SourceExpressionsMap
    SourceExpressionsMap[I] = removeAmpersand(Expression);
  }
}

AnalysisKey SourceExpressionAnalysis::Key;

SourceExpressionAnalysis::Result
SourceExpressionAnalysis::run(Function &F, FunctionAnalysisManager &) {
  LoadStoreSourceExpression PI(
      F); // Create an instance of LoadStoreSourceExpression
  ReversePostOrderTraversal<Function *> RPOT(&F);
  for (BasicBlock *BB : RPOT) {
    for (Instruction &I : *BB) {
      if (auto *loadInst = dyn_cast<LoadInst>(&I)) {
        // Process the LoadInst and generate the source expressions
        PI.processLoadInst(loadInst);
      } else if (auto *storeInst = dyn_cast<StoreInst>(&I)) {
        // Process the StoreInst and generate the source expressions
        PI.processStoreInst(storeInst);
      }
    }
  }

  return PI; // Return the built LoadStoreSourceExpression instance
}

void LoadStoreSourceExpression::print(raw_ostream &OS) const {

  for (const auto &Entry : SourceExpressionsMap) {
    Value *Key = Entry.first;
    std::string Value = Entry.second;

    if (Instruction *KeyInst = dyn_cast<Instruction>(Key)) {
      KeyInst->printAsOperand(dbgs(), /*PrintType=*/false);
    } else {
      OS << "<unknown>";
    }
    OS << " = " << Value;

    OS << "\n";
  }
}

PreservedAnalyses
SourceExpressionAnalysisPrinterPass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  dbgs() << "Load Store Expression " << F.getName() << "\n";

  SourceExpressionAnalysis::Result &PI = AM.getResult<SourceExpressionAnalysis>(
      F); // Retrieve the correct analysis result type
  ReversePostOrderTraversal<Function *> RPOT(&F);
  for (BasicBlock *BB : RPOT) {
    for (Instruction &I : *BB) {
      if (auto *loadInst = dyn_cast<LoadInst>(&I)) {
        // Process the LoadInst and generate the source expressions
        PI.processLoadInst(loadInst);
      } else if (auto *storeInst = dyn_cast<StoreInst>(&I)) {
        // Process the StoreInst and generate the source expressions
        PI.processStoreInst(storeInst);
      }
    }
  }

  PI.print(OS);
  return PreservedAnalyses::all();
}
