#include "llvm/Pass.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"
#include <stack>

#include <unordered_map>

using namespace llvm;

#define DEBUG_TYPE "source_pass"

namespace {

void load(Instruction &I) {

  if (auto *loadInst = llvm::dyn_cast<LoadInst>(&I)) {

    llvm::DebugLoc debugLoc = loadInst->getDebugLoc();
    if (debugLoc) {
      llvm::Metadata *metadata = debugLoc.get();

      llvm::Value *pointerOperand = loadInst->getPointerOperand();
      if (auto *gepInst = llvm::dyn_cast<GetElementPtrInst>(pointerOperand)) {
        dbgs() << "gepppp";
        llvm::Value *basePointer = gepInst->getPointerOperand();
        llvm::SmallVector<llvm::Value *, 4> indices(gepInst->idx_begin(),
                                                    gepInst->idx_end());
      }
      // }
    }
  }
}

std::string getExpressionFromOpcode(const std::string &opcode) {
  // Map LLVM opcodes to source-level expressions
  if (opcode == "add") {
    return "+";
  } else if (opcode == "sub") {
    return "-";
  } else if (opcode == "mul") {
    return "*";
  } else if (opcode == "div") {
    return "/";
  } else if (opcode == "shl") {
    return "<<";
  } else {
    // Handle unknown opcodes or unsupported operations
    return "unknown";
  }
}

std::unordered_map<llvm::Value *, std::string> sourceExpressions;
std::string buildSourceLevelExpressionFromIntrinsic(llvm::Instruction &I,
                                                    const std::string &symbol) {
  if (auto *dbgVal = dyn_cast<DbgValueInst>(&I)) {
    Value *llvmVal = dbgVal->getValue();
    DILocalVariable *localVar = dbgVal->getVariable();
    DIExpression *expr = dbgVal->getExpression();
      std::vector<std::string> sourceExpressions;
    std::string varName = localVar->getName().str();
   
    if (!expr->getNumElements()) {
      std::string value;
      std::string SourceExpression;
      llvm::raw_string_ostream valueStream(value);
      llvmVal->printAsOperand(valueStream, false /* PrintType */);
      value = valueStream.str();
     SourceExpression = localVar->getType()->getName().str() + " " + varName + " = " + value + "\n";
    //  dbgs() << "Source-level expression is "
        //     << localVar->getType()->getName().str() << " " << SourceExpression
          //   << " = " << value << "\n";
            sourceExpressions.push_back(SourceExpression);
   //         return  "";
     // return SourceExpression;
    }  
     
     else {
      std::string exprStr;
      llvm::raw_string_ostream exprStream(exprStr);
      llvmVal->printAsOperand(exprStream, false /* PrintType */);

      std::stack<uint64_t> exprStack;

      for (int i = 0; i < expr->getNumElements(); ++i) {
        uint64_t element = expr->getElement(i);
        exprStack.push(element);
      }

      while (!exprStack.empty()) {
        uint64_t element = exprStack.top();
        uint64_t store;

        if (element == llvm::dwarf::DW_OP_constu) {
          if (!exprStack.empty()) {
            store = exprStack.top();
            exprStream << store;
          }
          exprStack.pop();
        } else if (element == llvm::dwarf::DW_OP_mul) {
          exprStream << " * ";
          exprStack.pop();
        } else if (element == llvm::dwarf::DW_OP_minus) {
          exprStream << " - ";
          exprStack.pop();
        } else if (element == llvm::dwarf::DW_OP_stack_value) {
          exprStack.pop();
        } else {
          store = element;
          exprStack.pop();
        }
      }

      std::string expr = varName + " = " + exprStream.str();
      sourceExpressions.push_back(expr);
   //   dbgs() << "Source-level expression is " << SourceExpression << " = "
          //   << expr << "\n";
     // return "";
    }
    for (const std::string &expression : sourceExpressions) {
          dbgs() << "Source-level expression is: " << expression << "\n";
        }

        return "";

  } else if (auto *dbgDeclare = dyn_cast<DbgDeclareInst>(&I)) {
    // errs() << "Decl";
    DIVariable *localVar = dbgDeclare->getVariable();
    StringRef variableName = localVar->getName();

    AllocaInst *allocaInst = cast<AllocaInst>(dbgDeclare->getAddress());
    std::vector<std::string> sourceExpressions;
    for (User *user : allocaInst->users()) {
      if (auto *storeInst = dyn_cast<StoreInst>(user)) {
        Value *storedValue = storeInst->getValueOperand();
        std::string sourceExpression;

        if (auto *loadInst = dyn_cast<LoadInst>(storedValue)) {
          Value *loadedValue = loadInst->getPointerOperand();
          sourceExpression =
              variableName.str() + " = " + loadedValue->getNameOrAsOperand();
          sourceExpressions.push_back(sourceExpression);
        } else {
          sourceExpression =
              variableName.str() + " = " + storedValue->getNameOrAsOperand();
          sourceExpressions.push_back(sourceExpression);
        }
      }
    }


        for (const std::string &expression : sourceExpressions) {
          dbgs() << "Source-level expression is: " << expression << "\n";
        }
        return "";
  }
  return "";
}

std::string
buildSourceLevelExpressionWithRawInstruction(llvm::Instruction &I,
                                             const std::string &symbol) {
  std::string sourceExpression = "nullptr";

  if (llvm::BinaryOperator *binaryOp =
          llvm::dyn_cast<llvm::BinaryOperator>(&I)) {
    // Binary operator - build source expression using two operands
    llvm::Value *operand1 = binaryOp->getOperand(0);
    llvm::Value *operand2 = binaryOp->getOperand(1);

    // Check if source expressions exist for the operands
    std::string name1 = sourceExpressions.count(operand1)
                            ? sourceExpressions[operand1]
                            : operand1->getNameOrAsOperand();
    std::string name2 = sourceExpressions.count(operand2)
                            ? sourceExpressions[operand2]
                            : operand2->getNameOrAsOperand();

    // Construct the source expression using variable names and the symbol
    sourceExpression = "(" + name1 + " " + symbol + " " + name2 + ")";

    // Store the source expression for the current instruction
    sourceExpressions[&I] = sourceExpression;
  } else if (llvm::GetElementPtrInst *gepInst =
                 llvm::dyn_cast<llvm::GetElementPtrInst>(&I)) {

    llvm::Value *basePointer = gepInst->getOperand(0);
    llvm::Value *offset = gepInst->getOperand(gepInst->getNumIndices());

    std::string basePointerName = sourceExpressions.count(basePointer)
                                      ? sourceExpressions[basePointer]
                                      : basePointer->getNameOrAsOperand();
    std::string offsetName = sourceExpressions.count(offset)
                                 ? sourceExpressions[offset]
                                 : offset->getNameOrAsOperand();

    // Construct the source expression for the address computation
    sourceExpression = "&" + basePointerName + "[" + offsetName + "]";

    // Store the source expression for the current instruction
    sourceExpressions[&I] = sourceExpression;
  } else if (llvm::LoadInst *loadInst = llvm::dyn_cast<llvm::LoadInst>(&I)) {
    llvm::Value *address = loadInst->getOperand(0);

    // Check if source expressions exist for the address operand
    std::string addressName = sourceExpressions.count(address)
                                  ? sourceExpressions[address]
                                  : address->getNameOrAsOperand();

    // Construct the source expression for accessing the value
    sourceExpression = "*(" + addressName + ")";

    // Store the source expression for the current instruction
    sourceExpressions[&I] = sourceExpression;
  } else if (llvm::StoreInst *storeInst = llvm::dyn_cast<llvm::StoreInst>(&I)) {
    llvm::Value *value = storeInst->getValueOperand();
    llvm::Value *address = storeInst->getPointerOperand();

    std::string valueName = sourceExpressions.count(value)
                                ? sourceExpressions[value]
                                : value->getNameOrAsOperand();
    std::string addressName = sourceExpressions.count(address)
                                  ? sourceExpressions[address]
                                  : address->getNameOrAsOperand();

    sourceExpression = /*"*" +*/ addressName + " = " + valueName;
  } else if (llvm::UnaryOperator *unaryOp =
                 llvm::dyn_cast<llvm::UnaryOperator>(&I)) {
    dbgs() << "gelement";
    llvm::Value *operand = unaryOp->getOperand(0);

    std::string name = operand->getNameOrAsOperand();

    // Construct the source expression using the variable name and the symbol
    sourceExpression = name + symbol;
  }

  return sourceExpression;
}

// This method implements what the pass does
void visitor(Function &F) {
  errs() << "(llvm-tutor) Hello from: " << F.getName() << "\n";
  errs() << "(llvm-tutor)   number of arguments: " << F.arg_size() << "\n";
  std::string SourceExpression = "null";
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      //     errs() << I << "foo";
      std::string operation = llvm::Instruction::getOpcodeName(I.getOpcode());
      std::string symbol = getExpressionFromOpcode(operation);

      std::string srcExx = buildSourceLevelExpressionFromIntrinsic(I, symbol);
    }

  }
}

// New PM implementation
struct SourceExpressionPass : PassInfoMixin<SourceExpressionPass> {
  // Main entry point, takes IR unit to run the pass on (&F) and the
  // corresponding pass manager (to be queried if need be)
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    visitor(F);
    return PreservedAnalyses::all();
  }

  // Without isRequired returning true, this pass will be skipped for functions
  // decorated with the optnone LLVM attribute. Note that clang -O0 decorates
  // all functions with optnone.
  static bool isRequired() { return true; }
};

} // namespace

llvm::PassPluginLibraryInfo getSourceExprPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "SourceExprMap", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "source-expr") {
                    FPM.addPass(SourceExpressionPass());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getSourceExprPluginInfo();
}