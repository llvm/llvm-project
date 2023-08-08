//===- mlir-lsp-server.cpp - MLIR Language Server -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/AsmParser/CodeComplete.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Tools/lsp-server-support/Logging.h"
#include "mlir/Tools/lsp-server-support/SourceMgrUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/SourceMgr.h"
#include <fstream>
#include <iostream>
#include <json.hpp>
#include <optional>
#include <sstream>

using namespace mlir;
using namespace llvm;

int IndentLevel = 0;

llvm::raw_ostream &printIndent() {
#if false
  for (int i = 0; i < IndentLevel; ++i)
    llvm::outs() << "  ";
  return llvm::outs();
#else
  for (int i = 0; i < IndentLevel; ++i)
    llvm::dbgs() << "  ";
  return llvm::dbgs();
#endif
}

class IndentAdjustor {
public:
  IndentAdjustor() { IndentLevel++; }
  ~IndentAdjustor() { IndentLevel--; }
};

IndentAdjustor pushIndent() { return IndentAdjustor(); }

typedef std::unordered_map<int64_t, std::vector<std::pair<int64_t, int64_t>>>
    BlockArgsMap;
void printOperation(Operation *op, BlockArgsMap &blockArgsMap,
                    nlohmann::json &j);
void printBlock(Block &block, BlockArgsMap &blockArgsMap, nlohmann::json &j);

void printRegion(Region &region, BlockArgsMap &blockArgsMap,
                 nlohmann::json &j) {
  // A region does not hold anything by itself other than a list of blocks.
  LLVM_DEBUG(printIndent() << "Region with " << region.getBlocks().size()
                           << " blocks:\n");
  auto indent = pushIndent();

  // Print the operation attributes
  /*if (!region.getParentOp()->getAttrs().empty()) {
    printIndent() << region.getParentOp()->getAttrs().size() << " attributes:\n";
    for (NamedAttribute attr : region.getParentOp()->getAttrs())
      printIndent() << " - '" << attr.getName() << "' : '" << attr.getValue()
                    << "'\n";
  }*/

  /*printIndent() << "  region arg count: " << region.getNumArguments() << "\n";
  for (auto arg : region.getArguments()) {
    printIndent() << "     - region arg " << arg.getArgNumber() << ": " << arg
                  << "\n";
  }*/

  nlohmann::json regionj;
  for (Block &block : region.getBlocks())
    printBlock(block, blockArgsMap, regionj);
  j["Regions"].push_back(regionj);
}

StringRef getBlockName(Block &block);

void printBlock(Block &block, BlockArgsMap &blockArgsMap, nlohmann::json &j) {
  // Print the block intrinsics properties (basically: argument list)
  LLVM_DEBUG(printIndent()
      << "Block with " << block.getNumArguments() << " arguments, "
                << block.getNumSuccessors()
                << " successors, and "
                // Note, this `.size()` is traversing a linked-list and is O(n).
                << block.getOperations().size() << " operations\n");
  //auto blockName = getBlockName(block);
  //printIndent() << "=> block name: " << blockName;

  // A block main role is to hold a list of Operations: let's recurse into
  // printing each operation.
  auto indent = pushIndent();
  nlohmann::json blockj;
  blockj["Id"] = (int64_t)&block;
  blockj["Arguments"] = nlohmann::json::array();
  auto entry = blockArgsMap.find((int64_t)&block);

  if (entry != blockArgsMap.end()) {
    int index = 0;

    for (auto &blockArg : block.getArguments()) {
      //
      nlohmann::json usesj;
      LLVM_DEBUG(printIndent() << "at arg " << blockArg.getArgNumber() << ", "
                               << blockArg << "\n");

      for (auto &useOp : blockArg.getUses()) {
        usesj.push_back({{"UseId", (int64_t)&useOp},
                         {"UserId", (int64_t)useOp.getOwner()}});
      }

      //
      nlohmann::json incomingj;

      if (!block.isEntryBlock()) {
        for (auto predBlock = block.pred_begin(); predBlock != block.pred_end();
             predBlock++) {
          auto terminator = (*predBlock)->getTerminator();
          auto branch = dyn_cast<BranchOpInterface>(terminator);

          if (branch) {
            auto succIndex = predBlock.getSuccessorIndex();
            auto incomingOp =
                branch.getSuccessorOperands(succIndex)[blockArg.getArgNumber()];
            auto incomingArg =
                branch.getSuccessorBlockArgument(blockArg.getArgNumber());
            auto incomingOpIndex =
                branch.getSuccessorOperands(succIndex).getOperandIndex(
                    blockArg.getArgNumber());
            auto& incomingOpOp = terminator->getOpOperand(incomingOpIndex);

           /* if (incomingArg.has_value()) {
              printIndent() << " > incoming arg " << incomingArg.value();
              printIndent() << "    opop at index " << incomingOpIndex << ": "
                            << incomingOpOp.getOperandNumber();
            }

            printIndent() << "  > incoming op from succIndex " << succIndex
                          << ": " << incomingOp;*/
            
            incomingj.push_back( 
                {{"BlockId", (int64_t)(*predBlock)},
                 {"OperandId", (int64_t)&incomingOpOp}});
          }
        }
      } else {
        
      }

      nlohmann::json sourcej{{"Id", (int64_t)blockArg.getAsOpaquePointer()},
                             {"Uses", usesj},
                             {"LineNumber", 0},
                             {"StartOffset", entry->second[index].first},
                             {"EndOffset", entry->second[index].second}};
      blockj["Arguments"].push_back({
          {"Argument",sourcej},
          { "IncomingValues", incomingj}
      });

      index++;
    }
  }

  blockj["Predecessors"] = nlohmann::json::array();
  for (auto predBlock : block.getPredecessors()) {
    blockj["Predecessors"].push_back((int64_t)predBlock);
  }

  blockj["Successors"] = nlohmann::json::array();
  for (auto succBlock : block.getSuccessors()) {
    blockj["Successors"].push_back((int64_t)succBlock);
  }

  blockj["Operations"] = nlohmann::json::array();
  for (Operation &op : block.getOperations())
    printOperation(&op, blockArgsMap, blockj);
  j["Blocks"].push_back(blockj);
}

void printOperation(Operation *op, BlockArgsMap &blockArgsMap,
                    nlohmann::json &j) {
  // Print the operation itself and some of its properties
  LLVM_DEBUG(printIndent() << "\no visiting op " << (int64_t)op << ": '"
                          << op->getName()
                << "' with " << op->getNumOperands() << " operands and "
                << op->getNumResults() << " results\n");

  nlohmann::json opj;
  opj["Opcode"] = op->getName().getStringRef().str();
  opj["Id"] = (int64_t)op;

  // Print the operation attributes
  /*if (!op->getAttrs().empty()) {
    printIndent() << op->getAttrs().size() << " attributes:\n";
    for (NamedAttribute attr : op->getAttrs())
      printIndent() << " - '" << attr.getName() << "' : '" << attr.getValue()
                    << "'\n";
  }*/

  if (auto startOffsetAttr =
          op->getAttrOfType<IntegerAttr>("irx_start_offset")) {
    opj["StartOffset"] = startOffsetAttr.getUInt();
  }

  if (auto endOffsetAttr = op->getAttrOfType<IntegerAttr>("irx_end_offset")) {
    opj["EndOffset"] = endOffsetAttr.getUInt();
  }

  unsigned lineNumber = 0;

  if (auto lineAttr = op->getAttrOfType<IntegerAttr>("irx_line")) {
    lineNumber = lineAttr.getUInt();
    opj["LineNumber"] = lineAttr.getUInt();
  }

  opj["Sources"] = nlohmann::json::array();

  auto sourceAttr = op->getAttrOfType<ArrayAttr>("irx_ops");
  int index = 0;

  for (auto &sourceOp : op->getOpOperands()) {
    auto defOp = sourceOp.get().getDefiningOp();

    nlohmann::json sourcej{
        {"Id", (int64_t)&sourceOp},
        {"DefinitionId",
         defOp ? (int64_t)defOp : (int64_t)sourceOp.get().getAsOpaquePointer()},
        {"LineNumber", 0}};

    if (sourceAttr) {
      sourcej["StartOffset"] =
          sourceAttr[2 * index].cast<IntegerAttr>().getUInt();
      sourcej["EndOffset"] =
          sourceAttr[2 * index + 1].cast<IntegerAttr>().getUInt();
    }

    opj["Sources"].push_back(sourcej);

    LLVM_DEBUG(printIndent()
               << "     - source " << sourceOp.getOperandNumber() << " "
               << (int64_t)sourceOp.get().getAsOpaquePointer() << ", def "
               << (int64_t)sourceOp.get().getDefiningOp() << ": "
               << sourceOp.get() << "\n");

    index++;
  }

  auto resultsAttr = op->getAttrOfType<ArrayAttr>("irx_results");
  opj["Results"] = nlohmann::json::array();
  index = 0;

  for (auto destOp : op->getResults()) {
    nlohmann::json usesj;

    for (auto &useOp : destOp.getUses()) {
      usesj.push_back({{"UseId", (int64_t)&useOp},
                       {"UserId", (int64_t)useOp.getOwner()}});
    }

    nlohmann::json resultj{
        //{"Number", destOp.getResultNumber()},
        {"Id", (int64_t)destOp.getAsOpaquePointer()},
        {"Uses", usesj},
    };

    if (resultsAttr) {
      resultj["StartOffset"] =
          resultsAttr[2 * index].cast<IntegerAttr>().getUInt();
      resultj["EndOffset"] =
          resultsAttr[2 * index + 1].cast<IntegerAttr>().getUInt();
    }

    opj["Results"].push_back(resultj);

    LLVM_DEBUG(printIndent()
               << "     - dest " << (int64_t)destOp.getAsOpaquePointer()
               << ", def " << (int64_t)destOp.getDefiningOp() << ": "
               << destOp.getResultNumber() << "\n ");
  }

  // Recurse into each of the regions attached to the operation.
  auto indent = pushIndent();
  opj["Regions"] = nlohmann::json::array();

  for (Region &region : op->getRegions()) {
    for (auto arg : region.getArguments()) {
      LLVM_DEBUG(printIndent() << "     - region arg " << arg.getArgNumber()
                               << ": " << arg << "\n");
      nlohmann::json sourcej{
          {"Id", (int64_t)arg.getAsOpaquePointer()},
         // {"DefinitionId", (int64_t)arg.getAsOpaquePointer()},
      };
      opj["Results"].push_back(sourcej);
    }

    printRegion(region, blockArgsMap, opj);
  }

  j["Operations"].push_back(opj);
}

#ifdef MLIR_INCLUDE_TESTS
namespace test {
void registerTestDialect(DialectRegistry &);
void registerTestTransformDialectExtension(DialectRegistry &);
} // namespace test
#endif

mlir::ModuleOp getParentModule(mlir::Operation *op) {
  // Traverse up the operation's parent chain until a module operation is found.
  while (op && !mlir::isa<mlir::ModuleOp>(op))
    op = op->getParentOp();

  // If a module operation is found, return it.
  return mlir::dyn_cast_or_null<mlir::ModuleOp>(op);
}

void buildBlockArgsMap(ArrayAttr &blockArgs, BlockArgsMap &blockArgsMap) {
  auto blockCountAttr = blockArgs[0].dyn_cast<IntegerAttr>();
  int64_t blockCount = blockCountAttr.getUInt();
  uint64_t index = 1;

  for (int blockIndex = 0; blockIndex < blockCount; blockIndex++) {
    auto blockAttr = blockArgs[index].dyn_cast<IntegerAttr>();
    auto argCountAttr = blockArgs[index + 1].dyn_cast<IntegerAttr>();
    index += 2;

    auto &argsVector = blockArgsMap[blockAttr.getUInt()];

    for (int i = 0; i < argCountAttr.getUInt(); i++, index += 2) {
      auto startOffsetAttr = blockArgs[index].dyn_cast<IntegerAttr>();
      auto endOffsetAttr = blockArgs[index + 1].dyn_cast<IntegerAttr>();
      argsVector.push_back(
          std::make_pair(startOffsetAttr.getUInt(), endOffsetAttr.getUInt()));
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 3) {
    return -1;
  }

  std::ifstream inFile;
  char *filePath = argv[1];
  inFile.open(filePath, std::ios::binary); // open the input file

  std::string data((std::istreambuf_iterator<char>(inFile)),
                   std::istreambuf_iterator<char>());

  DialectRegistry registry;
  registerAllDialects(registry);
#ifdef MLIR_INCLUDE_TESTS
  ::test::registerTestDialect(registry);
  ::test::registerTestTransformDialectExtension(registry);
#endif

  MLIRContext context(registry, MLIRContext::Threading::DISABLED);
  StringRef contents(data.c_str());

  // Try to parsed the given IR string.
  auto memBuffer = llvm::MemoryBuffer::getMemBufferCopy(contents);
  if (!memBuffer) {
    return 0;
  }

  Block parsedIR;
  AsmParserState asmState;

  llvm::SourceMgr sourceMgr;
  FallbackAsmResourceMap fallbackResourceMap;
  ParserConfig config(&context, /*verifyAfterParse=*/false,
                      &fallbackResourceMap);
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), SMLoc());

  if (failed(parseAsmSourceFile(sourceMgr, &parsedIR, config, &asmState))) {
    // If parsing failed, clear out any of the current state.
    parsedIR.clear();
    asmState = AsmParserState();
    fallbackResourceMap = FallbackAsmResourceMap();
    return -1;
  }

  nlohmann::json j;
  auto module = getParentModule(parsedIR.getParentOp());
  j["Name"] = module ? module.getName().value().data() : "";
  j["Functions"] = nlohmann::json::array();

  for (Operation &op : parsedIR) {
    nlohmann::json funcj;

    funcj["Opcode"] = op.getName().getStringRef().data();
    funcj["Name"] = "Unknown";
    funcj["Regions"] = nlohmann::json::array();

    auto blocksAttr = op.getAttrOfType<ArrayAttr>("irx_blockargs");
    BlockArgsMap blockArgsMap;

    if (blocksAttr) {
      buildBlockArgsMap(blocksAttr, blockArgsMap);
    }

    for (auto &region : op.getRegions()) {
      printRegion(region, blockArgsMap, funcj);
    }

    // llvm::outs() << "\n-----------------\n";

    j["Functions"].push_back(funcj);
  }

  // printIndent() << "\nJSON:\n" << j.dump();

  std::ofstream outFile;
  char *outFilePath = argv[2];
  outFile.open(outFilePath, std::ios::binary); // open the input file
  outFile << j.dump();
  outFile.close();

  return 0;
}
