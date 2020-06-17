//===- ViewOpGraph.cpp - View/write op graphviz graphs --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/ViewOpGraph.h"
#include "PassDetail.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

/// Return the size limits for eliding large attributes.
static int64_t getLargeAttributeSizeLimit() {
  // Use the default from the printer flags if possible.
  if (Optional<int64_t> limit = OpPrintingFlags().getLargeElementsAttrLimit())
    return *limit;
  return 16;
}

namespace llvm {

// Specialize GraphTraits to treat Block as a graph of Operations as nodes and
// uses as edges.
template <> struct GraphTraits<Block *> {
  using GraphType = Block *;
  using NodeRef = Operation *;

  using ChildIteratorType = Operation::user_iterator;
  static ChildIteratorType child_begin(NodeRef n) { return n->user_begin(); }
  static ChildIteratorType child_end(NodeRef n) { return n->user_end(); }

  // Operation's destructor is private so use Operation* instead and use
  // mapped iterator.
  static Operation *AddressOf(Operation &op) { return &op; }
  using nodes_iterator = mapped_iterator<Block::iterator, decltype(&AddressOf)>;
  static nodes_iterator nodes_begin(Block *b) {
    return nodes_iterator(b->begin(), &AddressOf);
  }
  static nodes_iterator nodes_end(Block *b) {
    return nodes_iterator(b->end(), &AddressOf);
  }
};

// Specialize DOTGraphTraits to produce more readable output.
template <> struct DOTGraphTraits<Block *> : public DefaultDOTGraphTraits {
  using DefaultDOTGraphTraits::DefaultDOTGraphTraits;
  static std::string getNodeLabel(Operation *op, Block *);
};

std::string DOTGraphTraits<Block *>::getNodeLabel(Operation *op, Block *b) {
  // Reuse the print output for the node labels.
  std::string ostr;
  raw_string_ostream os(ostr);
  os << op->getName() << "\n";

  if (!op->getLoc().isa<UnknownLoc>()) {
    os << op->getLoc() << "\n";
  }

  // Print resultant types
  llvm::interleaveComma(op->getResultTypes(), os);
  os << "\n";

  // A value used to elide large container attribute.
  int64_t largeAttrLimit = getLargeAttributeSizeLimit();
  for (auto attr : op->getAttrs()) {
    os << '\n' << attr.first << ": ";
    // Always emit splat attributes.
    if (attr.second.isa<SplatElementsAttr>()) {
      attr.second.print(os);
      continue;
    }

    // Elide "big" elements attributes.
    auto elements = attr.second.dyn_cast<ElementsAttr>();
    if (elements && elements.getNumElements() > largeAttrLimit) {
      os << std::string(elements.getType().getRank(), '[') << "..."
         << std::string(elements.getType().getRank(), ']') << " : "
         << elements.getType();
      continue;
    }

    auto array = attr.second.dyn_cast<ArrayAttr>();
    if (array && static_cast<int64_t>(array.size()) > largeAttrLimit) {
      os << "[...]";
      continue;
    }

    // Print all other attributes.
    attr.second.print(os);
  }
  return os.str();
}

} // end namespace llvm

namespace {
// PrintOpPass is simple pass to write graph per function.
// Note: this is a module pass only to avoid interleaving on the same ostream
// due to multi-threading over functions.
struct PrintOpPass : public PrintOpBase<PrintOpPass> {
  explicit PrintOpPass(raw_ostream &os = llvm::errs(), bool short_names = false,
                       const Twine &title = "")
      : os(os), title(title.str()), short_names(short_names) {}

  std::string getOpName(Operation &op) {
    auto symbolAttr =
        op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    if (symbolAttr)
      return std::string(symbolAttr.getValue());
    ++unnamedOpCtr;
    return (op.getName().getStringRef() + llvm::utostr(unnamedOpCtr)).str();
  }

  // Print all the ops in a module.
  void processModule(ModuleOp module) {
    for (Operation &op : module) {
      // Modules may actually be nested, recurse on nesting.
      if (auto nestedModule = dyn_cast<ModuleOp>(op)) {
        processModule(nestedModule);
        continue;
      }
      auto opName = getOpName(op);
      for (Region &region : op.getRegions()) {
        for (auto indexed_block : llvm::enumerate(region)) {
          // Suffix block number if there are more than 1 block.
          auto blockName = llvm::hasSingleElement(region)
                               ? ""
                               : ("__" + llvm::utostr(indexed_block.index()));
          llvm::WriteGraph(os, &indexed_block.value(), short_names,
                           Twine(title) + opName + blockName);
        }
      }
    }
  }

  void runOnOperation() override { processModule(getOperation()); }

private:
  raw_ostream &os;
  std::string title;
  int unnamedOpCtr = 0;
  bool short_names;
};
} // namespace

void mlir::viewGraph(Block &block, const Twine &name, bool shortNames,
                     const Twine &title, llvm::GraphProgram::Name program) {
  llvm::ViewGraph(&block, name, shortNames, title, program);
}

raw_ostream &mlir::writeGraph(raw_ostream &os, Block &block, bool shortNames,
                              const Twine &title) {
  return llvm::WriteGraph(os, &block, shortNames, title);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createPrintOpGraphPass(raw_ostream &os, bool shortNames,
                             const Twine &title) {
  return std::make_unique<PrintOpPass>(os, shortNames, title);
}
