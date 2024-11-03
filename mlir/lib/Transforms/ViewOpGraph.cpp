//===- ViewOpGraph.cpp - View/write op graphviz graphs --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/ViewOpGraph.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/GraphWriter.h"
#include <map>
#include <optional>
#include <utility>

namespace mlir {
#define GEN_PASS_DEF_VIEWOPGRAPH
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

static const StringRef kLineStyleControlFlow = "dashed";
static const StringRef kLineStyleDataFlow = "solid";
static const StringRef kShapeNode = "ellipse";
static const StringRef kShapeNone = "plain";

/// Return the size limits for eliding large attributes.
static int64_t getLargeAttributeSizeLimit() {
  // Use the default from the printer flags if possible.
  if (std::optional<int64_t> limit =
          OpPrintingFlags().getLargeElementsAttrLimit())
    return *limit;
  return 16;
}

/// Return all values printed onto a stream as a string.
static std::string strFromOs(function_ref<void(raw_ostream &)> func) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  func(os);
  return os.str();
}

/// Escape special characters such as '\n' and quotation marks.
static std::string escapeString(std::string str) {
  return strFromOs([&](raw_ostream &os) { os.write_escaped(str); });
}

/// Put quotation marks around a given string.
static std::string quoteString(const std::string &str) {
  return "\"" + str + "\"";
}

using AttributeMap = std::map<std::string, std::string>;

namespace {

/// This struct represents a node in the DOT language. Each node has an
/// identifier and an optional identifier for the cluster (subgraph) that
/// contains the node.
/// Note: In the DOT language, edges can be drawn only from nodes to nodes, but
/// not between clusters. However, edges can be clipped to the boundary of a
/// cluster with `lhead` and `ltail` attributes. Therefore, when creating a new
/// cluster, an invisible "anchor" node is created.
struct Node {
public:
  Node(int id = 0, std::optional<int> clusterId = std::nullopt)
      : id(id), clusterId(clusterId) {}

  int id;
  std::optional<int> clusterId;
};

/// This pass generates a Graphviz dataflow visualization of an MLIR operation.
/// Note: See https://www.graphviz.org/doc/info/lang.html for more information
/// about the Graphviz DOT language.
class PrintOpPass : public impl::ViewOpGraphBase<PrintOpPass> {
public:
  PrintOpPass(raw_ostream &os) : os(os) {}
  PrintOpPass(const PrintOpPass &o) : PrintOpPass(o.os.getOStream()) {}

  void runOnOperation() override {
    emitGraph([&]() {
      processOperation(getOperation());
      emitAllEdgeStmts();
    });
  }

  /// Create a CFG graph for a region. Used in `Region::viewGraph`.
  void emitRegionCFG(Region &region) {
    printControlFlowEdges = true;
    printDataFlowEdges = false;
    emitGraph([&]() { processRegion(region); });
  }

private:
  /// Emit all edges. This function should be called after all nodes have been
  /// emitted.
  void emitAllEdgeStmts() {
    for (const std::string &edge : edges)
      os << edge << ";\n";
    edges.clear();
  }

  /// Emit a cluster (subgraph). The specified builder generates the body of the
  /// cluster. Return the anchor node of the cluster.
  Node emitClusterStmt(function_ref<void()> builder, std::string label = "") {
    int clusterId = ++counter;
    os << "subgraph cluster_" << clusterId << " {\n";
    os.indent();
    // Emit invisible anchor node from/to which arrows can be drawn.
    Node anchorNode = emitNodeStmt(" ", kShapeNone);
    os << attrStmt("label", quoteString(escapeString(std::move(label))))
       << ";\n";
    builder();
    os.unindent();
    os << "}\n";
    return Node(anchorNode.id, clusterId);
  }

  /// Generate an attribute statement.
  std::string attrStmt(const Twine &key, const Twine &value) {
    return (key + " = " + value).str();
  }

  /// Emit an attribute list.
  void emitAttrList(raw_ostream &os, const AttributeMap &map) {
    os << "[";
    interleaveComma(map, os, [&](const auto &it) {
      os << this->attrStmt(it.first, it.second);
    });
    os << "]";
  }

  // Print an MLIR attribute to `os`. Large attributes are truncated.
  void emitMlirAttr(raw_ostream &os, Attribute attr) {
    // A value used to elide large container attribute.
    int64_t largeAttrLimit = getLargeAttributeSizeLimit();

    // Always emit splat attributes.
    if (attr.isa<SplatElementsAttr>()) {
      attr.print(os);
      return;
    }

    // Elide "big" elements attributes.
    auto elements = attr.dyn_cast<ElementsAttr>();
    if (elements && elements.getNumElements() > largeAttrLimit) {
      os << std::string(elements.getType().getRank(), '[') << "..."
         << std::string(elements.getType().getRank(), ']') << " : "
         << elements.getType();
      return;
    }

    auto array = attr.dyn_cast<ArrayAttr>();
    if (array && static_cast<int64_t>(array.size()) > largeAttrLimit) {
      os << "[...]";
      return;
    }

    // Print all other attributes.
    std::string buf;
    llvm::raw_string_ostream ss(buf);
    attr.print(ss);
    os << truncateString(ss.str());
  }

  /// Append an edge to the list of edges.
  /// Note: Edges are written to the output stream via `emitAllEdgeStmts`.
  void emitEdgeStmt(Node n1, Node n2, std::string label, StringRef style) {
    AttributeMap attrs;
    attrs["style"] = style.str();
    // Do not label edges that start/end at a cluster boundary. Such edges are
    // clipped at the boundary, but labels are not. This can lead to labels
    // floating around without any edge next to them.
    if (!n1.clusterId && !n2.clusterId)
      attrs["label"] = quoteString(escapeString(std::move(label)));
    // Use `ltail` and `lhead` to draw edges between clusters.
    if (n1.clusterId)
      attrs["ltail"] = "cluster_" + std::to_string(*n1.clusterId);
    if (n2.clusterId)
      attrs["lhead"] = "cluster_" + std::to_string(*n2.clusterId);

    edges.push_back(strFromOs([&](raw_ostream &os) {
      os << llvm::format("v%i -> v%i ", n1.id, n2.id);
      emitAttrList(os, attrs);
    }));
  }

  /// Emit a graph. The specified builder generates the body of the graph.
  void emitGraph(function_ref<void()> builder) {
    os << "digraph G {\n";
    os.indent();
    // Edges between clusters are allowed only in compound mode.
    os << attrStmt("compound", "true") << ";\n";
    builder();
    os.unindent();
    os << "}\n";
  }

  /// Emit a node statement.
  Node emitNodeStmt(std::string label, StringRef shape = kShapeNode) {
    int nodeId = ++counter;
    AttributeMap attrs;
    attrs["label"] = quoteString(escapeString(std::move(label)));
    attrs["shape"] = shape.str();
    os << llvm::format("v%i ", nodeId);
    emitAttrList(os, attrs);
    os << ";\n";
    return Node(nodeId);
  }

  /// Generate a label for an operation.
  std::string getLabel(Operation *op) {
    return strFromOs([&](raw_ostream &os) {
      // Print operation name and type.
      os << op->getName();
      if (printResultTypes) {
        os << " : (";
        std::string buf;
        llvm::raw_string_ostream ss(buf);
        interleaveComma(op->getResultTypes(), ss);
        os << truncateString(ss.str()) << ")";
        os << ")";
      }

      // Print attributes.
      if (printAttrs) {
        os << "\n";
        for (const NamedAttribute &attr : op->getAttrs()) {
          os << '\n' << attr.getName().getValue() << ": ";
          emitMlirAttr(os, attr.getValue());
        }
      }
    });
  }

  /// Generate a label for a block argument.
  std::string getLabel(BlockArgument arg) {
    return "arg" + std::to_string(arg.getArgNumber());
  }

  /// Process a block. Emit a cluster and one node per block argument and
  /// operation inside the cluster.
  void processBlock(Block &block) {
    emitClusterStmt([&]() {
      for (BlockArgument &blockArg : block.getArguments())
        valueToNode[blockArg] = emitNodeStmt(getLabel(blockArg));

      // Emit a node for each operation.
      std::optional<Node> prevNode;
      for (Operation &op : block) {
        Node nextNode = processOperation(&op);
        if (printControlFlowEdges && prevNode)
          emitEdgeStmt(*prevNode, nextNode, /*label=*/"",
                       kLineStyleControlFlow);
        prevNode = nextNode;
      }
    });
  }

  /// Process an operation. If the operation has regions, emit a cluster.
  /// Otherwise, emit a node.
  Node processOperation(Operation *op) {
    Node node;
    if (op->getNumRegions() > 0) {
      // Emit cluster for op with regions.
      node = emitClusterStmt(
          [&]() {
            for (Region &region : op->getRegions())
              processRegion(region);
          },
          getLabel(op));
    } else {
      node = emitNodeStmt(getLabel(op));
    }

    // Insert data flow edges originating from each operand.
    if (printDataFlowEdges) {
      unsigned numOperands = op->getNumOperands();
      for (unsigned i = 0; i < numOperands; i++)
        emitEdgeStmt(valueToNode[op->getOperand(i)], node,
                     /*label=*/numOperands == 1 ? "" : std::to_string(i),
                     kLineStyleDataFlow);
    }

    for (Value result : op->getResults())
      valueToNode[result] = node;

    return node;
  }

  /// Process a region.
  void processRegion(Region &region) {
    for (Block &block : region.getBlocks())
      processBlock(block);
  }

  /// Truncate long strings.
  std::string truncateString(std::string str) {
    if (str.length() <= maxLabelLen)
      return str;
    return str.substr(0, maxLabelLen) + "...";
  }

  /// Output stream to write DOT file to.
  raw_indented_ostream os;
  /// A list of edges. For simplicity, should be emitted after all nodes were
  /// emitted.
  std::vector<std::string> edges;
  /// Mapping of SSA values to Graphviz nodes/clusters.
  DenseMap<Value, Node> valueToNode;
  /// Counter for generating unique node/subgraph identifiers.
  int counter = 0;
};

} // namespace

std::unique_ptr<Pass> mlir::createPrintOpGraphPass(raw_ostream &os) {
  return std::make_unique<PrintOpPass>(os);
}

/// Generate a CFG for a region and show it in a window.
static void llvmViewGraph(Region &region, const Twine &name) {
  int fd;
  std::string filename = llvm::createGraphFilename(name.str(), fd);
  {
    llvm::raw_fd_ostream os(fd, /*shouldClose=*/true);
    if (fd == -1) {
      llvm::errs() << "error opening file '" << filename << "' for writing\n";
      return;
    }
    PrintOpPass pass(os);
    pass.emitRegionCFG(region);
  }
  llvm::DisplayGraph(filename, /*wait=*/false, llvm::GraphProgram::DOT);
}

void mlir::Region::viewGraph(const Twine &regionName) {
  llvmViewGraph(*this, regionName);
}

void mlir::Region::viewGraph() { viewGraph("region"); }
