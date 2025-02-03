//===- ViewOpGraph.cpp - View/write op graphviz graphs --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/ViewOpGraph.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
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
static const StringRef kShapeNode = "Mrecord";
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
  return buf;
}

/// Put quotation marks around a given string.
static std::string quoteString(const std::string &str) {
  return "\"" + str + "\"";
}

/// For Graphviz record nodes:
/// " Braces, vertical bars and angle brackets must be escaped with a backslash
/// character if you wish them to appear as a literal character "
std::string escapeLabelString(const std::string &str) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  llvm::DenseSet<char> shouldEscape = {'{', '|', '<', '}', '>', '\n', '"'};
  for (char c : str) {
    if (shouldEscape.contains(c)) {
      os << '\\';
    }
    os << c;
  }
  return buf;
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

struct DataFlowEdge {
  Value value;
  Node node;
  std::string port;
};

/// This pass generates a Graphviz dataflow visualization of an MLIR operation.
/// Note: See https://www.graphviz.org/doc/info/lang.html for more information
/// about the Graphviz DOT language.
class PrintOpPass : public impl::ViewOpGraphBase<PrintOpPass> {
public:
  PrintOpPass(raw_ostream &os) : os(os) {}
  PrintOpPass(const PrintOpPass &o) : PrintOpPass(o.os.getOStream()) {}

  void runOnOperation() override {
    initColorMapping(*getOperation());
    emitGraph([&]() {
      processOperation(getOperation());
      emitAllEdgeStmts();
    });
    markAllAnalysesPreserved();
  }

  /// Create a CFG graph for a region. Used in `Region::viewGraph`.
  void emitRegionCFG(Region &region) {
    printControlFlowEdges = true;
    printDataFlowEdges = false;
    initColorMapping(region);
    emitGraph([&]() { processRegion(region); });
  }

private:
  /// Generate a color mapping that will color every operation with the same
  /// name the same way. It'll interpolate the hue in the HSV color-space,
  /// attempting to keep the contrast suitable for black text.
  template <typename T>
  void initColorMapping(T &irEntity) {
    backgroundColors.clear();
    SmallVector<Operation *> ops;
    irEntity.walk([&](Operation *op) {
      auto &entry = backgroundColors[op->getName()];
      if (entry.first == 0)
        ops.push_back(op);
      ++entry.first;
    });
    for (auto indexedOps : llvm::enumerate(ops)) {
      double hue = ((double)indexedOps.index()) / ops.size();
      backgroundColors[indexedOps.value()->getName()].second =
          std::to_string(hue) + " 1.0 1.0";
    }
  }

  /// Emit all edges. This function should be called after all nodes have been
  /// emitted.
  void emitAllEdgeStmts() {
    if (printDataFlowEdges) {
      for (const auto &e : dataFlowEdges) {
        emitEdgeStmt(valueToNode[e.value], e.node, e.port, kLineStyleDataFlow);
      }
    }

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
    os << attrStmt("label", quoteString(label)) << ";\n";
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
    if (isa<SplatElementsAttr>(attr)) {
      attr.print(os);
      return;
    }

    // Elide "big" elements attributes.
    auto elements = dyn_cast<ElementsAttr>(attr);
    if (elements && elements.getNumElements() > largeAttrLimit) {
      os << std::string(elements.getShapedType().getRank(), '[') << "..."
         << std::string(elements.getShapedType().getRank(), ']') << " : "
         << elements.getType();
      return;
    }

    auto array = dyn_cast<ArrayAttr>(attr);
    if (array && static_cast<int64_t>(array.size()) > largeAttrLimit) {
      os << "[...]";
      return;
    }

    // Print all other attributes.
    std::string buf;
    llvm::raw_string_ostream ss(buf);
    attr.print(ss);
    os << escapeLabelString(truncateString(buf));
  }

  /// Append an edge to the list of edges.
  /// Note: Edges are written to the output stream via `emitAllEdgeStmts`.
  void emitEdgeStmt(Node n1, Node n2, std::string port, StringRef style) {
    AttributeMap attrs;
    attrs["style"] = style.str();
    // Use `ltail` and `lhead` to draw edges between clusters.
    if (n1.clusterId)
      attrs["ltail"] = "cluster_" + std::to_string(*n1.clusterId);
    if (n2.clusterId)
      attrs["lhead"] = "cluster_" + std::to_string(*n2.clusterId);

    edges.push_back(strFromOs([&](raw_ostream &os) {
      os << "v" << n1.id;
      if (!port.empty())
        // Attach edge to south compass point of the result
        os << ":" << port << ":s";
      os << " -> ";
      os << "v" << n2.id;
      if (!port.empty())
        // Attach edge to north compass point of the operand
        os << ":" << port << ":n";
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
  Node emitNodeStmt(std::string label, StringRef shape = kShapeNode,
                    StringRef background = "") {
    int nodeId = ++counter;
    AttributeMap attrs;
    attrs["label"] = quoteString(label);
    attrs["shape"] = shape.str();
    if (!background.empty()) {
      attrs["style"] = "filled";
      attrs["fillcolor"] = ("\"" + background + "\"").str();
    }
    os << llvm::format("v%i ", nodeId);
    emitAttrList(os, attrs);
    os << ";\n";
    return Node(nodeId);
  }

  std::string getValuePortName(Value operand) {
    // Print value as an operand and omit the leading '%' character.
    auto str = strFromOs([&](raw_ostream &os) {
      operand.printAsOperand(os, OpPrintingFlags());
    });
    // Replace % and # with _
    std::replace(str.begin(), str.end(), '%', '_');
    std::replace(str.begin(), str.end(), '#', '_');
    return str;
  }

  std::string getClusterLabel(Operation *op) {
    return strFromOs([&](raw_ostream &os) {
      // Print operation name and type.
      os << op->getName();
      if (printResultTypes) {
        os << " : (";
        std::string buf;
        llvm::raw_string_ostream ss(buf);
        interleaveComma(op->getResultTypes(), ss);
        os << truncateString(buf) << ")";
      }

      // Print attributes.
      if (printAttrs) {
        os << "\\l";
        for (const NamedAttribute &attr : op->getAttrs()) {
          os << attr.getName().getValue() << ": ";
          emitMlirAttr(os, attr.getValue());
          os << "\\l";
        }
      }
    });
  }

  /// Generate a label for an operation.
  std::string getRecordLabel(Operation *op) {
    return strFromOs([&](raw_ostream &os) {
      os << "{";

      // Print operation inputs.
      if (op->getNumOperands() > 0) {
        os << "{";
        auto operandToPort = [&](Value operand) {
          os << "<" << getValuePortName(operand) << "> ";
          operand.printAsOperand(os, OpPrintingFlags());
        };
        interleave(op->getOperands(), os, operandToPort, "|");
        os << "}|";
      }
      // Print operation name and type.
      os << op->getName() << "\\l";

      // Print attributes.
      if (printAttrs && !op->getAttrs().empty()) {
        // Extra line break to separate attributes from the operation name.
        os << "\\l";
        for (const NamedAttribute &attr : op->getAttrs()) {
          os << attr.getName().getValue() << ": ";
          emitMlirAttr(os, attr.getValue());
          os << "\\l";
        }
      }

      if (op->getNumResults() > 0) {
        os << "|{";
        auto resultToPort = [&](Value result) {
          os << "<" << getValuePortName(result) << "> ";
          result.printAsOperand(os, OpPrintingFlags());
          if (printResultTypes)
            os << " "
               << truncateString(escapeLabelString(strFromOs(
                      [&](raw_ostream &os) { os << result.getType(); })));
        };
        interleave(op->getResults(), os, resultToPort, "|");
        os << "}";
      }

      os << "}";
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
      for (BlockArgument &blockArg : block.getArguments()) {
        valueToNode[blockArg] = emitNodeStmt(getLabel(blockArg));
      }
      // Emit a node for each operation.
      std::optional<Node> prevNode;
      for (Operation &op : block) {
        Node nextNode = processOperation(&op);
        if (printControlFlowEdges && prevNode)
          emitEdgeStmt(*prevNode, nextNode, /*port=*/"", kLineStyleControlFlow);
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
          getClusterLabel(op));
    } else {
      node = emitNodeStmt(getRecordLabel(op), kShapeNode,
                          backgroundColors[op->getName()].second);
    }

    // Insert data flow edges originating from each operand.
    if (printDataFlowEdges) {
      unsigned numOperands = op->getNumOperands();
      for (unsigned i = 0; i < numOperands; i++) {
        auto operand = op->getOperand(i);
        dataFlowEdges.push_back({operand, node, getValuePortName(operand)});
      }
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
  /// Output for data flow edges is delayed until the end to handle cycles
  std::vector<DataFlowEdge> dataFlowEdges;
  /// Counter for generating unique node/subgraph identifiers.
  int counter = 0;

  DenseMap<OperationName, std::pair<int, std::string>> backgroundColors;
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
