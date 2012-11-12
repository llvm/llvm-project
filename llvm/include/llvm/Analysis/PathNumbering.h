//===- PathNumbering.h ----------------------------------------*- C++ -*---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Ball-Larus path numbers uniquely identify paths through a directed acyclic
// graph (DAG) [Ball96].  For a CFG backedges are removed and replaced by phony
// edges to obtain a DAG, and thus the unique path numbers [Ball96].
//
// The purpose of this analysis is to enumerate the edges in a CFG in order
// to obtain paths from path numbers in a convenient manner.  As described in
// [Ball96] edges can be enumerated such that given a path number by following
// the CFG and updating the path number, the path is obtained.
//
// [Ball96]
//  T. Ball and J. R. Larus. "Efficient Path Profiling."
//  International Symposium on Microarchitecture, pages 46-57, 1996.
//  http://portal.acm.org/citation.cfm?id=243857
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PATH_NUMBERING_H
#define LLVM_PATH_NUMBERING_H

#include "llvm/BasicBlock.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/CFG.h"
#include "llvm/Analysis/ProfileInfoTypes.h"
#include <map>
#include <stack>
#include <vector>

namespace llvm {
class BallLarusNode;
class BallLarusEdge;
class BallLarusDag;

// typedefs for storage/ interators of various DAG components
typedef std::vector<BallLarusNode*> BLNodeVector;
typedef std::vector<BallLarusNode*>::iterator BLNodeIterator;
typedef std::vector<BallLarusEdge*> BLEdgeVector;
typedef std::vector<BallLarusEdge*>::iterator BLEdgeIterator;
typedef std::map<BasicBlock*, BallLarusNode*> BLBlockNodeMap;
typedef std::stack<BallLarusNode*> BLNodeStack;

// Represents a basic block with information necessary for the BallLarus
// algorithms.
class BallLarusNode {
public:
  enum NodeColor { WHITE, GRAY, BLACK };

  // Constructor: Initializes a new Node for the given BasicBlock
  BallLarusNode(BasicBlock* BB) :
    _basicBlock(BB), _numberPaths(0), _color(WHITE) {
    static unsigned nextUID = 0;
    _uid = nextUID++;
  }

  // Returns the basic block for the BallLarusNode
  BasicBlock* getBlock();

  // Get/set the number of paths to the exit starting at the node.
  unsigned getNumberPaths();
  void setNumberPaths(unsigned numberPaths);

  // Get/set the NodeColor used in graph algorithms.
  NodeColor getColor();
  void setColor(NodeColor color);

  // Iterator information for predecessor edges. Includes phony and
  // backedges.
  BLEdgeIterator predBegin();
  BLEdgeIterator predEnd();
  unsigned getNumberPredEdges();

  // Iterator information for successor edges. Includes phony and
  // backedges.
  BLEdgeIterator succBegin();
  BLEdgeIterator succEnd();
  unsigned getNumberSuccEdges();

  // Add an edge to the predecessor list.
  void addPredEdge(BallLarusEdge* edge);

  // Remove an edge from the predecessor list.
  void removePredEdge(BallLarusEdge* edge);

  // Add an edge to the successor list.
  void addSuccEdge(BallLarusEdge* edge);

  // Remove an edge from the successor list.
  void removeSuccEdge(BallLarusEdge* edge);

  // Returns the name of the BasicBlock being represented.  If BasicBlock
  // is null then returns "<null>".  If BasicBlock has no name, then
  // "<unnamed>" is returned.  Intended for use with debug output.
  std::string getName();

private:
  // The corresponding underlying BB.
  BasicBlock* _basicBlock;

  // Holds the predecessor edges of this node.
  BLEdgeVector _predEdges;

  // Holds the successor edges of this node.
  BLEdgeVector _succEdges;

  // The number of paths from the node to the exit.
  unsigned _numberPaths;

  // 'Color' used by graph algorithms to mark the node.
  NodeColor _color;

  // Unique ID to ensure naming difference with dotgraphs
  unsigned _uid;

  // Removes an edge from an edgeVector.  Used by removePredEdge and
  // removeSuccEdge.
  void removeEdge(BLEdgeVector& v, BallLarusEdge* e);
};

// Represents an edge in the Dag.  For an edge, v -> w, v is the source, and
// w is the target.
class BallLarusEdge {
public:
  enum EdgeType { NORMAL, BACKEDGE, SPLITEDGE,
    BACKEDGE_PHONY, SPLITEDGE_PHONY, CALLEDGE_PHONY };

  // Constructor: Initializes an BallLarusEdge with a source and target.
  BallLarusEdge(BallLarusNode* source, BallLarusNode* target,
                                unsigned duplicateNumber)
    : _source(source), _target(target), _weight(0), _edgeType(NORMAL),
      _realEdge(NULL), _duplicateNumber(duplicateNumber) {}

  // Returns the source/ target node of this edge.
  BallLarusNode* getSource() const;
  BallLarusNode* getTarget() const;

  // Sets the type of the edge.
  EdgeType getType() const;

  // Gets the type of the edge.
  void setType(EdgeType type);

  // Returns the weight of this edge.  Used to decode path numbers to
  // sequences of basic blocks.
  unsigned getWeight();

  // Sets the weight of the edge.  Used during path numbering.
  void setWeight(unsigned weight);

  // Gets/sets the phony edge originating at the root.
  BallLarusEdge* getPhonyRoot();
  void setPhonyRoot(BallLarusEdge* phonyRoot);

  // Gets/sets the phony edge terminating at the exit.
  BallLarusEdge* getPhonyExit();
  void setPhonyExit(BallLarusEdge* phonyExit);

  // Gets/sets the associated real edge if this is a phony edge.
  BallLarusEdge* getRealEdge();
  void setRealEdge(BallLarusEdge* realEdge);

  // Returns the duplicate number of the edge.
  unsigned getDuplicateNumber();

protected:
  // Source node for this edge.
  BallLarusNode* _source;

  // Target node for this edge.
  BallLarusNode* _target;

private:
  // Edge weight cooresponding to path number increments before removing
  // increments along a spanning tree. The sum over the edge weights gives
  // the path number.
  unsigned _weight;

  // Type to represent for what this edge is intended
  EdgeType _edgeType;

  // For backedges and split-edges, the phony edge which is linked to the
  // root node of the DAG. This contains a path number initialization.
  BallLarusEdge* _phonyRoot;

  // For backedges and split-edges, the phony edge which is linked to the
  // exit node of the DAG. This contains a path counter increment, and
  // potentially a path number increment.
  BallLarusEdge* _phonyExit;

  // If this is a phony edge, _realEdge is a link to the back or split
  // edge. Otherwise, this is null.
  BallLarusEdge* _realEdge;

  // An ID to differentiate between those edges which have the same source
  // and destination blocks.
  unsigned _duplicateNumber;
};

// Represents the Ball Larus DAG for a given Function.  Can calculate
// various properties required for instrumentation or analysis.  E.g. the
// edge weights that determine the path number.
class BallLarusDag {
public:
  // Initializes a BallLarusDag from the CFG of a given function.  Must
  // call init() after creation, since some initialization requires
  // virtual functions.
  BallLarusDag(Function &F)
    : _root(NULL), _exit(NULL), _function(F) {}

  // Initialization that requires virtual functions which are not fully
  // functional in the constructor.
  void init();

  // Frees all memory associated with the DAG.
  virtual ~BallLarusDag();

  // Calculate the path numbers by assigning edge increments as prescribed
  // in Ball-Larus path profiling.
  void calculatePathNumbers();

  // Returns the number of paths for the DAG.
  unsigned getNumberOfPaths();

  // Returns the root (i.e. entry) node for the DAG.
  BallLarusNode* getRoot();

  // Returns the exit node for the DAG.
  BallLarusNode* getExit();

  // Returns the function for the DAG.
  Function& getFunction();

  // Clears the node colors.
  void clearColors(BallLarusNode::NodeColor color);

protected:
  // All nodes in the DAG.
  BLNodeVector _nodes;

  // All edges in the DAG.
  BLEdgeVector _edges;

  // All backedges in the DAG.
  BLEdgeVector _backEdges;

  // Allows subclasses to determine which type of Node is created.
  // Override this method to produce subclasses of BallLarusNode if
  // necessary. The destructor of BallLarusDag will call free on each pointer
  // created.
  virtual BallLarusNode* createNode(BasicBlock* BB);

  // Allows subclasses to determine which type of Edge is created.
  // Override this method to produce subclasses of BallLarusEdge if
  // necessary.  Parameters source and target will have been created by
  // createNode and can be cast to the subclass of BallLarusNode*
  // returned by createNode. The destructor of BallLarusDag will call free
  // on each pointer created.
  virtual BallLarusEdge* createEdge(BallLarusNode* source, BallLarusNode*
                                    target, unsigned duplicateNumber);

  // Proxy to node's constructor.  Updates the DAG state.
  BallLarusNode* addNode(BasicBlock* BB);

  // Proxy to edge's constructor.  Updates the DAG state.
  BallLarusEdge* addEdge(BallLarusNode* source, BallLarusNode* target,
                         unsigned duplicateNumber);

private:
  // The root (i.e. entry) node for this DAG.
  BallLarusNode* _root;

  // The exit node for this DAG.
  BallLarusNode* _exit;

  // The function represented by this DAG.
  Function& _function;

  // Processes one node and its imediate edges for building the DAG.
  void buildNode(BLBlockNodeMap& inDag, std::stack<BallLarusNode*>& dfsStack);

  // Process an edge in the CFG for DAG building.
  void buildEdge(BLBlockNodeMap& inDag, std::stack<BallLarusNode*>& dfsStack,
                 BallLarusNode* currentNode, BasicBlock* succBB,
                 unsigned duplicateNumber);

  // The weight on each edge is the increment required along any path that
  // contains that edge.
  void calculatePathNumbersFrom(BallLarusNode* node);

  // Adds a backedge with its phony edges.  Updates the DAG state.
  void addBackedge(BallLarusNode* source, BallLarusNode* target,
                   unsigned duplicateCount);
};
} // end namespace llvm

#endif
