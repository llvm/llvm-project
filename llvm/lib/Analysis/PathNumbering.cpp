//===- PathNumbering.cpp --------------------------------------*- C++ -*---===//
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
#define DEBUG_TYPE "ball-larus-numbering"

#include "llvm/Analysis/PathNumbering.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/InstrTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/TypeBuilder.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <queue>
#include <stack>
#include <string>
#include <utility>
#include <sstream>

using namespace llvm;

// Are we enabling early termination
static cl::opt<bool> ProcessEarlyTermination(
  "path-profile-early-termination", cl::Hidden,
  cl::desc("In path profiling, insert extra instrumentation to account for "
           "unexpected function termination."));

// Returns the basic block for the BallLarusNode
BasicBlock* BallLarusNode::getBlock() {
  return(_basicBlock);
}

// Returns the number of paths to the exit starting at the node.
unsigned BallLarusNode::getNumberPaths() {
  return(_numberPaths);
}

// Sets the number of paths to the exit starting at the node.
void BallLarusNode::setNumberPaths(unsigned numberPaths) {
  _numberPaths = numberPaths;
}

// Gets the NodeColor used in graph algorithms.
BallLarusNode::NodeColor BallLarusNode::getColor() {
  return(_color);
}

// Sets the NodeColor used in graph algorithms.
void BallLarusNode::setColor(BallLarusNode::NodeColor color) {
  _color = color;
}

// Returns an iterator over predecessor edges. Includes phony and
// backedges.
BLEdgeIterator BallLarusNode::predBegin() {
  return(_predEdges.begin());
}

// Returns the end sentinel for the predecessor iterator.
BLEdgeIterator BallLarusNode::predEnd() {
  return(_predEdges.end());
}

// Returns the number of predecessor edges.  Includes phony and
// backedges.
unsigned BallLarusNode::getNumberPredEdges() {
  return(_predEdges.size());
}

// Returns an iterator over successor edges. Includes phony and
// backedges.
BLEdgeIterator BallLarusNode::succBegin() {
  return(_succEdges.begin());
}

// Returns the end sentinel for the successor iterator.
BLEdgeIterator BallLarusNode::succEnd() {
  return(_succEdges.end());
}

// Returns the number of successor edges.  Includes phony and
// backedges.
unsigned BallLarusNode::getNumberSuccEdges() {
  return(_succEdges.size());
}

// Add an edge to the predecessor list.
void BallLarusNode::addPredEdge(BallLarusEdge* edge) {
  _predEdges.push_back(edge);
}

// Remove an edge from the predecessor list.
void BallLarusNode::removePredEdge(BallLarusEdge* edge) {
  removeEdge(_predEdges, edge);
}

// Add an edge to the successor list.
void BallLarusNode::addSuccEdge(BallLarusEdge* edge) {
  _succEdges.push_back(edge);
}

// Remove an edge from the successor list.
void BallLarusNode::removeSuccEdge(BallLarusEdge* edge) {
  removeEdge(_succEdges, edge);
}

// Returns the name of the BasicBlock being represented.  If BasicBlock
// is null then returns "<null>".  If BasicBlock has no name, then
// "<unnamed>" is returned.  Intended for use with debug output.
std::string BallLarusNode::getName() {
  std::stringstream name;

  if(getBlock() != NULL) {
    if(getBlock()->hasName()) {
      std::string tempName(getBlock()->getName());
      name << tempName.c_str() << " (" << _uid << ")";
    } else
      name << "<unnamed> (" << _uid << ")";
  } else
    name << "<null> (" << _uid << ")";

  return name.str();
}

// Removes an edge from an edgeVector.  Used by removePredEdge and
// removeSuccEdge.
void BallLarusNode::removeEdge(BLEdgeVector& v, BallLarusEdge* e) {
  // TODO: Avoid linear scan by using a set instead
  for(BLEdgeIterator i = v.begin(),
        end = v.end();
      i != end;
      ++i) {
    if((*i) == e) {
      v.erase(i);
      break;
    }
  }
}

// Returns the source node of this edge.
BallLarusNode* BallLarusEdge::getSource() const {
  return(_source);
}

// Returns the target node of this edge.
BallLarusNode* BallLarusEdge::getTarget() const {
  return(_target);
}

// Sets the type of the edge.
BallLarusEdge::EdgeType BallLarusEdge::getType() const {
  return _edgeType;
}

// Gets the type of the edge.
void BallLarusEdge::setType(EdgeType type) {
  _edgeType = type;
}

// Returns the weight of this edge.  Used to decode path numbers to sequences
// of basic blocks.
unsigned BallLarusEdge::getWeight() {
  return(_weight);
}

// Sets the weight of the edge.  Used during path numbering.
void BallLarusEdge::setWeight(unsigned weight) {
  _weight = weight;
}

// Gets the phony edge originating at the root.
BallLarusEdge* BallLarusEdge::getPhonyRoot() {
  return _phonyRoot;
}

// Sets the phony edge originating at the root.
void BallLarusEdge::setPhonyRoot(BallLarusEdge* phonyRoot) {
  _phonyRoot = phonyRoot;
}

// Gets the phony edge terminating at the exit.
BallLarusEdge* BallLarusEdge::getPhonyExit() {
  return _phonyExit;
}

// Sets the phony edge terminating at the exit.
void BallLarusEdge::setPhonyExit(BallLarusEdge* phonyExit) {
  _phonyExit = phonyExit;
}

// Gets the associated real edge if this is a phony edge.
BallLarusEdge* BallLarusEdge::getRealEdge() {
  return _realEdge;
}

// Sets the associated real edge if this is a phony edge.
void BallLarusEdge::setRealEdge(BallLarusEdge* realEdge) {
  _realEdge = realEdge;
}

// Returns the duplicate number of the edge.
unsigned BallLarusEdge::getDuplicateNumber() {
  return(_duplicateNumber);
}

// Initialization that requires virtual functions which are not fully
// functional in the constructor.
void BallLarusDag::init() {
  BLBlockNodeMap inDag;
  std::stack<BallLarusNode*> dfsStack;

  _root = addNode(&(_function.getEntryBlock()));
  _exit = addNode(NULL);

  // start search from root
  dfsStack.push(getRoot());

  // dfs to add each bb into the dag
  while(dfsStack.size())
    buildNode(inDag, dfsStack);

  // put in the final edge
  addEdge(getExit(),getRoot(),0);
}

// Frees all memory associated with the DAG.
BallLarusDag::~BallLarusDag() {
  for(BLEdgeIterator edge = _edges.begin(), end = _edges.end(); edge != end;
      ++edge)
    delete (*edge);

  for(BLNodeIterator node = _nodes.begin(), end = _nodes.end(); node != end;
      ++node)
    delete (*node);
}

// Calculate the path numbers by assigning edge increments as prescribed
// in Ball-Larus path profiling.
void BallLarusDag::calculatePathNumbers() {
  BallLarusNode* node;
  std::queue<BallLarusNode*> bfsQueue;
  bfsQueue.push(getExit());

  while(bfsQueue.size() > 0) {
    node = bfsQueue.front();

    DEBUG(dbgs() << "calculatePathNumbers on " << node->getName() << "\n");

    bfsQueue.pop();
    unsigned prevPathNumber = node->getNumberPaths();
    calculatePathNumbersFrom(node);

    // Check for DAG splitting
    if( node->getNumberPaths() > 100000000 && node != getRoot() ) {
      // Add new phony edge from the split-node to the DAG's exit
      BallLarusEdge* exitEdge = addEdge(node, getExit(), 0);
      exitEdge->setType(BallLarusEdge::SPLITEDGE_PHONY);

      // Counters to handle the possibility of a multi-graph
      BasicBlock* oldTarget = 0;
      unsigned duplicateNumber = 0;

      // Iterate through each successor edge, adding phony edges
      for( BLEdgeIterator succ = node->succBegin(), end = node->succEnd();
           succ != end; oldTarget = (*succ)->getTarget()->getBlock(), succ++ ) {

        if( (*succ)->getType() == BallLarusEdge::NORMAL ) {
          // is this edge a duplicate?
          if( oldTarget != (*succ)->getTarget()->getBlock() )
            duplicateNumber = 0;

          // create the new phony edge: root -> succ
          BallLarusEdge* rootEdge =
            addEdge(getRoot(), (*succ)->getTarget(), duplicateNumber++);
          rootEdge->setType(BallLarusEdge::SPLITEDGE_PHONY);
          rootEdge->setRealEdge(*succ);

          // split on this edge and reference it's exit/root phony edges
          (*succ)->setType(BallLarusEdge::SPLITEDGE);
          (*succ)->setPhonyRoot(rootEdge);
          (*succ)->setPhonyExit(exitEdge);
          (*succ)->setWeight(0);
        }
      }

      calculatePathNumbersFrom(node);
    }

    DEBUG(dbgs() << "prev, new number paths " << prevPathNumber << ", "
          << node->getNumberPaths() << ".\n");

    if(prevPathNumber == 0 && node->getNumberPaths() != 0) {
      DEBUG(dbgs() << "node ready : " << node->getName() << "\n");
      for(BLEdgeIterator pred = node->predBegin(), end = node->predEnd();
          pred != end; pred++) {
        if( (*pred)->getType() == BallLarusEdge::BACKEDGE ||
            (*pred)->getType() == BallLarusEdge::SPLITEDGE )
          continue;

        BallLarusNode* nextNode = (*pred)->getSource();
        // not yet visited?
        if(nextNode->getNumberPaths() == 0)
          bfsQueue.push(nextNode);
      }
    }
  }

  DEBUG(dbgs() << "\tNumber of paths: " << getRoot()->getNumberPaths() << "\n");
}

// Returns the number of paths for the Dag.
unsigned BallLarusDag::getNumberOfPaths() {
  return(getRoot()->getNumberPaths());
}

// Returns the root (i.e. entry) node for the DAG.
BallLarusNode* BallLarusDag::getRoot() {
  return _root;
}

// Returns the exit node for the DAG.
BallLarusNode* BallLarusDag::getExit() {
  return _exit;
}

// Returns the function for the DAG.
Function& BallLarusDag::getFunction() {
  return(_function);
}

// Clears the node colors.
void BallLarusDag::clearColors(BallLarusNode::NodeColor color) {
  for (BLNodeIterator nodeIt = _nodes.begin(); nodeIt != _nodes.end(); nodeIt++)
    (*nodeIt)->setColor(color);
}

// Processes one node and its imediate edges for building the DAG.
void BallLarusDag::buildNode(BLBlockNodeMap& inDag, BLNodeStack& dfsStack) {
  BallLarusNode* currentNode = dfsStack.top();
  BasicBlock* currentBlock = currentNode->getBlock();

  if(currentNode->getColor() != BallLarusNode::WHITE) {
    // we have already visited this node
    dfsStack.pop();
    currentNode->setColor(BallLarusNode::BLACK);
  } else {
    // are there any external procedure calls?
    if( ProcessEarlyTermination ) {
      for( BasicBlock::iterator bbCurrent = currentNode->getBlock()->begin(),
             bbEnd = currentNode->getBlock()->end(); bbCurrent != bbEnd;
           bbCurrent++ ) {
        Instruction& instr = *bbCurrent;
        if( instr.getOpcode() == Instruction::Call ) {
          BallLarusEdge* callEdge = addEdge(currentNode, getExit(), 0);
          callEdge->setType(BallLarusEdge::CALLEDGE_PHONY);
          break;
        }
      }
    }

    TerminatorInst* terminator = currentNode->getBlock()->getTerminator();
    if(isa<ReturnInst>(terminator) || isa<UnreachableInst>(terminator) ||
       isa<ResumeInst>(terminator))
      addEdge(currentNode, getExit(),0);

    currentNode->setColor(BallLarusNode::GRAY);
    inDag[currentBlock] = currentNode;

    BasicBlock* oldSuccessor = 0;
    unsigned duplicateNumber = 0;

    // iterate through this node's successors
    for(succ_iterator successor = succ_begin(currentBlock),
          succEnd = succ_end(currentBlock); successor != succEnd;
        oldSuccessor = *successor, ++successor ) {
      BasicBlock* succBB = *successor;

      // is this edge a duplicate?
      if (oldSuccessor == succBB)
        duplicateNumber++;
      else
        duplicateNumber = 0;

      buildEdge(inDag, dfsStack, currentNode, succBB, duplicateNumber);
    }
  }
}

// Process an edge in the CFG for DAG building.
void BallLarusDag::buildEdge(BLBlockNodeMap& inDag, std::stack<BallLarusNode*>&
                             dfsStack, BallLarusNode* currentNode,
                             BasicBlock* succBB, unsigned duplicateCount) {
  BallLarusNode* succNode = inDag[succBB];

  if(succNode && succNode->getColor() == BallLarusNode::BLACK) {
    // visited node and forward edge
    addEdge(currentNode, succNode, duplicateCount);
  } else if(succNode && succNode->getColor() == BallLarusNode::GRAY) {
    // visited node and back edge
    DEBUG(dbgs() << "Backedge detected.\n");
    addBackedge(currentNode, succNode, duplicateCount);
  } else {
    BallLarusNode* childNode;
    // not visited node and forward edge
    if(succNode) // an unvisited node that is child of a gray node
      childNode = succNode;
    else { // an unvisited node that is a child of a an unvisted node
      childNode = addNode(succBB);
      inDag[succBB] = childNode;
    }
    addEdge(currentNode, childNode, duplicateCount);
    dfsStack.push(childNode);
  }
}

// The weight on each edge is the increment required along any path that
// contains that edge.
void BallLarusDag::calculatePathNumbersFrom(BallLarusNode* node) {
  if(node == getExit())
    // The Exit node must be base case
    node->setNumberPaths(1);
  else {
    unsigned sumPaths = 0;
    BallLarusNode* succNode;

    for(BLEdgeIterator succ = node->succBegin(), end = node->succEnd();
        succ != end; succ++) {
      if( (*succ)->getType() == BallLarusEdge::BACKEDGE ||
          (*succ)->getType() == BallLarusEdge::SPLITEDGE )
        continue;

      (*succ)->setWeight(sumPaths);
      succNode = (*succ)->getTarget();

      if( !succNode->getNumberPaths() )
        return;
      sumPaths += succNode->getNumberPaths();
    }

    node->setNumberPaths(sumPaths);
  }
}

// Allows subclasses to determine which type of Node is created.
// Override this method to produce subclasses of BallLarusNode if
// necessary. The destructor of BallLarusDag will call free on each
// pointer created.
BallLarusNode* BallLarusDag::createNode(BasicBlock* BB) {
  return( new BallLarusNode(BB) );
}

// Allows subclasses to determine which type of Edge is created.
// Override this method to produce subclasses of BallLarusEdge if
// necessary. The destructor of BallLarusDag will call free on each
// pointer created.
BallLarusEdge* BallLarusDag::createEdge(BallLarusNode* source,
                                        BallLarusNode* target,
                                        unsigned duplicateCount) {
  return( new BallLarusEdge(source, target, duplicateCount) );
}

// Proxy to node's constructor.  Updates the DAG state.
BallLarusNode* BallLarusDag::addNode(BasicBlock* BB) {
  BallLarusNode* newNode = createNode(BB);
  _nodes.push_back(newNode);
  return( newNode );
}

// Proxy to edge's constructor. Updates the DAG state.
BallLarusEdge* BallLarusDag::addEdge(BallLarusNode* source,
                                     BallLarusNode* target,
                                     unsigned duplicateCount) {
  BallLarusEdge* newEdge = createEdge(source, target, duplicateCount);
  _edges.push_back(newEdge);
  source->addSuccEdge(newEdge);
  target->addPredEdge(newEdge);
  return(newEdge);
}

// Adds a backedge with its phony edges. Updates the DAG state.
void BallLarusDag::addBackedge(BallLarusNode* source, BallLarusNode* target,
                               unsigned duplicateCount) {
  BallLarusEdge* childEdge = addEdge(source, target, duplicateCount);
  childEdge->setType(BallLarusEdge::BACKEDGE);

  childEdge->setPhonyRoot(addEdge(getRoot(), target,0));
  childEdge->setPhonyExit(addEdge(source, getExit(),0));

  childEdge->getPhonyRoot()->setRealEdge(childEdge);
  childEdge->getPhonyRoot()->setType(BallLarusEdge::BACKEDGE_PHONY);

  childEdge->getPhonyExit()->setRealEdge(childEdge);
  childEdge->getPhonyExit()->setType(BallLarusEdge::BACKEDGE_PHONY);
  _backEdges.push_back(childEdge);
}
