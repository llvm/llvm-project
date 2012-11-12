//===-- Briggs.h --- Briggs Heuristic for PBQP ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements the Briggs test for "allocability" of nodes in a
// PBQP graph representing a register allocation problem. Nodes which can be
// proven allocable (by a safe and relatively accurate test) are removed from
// the PBQP graph first. If no provably allocable node is present in the graph
// then the node with the minimal spill-cost to degree ratio is removed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PBQP_HEURISTICS_BRIGGS_H
#define LLVM_CODEGEN_PBQP_HEURISTICS_BRIGGS_H

#include "../HeuristicSolver.h"
#include "../HeuristicBase.h"

#include <limits>

namespace PBQP {
  namespace Heuristics {

    /// \brief PBQP Heuristic which applies an allocability test based on
    ///        Briggs.
    /// 
    /// This heuristic assumes that the elements of cost vectors in the PBQP
    /// problem represent storage options, with the first being the spill
    /// option and subsequent elements representing legal registers for the
    /// corresponding node. Edge cost matrices are likewise assumed to represent
    /// register constraints.
    /// If one or more nodes can be proven allocable by this heuristic (by
    /// inspection of their constraint matrices) then the allocable node of
    /// highest degree is selected for the next reduction and pushed to the
    /// solver stack. If no nodes can be proven allocable then the node with
    /// the lowest estimated spill cost is selected and push to the solver stack
    /// instead.
    /// 
    /// This implementation is built on top of HeuristicBase.       
    class Briggs : public HeuristicBase<Briggs> {
    private:

      class LinkDegreeComparator {
      public:
        LinkDegreeComparator(HeuristicSolverImpl<Briggs> &s) : s(&s) {}
        bool operator()(Graph::NodeItr n1Itr, Graph::NodeItr n2Itr) const {
          if (s->getSolverDegree(n1Itr) > s->getSolverDegree(n2Itr))
            return true;
          return false;
        }
      private:
        HeuristicSolverImpl<Briggs> *s;
      };

      class SpillCostComparator {
      public:
        SpillCostComparator(HeuristicSolverImpl<Briggs> &s)
          : s(&s), g(&s.getGraph()) {}
        bool operator()(Graph::NodeItr n1Itr, Graph::NodeItr n2Itr) const {
          const PBQP::Vector &cv1 = g->getNodeCosts(n1Itr);
          const PBQP::Vector &cv2 = g->getNodeCosts(n2Itr);

          PBQPNum cost1 = cv1[0] / s->getSolverDegree(n1Itr);
          PBQPNum cost2 = cv2[0] / s->getSolverDegree(n2Itr);

          if (cost1 < cost2)
            return true;
          return false;
        }

      private:
        HeuristicSolverImpl<Briggs> *s;
        Graph *g;
      };

      typedef std::list<Graph::NodeItr> RNAllocableList;
      typedef RNAllocableList::iterator RNAllocableListItr;

      typedef std::list<Graph::NodeItr> RNUnallocableList;  
      typedef RNUnallocableList::iterator RNUnallocableListItr;

    public:

      struct NodeData {
        typedef std::vector<unsigned> UnsafeDegreesArray;
        bool isHeuristic, isAllocable, isInitialized;
        unsigned numDenied, numSafe;
        UnsafeDegreesArray unsafeDegrees;
        RNAllocableListItr rnaItr;
        RNUnallocableListItr rnuItr;

        NodeData()
          : isHeuristic(false), isAllocable(false), isInitialized(false),
            numDenied(0), numSafe(0) { }
      };

      struct EdgeData {
        typedef std::vector<unsigned> UnsafeArray;
        unsigned worst, reverseWorst;
        UnsafeArray unsafe, reverseUnsafe;
        bool isUpToDate;

        EdgeData() : worst(0), reverseWorst(0), isUpToDate(false) {}
      };

      /// \brief Construct an instance of the Briggs heuristic.
      /// @param solver A reference to the solver which is using this heuristic.
      Briggs(HeuristicSolverImpl<Briggs> &solver) :
        HeuristicBase<Briggs>(solver) {}

      /// \brief Determine whether a node should be reduced using optimal
      ///        reduction.
      /// @param nItr Node iterator to be considered.
      /// @return True if the given node should be optimally reduced, false
      ///         otherwise.
      ///
      /// Selects nodes of degree 0, 1 or 2 for optimal reduction, with one
      /// exception. Nodes whose spill cost (element 0 of their cost vector) is
      /// infinite are checked for allocability first. Allocable nodes may be
      /// optimally reduced, but nodes whose allocability cannot be proven are
      /// selected for heuristic reduction instead.
      bool shouldOptimallyReduce(Graph::NodeItr nItr) {
        if (getSolver().getSolverDegree(nItr) < 3) {
          return true;
        }
        // else
        return false;
      }

      /// \brief Add a node to the heuristic reduce list.
      /// @param nItr Node iterator to add to the heuristic reduce list.
      void addToHeuristicReduceList(Graph::NodeItr nItr) {
        NodeData &nd = getHeuristicNodeData(nItr);
        initializeNode(nItr);
        nd.isHeuristic = true;
        if (nd.isAllocable) {
          nd.rnaItr = rnAllocableList.insert(rnAllocableList.end(), nItr);
        } else {
          nd.rnuItr = rnUnallocableList.insert(rnUnallocableList.end(), nItr);
        }
      }

      /// \brief Heuristically reduce one of the nodes in the heuristic
      ///        reduce list.
      /// @return True if a reduction takes place, false if the heuristic reduce
      ///         list is empty.
      ///
      /// If the list of allocable nodes is non-empty a node is selected
      /// from it and pushed to the stack. Otherwise if the non-allocable list
      /// is non-empty a node is selected from it and pushed to the stack.
      /// If both lists are empty the method simply returns false with no action
      /// taken.
      bool heuristicReduce() {
        if (!rnAllocableList.empty()) {
          RNAllocableListItr rnaItr =
            min_element(rnAllocableList.begin(), rnAllocableList.end(),
                        LinkDegreeComparator(getSolver()));
          Graph::NodeItr nItr = *rnaItr;
          rnAllocableList.erase(rnaItr);
          handleRemoveNode(nItr);
          getSolver().pushToStack(nItr);
          return true;
        } else if (!rnUnallocableList.empty()) {
          RNUnallocableListItr rnuItr =
            min_element(rnUnallocableList.begin(), rnUnallocableList.end(),
                        SpillCostComparator(getSolver()));
          Graph::NodeItr nItr = *rnuItr;
          rnUnallocableList.erase(rnuItr);
          handleRemoveNode(nItr);
          getSolver().pushToStack(nItr);
          return true;
        }
        // else
        return false;
      }

      /// \brief Prepare a change in the costs on the given edge.
      /// @param eItr Edge iterator.    
      void preUpdateEdgeCosts(Graph::EdgeItr eItr) {
        Graph &g = getGraph();
        Graph::NodeItr n1Itr = g.getEdgeNode1(eItr),
                       n2Itr = g.getEdgeNode2(eItr);
        NodeData &n1 = getHeuristicNodeData(n1Itr),
                 &n2 = getHeuristicNodeData(n2Itr);

        if (n1.isHeuristic)
          subtractEdgeContributions(eItr, getGraph().getEdgeNode1(eItr));
        if (n2.isHeuristic)
          subtractEdgeContributions(eItr, getGraph().getEdgeNode2(eItr));

        EdgeData &ed = getHeuristicEdgeData(eItr);
        ed.isUpToDate = false;
      }

      /// \brief Handle the change in the costs on the given edge.
      /// @param eItr Edge iterator.
      void postUpdateEdgeCosts(Graph::EdgeItr eItr) {
        // This is effectively the same as adding a new edge now, since
        // we've factored out the costs of the old one.
        handleAddEdge(eItr);
      }

      /// \brief Handle the addition of a new edge into the PBQP graph.
      /// @param eItr Edge iterator for the added edge.
      ///
      /// Updates allocability of any nodes connected by this edge which are
      /// being managed by the heuristic. If allocability changes they are
      /// moved to the appropriate list.
      void handleAddEdge(Graph::EdgeItr eItr) {
        Graph &g = getGraph();
        Graph::NodeItr n1Itr = g.getEdgeNode1(eItr),
                       n2Itr = g.getEdgeNode2(eItr);
        NodeData &n1 = getHeuristicNodeData(n1Itr),
                 &n2 = getHeuristicNodeData(n2Itr);

        // If neither node is managed by the heuristic there's nothing to be
        // done.
        if (!n1.isHeuristic && !n2.isHeuristic)
          return;

        // Ok - we need to update at least one node.
        computeEdgeContributions(eItr);

        // Update node 1 if it's managed by the heuristic.
        if (n1.isHeuristic) {
          bool n1WasAllocable = n1.isAllocable;
          addEdgeContributions(eItr, n1Itr);
          updateAllocability(n1Itr);
          if (n1WasAllocable && !n1.isAllocable) {
            rnAllocableList.erase(n1.rnaItr);
            n1.rnuItr =
              rnUnallocableList.insert(rnUnallocableList.end(), n1Itr);
          }
        }

        // Likewise for node 2.
        if (n2.isHeuristic) {
          bool n2WasAllocable = n2.isAllocable;
          addEdgeContributions(eItr, n2Itr);
          updateAllocability(n2Itr);
          if (n2WasAllocable && !n2.isAllocable) {
            rnAllocableList.erase(n2.rnaItr);
            n2.rnuItr =
              rnUnallocableList.insert(rnUnallocableList.end(), n2Itr);
          }
        }
      }

      /// \brief Handle disconnection of an edge from a node.
      /// @param eItr Edge iterator for edge being disconnected.
      /// @param nItr Node iterator for the node being disconnected from.
      ///
      /// Updates allocability of the given node and, if appropriate, moves the
      /// node to a new list.
      void handleRemoveEdge(Graph::EdgeItr eItr, Graph::NodeItr nItr) {
        NodeData &nd = getHeuristicNodeData(nItr);

        // If the node is not managed by the heuristic there's nothing to be
        // done.
        if (!nd.isHeuristic)
          return;

        EdgeData &ed = getHeuristicEdgeData(eItr);
        (void)ed;
        assert(ed.isUpToDate && "Edge data is not up to date.");

        // Update node.
        bool ndWasAllocable = nd.isAllocable;
        subtractEdgeContributions(eItr, nItr);
        updateAllocability(nItr);

        // If the node has gone optimal...
        if (shouldOptimallyReduce(nItr)) {
          nd.isHeuristic = false;
          addToOptimalReduceList(nItr);
          if (ndWasAllocable) {
            rnAllocableList.erase(nd.rnaItr);
          } else {
            rnUnallocableList.erase(nd.rnuItr);
          }
        } else {
          // Node didn't go optimal, but we might have to move it
          // from "unallocable" to "allocable".
          if (!ndWasAllocable && nd.isAllocable) {
            rnUnallocableList.erase(nd.rnuItr);
            nd.rnaItr = rnAllocableList.insert(rnAllocableList.end(), nItr);
          }
        }
      }

    private:

      NodeData& getHeuristicNodeData(Graph::NodeItr nItr) {
        return getSolver().getHeuristicNodeData(nItr);
      }

      EdgeData& getHeuristicEdgeData(Graph::EdgeItr eItr) {
        return getSolver().getHeuristicEdgeData(eItr);
      }

      // Work out what this edge will contribute to the allocability of the
      // nodes connected to it.
      void computeEdgeContributions(Graph::EdgeItr eItr) {
        EdgeData &ed = getHeuristicEdgeData(eItr);

        if (ed.isUpToDate)
          return; // Edge data is already up to date.

        Matrix &eCosts = getGraph().getEdgeCosts(eItr);

        unsigned numRegs = eCosts.getRows() - 1,
                 numReverseRegs = eCosts.getCols() - 1;

        std::vector<unsigned> rowInfCounts(numRegs, 0),
                              colInfCounts(numReverseRegs, 0);        

        ed.worst = 0;
        ed.reverseWorst = 0;
        ed.unsafe.clear();
        ed.unsafe.resize(numRegs, 0);
        ed.reverseUnsafe.clear();
        ed.reverseUnsafe.resize(numReverseRegs, 0);

        for (unsigned i = 0; i < numRegs; ++i) {
          for (unsigned j = 0; j < numReverseRegs; ++j) {
            if (eCosts[i + 1][j + 1] ==
                  std::numeric_limits<PBQPNum>::infinity()) {
              ed.unsafe[i] = 1;
              ed.reverseUnsafe[j] = 1;
              ++rowInfCounts[i];
              ++colInfCounts[j];

              if (colInfCounts[j] > ed.worst) {
                ed.worst = colInfCounts[j];
              }

              if (rowInfCounts[i] > ed.reverseWorst) {
                ed.reverseWorst = rowInfCounts[i];
              }
            }
          }
        }

        ed.isUpToDate = true;
      }

      // Add the contributions of the given edge to the given node's 
      // numDenied and safe members. No action is taken other than to update
      // these member values. Once updated these numbers can be used by clients
      // to update the node's allocability.
      void addEdgeContributions(Graph::EdgeItr eItr, Graph::NodeItr nItr) {
        EdgeData &ed = getHeuristicEdgeData(eItr);

        assert(ed.isUpToDate && "Using out-of-date edge numbers.");

        NodeData &nd = getHeuristicNodeData(nItr);
        unsigned numRegs = getGraph().getNodeCosts(nItr).getLength() - 1;
        
        bool nIsNode1 = nItr == getGraph().getEdgeNode1(eItr);
        EdgeData::UnsafeArray &unsafe =
          nIsNode1 ? ed.unsafe : ed.reverseUnsafe;
        nd.numDenied += nIsNode1 ? ed.worst : ed.reverseWorst;

        for (unsigned r = 0; r < numRegs; ++r) {
          if (unsafe[r]) {
            if (nd.unsafeDegrees[r]==0) {
              --nd.numSafe;
            }
            ++nd.unsafeDegrees[r];
          }
        }
      }

      // Subtract the contributions of the given edge to the given node's 
      // numDenied and safe members. No action is taken other than to update
      // these member values. Once updated these numbers can be used by clients
      // to update the node's allocability.
      void subtractEdgeContributions(Graph::EdgeItr eItr, Graph::NodeItr nItr) {
        EdgeData &ed = getHeuristicEdgeData(eItr);

        assert(ed.isUpToDate && "Using out-of-date edge numbers.");

        NodeData &nd = getHeuristicNodeData(nItr);
        unsigned numRegs = getGraph().getNodeCosts(nItr).getLength() - 1;
        
        bool nIsNode1 = nItr == getGraph().getEdgeNode1(eItr);
        EdgeData::UnsafeArray &unsafe =
          nIsNode1 ? ed.unsafe : ed.reverseUnsafe;
        nd.numDenied -= nIsNode1 ? ed.worst : ed.reverseWorst;

        for (unsigned r = 0; r < numRegs; ++r) {
          if (unsafe[r]) { 
            if (nd.unsafeDegrees[r] == 1) {
              ++nd.numSafe;
            }
            --nd.unsafeDegrees[r];
          }
        }
      }

      void updateAllocability(Graph::NodeItr nItr) {
        NodeData &nd = getHeuristicNodeData(nItr);
        unsigned numRegs = getGraph().getNodeCosts(nItr).getLength() - 1;
        nd.isAllocable = nd.numDenied < numRegs || nd.numSafe > 0;
      }

      void initializeNode(Graph::NodeItr nItr) {
        NodeData &nd = getHeuristicNodeData(nItr);

        if (nd.isInitialized)
          return; // Node data is already up to date.

        unsigned numRegs = getGraph().getNodeCosts(nItr).getLength() - 1;

        nd.numDenied = 0;
        const Vector& nCosts = getGraph().getNodeCosts(nItr);
        for (unsigned i = 1; i < nCosts.getLength(); ++i) {
          if (nCosts[i] == std::numeric_limits<PBQPNum>::infinity())
            ++nd.numDenied;
        }

        nd.numSafe = numRegs;
        nd.unsafeDegrees.resize(numRegs, 0);

        typedef HeuristicSolverImpl<Briggs>::SolverEdgeItr SolverEdgeItr;

        for (SolverEdgeItr aeItr = getSolver().solverEdgesBegin(nItr),
                           aeEnd = getSolver().solverEdgesEnd(nItr);
             aeItr != aeEnd; ++aeItr) {
          
          Graph::EdgeItr eItr = *aeItr;
          computeEdgeContributions(eItr);
          addEdgeContributions(eItr, nItr);
        }

        updateAllocability(nItr);
        nd.isInitialized = true;
      }

      void handleRemoveNode(Graph::NodeItr xnItr) {
        typedef HeuristicSolverImpl<Briggs>::SolverEdgeItr SolverEdgeItr;
        std::vector<Graph::EdgeItr> edgesToRemove;
        for (SolverEdgeItr aeItr = getSolver().solverEdgesBegin(xnItr),
                           aeEnd = getSolver().solverEdgesEnd(xnItr);
             aeItr != aeEnd; ++aeItr) {
          Graph::NodeItr ynItr = getGraph().getEdgeOtherNode(*aeItr, xnItr);
          handleRemoveEdge(*aeItr, ynItr);
          edgesToRemove.push_back(*aeItr);
        }
        while (!edgesToRemove.empty()) {
          getSolver().removeSolverEdge(edgesToRemove.back());
          edgesToRemove.pop_back();
        }
      }

      RNAllocableList rnAllocableList;
      RNUnallocableList rnUnallocableList;
    };

  }
}


#endif // LLVM_CODEGEN_PBQP_HEURISTICS_BRIGGS_H
