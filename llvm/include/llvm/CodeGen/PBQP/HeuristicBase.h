//===-- HeuristcBase.h --- Heuristic base class for PBQP --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PBQP_HEURISTICBASE_H
#define LLVM_CODEGEN_PBQP_HEURISTICBASE_H

#include "HeuristicSolver.h"

namespace PBQP {

  /// \brief Abstract base class for heuristic implementations.
  ///
  /// This class provides a handy base for heuristic implementations with common
  /// solver behaviour implemented for a number of methods.
  ///
  /// To implement your own heuristic using this class as a base you'll have to
  /// implement, as a minimum, the following methods:
  /// <ul>
  ///   <li> void addToHeuristicList(Graph::NodeItr) : Add a node to the
  ///        heuristic reduction list.
  ///   <li> void heuristicReduce() : Perform a single heuristic reduction.
  ///   <li> void preUpdateEdgeCosts(Graph::EdgeItr) : Handle the (imminent)
  ///        change to the cost matrix on the given edge (by R2).
  ///   <li> void postUpdateEdgeCostts(Graph::EdgeItr) : Handle the new 
  ///        costs on the given edge.
  ///   <li> void handleAddEdge(Graph::EdgeItr) : Handle the addition of a new
  ///        edge into the PBQP graph (by R2).
  ///   <li> void handleRemoveEdge(Graph::EdgeItr, Graph::NodeItr) : Handle the
  ///        disconnection of the given edge from the given node.
  ///   <li> A constructor for your derived class : to pass back a reference to
  ///        the solver which is using this heuristic.
  /// </ul>
  ///
  /// These methods are implemented in this class for documentation purposes,
  /// but will assert if called.
  /// 
  /// Note that this class uses the curiously recursive template idiom to
  /// forward calls to the derived class. These methods need not be made
  /// virtual, and indeed probably shouldn't for performance reasons.
  ///
  /// You'll also need to provide NodeData and EdgeData structs in your class.
  /// These can be used to attach data relevant to your heuristic to each
  /// node/edge in the PBQP graph.

  template <typename HImpl>
  class HeuristicBase {
  private:

    typedef std::list<Graph::NodeItr> OptimalList;

    HeuristicSolverImpl<HImpl> &s;
    Graph &g;
    OptimalList optimalList;

    // Return a reference to the derived heuristic.
    HImpl& impl() { return static_cast<HImpl&>(*this); }

    // Add the given node to the optimal reductions list. Keep an iterator to
    // its location for fast removal. 
    void addToOptimalReductionList(Graph::NodeItr nItr) {
      optimalList.insert(optimalList.end(), nItr);
    }

  public:

    /// \brief Construct an instance with a reference to the given solver.
    /// @param solver The solver which is using this heuristic instance.
    HeuristicBase(HeuristicSolverImpl<HImpl> &solver)
      : s(solver), g(s.getGraph()) { }

    /// \brief Get the solver which is using this heuristic instance.
    /// @return The solver which is using this heuristic instance.
    ///
    /// You can use this method to get access to the solver in your derived
    /// heuristic implementation.
    HeuristicSolverImpl<HImpl>& getSolver() { return s; }

    /// \brief Get the graph representing the problem to be solved.
    /// @return The graph representing the problem to be solved.
    Graph& getGraph() { return g; }

    /// \brief Tell the solver to simplify the graph before the reduction phase.
    /// @return Whether or not the solver should run a simplification phase
    ///         prior to the main setup and reduction.
    ///
    /// HeuristicBase returns true from this method as it's a sensible default,
    /// however you can over-ride it in your derived class if you want different
    /// behaviour.
    bool solverRunSimplify() const { return true; }

    /// \brief Decide whether a node should be optimally or heuristically 
    ///        reduced.
    /// @return Whether or not the given node should be listed for optimal
    ///         reduction (via R0, R1 or R2).
    ///
    /// HeuristicBase returns true for any node with degree less than 3. This is
    /// sane and sensible for many situations, but not all. You can over-ride
    /// this method in your derived class if you want a different selection
    /// criteria. Note however that your criteria for selecting optimal nodes
    /// should be <i>at least</i> as strong as this. I.e. Nodes of degree 3 or
    /// higher should not be selected under any circumstances.
    bool shouldOptimallyReduce(Graph::NodeItr nItr) {
      if (g.getNodeDegree(nItr) < 3)
        return true;
      // else
      return false;
    }

    /// \brief Add the given node to the list of nodes to be optimally reduced.
    /// @param nItr Node iterator to be added.
    ///
    /// You probably don't want to over-ride this, except perhaps to record
    /// statistics before calling this implementation. HeuristicBase relies on
    /// its behaviour.
    void addToOptimalReduceList(Graph::NodeItr nItr) {
      optimalList.push_back(nItr);
    }

    /// \brief Initialise the heuristic.
    ///
    /// HeuristicBase iterates over all nodes in the problem and adds them to
    /// the appropriate list using addToOptimalReduceList or
    /// addToHeuristicReduceList based on the result of shouldOptimallyReduce.
    ///
    /// This behaviour should be fine for most situations.
    void setup() {
      for (Graph::NodeItr nItr = g.nodesBegin(), nEnd = g.nodesEnd();
           nItr != nEnd; ++nItr) {
        if (impl().shouldOptimallyReduce(nItr)) {
          addToOptimalReduceList(nItr);
        } else {
          impl().addToHeuristicReduceList(nItr);
        }
      }
    }

    /// \brief Optimally reduce one of the nodes in the optimal reduce list.
    /// @return True if a reduction takes place, false if the optimal reduce
    ///         list is empty.
    ///
    /// Selects a node from the optimal reduce list and removes it, applying
    /// R0, R1 or R2 as appropriate based on the selected node's degree.
    bool optimalReduce() {
      if (optimalList.empty())
        return false;

      Graph::NodeItr nItr = optimalList.front();
      optimalList.pop_front();

      switch (s.getSolverDegree(nItr)) {
        case 0: s.applyR0(nItr); break;
        case 1: s.applyR1(nItr); break;
        case 2: s.applyR2(nItr); break;
        default: llvm_unreachable(
                        "Optimal reductions of degree > 2 nodes is invalid.");
      }

      return true;
    }

    /// \brief Perform the PBQP reduction process.
    ///
    /// Reduces the problem to the empty graph by repeated application of the
    /// reduction rules R0, R1, R2 and RN.
    /// R0, R1 or R2 are always applied if possible before RN is used.
    void reduce() {
      bool finished = false;

      while (!finished) {
        if (!optimalReduce()) {
          if (impl().heuristicReduce()) {
            getSolver().recordRN();
          } else {
            finished = true;
          }
        }
      }
    }

    /// \brief Add a node to the heuristic reduce list.
    /// @param nItr Node iterator to add to the heuristic reduce list.
    void addToHeuristicList(Graph::NodeItr nItr) {
      llvm_unreachable("Must be implemented in derived class.");
    }

    /// \brief Heuristically reduce one of the nodes in the heuristic
    ///        reduce list.
    /// @return True if a reduction takes place, false if the heuristic reduce
    ///         list is empty.
    bool heuristicReduce() {
      llvm_unreachable("Must be implemented in derived class.");
      return false;
    }

    /// \brief Prepare a change in the costs on the given edge.
    /// @param eItr Edge iterator.    
    void preUpdateEdgeCosts(Graph::EdgeItr eItr) {
      llvm_unreachable("Must be implemented in derived class.");
    }

    /// \brief Handle the change in the costs on the given edge.
    /// @param eItr Edge iterator.
    void postUpdateEdgeCostts(Graph::EdgeItr eItr) {
      llvm_unreachable("Must be implemented in derived class.");
    }

    /// \brief Handle the addition of a new edge into the PBQP graph.
    /// @param eItr Edge iterator for the added edge.
    void handleAddEdge(Graph::EdgeItr eItr) {
      llvm_unreachable("Must be implemented in derived class.");
    }

    /// \brief Handle disconnection of an edge from a node.
    /// @param eItr Edge iterator for edge being disconnected.
    /// @param nItr Node iterator for the node being disconnected from.
    ///
    /// Edges are frequently removed due to the removal of a node. This
    /// method allows for the effect to be computed only for the remaining
    /// node in the graph.
    void handleRemoveEdge(Graph::EdgeItr eItr, Graph::NodeItr nItr) {
      llvm_unreachable("Must be implemented in derived class.");
    }

    /// \brief Clean up any structures used by HeuristicBase.
    ///
    /// At present this just performs a sanity check: that the optimal reduce
    /// list is empty now that reduction has completed.
    ///
    /// If your derived class has more complex structures which need tearing
    /// down you should over-ride this method but include a call back to this
    /// implementation.
    void cleanup() {
      assert(optimalList.empty() && "Nodes left over in optimal reduce list?");
    }

  };

}


#endif // LLVM_CODEGEN_PBQP_HEURISTICBASE_H
