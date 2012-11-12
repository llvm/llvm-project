//===-- HeuristicSolver.h - Heuristic PBQP Solver --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Heuristic PBQP solver. This solver is able to perform optimal reductions for
// nodes of degree 0, 1 or 2. For nodes of degree >2 a plugable heuristic is
// used to select a node for reduction. 
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PBQP_HEURISTICSOLVER_H
#define LLVM_CODEGEN_PBQP_HEURISTICSOLVER_H

#include "Graph.h"
#include "Solution.h"
#include <vector>
#include <limits>

namespace PBQP {

  /// \brief Heuristic PBQP solver implementation.
  ///
  /// This class should usually be created (and destroyed) indirectly via a call
  /// to HeuristicSolver<HImpl>::solve(Graph&).
  /// See the comments for HeuristicSolver.
  ///
  /// HeuristicSolverImpl provides the R0, R1 and R2 reduction rules,
  /// backpropagation phase, and maintains the internal copy of the graph on
  /// which the reduction is carried out (the original being kept to facilitate
  /// backpropagation).
  template <typename HImpl>
  class HeuristicSolverImpl {
  private:

    typedef typename HImpl::NodeData HeuristicNodeData;
    typedef typename HImpl::EdgeData HeuristicEdgeData;

    typedef std::list<Graph::EdgeItr> SolverEdges;

  public:
  
    /// \brief Iterator type for edges in the solver graph.
    typedef SolverEdges::iterator SolverEdgeItr;

  private:

    class NodeData {
    public:
      NodeData() : solverDegree(0) {}

      HeuristicNodeData& getHeuristicData() { return hData; }

      SolverEdgeItr addSolverEdge(Graph::EdgeItr eItr) {
        ++solverDegree;
        return solverEdges.insert(solverEdges.end(), eItr);
      }

      void removeSolverEdge(SolverEdgeItr seItr) {
        --solverDegree;
        solverEdges.erase(seItr);
      }

      SolverEdgeItr solverEdgesBegin() { return solverEdges.begin(); }
      SolverEdgeItr solverEdgesEnd() { return solverEdges.end(); }
      unsigned getSolverDegree() const { return solverDegree; }
      void clearSolverEdges() {
        solverDegree = 0;
        solverEdges.clear(); 
      }
      
    private:
      HeuristicNodeData hData;
      unsigned solverDegree;
      SolverEdges solverEdges;
    };
 
    class EdgeData {
    public:
      HeuristicEdgeData& getHeuristicData() { return hData; }

      void setN1SolverEdgeItr(SolverEdgeItr n1SolverEdgeItr) {
        this->n1SolverEdgeItr = n1SolverEdgeItr;
      }

      SolverEdgeItr getN1SolverEdgeItr() { return n1SolverEdgeItr; }

      void setN2SolverEdgeItr(SolverEdgeItr n2SolverEdgeItr){
        this->n2SolverEdgeItr = n2SolverEdgeItr;
      }

      SolverEdgeItr getN2SolverEdgeItr() { return n2SolverEdgeItr; }

    private:

      HeuristicEdgeData hData;
      SolverEdgeItr n1SolverEdgeItr, n2SolverEdgeItr;
    };

    Graph &g;
    HImpl h;
    Solution s;
    std::vector<Graph::NodeItr> stack;

    typedef std::list<NodeData> NodeDataList;
    NodeDataList nodeDataList;

    typedef std::list<EdgeData> EdgeDataList;
    EdgeDataList edgeDataList;

  public:

    /// \brief Construct a heuristic solver implementation to solve the given
    ///        graph.
    /// @param g The graph representing the problem instance to be solved.
    HeuristicSolverImpl(Graph &g) : g(g), h(*this) {}  

    /// \brief Get the graph being solved by this solver.
    /// @return The graph representing the problem instance being solved by this
    ///         solver.
    Graph& getGraph() { return g; }

    /// \brief Get the heuristic data attached to the given node.
    /// @param nItr Node iterator.
    /// @return The heuristic data attached to the given node.
    HeuristicNodeData& getHeuristicNodeData(Graph::NodeItr nItr) {
      return getSolverNodeData(nItr).getHeuristicData();
    }

    /// \brief Get the heuristic data attached to the given edge.
    /// @param eItr Edge iterator.
    /// @return The heuristic data attached to the given node.
    HeuristicEdgeData& getHeuristicEdgeData(Graph::EdgeItr eItr) {
      return getSolverEdgeData(eItr).getHeuristicData();
    }

    /// \brief Begin iterator for the set of edges adjacent to the given node in
    ///        the solver graph.
    /// @param nItr Node iterator.
    /// @return Begin iterator for the set of edges adjacent to the given node
    ///         in the solver graph. 
    SolverEdgeItr solverEdgesBegin(Graph::NodeItr nItr) {
      return getSolverNodeData(nItr).solverEdgesBegin();
    }

    /// \brief End iterator for the set of edges adjacent to the given node in
    ///        the solver graph.
    /// @param nItr Node iterator.
    /// @return End iterator for the set of edges adjacent to the given node in
    ///         the solver graph. 
    SolverEdgeItr solverEdgesEnd(Graph::NodeItr nItr) {
      return getSolverNodeData(nItr).solverEdgesEnd();
    }

    /// \brief Remove a node from the solver graph.
    /// @param eItr Edge iterator for edge to be removed.
    ///
    /// Does <i>not</i> notify the heuristic of the removal. That should be
    /// done manually if necessary.
    void removeSolverEdge(Graph::EdgeItr eItr) {
      EdgeData &eData = getSolverEdgeData(eItr);
      NodeData &n1Data = getSolverNodeData(g.getEdgeNode1(eItr)),
               &n2Data = getSolverNodeData(g.getEdgeNode2(eItr));

      n1Data.removeSolverEdge(eData.getN1SolverEdgeItr());
      n2Data.removeSolverEdge(eData.getN2SolverEdgeItr());
    }

    /// \brief Compute a solution to the PBQP problem instance with which this
    ///        heuristic solver was constructed.
    /// @return A solution to the PBQP problem.
    ///
    /// Performs the full PBQP heuristic solver algorithm, including setup,
    /// calls to the heuristic (which will call back to the reduction rules in
    /// this class), and cleanup.
    Solution computeSolution() {
      setup();
      h.setup();
      h.reduce();
      backpropagate();
      h.cleanup();
      cleanup();
      return s;
    }

    /// \brief Add to the end of the stack.
    /// @param nItr Node iterator to add to the reduction stack.
    void pushToStack(Graph::NodeItr nItr) {
      getSolverNodeData(nItr).clearSolverEdges();
      stack.push_back(nItr);
    }

    /// \brief Returns the solver degree of the given node.
    /// @param nItr Node iterator for which degree is requested.
    /// @return Node degree in the <i>solver</i> graph (not the original graph).
    unsigned getSolverDegree(Graph::NodeItr nItr) {
      return  getSolverNodeData(nItr).getSolverDegree();
    }

    /// \brief Set the solution of the given node.
    /// @param nItr Node iterator to set solution for.
    /// @param selection Selection for node.
    void setSolution(const Graph::NodeItr &nItr, unsigned selection) {
      s.setSelection(nItr, selection);

      for (Graph::AdjEdgeItr aeItr = g.adjEdgesBegin(nItr),
                             aeEnd = g.adjEdgesEnd(nItr);
           aeItr != aeEnd; ++aeItr) {
        Graph::EdgeItr eItr(*aeItr);
        Graph::NodeItr anItr(g.getEdgeOtherNode(eItr, nItr));
        getSolverNodeData(anItr).addSolverEdge(eItr);
      }
    }

    /// \brief Apply rule R0.
    /// @param nItr Node iterator for node to apply R0 to.
    ///
    /// Node will be automatically pushed to the solver stack.
    void applyR0(Graph::NodeItr nItr) {
      assert(getSolverNodeData(nItr).getSolverDegree() == 0 &&
             "R0 applied to node with degree != 0.");

      // Nothing to do. Just push the node onto the reduction stack.
      pushToStack(nItr);

      s.recordR0();
    }

    /// \brief Apply rule R1.
    /// @param xnItr Node iterator for node to apply R1 to.
    ///
    /// Node will be automatically pushed to the solver stack.
    void applyR1(Graph::NodeItr xnItr) {
      NodeData &nd = getSolverNodeData(xnItr);
      assert(nd.getSolverDegree() == 1 &&
             "R1 applied to node with degree != 1.");

      Graph::EdgeItr eItr = *nd.solverEdgesBegin();

      const Matrix &eCosts = g.getEdgeCosts(eItr);
      const Vector &xCosts = g.getNodeCosts(xnItr);
      
      // Duplicate a little to avoid transposing matrices.
      if (xnItr == g.getEdgeNode1(eItr)) {
        Graph::NodeItr ynItr = g.getEdgeNode2(eItr);
        Vector &yCosts = g.getNodeCosts(ynItr);
        for (unsigned j = 0; j < yCosts.getLength(); ++j) {
          PBQPNum min = eCosts[0][j] + xCosts[0];
          for (unsigned i = 1; i < xCosts.getLength(); ++i) {
            PBQPNum c = eCosts[i][j] + xCosts[i];
            if (c < min)
              min = c;
          }
          yCosts[j] += min;
        }
        h.handleRemoveEdge(eItr, ynItr);
     } else {
        Graph::NodeItr ynItr = g.getEdgeNode1(eItr);
        Vector &yCosts = g.getNodeCosts(ynItr);
        for (unsigned i = 0; i < yCosts.getLength(); ++i) {
          PBQPNum min = eCosts[i][0] + xCosts[0];
          for (unsigned j = 1; j < xCosts.getLength(); ++j) {
            PBQPNum c = eCosts[i][j] + xCosts[j];
            if (c < min)
              min = c;
          }
          yCosts[i] += min;
        }
        h.handleRemoveEdge(eItr, ynItr);
      }
      removeSolverEdge(eItr);
      assert(nd.getSolverDegree() == 0 &&
             "Degree 1 with edge removed should be 0.");
      pushToStack(xnItr);
      s.recordR1();
    }

    /// \brief Apply rule R2.
    /// @param xnItr Node iterator for node to apply R2 to.
    ///
    /// Node will be automatically pushed to the solver stack.
    void applyR2(Graph::NodeItr xnItr) {
      assert(getSolverNodeData(xnItr).getSolverDegree() == 2 &&
             "R2 applied to node with degree != 2.");

      NodeData &nd = getSolverNodeData(xnItr);
      const Vector &xCosts = g.getNodeCosts(xnItr);

      SolverEdgeItr aeItr = nd.solverEdgesBegin();
      Graph::EdgeItr yxeItr = *aeItr,
                     zxeItr = *(++aeItr);

      Graph::NodeItr ynItr = g.getEdgeOtherNode(yxeItr, xnItr),
                     znItr = g.getEdgeOtherNode(zxeItr, xnItr);

      bool flipEdge1 = (g.getEdgeNode1(yxeItr) == xnItr),
           flipEdge2 = (g.getEdgeNode1(zxeItr) == xnItr);

      const Matrix *yxeCosts = flipEdge1 ?
        new Matrix(g.getEdgeCosts(yxeItr).transpose()) :
        &g.getEdgeCosts(yxeItr);

      const Matrix *zxeCosts = flipEdge2 ?
        new Matrix(g.getEdgeCosts(zxeItr).transpose()) :
        &g.getEdgeCosts(zxeItr);

      unsigned xLen = xCosts.getLength(),
               yLen = yxeCosts->getRows(),
               zLen = zxeCosts->getRows();
               
      Matrix delta(yLen, zLen);

      for (unsigned i = 0; i < yLen; ++i) {
        for (unsigned j = 0; j < zLen; ++j) {
          PBQPNum min = (*yxeCosts)[i][0] + (*zxeCosts)[j][0] + xCosts[0];
          for (unsigned k = 1; k < xLen; ++k) {
            PBQPNum c = (*yxeCosts)[i][k] + (*zxeCosts)[j][k] + xCosts[k];
            if (c < min) {
              min = c;
            }
          }
          delta[i][j] = min;
        }
      }

      if (flipEdge1)
        delete yxeCosts;

      if (flipEdge2)
        delete zxeCosts;

      Graph::EdgeItr yzeItr = g.findEdge(ynItr, znItr);
      bool addedEdge = false;

      if (yzeItr == g.edgesEnd()) {
        yzeItr = g.addEdge(ynItr, znItr, delta);
        addedEdge = true;
      } else {
        Matrix &yzeCosts = g.getEdgeCosts(yzeItr);
        h.preUpdateEdgeCosts(yzeItr);
        if (ynItr == g.getEdgeNode1(yzeItr)) {
          yzeCosts += delta;
        } else {
          yzeCosts += delta.transpose();
        }
      }

      bool nullCostEdge = tryNormaliseEdgeMatrix(yzeItr);

      if (!addedEdge) {
        // If we modified the edge costs let the heuristic know.
        h.postUpdateEdgeCosts(yzeItr);
      }
 
      if (nullCostEdge) {
        // If this edge ended up null remove it.
        if (!addedEdge) {
          // We didn't just add it, so we need to notify the heuristic
          // and remove it from the solver.
          h.handleRemoveEdge(yzeItr, ynItr);
          h.handleRemoveEdge(yzeItr, znItr);
          removeSolverEdge(yzeItr);
        }
        g.removeEdge(yzeItr);
      } else if (addedEdge) {
        // If the edge was added, and non-null, finish setting it up, add it to
        // the solver & notify heuristic.
        edgeDataList.push_back(EdgeData());
        g.setEdgeData(yzeItr, &edgeDataList.back());
        addSolverEdge(yzeItr);
        h.handleAddEdge(yzeItr);
      }

      h.handleRemoveEdge(yxeItr, ynItr);
      removeSolverEdge(yxeItr);
      h.handleRemoveEdge(zxeItr, znItr);
      removeSolverEdge(zxeItr);

      pushToStack(xnItr);
      s.recordR2();
    }

    /// \brief Record an application of the RN rule.
    ///
    /// For use by the HeuristicBase.
    void recordRN() { s.recordRN(); } 

  private:

    NodeData& getSolverNodeData(Graph::NodeItr nItr) {
      return *static_cast<NodeData*>(g.getNodeData(nItr));
    }

    EdgeData& getSolverEdgeData(Graph::EdgeItr eItr) {
      return *static_cast<EdgeData*>(g.getEdgeData(eItr));
    }

    void addSolverEdge(Graph::EdgeItr eItr) {
      EdgeData &eData = getSolverEdgeData(eItr);
      NodeData &n1Data = getSolverNodeData(g.getEdgeNode1(eItr)),
               &n2Data = getSolverNodeData(g.getEdgeNode2(eItr));

      eData.setN1SolverEdgeItr(n1Data.addSolverEdge(eItr));
      eData.setN2SolverEdgeItr(n2Data.addSolverEdge(eItr));
    }

    void setup() {
      if (h.solverRunSimplify()) {
        simplify();
      }

      // Create node data objects.
      for (Graph::NodeItr nItr = g.nodesBegin(), nEnd = g.nodesEnd();
           nItr != nEnd; ++nItr) {
        nodeDataList.push_back(NodeData());
        g.setNodeData(nItr, &nodeDataList.back());
      }

      // Create edge data objects.
      for (Graph::EdgeItr eItr = g.edgesBegin(), eEnd = g.edgesEnd();
           eItr != eEnd; ++eItr) {
        edgeDataList.push_back(EdgeData());
        g.setEdgeData(eItr, &edgeDataList.back());
        addSolverEdge(eItr);
      }
    }

    void simplify() {
      disconnectTrivialNodes();
      eliminateIndependentEdges();
    }

    // Eliminate trivial nodes.
    void disconnectTrivialNodes() {
      unsigned numDisconnected = 0;

      for (Graph::NodeItr nItr = g.nodesBegin(), nEnd = g.nodesEnd();
           nItr != nEnd; ++nItr) {

        if (g.getNodeCosts(nItr).getLength() == 1) {

          std::vector<Graph::EdgeItr> edgesToRemove;

          for (Graph::AdjEdgeItr aeItr = g.adjEdgesBegin(nItr),
                                 aeEnd = g.adjEdgesEnd(nItr);
               aeItr != aeEnd; ++aeItr) {

            Graph::EdgeItr eItr = *aeItr;

            if (g.getEdgeNode1(eItr) == nItr) {
              Graph::NodeItr otherNodeItr = g.getEdgeNode2(eItr);
              g.getNodeCosts(otherNodeItr) +=
                g.getEdgeCosts(eItr).getRowAsVector(0);
            }
            else {
              Graph::NodeItr otherNodeItr = g.getEdgeNode1(eItr);
              g.getNodeCosts(otherNodeItr) +=
                g.getEdgeCosts(eItr).getColAsVector(0);
            }

            edgesToRemove.push_back(eItr);
          }

          if (!edgesToRemove.empty())
            ++numDisconnected;

          while (!edgesToRemove.empty()) {
            g.removeEdge(edgesToRemove.back());
            edgesToRemove.pop_back();
          }
        }
      }
    }

    void eliminateIndependentEdges() {
      std::vector<Graph::EdgeItr> edgesToProcess;
      unsigned numEliminated = 0;

      for (Graph::EdgeItr eItr = g.edgesBegin(), eEnd = g.edgesEnd();
           eItr != eEnd; ++eItr) {
        edgesToProcess.push_back(eItr);
      }

      while (!edgesToProcess.empty()) {
        if (tryToEliminateEdge(edgesToProcess.back()))
          ++numEliminated;
        edgesToProcess.pop_back();
      }
    }

    bool tryToEliminateEdge(Graph::EdgeItr eItr) {
      if (tryNormaliseEdgeMatrix(eItr)) {
        g.removeEdge(eItr);
        return true; 
      }
      return false;
    }

    bool tryNormaliseEdgeMatrix(Graph::EdgeItr &eItr) {

      const PBQPNum infinity = std::numeric_limits<PBQPNum>::infinity();

      Matrix &edgeCosts = g.getEdgeCosts(eItr);
      Vector &uCosts = g.getNodeCosts(g.getEdgeNode1(eItr)),
             &vCosts = g.getNodeCosts(g.getEdgeNode2(eItr));

      for (unsigned r = 0; r < edgeCosts.getRows(); ++r) {
        PBQPNum rowMin = infinity;

        for (unsigned c = 0; c < edgeCosts.getCols(); ++c) {
          if (vCosts[c] != infinity && edgeCosts[r][c] < rowMin)
            rowMin = edgeCosts[r][c];
        }

        uCosts[r] += rowMin;

        if (rowMin != infinity) {
          edgeCosts.subFromRow(r, rowMin);
        }
        else {
          edgeCosts.setRow(r, 0);
        }
      }

      for (unsigned c = 0; c < edgeCosts.getCols(); ++c) {
        PBQPNum colMin = infinity;

        for (unsigned r = 0; r < edgeCosts.getRows(); ++r) {
          if (uCosts[r] != infinity && edgeCosts[r][c] < colMin)
            colMin = edgeCosts[r][c];
        }

        vCosts[c] += colMin;

        if (colMin != infinity) {
          edgeCosts.subFromCol(c, colMin);
        }
        else {
          edgeCosts.setCol(c, 0);
        }
      }

      return edgeCosts.isZero();
    }

    void backpropagate() {
      while (!stack.empty()) {
        computeSolution(stack.back());
        stack.pop_back();
      }
    }

    void computeSolution(Graph::NodeItr nItr) {

      NodeData &nodeData = getSolverNodeData(nItr);

      Vector v(g.getNodeCosts(nItr));

      // Solve based on existing solved edges.
      for (SolverEdgeItr solvedEdgeItr = nodeData.solverEdgesBegin(),
                         solvedEdgeEnd = nodeData.solverEdgesEnd();
           solvedEdgeItr != solvedEdgeEnd; ++solvedEdgeItr) {

        Graph::EdgeItr eItr(*solvedEdgeItr);
        Matrix &edgeCosts = g.getEdgeCosts(eItr);

        if (nItr == g.getEdgeNode1(eItr)) {
          Graph::NodeItr adjNode(g.getEdgeNode2(eItr));
          unsigned adjSolution = s.getSelection(adjNode);
          v += edgeCosts.getColAsVector(adjSolution);
        }
        else {
          Graph::NodeItr adjNode(g.getEdgeNode1(eItr));
          unsigned adjSolution = s.getSelection(adjNode);
          v += edgeCosts.getRowAsVector(adjSolution);
        }

      }

      setSolution(nItr, v.minIndex());
    }

    void cleanup() {
      h.cleanup();
      nodeDataList.clear();
      edgeDataList.clear();
    }
  };

  /// \brief PBQP heuristic solver class.
  ///
  /// Given a PBQP Graph g representing a PBQP problem, you can find a solution
  /// by calling
  /// <tt>Solution s = HeuristicSolver<H>::solve(g);</tt>
  ///
  /// The choice of heuristic for the H parameter will affect both the solver
  /// speed and solution quality. The heuristic should be chosen based on the
  /// nature of the problem being solved.
  /// Currently the only solver included with LLVM is the Briggs heuristic for
  /// register allocation.
  template <typename HImpl>
  class HeuristicSolver {
  public:
    static Solution solve(Graph &g) {
      HeuristicSolverImpl<HImpl> hs(g);
      return hs.computeSolution();
    }
  };

}

#endif // LLVM_CODEGEN_PBQP_HEURISTICSOLVER_H
