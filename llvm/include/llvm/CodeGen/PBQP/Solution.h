//===-- Solution.h ------- PBQP Solution ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// PBQP Solution class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PBQP_SOLUTION_H
#define LLVM_CODEGEN_PBQP_SOLUTION_H

#include "Graph.h"
#include "Math.h"
#include <map>

namespace llvm {
namespace PBQP {

  /// \brief Represents a solution to a PBQP problem.
  ///
  /// To get the selection for each node in the problem use the getSelection method.
  class Solution {
  private:

    typedef std::map<GraphBase::NodeId, unsigned> SelectionsMap;
    SelectionsMap selections;

    unsigned r0Reductions, r1Reductions, r2Reductions, rNReductions;

  public:

    /// \brief Initialise an empty solution.
    Solution()
      : r0Reductions(0), r1Reductions(0), r2Reductions(0), rNReductions(0) {}

    /// \brief Set the selection for a given node.
    /// @param nodeId Node id.
    /// @param selection Selection for nodeId.
    void setSelection(GraphBase::NodeId nodeId, unsigned selection) {
      selections[nodeId] = selection;
    }

    /// \brief Get a node's selection.
    /// @param nodeId Node id.
    /// @return The selection for nodeId;
    unsigned getSelection(GraphBase::NodeId nodeId) const {
      SelectionsMap::const_iterator sItr = selections.find(nodeId);
      assert(sItr != selections.end() && "No selection for node.");
      return sItr->second;
    }

  };

} // namespace PBQP
} // namespace llvm

#endif // LLVM_CODEGEN_PBQP_SOLUTION_H
