//===- ScheduleDAGILP.h - ILP metric for ScheduleDAGInstrs ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Definition of an ILP metric for machine level instruction scheduling.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SCHEDULEDAGILP_H
#define LLVM_CODEGEN_SCHEDULEDAGILP_H

#include "llvm/Support/DataTypes.h"
#include <vector>

namespace llvm {

class raw_ostream;
class ScheduleDAGInstrs;
class SUnit;

/// \brief Represent the ILP of the subDAG rooted at a DAG node.
struct ILPValue {
  unsigned InstrCount;
  unsigned Cycles;

  ILPValue(): InstrCount(0), Cycles(0) {}

  ILPValue(unsigned count, unsigned cycles):
    InstrCount(count), Cycles(cycles) {}

  bool isValid() const { return Cycles > 0; }

  // Order by the ILP metric's value.
  bool operator<(ILPValue RHS) const {
    return (uint64_t)InstrCount * RHS.Cycles
      < (uint64_t)Cycles * RHS.InstrCount;
  }
  bool operator>(ILPValue RHS) const {
    return RHS < *this;
  }
  bool operator<=(ILPValue RHS) const {
    return (uint64_t)InstrCount * RHS.Cycles
      <= (uint64_t)Cycles * RHS.InstrCount;
  }
  bool operator>=(ILPValue RHS) const {
    return RHS <= *this;
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void print(raw_ostream &OS) const;

  void dump() const;
#endif
};

/// \brief Compute the values of each DAG node for an ILP metric.
///
/// This metric assumes that the DAG is a forest of trees with roots at the
/// bottom of the schedule.
class ScheduleDAGILP {
  bool IsBottomUp;
  std::vector<ILPValue> ILPValues;

public:
  ScheduleDAGILP(bool IsBU): IsBottomUp(IsBU) {}

  /// \brief Initialize the result data with the size of the DAG.
  void resize(unsigned NumSUnits);

  /// \brief Compute the ILP metric for the subDAG at this root.
  void computeILP(const SUnit *Root);

  /// \brief Get the ILP value for a DAG node.
  ILPValue getILP(const SUnit *SU);
};

raw_ostream &operator<<(raw_ostream &OS, const ILPValue &Val);

} // namespace llvm

#endif
