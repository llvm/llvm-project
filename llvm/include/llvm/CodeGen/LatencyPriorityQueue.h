//===---- LatencyPriorityQueue.h - A latency-oriented priority queue ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the LatencyPriorityQueue class, which is a
// SchedulingPriorityQueue that schedules using latency information to
// reduce the length of the critical path through the basic block.
//
//===----------------------------------------------------------------------===//

#ifndef LATENCY_PRIORITY_QUEUE_H
#define LATENCY_PRIORITY_QUEUE_H

#include "llvm/CodeGen/ScheduleDAG.h"

namespace llvm {
  class LatencyPriorityQueue;

  /// Sorting functions for the Available queue.
  struct latency_sort : public std::binary_function<SUnit*, SUnit*, bool> {
    LatencyPriorityQueue *PQ;
    explicit latency_sort(LatencyPriorityQueue *pq) : PQ(pq) {}

    bool operator()(const SUnit* left, const SUnit* right) const;
  };

  class LatencyPriorityQueue : public SchedulingPriorityQueue {
    // SUnits - The SUnits for the current graph.
    std::vector<SUnit> *SUnits;

    /// NumNodesSolelyBlocking - This vector contains, for every node in the
    /// Queue, the number of nodes that the node is the sole unscheduled
    /// predecessor for.  This is used as a tie-breaker heuristic for better
    /// mobility.
    std::vector<unsigned> NumNodesSolelyBlocking;

    /// Queue - The queue.
    std::vector<SUnit*> Queue;
    latency_sort Picker;

  public:
    LatencyPriorityQueue() : Picker(this) {
    }

    bool isBottomUp() const { return false; }

    void initNodes(std::vector<SUnit> &sunits) {
      SUnits = &sunits;
      NumNodesSolelyBlocking.resize(SUnits->size(), 0);
    }

    void addNode(const SUnit *SU) {
      NumNodesSolelyBlocking.resize(SUnits->size(), 0);
    }

    void updateNode(const SUnit *SU) {
    }

    void releaseState() {
      SUnits = 0;
    }

    unsigned getLatency(unsigned NodeNum) const {
      assert(NodeNum < (*SUnits).size());
      return (*SUnits)[NodeNum].getHeight();
    }

    unsigned getNumSolelyBlockNodes(unsigned NodeNum) const {
      assert(NodeNum < NumNodesSolelyBlocking.size());
      return NumNodesSolelyBlocking[NodeNum];
    }

    bool empty() const { return Queue.empty(); }

    virtual void push(SUnit *U);

    virtual SUnit *pop();

    virtual void remove(SUnit *SU);

    virtual void dump(ScheduleDAG* DAG) const;

    // scheduledNode - As nodes are scheduled, we look to see if there are any
    // successor nodes that have a single unscheduled predecessor.  If so, that
    // single predecessor has a higher priority, since scheduling it will make
    // the node available.
    void scheduledNode(SUnit *Node);

private:
    void AdjustPriorityOfUnscheduledPreds(SUnit *SU);
    SUnit *getSingleUnscheduledPred(SUnit *SU);
  };
}

#endif
