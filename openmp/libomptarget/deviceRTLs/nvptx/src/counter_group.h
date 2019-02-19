//===------ counter_group.h - NVPTX OpenMP loop scheduling ------- CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// Interface to implement OpenMP loop scheduling
//
//===----------------------------------------------------------------------===//

#ifndef _OMPTARGET_NVPTX_COUNTER_GROUP_H_
#define _OMPTARGET_NVPTX_COUNTER_GROUP_H_

#include "option.h"

// counter group type for synchronizations
class omptarget_nvptx_CounterGroup {
public:
  // getters and setters
  INLINE Counter &Event() { return v_event; }
  INLINE volatile Counter &Start() { return v_start; }
  INLINE Counter &Init() { return v_init; }

  // Synchronization Interface

  INLINE void Clear();             // first time start=event
  INLINE void Reset();             // init = first
  INLINE void Init(Counter &priv); // priv = init
  INLINE Counter Next();           // just counts number of events

  // set priv to n, to be used in later waitOrRelease
  INLINE void Complete(Counter &priv, Counter n);

  // check priv and decide if we have to wait or can free the other warps
  INLINE void Release(Counter priv, Counter current_event_value);
  INLINE void WaitOrRelease(Counter priv, Counter current_event_value);

private:
  Counter v_event; // counter of events (atomic)

  // volatile is needed to force loads to read from global
  // memory or L2 cache and see the write by the last master
  volatile Counter v_start; // signal when events registered are finished

  Counter v_init; // used to initialize local thread variables
};

#endif /* SRC_COUNTER_GROUP_H_ */
