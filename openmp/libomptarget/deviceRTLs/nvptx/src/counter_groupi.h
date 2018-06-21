//===----- counter_groupi.h - NVPTX OpenMP loop scheduling ------- CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// Interface implementation for OpenMP loop scheduling
//
//===----------------------------------------------------------------------===//

#include "option.h"

INLINE void omptarget_nvptx_CounterGroup::Clear() {
  PRINT0(LD_SYNCD, "clear counters\n")
  v_event = 0;
  v_start = 0;
  // v_init does not need to be reset (its value is dead)
}

INLINE void omptarget_nvptx_CounterGroup::Reset() {
  // done by master before entering parallel
  ASSERT(LT_FUSSY, v_event == v_start,
         "error, entry %lld !=start %lld at reset\n", P64(v_event),
         P64(v_start));
  v_init = v_start;
}

INLINE void omptarget_nvptx_CounterGroup::Init(Counter &priv) {
  PRINT(LD_SYNCD, "init priv counter 0x%llx with val %lld\n", P64(&priv),
        P64(v_start));
  priv = v_start;
}

// just counts number of events
INLINE Counter omptarget_nvptx_CounterGroup::Next() {
  Counter oldVal = atomicAdd(&v_event, (Counter)1);
  PRINT(LD_SYNCD, "next event counter 0x%llx with val %lld->%lld\n",
        P64(&v_event), P64(oldVal), P64(oldVal + 1));

  return oldVal;
}

// set priv to n, to be used in later waitOrRelease
INLINE void omptarget_nvptx_CounterGroup::Complete(Counter &priv, Counter n) {
  PRINT(LD_SYNCD, "complete priv counter 0x%llx with val %llu->%llu (+%llu)\n",
        P64(&priv), P64(priv), P64(priv + n), n);
  priv += n;
}

INLINE void omptarget_nvptx_CounterGroup::Release(Counter priv,
                                                  Counter current_event_value) {
  if (priv - 1 == current_event_value) {
    PRINT(LD_SYNCD, "Release start counter 0x%llx with val %lld->%lld\n",
          P64(&v_start), P64(v_start), P64(priv));
    v_start = priv;
  }
}

// check priv and decide if we have to wait or can free the other warps
INLINE void
omptarget_nvptx_CounterGroup::WaitOrRelease(Counter priv,
                                            Counter current_event_value) {
  if (priv - 1 == current_event_value) {
    PRINT(LD_SYNCD, "Release start counter 0x%llx with val %lld->%lld\n",
          P64(&v_start), P64(v_start), P64(priv));
    v_start = priv;
  } else {
    PRINT(LD_SYNCD,
          "Start waiting while start counter 0x%llx with val %lld < %lld\n",
          P64(&v_start), P64(v_start), P64(priv));
    while (priv > v_start) {
      // IDLE LOOP
      // start is volatile: it will be re-loaded at each while loop
    }
    PRINT(LD_SYNCD,
          "Done waiting as start counter 0x%llx with val %lld >= %lld\n",
          P64(&v_start), P64(v_start), P64(priv));
  }
}
