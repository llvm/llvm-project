//===--------- statequeue.h - NVPTX OpenMP GPU State Queue ------- CUDA -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a queue to hand out OpenMP state objects to teams of
// one or more kernels.
//
// Reference:
// Thomas R.W. Scogland and Wu-chun Feng. 2015.
// Design and Evaluation of Scalable Concurrent Queues for Many-Core
// Architectures. International Conference on Performance Engineering.
//
//===----------------------------------------------------------------------===//

#ifndef __STATE_QUEUE_H
#define __STATE_QUEUE_H

#include <stdint.h>

#include "option.h" // choices we have

template <typename ElementType, uint32_t SIZE> class omptarget_nvptx_Queue {
private:
  ElementType elements[SIZE];
  volatile ElementType *elementQueue[SIZE];
  volatile uint32_t head;
  volatile uint32_t ids[SIZE];
  volatile uint32_t tail;

  static const uint32_t MAX_ID = (1u << 31) / SIZE / 2;
  INLINE uint32_t ENQUEUE_TICKET();
  INLINE uint32_t DEQUEUE_TICKET();
  INLINE static uint32_t ID(uint32_t ticket);
  INLINE bool IsServing(uint32_t slot, uint32_t id);
  INLINE void PushElement(uint32_t slot, ElementType *element);
  INLINE ElementType *PopElement(uint32_t slot);
  INLINE void DoneServing(uint32_t slot, uint32_t id);

public:
  INLINE omptarget_nvptx_Queue() {}
  INLINE void Enqueue(ElementType *element);
  INLINE ElementType *Dequeue();
};

#include "state-queuei.h"

#endif
