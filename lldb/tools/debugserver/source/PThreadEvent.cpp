//===-- PThreadEvent.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 6/16/07.
//
//===----------------------------------------------------------------------===//

#include "PThreadEvent.h"
#include "DNBLog.h"
#include <cerrno>

PThreadEvent::PThreadEvent(uint32_t bits, uint32_t validBits)
    : m_mutex(), m_set_condition(), m_reset_condition(), m_bits(bits),
      m_validBits(validBits), m_reset_ack_mask(0) {
  // DNBLogThreadedIf(LOG_EVENTS, "%p PThreadEvent::%s (0x%8.8x, 0x%8.8x)",
  // this, __FUNCTION__, bits, validBits);
}

PThreadEvent::~PThreadEvent() {
  // DNBLogThreadedIf(LOG_EVENTS, "%p %s", this, LLVM_PRETTY_FUNCTION);
}

uint32_t PThreadEvent::NewEventBit() {
  // DNBLogThreadedIf(LOG_EVENTS, "%p %s", this, LLVM_PRETTY_FUNCTION);
  PTHREAD_MUTEX_LOCKER(locker, m_mutex);
  uint32_t mask = 1;
  while (mask & m_validBits)
    mask <<= 1;
  m_validBits |= mask;
  return mask;
}

void PThreadEvent::FreeEventBits(const uint32_t mask) {
  // DNBLogThreadedIf(LOG_EVENTS, "%p PThreadEvent::%s (0x%8.8x)", this,
  // __FUNCTION__, mask);
  if (mask) {
    PTHREAD_MUTEX_LOCKER(locker, m_mutex);
    m_bits &= ~mask;
    m_validBits &= ~mask;
  }
}

uint32_t PThreadEvent::GetEventBits() const {
  // DNBLogThreadedIf(LOG_EVENTS, "%p %s", this, LLVM_PRETTY_FUNCTION);
  PTHREAD_MUTEX_LOCKER(locker, m_mutex);
  uint32_t bits = m_bits;
  return bits;
}

// Replace the event bits with a new bitmask value
void PThreadEvent::ReplaceEventBits(const uint32_t bits) {
  // DNBLogThreadedIf(LOG_EVENTS, "%p PThreadEvent::%s (0x%8.8x)", this,
  // __FUNCTION__, bits);
  PTHREAD_MUTEX_LOCKER(locker, m_mutex);
  // Make sure we have some bits and that they aren't already set...
  if (m_bits != bits) {
    // Figure out which bits are changing
    uint32_t changed_bits = m_bits ^ bits;
    // Set the new bit values
    m_bits = bits;
    // If any new bits are set, then broadcast
    if (changed_bits & m_bits)
      m_set_condition.Broadcast();
  }
}

// Set one or more event bits and broadcast if any new event bits get set
// that weren't already set.

void PThreadEvent::SetEvents(const uint32_t mask) {
  // DNBLogThreadedIf(LOG_EVENTS, "%p PThreadEvent::%s (0x%8.8x)", this,
  // __FUNCTION__, mask);
  // Make sure we have some bits to set
  if (mask) {
    PTHREAD_MUTEX_LOCKER(locker, m_mutex);
    // Save the old event bit state so we can tell if things change
    uint32_t old = m_bits;
    // Set the all event bits that are set in 'mask'
    m_bits |= mask;
    // Broadcast only if any extra bits got set.
    if (old != m_bits)
      m_set_condition.Broadcast();
  }
}

// Reset one or more event bits
void PThreadEvent::ResetEvents(const uint32_t mask) {
  // DNBLogThreadedIf(LOG_EVENTS, "%p PThreadEvent::%s (0x%8.8x)", this,
  // __FUNCTION__, mask);
  if (mask) {
    PTHREAD_MUTEX_LOCKER(locker, m_mutex);

    // Save the old event bit state so we can tell if things change
    uint32_t old = m_bits;
    // Clear the all event bits that are set in 'mask'
    m_bits &= ~mask;
    // Broadcast only if any extra bits got reset.
    if (old != m_bits)
      m_reset_condition.Broadcast();
  }
}

// Wait until 'timeout_abstime' for any events that are set in
// 'mask'. If 'timeout_abstime' is NULL, then wait forever.
uint32_t
PThreadEvent::WaitForEventsImpl(const uint32_t mask,
                                const struct timespec *timeout_abstime,
                                std::function<bool()> predicate) const {
  // DNBLogThreadedIf(LOG_EVENTS, "%p PThreadEvent::%s (0x%8.8x, %p)", this,
  // __FUNCTION__, mask, timeout_abstime);

  int err = 0;

  // pthread_cond_timedwait() or pthread_cond_wait() will atomically
  // unlock the mutex and wait for the condition to be set. When either
  // function returns, they will re-lock the mutex. We use an auto lock/unlock
  // class (PThreadMutex::Locker) to allow us to return at any point in this
  // function and not have to worry about unlocking the mutex.
  PTHREAD_MUTEX_LOCKER(locker, m_mutex);

  // Check the predicate and the error code. The functions below do not return
  // EINTR so that's not something we need to handle.
  while (!predicate() && err == 0) {
    if (timeout_abstime) {
      // Wait for condition to get broadcast, or for a timeout. If we get
      // a timeout we will drop out of the loop on the next iteration and we
      // will recompute the mask in case of a race between the condition and the
      // timeout.
      err = ::pthread_cond_timedwait(m_set_condition.Condition(),
                                     m_mutex.Mutex(), timeout_abstime);
    } else {
      // Wait for condition to get broadcast.
      err = ::pthread_cond_wait(m_set_condition.Condition(), m_mutex.Mutex());
    }
  }

  // Either the predicate passed, we hit the specified timeout (ETIMEDOUT) or we
  // encountered an unrecoverable error (EINVAL, EPERM). Regardless of how we
  // got here, recompute and return the mask indicating which bits (if any) are
  // set.
  return GetBitsMasked(mask);
}

uint32_t
PThreadEvent::WaitForSetEvents(const uint32_t mask,
                               const struct timespec *timeout_abstime) const {
  auto predicate = [&]() -> uint32_t { return GetBitsMasked(mask) != 0; };
  return WaitForEventsImpl(mask, timeout_abstime, predicate);
}

uint32_t PThreadEvent::WaitForEventsToReset(
    const uint32_t mask, const struct timespec *timeout_abstime) const {
  auto predicate = [&]() -> uint32_t { return GetBitsMasked(mask) == 0; };
  return WaitForEventsImpl(mask, timeout_abstime, predicate);
}

uint32_t
PThreadEvent::WaitForResetAck(const uint32_t mask,
                              const struct timespec *timeout_abstime) const {
  if (mask & m_reset_ack_mask) {
    // DNBLogThreadedIf(LOG_EVENTS, "%p PThreadEvent::%s (0x%8.8x, %p)", this,
    // __FUNCTION__, mask, timeout_abstime);
    return WaitForEventsToReset(mask & m_reset_ack_mask, timeout_abstime);
  }
  return 0;
}
