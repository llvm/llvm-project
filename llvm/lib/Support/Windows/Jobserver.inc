//==- llvm/Support/Windows/Jobserver.inc - Windows Jobserver Impl -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Windows-specific parts of the JobserverClient class.
// On Windows, the jobserver is implemented using a named semaphore.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Windows/WindowsSupport.h"
#include <atomic>
#include <cassert>

namespace llvm {
/// The constructor for the Windows jobserver client. It attempts to open a
/// handle to an existing named semaphore, the name of which is provided by
/// GNU make in the --jobserver-auth argument. If the semaphore is opened
/// successfully, the client is marked as initialized.
JobserverClientImpl::JobserverClientImpl(const JobserverConfig &Config) {
  Semaphore = (void *)::OpenSemaphoreA(SEMAPHORE_MODIFY_STATE | SYNCHRONIZE,
                                       FALSE, Config.Path.c_str());
  if (Semaphore != nullptr)
    IsInitialized = true;
}

/// The destructor closes the handle to the semaphore, releasing the resource.
JobserverClientImpl::~JobserverClientImpl() {
  if (Semaphore != nullptr)
    ::CloseHandle((HANDLE)Semaphore);
}

/// Tries to acquire a job slot. The first call always returns the implicit
/// slot. Subsequent calls use a non-blocking wait on the semaphore
/// (`WaitForSingleObject` with a timeout of 0). If the wait succeeds, the
/// semaphore's count is decremented, and an explicit job slot is acquired.
/// If the wait times out, it means no slots are available, and an invalid
/// slot is returned.
JobSlot JobserverClientImpl::tryAcquire() {
  if (!IsInitialized)
    return JobSlot();

  // First, grant the implicit slot.
  if (HasImplicitSlot.exchange(false, std::memory_order_acquire)) {
    return JobSlot::createImplicit();
  }

  // Try to acquire a slot from the semaphore without blocking.
  if (::WaitForSingleObject((HANDLE)Semaphore, 0) == WAIT_OBJECT_0) {
    // The explicit token value is arbitrary on Windows, as the semaphore
    // count is the real resource.
    return JobSlot::createExplicit(1);
  }

  return JobSlot(); // Invalid slot
}

/// Releases a job slot back to the pool. If the slot is implicit, it simply
/// resets a flag. For an explicit slot, it increments the semaphore's count
/// by one using `ReleaseSemaphore`, making the slot available to other
/// processes.
void JobserverClientImpl::release(JobSlot Slot) {
  if (!IsInitialized || !Slot.isValid())
    return;

  if (Slot.isImplicit()) {
    [[maybe_unused]] bool was_already_released =
        HasImplicitSlot.exchange(true, std::memory_order_release);
    assert(!was_already_released && "Implicit slot released twice");
    return;
  }

  // Release the slot by incrementing the semaphore count.
  (void)::ReleaseSemaphore((HANDLE)Semaphore, 1, NULL);
}
} // namespace llvm
