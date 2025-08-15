//===----------------- LockGuarded.h ----------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2024 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SwiftLockGuarded_h_
#define liblldb_SwiftLockGuarded_h_

#include <mutex>

namespace lldb_private {
/// A generic wrapper around a resource which holds a lock to ensure
/// exclusive access.
template <typename Resource> struct LockGuarded {
  LockGuarded(Resource *resource, std::recursive_mutex &mutex)
      : m_resource(resource), m_lock(mutex, std::adopt_lock) {}

  LockGuarded() = default;
  LockGuarded(const LockGuarded &) = delete;
  LockGuarded &operator=(const LockGuarded &) = delete;

  Resource *operator->() const { return m_resource; }

  Resource *operator*() const { return m_resource; }

  operator bool() const { return m_resource != nullptr; }

private:
  Resource *m_resource;
  std::unique_lock<std::recursive_mutex> m_lock;
};

} // namespace lldb_private
#endif
