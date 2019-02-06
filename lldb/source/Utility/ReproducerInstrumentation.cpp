//===-- ReproducerInstrumentation.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/ReproducerInstrumentation.h"
#include "lldb/Utility/Reproducer.h"

using namespace lldb_private;
using namespace lldb_private::repro;

void *IndexToObject::GetObjectForIndexImpl(unsigned idx) {
  return m_mapping.lookup(idx);
}

void IndexToObject::AddObjectForIndexImpl(unsigned idx, void *object) {
  assert(idx != 0 && "Cannot add object for sentinel");
  m_mapping[idx] = object;
}

template <> char *Deserializer::Deserialize<char *>() {
  return const_cast<char *>(Deserialize<const char *>());
}

template <> const char *Deserializer::Deserialize<const char *>() {
  auto pos = m_buffer.find('\0');
  if (pos == llvm::StringRef::npos)
    return nullptr;
  const char *str = m_buffer.data();
  m_buffer = m_buffer.drop_front(pos + 1);
  return str;
}

unsigned ObjectToIndex::GetIndexForObjectImpl(void *object) {
  std::lock_guard<std::mutex> guard(m_mutex);
  unsigned index = m_mapping.size() + 1;
  auto it = m_mapping.find(object);
  if (it == m_mapping.end())
    m_mapping[object] = index;
  return m_mapping[object];
}
