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

bool Registry::Replay(const FileSpec &file) {
  auto error_or_file = llvm::MemoryBuffer::getFile(file.GetPath());
  if (auto err = error_or_file.getError())
    return false;

  return Replay((*error_or_file)->getBuffer());
}

bool Registry::Replay(llvm::StringRef buffer) {
  Log *log = GetLogIfAllCategoriesSet(LIBLLDB_LOG_API);

  Deserializer deserializer(buffer);
  while (deserializer.HasData(1)) {
    unsigned id = deserializer.Deserialize<unsigned>();
    LLDB_LOG(log, "Replaying function #{0}", id);
    m_ids[id]->operator()(deserializer);
  }

  return true;
}

void Registry::DoRegister(uintptr_t RunID, std::unique_ptr<Replayer> replayer) {
  const unsigned id = m_replayers.size() + 1;
  assert(m_replayers.find(RunID) == m_replayers.end());
  m_replayers[RunID] = std::make_pair(std::move(replayer), id);
  m_ids[id] = m_replayers[RunID].first.get();
}

unsigned Registry::GetID(uintptr_t addr) {
  unsigned id = m_replayers[addr].second;
  assert(id != 0 && "Forgot to add function to registry?");
  return id;
}

unsigned ObjectToIndex::GetIndexForObjectImpl(void *object) {
  unsigned index = m_mapping.size() + 1;
  auto it = m_mapping.find(object);
  if (it == m_mapping.end())
    m_mapping[object] = index;
  return m_mapping[object];
}

Recorder::Recorder(Serializer &serializer, Registry &registry,
                   llvm::StringRef pretty_func)
    : m_serializer(serializer), m_registry(registry),
      m_pretty_func(pretty_func), m_local_boundary(false),
      m_result_recorded(true) {
  if (!g_global_boundary) {
    g_global_boundary = true;
    m_local_boundary = true;
  }
}

Recorder::~Recorder() {
  assert(m_result_recorded && "Did you forget LLDB_RECORD_RESULT?");
  UpdateBoundary();
}

bool lldb_private::repro::Recorder::g_global_boundary;
