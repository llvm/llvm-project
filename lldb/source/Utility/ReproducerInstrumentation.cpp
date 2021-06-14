//===-- ReproducerInstrumentation.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/ReproducerInstrumentation.h"
#include "lldb/Utility/Reproducer.h"
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <thread>

using namespace lldb_private;
using namespace lldb_private::repro;

void *IndexToObject::GetObjectForIndexImpl(unsigned idx) {
  return m_mapping.lookup(idx);
}

void IndexToObject::AddObjectForIndexImpl(unsigned idx, void *object) {
  assert(idx != 0 && "Cannot add object for sentinel");
  m_mapping[idx] = object;
}

std::vector<void *> IndexToObject::GetAllObjects() const {
  std::vector<std::pair<unsigned, void *>> pairs;
  for (auto &e : m_mapping) {
    pairs.emplace_back(e.first, e.second);
  }

  // Sort based on index.
  std::sort(pairs.begin(), pairs.end(),
            [](auto &lhs, auto &rhs) { return lhs.first < rhs.first; });

  std::vector<void *> objects;
  objects.reserve(pairs.size());
  for (auto &p : pairs) {
    objects.push_back(p.second);
  }

  return objects;
}

template <> const uint8_t *Deserializer::Deserialize<const uint8_t *>() {
  return Deserialize<uint8_t *>();
}

template <> void *Deserializer::Deserialize<void *>() {
  return const_cast<void *>(Deserialize<const void *>());
}

template <> const void *Deserializer::Deserialize<const void *>() {
  return nullptr;
}

template <> char *Deserializer::Deserialize<char *>() {
  return const_cast<char *>(Deserialize<const char *>());
}

template <> const char *Deserializer::Deserialize<const char *>() {
  const size_t size = Deserialize<size_t>();
  if (size == std::numeric_limits<size_t>::max())
    return nullptr;
  assert(HasData(size + 1));
  const char *str = m_buffer.data();
  m_buffer = m_buffer.drop_front(size + 1);
#ifdef LLDB_REPRO_INSTR_TRACE
  llvm::errs() << "Deserializing with " << LLVM_PRETTY_FUNCTION << " -> \""
               << str << "\"\n";
#endif
  return str;
}

template <> const char **Deserializer::Deserialize<const char **>() {
  const size_t size = Deserialize<size_t>();
  if (size == 0)
    return nullptr;
  const char **r =
      reinterpret_cast<const char **>(calloc(size + 1, sizeof(char *)));
  for (size_t i = 0; i < size; ++i)
    r[i] = Deserialize<const char *>();
  return r;
}

void Deserializer::CheckSequence(unsigned sequence) {
  if (m_expected_sequence && *m_expected_sequence != sequence)
    llvm::report_fatal_error(
        "The result does not match the preceding "
        "function. This is probably the result of concurrent "
        "use of the SB API during capture, which is currently not "
        "supported.");
  m_expected_sequence.reset();
}

bool Registry::Replay(const FileSpec &file) {
  auto error_or_file = llvm::MemoryBuffer::getFile(file.GetPath());
  if (auto err = error_or_file.getError())
    return false;

  return Replay((*error_or_file)->getBuffer());
}

bool Registry::Replay(llvm::StringRef buffer) {
  Deserializer deserializer(buffer);
  return Replay(deserializer);
}

bool Registry::Replay(Deserializer &deserializer) {
#ifndef LLDB_REPRO_INSTR_TRACE
  Log *log = GetLogIfAllCategoriesSet(LIBLLDB_LOG_API);
#endif

  // Disable buffering stdout so that we approximate the way things get flushed
  // during an interactive session.
  setvbuf(stdout, nullptr, _IONBF, 0);

  while (deserializer.HasData(1)) {
    unsigned sequence = deserializer.Deserialize<unsigned>();
    unsigned id = deserializer.Deserialize<unsigned>();

#ifndef LLDB_REPRO_INSTR_TRACE
    LLDB_LOG(log, "Replaying {0}: {1}", id, GetSignature(id));
#else
    llvm::errs() << "Replaying " << id << ": " << GetSignature(id) << "\n";
#endif

    deserializer.SetExpectedSequence(sequence);
    GetReplayer(id)->operator()(deserializer);
  }

  // Add a small artificial delay to ensure that all asynchronous events have
  // completed before we exit.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  return true;
}

void Registry::DoRegister(uintptr_t RunID, std::unique_ptr<Replayer> replayer,
                          SignatureStr signature) {
  const unsigned id = m_replayers.size() + 1;
  assert(m_replayers.find(RunID) == m_replayers.end());
  m_replayers[RunID] = std::make_pair(std::move(replayer), id);
  m_ids[id] =
      std::make_pair(m_replayers[RunID].first.get(), std::move(signature));
}

unsigned Registry::GetID(uintptr_t addr) {
  unsigned id = m_replayers[addr].second;
  assert(id != 0 && "Forgot to add function to registry?");
  return id;
}

std::string Registry::GetSignature(unsigned id) {
  assert(m_ids.count(id) != 0 && "ID not in registry");
  return m_ids[id].second.ToString();
}

void Registry::CheckID(unsigned expected, unsigned actual) {
  if (expected != actual) {
    llvm::errs() << "Reproducer expected signature " << expected << ": '"
                 << GetSignature(expected) << "'\n";
    llvm::errs() << "Reproducer actual signature " << actual << ": '"
                 << GetSignature(actual) << "'\n";
    llvm::report_fatal_error(
        "Detected reproducer replay divergence. Refusing to continue.");
  }

#ifdef LLDB_REPRO_INSTR_TRACE
  llvm::errs() << "Replaying " << actual << ": " << GetSignature(actual)
               << "\n";
#endif
}

Replayer *Registry::GetReplayer(unsigned id) {
  assert(m_ids.count(id) != 0 && "ID not in registry");
  return m_ids[id].first;
}

std::string Registry::SignatureStr::ToString() const {
  return (result + (result.empty() ? "" : " ") + scope + "::" + name + args)
      .str();
}

unsigned ObjectToIndex::GetIndexForObjectImpl(const void *object) {
  unsigned index = m_mapping.size() + 1;
  auto it = m_mapping.find(object);
  if (it == m_mapping.end())
    m_mapping[object] = index;
  return m_mapping[object];
}

Recorder::Recorder()
    : m_pretty_func(), m_pretty_args(),

      m_sequence(std::numeric_limits<unsigned>::max()) {
  if (!g_global_boundary) {
    g_global_boundary = true;
    m_local_boundary = true;
    m_sequence = GetNextSequenceNumber();
  }
}

Recorder::Recorder(llvm::StringRef pretty_func, std::string &&pretty_args)
    : m_serializer(nullptr), m_pretty_func(pretty_func),
      m_pretty_args(pretty_args), m_local_boundary(false),
      m_result_recorded(true),
      m_sequence(std::numeric_limits<unsigned>::max()) {
  if (!g_global_boundary) {
    g_global_boundary = true;
    m_local_boundary = true;
    m_sequence = GetNextSequenceNumber();
    LLDB_LOG(GetLogIfAllCategoriesSet(LIBLLDB_LOG_API), "{0} ({1})",
             m_pretty_func, m_pretty_args);
  }
}

Recorder::~Recorder() {
  assert(m_result_recorded && "Did you forget LLDB_RECORD_RESULT?");
  UpdateBoundary();
}

unsigned Recorder::GetSequenceNumber() const {
  assert(m_sequence != std::numeric_limits<unsigned>::max());
  return m_sequence;
}

void InstrumentationData::Initialize(Serializer &serializer,
                                     Registry &registry) {
  InstanceImpl().emplace(serializer, registry);
}

void InstrumentationData::Initialize(Deserializer &deserializer,
                                     Registry &registry) {
  InstanceImpl().emplace(deserializer, registry);
}

InstrumentationData &InstrumentationData::Instance() {
  if (!InstanceImpl())
    InstanceImpl().emplace();
  return *InstanceImpl();
}

llvm::Optional<InstrumentationData> &InstrumentationData::InstanceImpl() {
  static llvm::Optional<InstrumentationData> g_instrumentation_data;
  return g_instrumentation_data;
}

thread_local bool lldb_private::repro::Recorder::g_global_boundary = false;
std::atomic<unsigned> lldb_private::repro::Recorder::g_sequence;
std::mutex lldb_private::repro::Recorder::g_mutex;
