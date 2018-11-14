//===-- Reproducer.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/Reproducer.h"
#include "lldb/Host/HostInfo.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

using namespace lldb_private;
using namespace lldb_private::repro;
using namespace llvm;
using namespace llvm::yaml;

Reproducer &Reproducer::Instance() {
  static Reproducer g_reproducer;
  return g_reproducer;
}

const Generator *Reproducer::GetGenerator() const {
  std::lock_guard<std::mutex> guard(m_mutex);
  if (m_generate_reproducer)
    return &m_generator;
  return nullptr;
}

const Loader *Reproducer::GetLoader() const {
  std::lock_guard<std::mutex> guard(m_mutex);
  if (m_replay_reproducer)
    return &m_loader;
  return nullptr;
}

Generator *Reproducer::GetGenerator() {
  std::lock_guard<std::mutex> guard(m_mutex);
  if (m_generate_reproducer)
    return &m_generator;
  return nullptr;
}

Loader *Reproducer::GetLoader() {
  std::lock_guard<std::mutex> guard(m_mutex);
  if (m_replay_reproducer)
    return &m_loader;
  return nullptr;
}

llvm::Error Reproducer::SetGenerateReproducer(bool value) {
  std::lock_guard<std::mutex> guard(m_mutex);

  if (value && m_replay_reproducer)
    return make_error<StringError>(
        "cannot generate a reproducer when replay one",
        inconvertibleErrorCode());

  m_generate_reproducer = value;
  m_generator.SetEnabled(value);

  return Error::success();
}

llvm::Error Reproducer::SetReplayReproducer(bool value) {
  std::lock_guard<std::mutex> guard(m_mutex);

  if (value && m_generate_reproducer)
    return make_error<StringError>(
        "cannot replay a reproducer when generating one",
        inconvertibleErrorCode());

  m_replay_reproducer = value;

  return Error::success();
}

llvm::Error Reproducer::SetReproducerPath(const FileSpec &path) {
  // Setting the path implies using the reproducer.
  if (auto e = SetReplayReproducer(true))
    return e;

  // Tell the reproducer to load the index form the given path.
  if (auto loader = GetLoader()) {
    if (auto e = loader->LoadIndex(path))
      return e;
  }

  return make_error<StringError>("unable to get loader",
                                 inconvertibleErrorCode());
}

FileSpec Reproducer::GetReproducerPath() const {
  if (auto g = GetGenerator())
    return g->GetDirectory();
  return {};
}

Generator::Generator() : m_enabled(false), m_done(false) {
  m_directory = HostInfo::GetReproducerTempDir();
}

Generator::~Generator() {}

Provider &Generator::Register(std::unique_ptr<Provider> provider) {
  std::lock_guard<std::mutex> lock(m_providers_mutex);

  AddProviderToIndex(provider->GetInfo());

  m_providers.push_back(std::move(provider));
  return *m_providers.back();
}

void Generator::Keep() {
  assert(!m_done);
  m_done = true;

  if (!m_enabled)
    return;

  for (auto &provider : m_providers)
    provider->Keep();
}

void Generator::Discard() {
  assert(!m_done);
  m_done = true;

  if (!m_enabled)
    return;

  for (auto &provider : m_providers)
    provider->Discard();

  llvm::sys::fs::remove_directories(m_directory.GetPath());
}

void Generator::ChangeDirectory(const FileSpec &directory) {
  assert(m_providers.empty() && "Changing the directory after providers have "
                                "been registered would invalidate the index.");
  m_directory = directory;
}

const FileSpec &Generator::GetDirectory() const { return m_directory; }

void Generator::AddProviderToIndex(const ProviderInfo &provider_info) {
  FileSpec index = m_directory;
  index.AppendPathComponent("index.yaml");

  std::error_code EC;
  auto strm = llvm::make_unique<raw_fd_ostream>(index.GetPath(), EC,
                                                sys::fs::OpenFlags::F_None);
  yaml::Output yout(*strm);
  yout << const_cast<ProviderInfo &>(provider_info);
}

Loader::Loader() : m_loaded(false) {}

llvm::Error Loader::LoadIndex(const FileSpec &directory) {
  if (m_loaded)
    return llvm::Error::success();

  FileSpec index = directory.CopyByAppendingPathComponent("index.yaml");

  auto error_or_file = MemoryBuffer::getFile(index.GetPath());
  if (auto err = error_or_file.getError())
    return errorCodeToError(err);

  std::vector<ProviderInfo> provider_info;
  yaml::Input yin((*error_or_file)->getBuffer());
  yin >> provider_info;

  if (auto err = yin.error())
    return errorCodeToError(err);

  for (auto &info : provider_info)
    m_provider_info[info.name] = info;

  m_directory = directory;
  m_loaded = true;

  return llvm::Error::success();
}

llvm::Optional<ProviderInfo> Loader::GetProviderInfo(StringRef name) {
  assert(m_loaded);

  auto it = m_provider_info.find(name);
  if (it == m_provider_info.end())
    return llvm::None;

  return it->second;
}
