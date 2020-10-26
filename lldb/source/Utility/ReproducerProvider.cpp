//===-- Reproducer.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/ReproducerProvider.h"
#include "lldb/Utility/ProcessInfo.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace lldb_private;
using namespace lldb_private::repro;
using namespace llvm;
using namespace llvm::yaml;

llvm::Expected<std::unique_ptr<DataRecorder>>
DataRecorder::Create(const FileSpec &filename) {
  std::error_code ec;
  auto recorder = std::make_unique<DataRecorder>(std::move(filename), ec);
  if (ec)
    return llvm::errorCodeToError(ec);
  return std::move(recorder);
}

llvm::Expected<std::unique_ptr<YamlRecorder>>
YamlRecorder::Create(const FileSpec &filename) {
  std::error_code ec;
  auto recorder = std::make_unique<YamlRecorder>(std::move(filename), ec);
  if (ec)
    return llvm::errorCodeToError(ec);
  return std::move(recorder);
}

void VersionProvider::Keep() {
  FileSpec file = GetRoot().CopyByAppendingPathComponent(Info::file);
  std::error_code ec;
  llvm::raw_fd_ostream os(file.GetPath(), ec, llvm::sys::fs::OF_Text);
  if (ec)
    return;
  os << m_version << "\n";
}

void FileProvider::RecordInterestingDirectory(const llvm::Twine &dir) {
  if (m_collector)
    m_collector->addFile(dir);
}

void FileProvider::RecordInterestingDirectoryRecursive(const llvm::Twine &dir) {
  if (m_collector)
    m_collector->addDirectory(dir);
}

llvm::Expected<std::unique_ptr<ProcessInfoRecorder>>
ProcessInfoRecorder::Create(const FileSpec &filename) {
  std::error_code ec;
  auto recorder =
      std::make_unique<ProcessInfoRecorder>(std::move(filename), ec);
  if (ec)
    return llvm::errorCodeToError(ec);
  return std::move(recorder);
}

void ProcessInfoProvider::Keep() {
  std::vector<std::string> files;
  for (auto &recorder : m_process_info_recorders) {
    recorder->Stop();
    files.push_back(recorder->GetFilename().GetPath());
  }

  FileSpec file = GetRoot().CopyByAppendingPathComponent(Info::file);
  std::error_code ec;
  llvm::raw_fd_ostream os(file.GetPath(), ec, llvm::sys::fs::OF_Text);
  if (ec)
    return;
  llvm::yaml::Output yout(os);
  yout << files;
}

void ProcessInfoProvider::Discard() { m_process_info_recorders.clear(); }

ProcessInfoRecorder *ProcessInfoProvider::GetNewProcessInfoRecorder() {
  std::size_t i = m_process_info_recorders.size() + 1;
  std::string filename = (llvm::Twine(Info::name) + llvm::Twine("-") +
                          llvm::Twine(i) + llvm::Twine(".yaml"))
                             .str();
  auto recorder_or_error = ProcessInfoRecorder::Create(
      GetRoot().CopyByAppendingPathComponent(filename));
  if (!recorder_or_error) {
    llvm::consumeError(recorder_or_error.takeError());
    return nullptr;
  }

  m_process_info_recorders.push_back(std::move(*recorder_or_error));
  return m_process_info_recorders.back().get();
}

void ProcessInfoRecorder::Record(const ProcessInstanceInfoList &process_infos) {
  if (!m_record)
    return;
  llvm::yaml::Output yout(m_os);
  yout << const_cast<ProcessInstanceInfoList &>(process_infos);
  m_os.flush();
}

void SymbolFileProvider::AddSymbolFile(const UUID *uuid,
                                       const FileSpec &module_file,
                                       const FileSpec &symbol_file) {
  if (!uuid || (!module_file && !symbol_file))
    return;
  m_symbol_files.emplace_back(uuid->GetAsString(), module_file.GetPath(),
                              symbol_file.GetPath());
}

void SymbolFileProvider::Keep() {
  FileSpec file = this->GetRoot().CopyByAppendingPathComponent(Info::file);
  std::error_code ec;
  llvm::raw_fd_ostream os(file.GetPath(), ec, llvm::sys::fs::OF_Text);
  if (ec)
    return;

  // Remove duplicates.
  llvm::sort(m_symbol_files.begin(), m_symbol_files.end());
  m_symbol_files.erase(
      std::unique(m_symbol_files.begin(), m_symbol_files.end()),
      m_symbol_files.end());

  llvm::yaml::Output yout(os);
  yout << m_symbol_files;
}

SymbolFileLoader::SymbolFileLoader(Loader *loader) {
  if (!loader)
    return;

  FileSpec file = loader->GetFile<SymbolFileProvider::Info>();
  if (!file)
    return;

  auto error_or_file = llvm::MemoryBuffer::getFile(file.GetPath());
  if (auto err = error_or_file.getError())
    return;

  llvm::yaml::Input yin((*error_or_file)->getBuffer());
  yin >> m_symbol_files;
}

std::pair<FileSpec, FileSpec>
SymbolFileLoader::GetPaths(const UUID *uuid) const {
  if (!uuid)
    return {};

  auto it = std::lower_bound(m_symbol_files.begin(), m_symbol_files.end(),
                             SymbolFileProvider::Entry(uuid->GetAsString()));
  if (it == m_symbol_files.end())
    return {};
  return std::make_pair<FileSpec, FileSpec>(FileSpec(it->module_path),
                                            FileSpec(it->symbol_path));
}

void ProviderBase::anchor() {}
char CommandProvider::ID = 0;
char FileProvider::ID = 0;
char ProviderBase::ID = 0;
char VersionProvider::ID = 0;
char WorkingDirectoryProvider::ID = 0;
char HomeDirectoryProvider::ID = 0;
char ProcessInfoProvider::ID = 0;
char SymbolFileProvider::ID = 0;
const char *CommandProvider::Info::file = "command-interpreter.yaml";
const char *CommandProvider::Info::name = "command-interpreter";
const char *FileProvider::Info::file = "files.yaml";
const char *FileProvider::Info::name = "files";
const char *VersionProvider::Info::file = "version.txt";
const char *VersionProvider::Info::name = "version";
const char *WorkingDirectoryProvider::Info::file = "cwd.txt";
const char *WorkingDirectoryProvider::Info::name = "cwd";
const char *HomeDirectoryProvider::Info::file = "home.txt";
const char *HomeDirectoryProvider::Info::name = "home";
const char *ProcessInfoProvider::Info::file = "process-info.yaml";
const char *ProcessInfoProvider::Info::name = "process-info";
const char *SymbolFileProvider::Info::file = "symbol-files.yaml";
const char *SymbolFileProvider::Info::name = "symbol-files";
