//===-- Diagnostics.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/Diagnostics.h"
#include "lldb/Utility/LLDBAssert.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace lldb_private;
using namespace lldb;
using namespace llvm;

void Diagnostics::Initialize() {
  lldbassert(!InstanceImpl() && "Already initialized.");
  InstanceImpl().emplace();
}

void Diagnostics::Terminate() {
  lldbassert(InstanceImpl() && "Already terminated.");
  InstanceImpl().reset();
}

Optional<Diagnostics> &Diagnostics::InstanceImpl() {
  static Optional<Diagnostics> g_diagnostics;
  return g_diagnostics;
}

Diagnostics &Diagnostics::Instance() { return *InstanceImpl(); }

Diagnostics::Diagnostics() {}

Diagnostics::~Diagnostics() {}

void Diagnostics::AddCallback(Callback callback) {
  std::lock_guard<std::mutex> guard(m_callbacks_mutex);
  m_callbacks.push_back(callback);
}

bool Diagnostics::Dump(raw_ostream &stream) {
  Expected<FileSpec> diagnostics_dir = CreateUniqueDirectory();
  if (!diagnostics_dir) {
    stream << "unable to create diagnostic dir: "
           << toString(diagnostics_dir.takeError()) << '\n';
    return false;
  }

  return Dump(stream, *diagnostics_dir);
}

bool Diagnostics::Dump(raw_ostream &stream, const FileSpec &dir) {
  stream << "LLDB diagnostics will be written to " << dir.GetPath() << "\n";
  stream << "Please include the directory content when filing a bug report\n";

  Error error = Create(dir);
  if (error) {
    stream << toString(std::move(error)) << '\n';
    return false;
  }

  return true;
}

llvm::Expected<FileSpec> Diagnostics::CreateUniqueDirectory() {
  SmallString<128> diagnostics_dir;
  std::error_code ec =
      sys::fs::createUniqueDirectory("diagnostics", diagnostics_dir);
  if (ec)
    return errorCodeToError(ec);
  return FileSpec(diagnostics_dir.str());
}

Error Diagnostics::Create(const FileSpec &dir) {
  for (Callback c : m_callbacks) {
    if (Error err = c(dir))
      return err;
  }
  return Error::success();
}
