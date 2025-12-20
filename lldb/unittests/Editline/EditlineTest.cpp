//===-- EditlineTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"
#include "lldb/Host/File.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/lldb-forward.h"
#include "llvm/Testing/Support/Error.h"

#if LLDB_ENABLE_LIBEDIT

#define EDITLINE_TEST_DUMP_OUTPUT 0

#include <stdio.h>
#include <unistd.h>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>
#include <thread>

#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Host/Editline.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/PseudoTerminal.h"
#include "lldb/Host/StreamFile.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/StringList.h"

using namespace lldb_private;

namespace {
const size_t TIMEOUT_MILLIS = 5000;
}

/**
 Wraps an Editline class, providing a simple way to feed
 input (as if from the keyboard) and receive output from Editline.
 */
class EditlineAdapter {
public:
  EditlineAdapter();

  void CloseInput();

  bool IsValid() const { return _editline_sp != nullptr; }

  lldb_private::Editline &GetEditline() { return *_editline_sp; }

  bool SendLine(const std::string &line);

  bool SendLines(const std::vector<std::string> &lines);

  bool GetLine(std::string &line, bool &interrupted, size_t timeout_millis);

  bool GetLines(lldb_private::StringList &lines, bool &interrupted,
                size_t timeout_millis);

  void ConsumeAllOutput();

private:
  bool IsInputComplete(lldb_private::Editline *editline,
                       lldb_private::StringList &lines);

  std::recursive_mutex output_mutex;
  std::unique_ptr<lldb_private::Editline> _editline_sp;

  lldb::FileSP _el_primary_file;
  lldb::FileSP _el_secondary_file;
};

EditlineAdapter::EditlineAdapter() : _editline_sp(), _el_secondary_file() {
  lldb_private::Status error;
  PseudoTerminal pty;

  // Open the first primary pty available.
  EXPECT_THAT_ERROR(pty.OpenFirstAvailablePrimary(O_RDWR), llvm::Succeeded());
  // Open the corresponding secondary pty.
  EXPECT_THAT_ERROR(pty.OpenSecondary(O_RDWR), llvm::Succeeded());

  // Grab the primary fd.  This is a file descriptor we will:
  // (1) write to when we want to send input to editline.
  // (2) read from when we want to see what editline sends back.
  _el_primary_file.reset(
      new NativeFile(pty.ReleasePrimaryFileDescriptor(),
                     lldb_private::NativeFile::eOpenOptionReadWrite, true));

  _el_secondary_file.reset(
      new NativeFile(pty.ReleaseSecondaryFileDescriptor(),
                     lldb_private::NativeFile::eOpenOptionReadWrite, true));

  lldb::LockableStreamFileSP output_stream_sp =
      std::make_shared<LockableStreamFile>(_el_secondary_file, output_mutex);
  lldb::LockableStreamFileSP error_stream_sp =
      std::make_shared<LockableStreamFile>(_el_secondary_file, output_mutex);

  // Create an Editline instance.
  _editline_sp.reset(new lldb_private::Editline(
      "gtest editor", _el_secondary_file->GetStream(), output_stream_sp,
      error_stream_sp,
      /*color=*/false));
  _editline_sp->SetPrompt("> ");

  // Hookup our input complete callback.
  auto input_complete_cb = [this](Editline *editline, StringList &lines) {
    return this->IsInputComplete(editline, lines);
  };
  _editline_sp->SetIsInputCompleteCallback(input_complete_cb);
}

void EditlineAdapter::CloseInput() {
  if (_el_secondary_file != nullptr)
    _el_secondary_file->Close();
}

bool EditlineAdapter::SendLine(const std::string &line) {
  // Ensure we're valid before proceeding.
  if (!IsValid())
    return false;

  std::string out = line + "\n";

  // Write the line out to the pipe connected to editline's input.
  size_t num_bytes = out.length() * sizeof(std::string::value_type);
  EXPECT_THAT_ERROR(_el_primary_file->Write(out.c_str(), num_bytes).takeError(),
                    llvm::Succeeded());
  EXPECT_EQ(num_bytes, out.length() * sizeof(std::string::value_type));
  return true;
}

bool EditlineAdapter::SendLines(const std::vector<std::string> &lines) {
  for (auto &line : lines) {
#if EDITLINE_TEST_DUMP_OUTPUT
    printf("<stdin> sending line \"%s\"\n", line.c_str());
#endif
    if (!SendLine(line))
      return false;
  }
  return true;
}

// We ignore the timeout for now.
bool EditlineAdapter::GetLine(std::string &line, bool &interrupted,
                              size_t /* timeout_millis */) {
  // Ensure we're valid before proceeding.
  if (!IsValid())
    return false;

  _editline_sp->GetLine(line, interrupted);
  return true;
}

bool EditlineAdapter::GetLines(lldb_private::StringList &lines,
                               bool &interrupted, size_t /* timeout_millis */) {
  // Ensure we're valid before proceeding.
  if (!IsValid())
    return false;

  _editline_sp->GetLines(1, lines, interrupted);
  return true;
}

bool EditlineAdapter::IsInputComplete(lldb_private::Editline *editline,
                                      lldb_private::StringList &lines) {
  // We'll call ourselves complete if we've received a balanced set of braces.
  int start_block_count = 0;
  int brace_balance = 0;

  for (const std::string &line : lines) {
    for (auto ch : line) {
      if (ch == '{') {
        ++start_block_count;
        ++brace_balance;
      } else if (ch == '}')
        --brace_balance;
    }
  }

  return (start_block_count > 0) && (brace_balance == 0);
}

void EditlineAdapter::ConsumeAllOutput() {
  FILE *output_file = _el_primary_file->GetStream();

  int ch;
  while ((ch = fgetc(output_file)) != EOF) {
#if EDITLINE_TEST_DUMP_OUTPUT
    char display_str[] = {0, 0, 0};
    switch (ch) {
    case '\t':
      display_str[0] = '\\';
      display_str[1] = 't';
      break;
    case '\n':
      display_str[0] = '\\';
      display_str[1] = 'n';
      break;
    case '\r':
      display_str[0] = '\\';
      display_str[1] = 'r';
      break;
    default:
      display_str[0] = ch;
      break;
    }
    printf("<stdout> 0x%02x (%03d) (%s)\n", ch, ch, display_str);
// putc(ch, stdout);
#endif
  }
}

class EditlineTestFixture : public ::testing::Test {
  SubsystemRAII<FileSystem, HostInfo> subsystems;
  EditlineAdapter _el_adapter;
  std::shared_ptr<std::thread> _sp_output_thread;

public:
  static void SetUpTestCase() {
    // We need a TERM set properly for editline to work as expected.
    setenv("TERM", "vt100", 1);
  }

  void SetUp() override {
    // Validate the editline adapter.
    EXPECT_TRUE(_el_adapter.IsValid());
    if (!_el_adapter.IsValid())
      return;

    // Dump output.
    _sp_output_thread =
        std::make_shared<std::thread>([&] { _el_adapter.ConsumeAllOutput(); });
  }

  void TearDown() override {
    _el_adapter.CloseInput();
    if (_sp_output_thread)
      _sp_output_thread->join();
  }

  EditlineAdapter &GetEditlineAdapter() { return _el_adapter; }
};

TEST_F(EditlineTestFixture, EditlineReceivesSingleLineText) {
  // Send it some text via our virtual keyboard.
  const std::string input_text("Hello, world");
  EXPECT_TRUE(GetEditlineAdapter().SendLine(input_text));

  // Verify editline sees what we put in.
  std::string el_reported_line;
  bool input_interrupted = false;
  const bool received_line = GetEditlineAdapter().GetLine(
      el_reported_line, input_interrupted, TIMEOUT_MILLIS);

  EXPECT_TRUE(received_line);
  EXPECT_FALSE(input_interrupted);
  EXPECT_EQ(input_text, el_reported_line);
}

TEST_F(EditlineTestFixture, EditlineReceivesMultiLineText) {
  // Send it some text via our virtual keyboard.
  std::vector<std::string> input_lines;
  input_lines.push_back("int foo()");
  input_lines.push_back("{");
  input_lines.push_back("printf(\"Hello, world\");");
  input_lines.push_back("}");
  input_lines.push_back("");

  EXPECT_TRUE(GetEditlineAdapter().SendLines(input_lines));

  // Verify editline sees what we put in.
  lldb_private::StringList el_reported_lines;
  bool input_interrupted = false;

  EXPECT_TRUE(GetEditlineAdapter().GetLines(el_reported_lines,
                                            input_interrupted, TIMEOUT_MILLIS));
  EXPECT_FALSE(input_interrupted);

  // Without any auto indentation support, our output should directly match our
  // input.
  std::vector<std::string> reported_lines;
  for (const std::string &line : el_reported_lines)
    reported_lines.push_back(line);

  EXPECT_THAT(reported_lines, testing::ContainerEq(input_lines));
}

#endif
