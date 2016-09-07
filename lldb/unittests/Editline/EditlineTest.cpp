//===-- EditlineTest.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_DISABLE_LIBEDIT

#define EDITLINE_TEST_DUMP_OUTPUT 0

#include <stdio.h>
#include <unistd.h>

#include <memory>
#include <thread>

#include "gtest/gtest.h"

#include "lldb/Core/Error.h"
#include "lldb/Core/StringList.h"
#include "lldb/Host/Editline.h"
#include "lldb/Host/Pipe.h"
#include "lldb/Utility/PseudoTerminal.h"

namespace {
const size_t TIMEOUT_MILLIS = 5000;
}

class FilePointer {
public:
  FilePointer() = delete;

  FilePointer(const FilePointer &) = delete;

  FilePointer(FILE *file_p) : _file_p(file_p) {}

  ~FilePointer() {
    if (_file_p != nullptr) {
      const int close_result = fclose(_file_p);
      EXPECT_EQ(0, close_result);
    }
  }

  operator FILE *() { return _file_p; }

private:
  FILE *_file_p;
};

/**
 Wraps an Editline class, providing a simple way to feed
 input (as if from the keyboard) and receive output from Editline.
 */
class EditlineAdapter {
public:
  EditlineAdapter();

  void CloseInput();

  bool IsValid() const { return _editline_sp.get() != nullptr; }

  lldb_private::Editline &GetEditline() { return *_editline_sp; }

  bool SendLine(const std::string &line);

  bool SendLines(const std::vector<std::string> &lines);

  bool GetLine(std::string &line, bool &interrupted, size_t timeout_millis);

  bool GetLines(lldb_private::StringList &lines, bool &interrupted,
                size_t timeout_millis);

  void ConsumeAllOutput();

private:
  static bool IsInputComplete(lldb_private::Editline *editline,
                              lldb_private::StringList &lines, void *baton);

  std::unique_ptr<lldb_private::Editline> _editline_sp;

  lldb_utility::PseudoTerminal _pty;
  int _pty_master_fd;
  int _pty_slave_fd;

  std::unique_ptr<FilePointer> _el_slave_file;
};

EditlineAdapter::EditlineAdapter()
    : _editline_sp(), _pty(), _pty_master_fd(-1), _pty_slave_fd(-1),
      _el_slave_file() {
  lldb_private::Error error;

  // Open the first master pty available.
  char error_string[256];
  error_string[0] = '\0';
  if (!_pty.OpenFirstAvailableMaster(O_RDWR, error_string,
                                     sizeof(error_string))) {
    fprintf(stderr, "failed to open first available master pty: '%s'\n",
            error_string);
    return;
  }

  // Grab the master fd.  This is a file descriptor we will:
  // (1) write to when we want to send input to editline.
  // (2) read from when we want to see what editline sends back.
  _pty_master_fd = _pty.GetMasterFileDescriptor();

  // Open the corresponding slave pty.
  if (!_pty.OpenSlave(O_RDWR, error_string, sizeof(error_string))) {
    fprintf(stderr, "failed to open slave pty: '%s'\n", error_string);
    return;
  }
  _pty_slave_fd = _pty.GetSlaveFileDescriptor();

  _el_slave_file.reset(new FilePointer(fdopen(_pty_slave_fd, "rw")));
  EXPECT_FALSE(nullptr == *_el_slave_file);
  if (*_el_slave_file == nullptr)
    return;

  // Create an Editline instance.
  _editline_sp.reset(new lldb_private::Editline("gtest editor", *_el_slave_file,
                                                *_el_slave_file,
                                                *_el_slave_file, false));
  _editline_sp->SetPrompt("> ");

  // Hookup our input complete callback.
  _editline_sp->SetIsInputCompleteCallback(IsInputComplete, this);
}

void EditlineAdapter::CloseInput() {
  if (_el_slave_file != nullptr)
    _el_slave_file.reset(nullptr);
}

bool EditlineAdapter::SendLine(const std::string &line) {
  // Ensure we're valid before proceeding.
  if (!IsValid())
    return false;

  // Write the line out to the pipe connected to editline's input.
  ssize_t input_bytes_written =
      ::write(_pty_master_fd, line.c_str(),
              line.length() * sizeof(std::string::value_type));

  const char *eoln = "\n";
  const size_t eoln_length = strlen(eoln);
  input_bytes_written =
      ::write(_pty_master_fd, eoln, eoln_length * sizeof(char));

  EXPECT_NE(-1, input_bytes_written) << strerror(errno);
  EXPECT_EQ(eoln_length * sizeof(char), size_t(input_bytes_written));
  return eoln_length * sizeof(char) == size_t(input_bytes_written);
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
                                      lldb_private::StringList &lines,
                                      void *baton) {
  // We'll call ourselves complete if we've received a balanced set of braces.
  int start_block_count = 0;
  int brace_balance = 0;

  for (size_t i = 0; i < lines.GetSize(); ++i) {
    for (auto ch : lines[i]) {
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
  FilePointer output_file(fdopen(_pty_master_fd, "r"));

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
private:
  EditlineAdapter _el_adapter;
  std::shared_ptr<std::thread> _sp_output_thread;

public:
  void SetUp() {
    // We need a TERM set properly for editline to work as expected.
    setenv("TERM", "vt100", 1);

    // Validate the editline adapter.
    EXPECT_TRUE(_el_adapter.IsValid());
    if (!_el_adapter.IsValid())
      return;

    // Dump output.
    _sp_output_thread.reset(
        new std::thread([&] { _el_adapter.ConsumeAllOutput(); }));
  }

  void TearDown() {
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
  EXPECT_EQ(input_lines.size(), el_reported_lines.GetSize());
  if (input_lines.size() == el_reported_lines.GetSize()) {
    for (size_t i = 0; i < input_lines.size(); ++i)
      EXPECT_EQ(input_lines[i], el_reported_lines[i]);
  }
}

#endif
