//===-- SBCommandInterpreterTest.cpp ------------------------===----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

// Use the umbrella header for -Wdocumentation.
#include "lldb/API/LLDB.h"

#include "TestingSupport/SubsystemRAII.h"
#include "lldb/API/SBDebugger.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <fcntl.h>
#include <poll.h>
#include <stdlib.h>
#include <unistd.h>
#endif

using namespace lldb;
using namespace lldb_private;

class SBCommandInterpreterTest : public testing::Test {
protected:
  void SetUp() override {
    debugger = SBDebugger::Create(/*source_init_files=*/false);
  }

  void TearDown() override { SBDebugger::Destroy(debugger); }

  SubsystemRAII<lldb::SBDebugger> subsystems;
  SBDebugger debugger;
};

class DummyCommand : public SBCommandPluginInterface {
public:
  DummyCommand(const char *message) : m_message(message) {}

  bool DoExecute(SBDebugger dbg, char **command,
                 SBCommandReturnObject &result) override {
    result.PutCString(m_message.c_str());
    result.SetStatus(eReturnStatusSuccessFinishResult);
    return result.Succeeded();
  }

private:
  std::string m_message;
};

TEST_F(SBCommandInterpreterTest, SingleWordCommand) {
  // We first test a command without autorepeat
  DummyCommand dummy("It worked");
  SBCommandInterpreter interp = debugger.GetCommandInterpreter();
  interp.AddCommand("dummy", &dummy, /*help=*/nullptr);
  {
    SBCommandReturnObject result;
    interp.HandleCommand("dummy", result, /*add_to_history=*/true);
    EXPECT_TRUE(result.Succeeded());
    EXPECT_STREQ(result.GetOutput(), "It worked\n");
  }
  {
    SBCommandReturnObject result;
    interp.HandleCommand("", result);
    EXPECT_FALSE(result.Succeeded());
    EXPECT_STREQ(result.GetError(), "error: no auto repeat\n");
  }

  // Now we test a command with autorepeat
  interp.AddCommand("dummy_with_autorepeat", &dummy, /*help=*/nullptr,
                    /*syntax=*/nullptr, /*auto_repeat_command=*/nullptr);
  {
    SBCommandReturnObject result;
    interp.HandleCommand("dummy_with_autorepeat", result,
                         /*add_to_history=*/true);
    EXPECT_TRUE(result.Succeeded());
    EXPECT_STREQ(result.GetOutput(), "It worked\n");
  }
  {
    SBCommandReturnObject result;
    interp.HandleCommand("", result);
    EXPECT_TRUE(result.Succeeded());
    EXPECT_STREQ(result.GetOutput(), "It worked\n");
  }
}

TEST_F(SBCommandInterpreterTest, MultiWordCommand) {
  SBCommandInterpreter interp = debugger.GetCommandInterpreter();
  auto command = interp.AddMultiwordCommand("multicommand", /*help=*/nullptr);
  // We first test a subcommand without autorepeat
  DummyCommand subcommand("It worked again");
  command.AddCommand("subcommand", &subcommand, /*help=*/nullptr);
  {
    SBCommandReturnObject result;
    interp.HandleCommand("multicommand subcommand", result,
                         /*add_to_history=*/true);
    EXPECT_TRUE(result.Succeeded());
    EXPECT_STREQ(result.GetOutput(), "It worked again\n");
  }
  {
    SBCommandReturnObject result;
    interp.HandleCommand("", result);
    EXPECT_FALSE(result.Succeeded());
    EXPECT_STREQ(result.GetError(), "error: no auto repeat\n");
  }

  // We first test a subcommand with autorepeat
  command.AddCommand("subcommand_with_autorepeat", &subcommand,
                     /*help=*/nullptr, /*syntax=*/nullptr,
                     /*auto_repeat_command=*/nullptr);
  {
    SBCommandReturnObject result;
    interp.HandleCommand("multicommand subcommand_with_autorepeat", result,
                         /*add_to_history=*/true);
    EXPECT_TRUE(result.Succeeded());
    EXPECT_STREQ(result.GetOutput(), "It worked again\n");
  }
  {
    SBCommandReturnObject result;
    interp.HandleCommand("", result);
    EXPECT_TRUE(result.Succeeded());
    EXPECT_STREQ(result.GetOutput(), "It worked again\n");
  }

  DummyCommand subcommand2("It worked again 2");
  // We now test a subcommand with autorepeat of the command name
  command.AddCommand(
      "subcommand_with_custom_autorepeat", &subcommand2, /*help=*/nullptr,
      /*syntax=*/nullptr,
      /*auto_repeat_command=*/"multicommand subcommand_with_autorepeat");
  {
    SBCommandReturnObject result;
    interp.HandleCommand("multicommand subcommand_with_custom_autorepeat",
                         result, /*add_to_history=*/true);
    EXPECT_TRUE(result.Succeeded());
    EXPECT_STREQ(result.GetOutput(), "It worked again 2\n");
  }
  {
    SBCommandReturnObject result;
    interp.HandleCommand("", result);
    EXPECT_TRUE(result.Succeeded());
    EXPECT_STREQ(result.GetOutput(), "It worked again\n");
  }
}

namespace {
#ifdef _WIN32
constexpr const char *kDevNull = "nul";
#else
constexpr const char *kDevNull = "/dev/null";
#endif

class CountingCommand : public SBCommandPluginInterface {
public:
  bool DoExecute(SBDebugger dbg, char **command,
                 SBCommandReturnObject &result) override {
    ++count;
    result.SetStatus(eReturnStatusSuccessFinishNoResult);
    return true;
  }

  std::atomic<int> count{0};
};

struct PipeSession {
  int read_fd = -1;
  int write_fd = -1;
  SBListener listener{"test.interpreter.listener"};

  bool Init(SBDebugger &debugger) {
    int fds[2] = {-1, -1};
#ifdef _WIN32
    if (::_pipe(fds, 4096, _O_BINARY) != 0)
      return false;
    read_fd = fds[0];
    write_fd = fds[1];
#if !defined(_DLL)
    int crt_read_fd =
        SBFile::OpenFdFromHandle(::_get_osfhandle(read_fd), _O_RDONLY);
    ::_close(read_fd);
    read_fd = crt_read_fd;
#endif
#else
    if (::pipe(fds) != 0)
      return false;
    read_fd = fds[0];
    write_fd = fds[1];
#endif

    FILE *out_null = std::fopen(kDevNull, "w");
    if (!out_null)
      return false;
    FILE *err_null = std::fopen(kDevNull, "w");
    if (!err_null) {
      std::fclose(out_null);
      return false;
    }

    debugger.SetInputFile(SBFile(read_fd, "r", /*transfer_ownership=*/true));
    read_fd = -1;
    debugger.SetOutputFile(SBFile(out_null, "w", /*transfer_ownership=*/true));
    debugger.SetErrorFile(SBFile(err_null, "w", /*transfer_ownership=*/true));

    listener.StartListeningForEvents(
        debugger.GetCommandInterpreter().GetBroadcaster(),
        SBCommandInterpreter::eBroadcastBitReadyForInput |
            SBCommandInterpreter::eBroadcastBitQuitCommandReceived);
    return true;
  }

  void Write(llvm::StringRef data) {
    ASSERT_NE(write_fd, -1);
#ifdef _WIN32
    int written =
        ::_write(write_fd, data.data(), static_cast<unsigned int>(data.size()));
#else
    ssize_t written = ::write(write_fd, data.data(), data.size());
#endif
    ASSERT_EQ(written, static_cast<decltype(written)>(data.size()));
  }

  void CloseWrite() {
    if (write_fd != -1) {
#ifdef _WIN32
      ::_close(write_fd);
#else
      ::close(write_fd);
#endif
      write_fd = -1;
    }
  }

  uint32_t WaitForEventBit(uint32_t mask, uint32_t timeout_sec = 5,
                           SBEvent *out_event = nullptr) {
    while (true) {
      SBEvent event;
      if (!listener.WaitForEvent(timeout_sec, event))
        return 0;
      const uint32_t type = event.GetType();
      if (type & mask) {
        if (out_event)
          *out_event = event;
        return type;
      }
    }
  }

  ~PipeSession() {
    CloseWrite();
    if (read_fd != -1) {
#ifdef _WIN32
      ::_close(read_fd);
#else
      ::close(read_fd);
#endif
    }
  }
};

SBCommandInterpreterRunOptions MakeSpawnedOptions() {
  SBCommandInterpreterRunOptions options;
  options.SetSpawnThread(true);
  options.SetEchoCommands(false);
  options.SetEchoCommentCommands(false);
  options.SetPrintResults(false);
  options.SetPrintErrors(false);
  return options;
}
} // namespace

TEST_F(SBCommandInterpreterTest, ReadyForInputBroadcastOnPipe) {
  PipeSession session;
  ASSERT_TRUE(session.Init(debugger));

  CountingCommand counting_cmd;
  SBCommandInterpreter interp = debugger.GetCommandInterpreter();
  interp.AddCommand("count_cmd", &counting_cmd, /*help=*/nullptr);

  debugger.RunCommandInterpreter(MakeSpawnedOptions());
  auto cleanup = llvm::scope_exit([&] { session.CloseWrite(); });

  // Initial prompt wait for broadcast eBroadcastBitReadyForInput with event
  // name "ready-for-input".
  SBEvent ready_event;
  ASSERT_EQ(
      session.WaitForEventBit(SBCommandInterpreter::eBroadcastBitReadyForInput,
                              5, &ready_event),
      static_cast<uint32_t>(SBCommandInterpreter::eBroadcastBitReadyForInput));
  SBStream desc;
  ASSERT_TRUE(ready_event.GetDescription(desc));
  EXPECT_NE(
      llvm::StringRef(desc.GetData(), desc.GetSize()).find("ready-for-input"),
      llvm::StringRef::npos);

  // Executing a single line runs the command and broadcasts
  // eBroadcastBitReadyForInput again when waiting for the next line.
  session.Write("count_cmd\n");
  ASSERT_EQ(
      session.WaitForEventBit(SBCommandInterpreter::eBroadcastBitReadyForInput),
      static_cast<uint32_t>(SBCommandInterpreter::eBroadcastBitReadyForInput));
  EXPECT_EQ(counting_cmd.count.load(), 1);

  // Writing two lines in a single write executes both commands, emitting
  // eBroadcastBitReadyForInput before each line is read.
  session.Write("count_cmd\ncount_cmd\n");
  ASSERT_EQ(
      session.WaitForEventBit(SBCommandInterpreter::eBroadcastBitReadyForInput),
      static_cast<uint32_t>(SBCommandInterpreter::eBroadcastBitReadyForInput));
  ASSERT_EQ(
      session.WaitForEventBit(SBCommandInterpreter::eBroadcastBitReadyForInput),
      static_cast<uint32_t>(SBCommandInterpreter::eBroadcastBitReadyForInput));
  EXPECT_EQ(counting_cmd.count.load(), 3);
  SBEvent extra_event;
  EXPECT_FALSE(session.listener.GetNextEvent(extra_event));

  // Sourcing commands from a separate file (`m_input_sp != GetInputFileSP()`)
  // should not broadcast eBroadcastBitReadyForInput for lines inside the
  // sourced file.
  llvm::Expected<llvm::sys::fs::TempFile> temp_file =
      llvm::sys::fs::TempFile::create("lldb-cmd-source-%%%%%%.lldb");
  ASSERT_THAT_EXPECTED(temp_file, llvm::Succeeded());
  auto discard_temp =
      llvm::scope_exit([&] { llvm::consumeError(temp_file->discard()); });
  {
    llvm::raw_fd_ostream os(temp_file->FD, /*shouldClose=*/false);
    os << "count_cmd\ncount_cmd\n";
  }

  std::string source_cmd = "command source \"" + temp_file->TmpName + "\"\n";
  session.Write(source_cmd);
  ASSERT_EQ(
      session.WaitForEventBit(SBCommandInterpreter::eBroadcastBitReadyForInput),
      static_cast<uint32_t>(SBCommandInterpreter::eBroadcastBitReadyForInput));
  EXPECT_EQ(counting_cmd.count.load(), 5);
  EXPECT_FALSE(session.listener.GetNextEvent(extra_event));

  // Quitting broadcasts QuitCommandReceived.
  session.Write("quit\n");
  EXPECT_EQ(session.WaitForEventBit(
                SBCommandInterpreter::eBroadcastBitQuitCommandReceived),
            static_cast<uint32_t>(
                SBCommandInterpreter::eBroadcastBitQuitCommandReceived));
  session.CloseWrite();
}

TEST_F(SBCommandInterpreterTest, ReadyForInputMultilineSubIOHandler) {
  PipeSession session;
  ASSERT_TRUE(session.Init(debugger));

  CountingCommand counting_cmd;
  SBCommandInterpreter interp = debugger.GetCommandInterpreter();
  interp.AddCommand("count_cmd", &counting_cmd, /*help=*/nullptr);

  debugger.RunCommandInterpreter(MakeSpawnedOptions());
  auto cleanup = llvm::scope_exit([&] { session.CloseWrite(); });

  ASSERT_EQ(
      session.WaitForEventBit(SBCommandInterpreter::eBroadcastBitReadyForInput),
      static_cast<uint32_t>(SBCommandInterpreter::eBroadcastBitReadyForInput));

  // Create a dummy-target breakpoint and enter multiline `breakpoint command
  // add`.
  SBBreakpoint bp = debugger.GetDummyTarget().BreakpointCreateByName("main");
  ASSERT_TRUE(bp.IsValid());

  std::string add_cmd = "breakpoint command add -D -s command " +
                        std::to_string(bp.GetID()) + "\n";
  session.Write(add_cmd);
  ASSERT_EQ(
      session.WaitForEventBit(SBCommandInterpreter::eBroadcastBitReadyForInput),
      static_cast<uint32_t>(SBCommandInterpreter::eBroadcastBitReadyForInput));

  // Sub-IOHandlerEditline prompts for subsequent lines using the debugger input
  // stream without executing `count_cmd` immediately.
  session.Write("count_cmd\n");
  ASSERT_EQ(
      session.WaitForEventBit(SBCommandInterpreter::eBroadcastBitReadyForInput),
      static_cast<uint32_t>(SBCommandInterpreter::eBroadcastBitReadyForInput));
  EXPECT_EQ(counting_cmd.count.load(), 0);

  // `DONE` exits the sub-IOHandler and returns to the main interpreter prompt.
  session.Write("DONE\n");
  ASSERT_EQ(
      session.WaitForEventBit(SBCommandInterpreter::eBroadcastBitReadyForInput),
      static_cast<uint32_t>(SBCommandInterpreter::eBroadcastBitReadyForInput));

  session.Write("quit\n");
  ASSERT_EQ(session.WaitForEventBit(
                SBCommandInterpreter::eBroadcastBitQuitCommandReceived),
            static_cast<uint32_t>(
                SBCommandInterpreter::eBroadcastBitQuitCommandReceived));
  session.CloseWrite();

  SBStringList commands;
  ASSERT_TRUE(bp.GetCommandLineCommands(commands));
  ASSERT_EQ(commands.GetSize(), 1u);
  EXPECT_STREQ(commands.GetStringAtIndex(0), "count_cmd");
}

#ifndef _WIN32
TEST_F(SBCommandInterpreterTest, ReadyForInputEditlinePty) {
  int primary_fd = ::posix_openpt(O_RDWR | O_NOCTTY);
  if (primary_fd < 0)
    GTEST_SKIP() << "posix_openpt unavailable";
  auto close_primary = llvm::scope_exit([&] {
    if (primary_fd != -1)
      ::close(primary_fd);
  });

  ASSERT_EQ(::grantpt(primary_fd), 0);
  ASSERT_EQ(::unlockpt(primary_fd), 0);
  const char *secondary_name = ::ptsname(primary_fd);
  ASSERT_NE(secondary_name, nullptr);
  int secondary_fd = ::open(secondary_name, O_RDWR | O_NOCTTY);
  ASSERT_GE(secondary_fd, 0);

  debugger.SetInputFile(SBFile(secondary_fd, "r", /*transfer_ownership=*/true));
  int out_fd = ::dup(secondary_fd);
  ASSERT_GE(out_fd, 0);
  int err_fd = ::dup(secondary_fd);
  ASSERT_GE(err_fd, 0);
  debugger.SetOutputFile(SBFile(out_fd, "w", /*transfer_ownership=*/true));
  debugger.SetErrorFile(SBFile(err_fd, "w", /*transfer_ownership=*/true));

  // Drain PTY output so editline writes never block.
  std::atomic<bool> stop_drain{false};
  std::thread drain_thread([fd = primary_fd, &stop_drain] {
    char buf[256];
    while (!stop_drain.load()) {
      struct pollfd pfd = {fd, POLLIN, 0};
      if (::poll(&pfd, 1, 10) > 0 && (pfd.revents & POLLIN)) {
        if (::read(fd, buf, sizeof(buf)) <= 0)
          break;
      }
    }
  });
  auto join_drain = llvm::scope_exit([&] {
    stop_drain.store(true);
    if (drain_thread.joinable())
      drain_thread.join();
  });

  CountingCommand counting_cmd;
  SBCommandInterpreter interp = debugger.GetCommandInterpreter();
  interp.AddCommand("count_cmd", &counting_cmd, /*help=*/nullptr);

  SBListener listener("test.pty.listener");
  listener.StartListeningForEvents(
      interp.GetBroadcaster(),
      SBCommandInterpreter::eBroadcastBitReadyForInput |
          SBCommandInterpreter::eBroadcastBitQuitCommandReceived);

  auto wait_for_bit = [&](uint32_t mask) -> uint32_t {
    while (true) {
      SBEvent event;
      if (!listener.WaitForEvent(5, event))
        return 0;
      if (event.GetType() & mask)
        return event.GetType();
    }
  };

  debugger.RunCommandInterpreter(MakeSpawnedOptions());

  ASSERT_EQ(
      wait_for_bit(SBCommandInterpreter::eBroadcastBitReadyForInput),
      static_cast<uint32_t>(SBCommandInterpreter::eBroadcastBitReadyForInput));

  llvm::StringRef cmd = "count_cmd\n";
  ASSERT_EQ(::write(primary_fd, cmd.data(), cmd.size()),
            static_cast<ssize_t>(cmd.size()));
  ASSERT_EQ(
      wait_for_bit(SBCommandInterpreter::eBroadcastBitReadyForInput),
      static_cast<uint32_t>(SBCommandInterpreter::eBroadcastBitReadyForInput));
  EXPECT_EQ(counting_cmd.count.load(), 1);

  llvm::StringRef quit_cmd = "quit\n";
  ASSERT_EQ(::write(primary_fd, quit_cmd.data(), quit_cmd.size()),
            static_cast<ssize_t>(quit_cmd.size()));
  EXPECT_EQ(
      wait_for_bit(SBCommandInterpreter::eBroadcastBitQuitCommandReceived),
      static_cast<uint32_t>(
          SBCommandInterpreter::eBroadcastBitQuitCommandReceived));
}
#endif

TEST_F(SBCommandInterpreterTest, ReadyForInputEmbeddedPythonInterpreter) {
  PipeSession session;
  ASSERT_TRUE(session.Init(debugger));

  SBCommandInterpreter interp = debugger.GetCommandInterpreter();
  {
    SBCommandReturnObject probe;
    interp.HandleCommand("script -l python -- 1 + 1", probe);
    if (!probe.Succeeded())
      GTEST_SKIP() << "Python script interpreter unavailable";
  }

  CountingCommand counting_cmd;
  interp.AddCommand("count_cmd", &counting_cmd, /*help=*/nullptr);

  debugger.RunCommandInterpreter(MakeSpawnedOptions());
  auto cleanup = llvm::scope_exit([&] { session.CloseWrite(); });

  ASSERT_EQ(
      session.WaitForEventBit(SBCommandInterpreter::eBroadcastBitReadyForInput),
      static_cast<uint32_t>(SBCommandInterpreter::eBroadcastBitReadyForInput));

  // Enter interactive `script` REPL and verify eBroadcastBitReadyForInput is
  // emitted on entry, across multiline continuation prompts, and after exiting
  // back to the command interpreter.
  session.Write("script -l python --\n");
  ASSERT_EQ(
      session.WaitForEventBit(SBCommandInterpreter::eBroadcastBitReadyForInput),
      static_cast<uint32_t>(SBCommandInterpreter::eBroadcastBitReadyForInput));

  session.Write("total = 0\n");
  ASSERT_EQ(
      session.WaitForEventBit(SBCommandInterpreter::eBroadcastBitReadyForInput),
      static_cast<uint32_t>(SBCommandInterpreter::eBroadcastBitReadyForInput));

  session.Write("for i in range(3):\n");
  ASSERT_EQ(
      session.WaitForEventBit(SBCommandInterpreter::eBroadcastBitReadyForInput),
      static_cast<uint32_t>(SBCommandInterpreter::eBroadcastBitReadyForInput));

  session.Write("    total += i\n");
  ASSERT_EQ(
      session.WaitForEventBit(SBCommandInterpreter::eBroadcastBitReadyForInput),
      static_cast<uint32_t>(SBCommandInterpreter::eBroadcastBitReadyForInput));

  session.Write("\n");
  ASSERT_EQ(
      session.WaitForEventBit(SBCommandInterpreter::eBroadcastBitReadyForInput),
      static_cast<uint32_t>(SBCommandInterpreter::eBroadcastBitReadyForInput));

  session.Write("quit()\n");
  ASSERT_EQ(
      session.WaitForEventBit(SBCommandInterpreter::eBroadcastBitReadyForInput),
      static_cast<uint32_t>(SBCommandInterpreter::eBroadcastBitReadyForInput));

  // Confirm we returned to the LLDB CommandInterpreter.
  session.Write("count_cmd\n");
  ASSERT_EQ(
      session.WaitForEventBit(SBCommandInterpreter::eBroadcastBitReadyForInput),
      static_cast<uint32_t>(SBCommandInterpreter::eBroadcastBitReadyForInput));
  EXPECT_EQ(counting_cmd.count.load(), 1);

  session.Write("quit\n");
  ASSERT_EQ(session.WaitForEventBit(
                SBCommandInterpreter::eBroadcastBitQuitCommandReceived),
            static_cast<uint32_t>(
                SBCommandInterpreter::eBroadcastBitQuitCommandReceived));
  session.CloseWrite();

  SBCommandReturnObject check;
  interp.HandleCommand("script -l python -- print(total)", check);
  ASSERT_TRUE(check.Succeeded());
  EXPECT_STREQ(check.GetOutput(), "3\n");
}
