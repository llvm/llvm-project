//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the interface for tools to be run in "daemon mode",
/// following the IPC protocol as described in docs/DaemonMode.rst.
///
//===----------------------------------------------------------------------===//

#include "llvm/Support/DaemonDriver.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <system_error>

#if defined _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace llvm;

// Windows emulated POSIX functions are prefixed by `_`.
#ifdef _WIN32
#define CLOSE_FN ::_close
#define DUP_FN ::_dup
#define DUP2_FN ::_dup2
#else
#define CLOSE_FN ::close
#define DUP_FN ::dup
#define DUP2_FN ::dup2
#endif

// Standard stream fileno macros are not defined on Windows.
#ifndef STDIN_FILENO
#define STDIN_FILENO 0
#endif
#ifndef STDOUT_FILENO
#define STDOUT_FILENO 1
#endif
#ifndef STDERR_FILENO
#define STDERR_FILENO 2
#endif

namespace {
/// RAII mechanism to redirect a file descriptor to the file pointed to by a
/// different file descriptor. The destructor will reset the file descriptor to
/// its original file.
class ScopedFileRedirect {
public:
  /// Redirect `FromFd` to the same file as `ToFd` for the lifetime of this
  /// object.
  ScopedFileRedirect(int FromFd, int ToFd) : FromFd(FromFd) {
    // Create a duplicate FD pointing to the original file so that we can
    // restore the original FD to the file pointed to by the duplicate.
    DuplicateFd = DUP_FN(FromFd);

    // Close the source FD and reopen it to the target FD.
    DUP2_FN(ToFd, FromFd);
  }

  ~ScopedFileRedirect() {
    // Close the source FD and reopen it to the original file.
    DUP2_FN(DuplicateFd, FromFd);

    // Close the duplicate file descriptor, as it's no longer needed.
    CLOSE_FN(DuplicateFd);
  }

  ScopedFileRedirect(const ScopedFileRedirect &) = delete;
  ScopedFileRedirect &operator=(const ScopedFileRedirect &) = delete;
  ScopedFileRedirect(ScopedFileRedirect &&) = delete;
  ScopedFileRedirect &operator=(const ScopedFileRedirect &&) = delete;

private:
  int FromFd;
  int DuplicateFd;
};

static ErrorOr<std::string> readNextLine(FILE *File) {
  std::string Result;
  raw_string_ostream ResultOS(Result);

  constexpr size_t BufSize = 512;
  char Buf[BufSize];

  // Read from the file, appending to the result, until a new line character
  // is found.
  while (std::fgets(Buf, BufSize, File)) {
    ResultOS << Buf;

    if (std::strchr(Buf, '\n'))
      break;
  }

  if (std::ferror(File))
    return std::make_error_code(static_cast<std::errc>(errno));

  return Result;
}

/// Status code returned if the daemon fails to initialize, for example due to
/// incorrect command line arguments.
constexpr int StatusInitError = 2;
/// Status code returned if the daemon receives a malformed command.
constexpr int StatusCommandError = 3;

/// This should only be used before the status pipe is set up - after,
/// errors are reported to the user via the status pipe.
[[noreturn]] static void reportInitError(const Twine &Err) {
  errs() << "[daemon] Error: " << Err << "\n";
  std::exit(StatusInitError);
}

/// Returns true if ``--daemon`` is passed as the first command line argument.
static bool detectDaemonArg(int Argc, char **Argv) {
  // `--daemon` must be the first argument.
  return Argc >= 2 && Argv[1] == StringRef("--daemon");
}

struct DaemonCommandLineOptions {
  bool DaemonModeEnabled;
  std::string StatusPipe;
};

// Creates the command line options for configuring the daemon. The returned
// reference points to a static struct where the option values will be stored.
static const DaemonCommandLineOptions &initializeDaemonCommandLineOptions() {
  // This is declared as a static local variables in this function rather than
  // a static global variable as is common in LLVM to avoid adding global
  // constructors and destructors to the program, which is forbidden in the
  // Support library.
  static DaemonCommandLineOptions Options;

  static cl::opt<bool, true> DaemonModeEnabledOpt(
      "daemon", cl::location(Options.DaemonModeEnabled), cl::init(false));

  static cl::opt<std::string, true> StatusPipeOpt(
      "daemon-status-pipe",
      cl::desc("File to which the daemon tool will send status messages. May "
               "be 'path:{filepath}', 'fd:{file descriptor}' or "
               "'handle:{Windows file handle}'"),
      cl::location(Options.StatusPipe), cl::init(""));

  return Options;
}

/// Command sent to invoke the tool.
struct RunCommand {
  static constexpr StringRef Prefix = "run";

  static Expected<RunCommand> parse(StringRef Remaining) {
    Remaining = Remaining.trim();

    if (!Remaining.empty())
      return createStringError("Unexpected trailing characters in command");

    return RunCommand();
  }
};

/// Command providing a command line argument for the tool as a framed string.
struct ArgCommand {
  static constexpr StringRef Prefix = "arg";

  static Expected<ArgCommand> parse(StringRef Remaining) {
    Remaining = Remaining.trim();

    size_t ExpectedLength;
    if (Remaining.consumeInteger(10, ExpectedLength) || ExpectedLength < 0)
      return createStringError("Expected non-negative integer");

    if (!Remaining.empty())
      return createStringError("Unexpected trailing characters in command");

    return ArgCommand{ExpectedLength};
  }

  size_t ExpectedLength;
};

// Command providing a framed string to use as standard input for the tool.
struct InputStringCommand {
  static constexpr StringRef Prefix = "input_string";

  static Expected<InputStringCommand> parse(StringRef Remaining) {
    Remaining = Remaining.trim();

    size_t ExpectedLength;
    if (Remaining.consumeInteger(10, ExpectedLength) || ExpectedLength < 0)
      return createStringError("Expected non-negative integer");

    if (!Remaining.trim().empty())
      return createStringError("Unexpected trailing characters in command");

    return InputStringCommand{ExpectedLength};
  }

  size_t ExpectedLength;
};

// Command providing a file path whose contents will be used as the standard
// input for the tool.
struct InputFileCommand {
  static constexpr StringRef Prefix = "input_file";

  static Expected<InputFileCommand> parse(StringRef Remaining) {
    Remaining = Remaining.trim();

    return InputFileCommand{Remaining};
  }

  StringRef Path;
};

// Command changing the current working directory for the daemon.
struct ChangeDirectoryCommand {
  static constexpr StringRef Prefix = "cd";

  static Expected<ChangeDirectoryCommand> parse(StringRef Remaining) {
    Remaining = Remaining.trim();

    return ChangeDirectoryCommand{Remaining};
  }

  StringRef Path;
};

/// Command telling the daemon to send the tool's standard error content through
/// its standard output stream, to maintain the ordering between stderr and
/// stdout.
struct RedirectStderrToStdoutCommand {
  static constexpr StringRef Prefix = "redirect_stderr_to_stdout";

  static Expected<RedirectStderrToStdoutCommand> parse(StringRef Remaining) {
    Remaining = Remaining.trim();

    if (!Remaining.empty())
      return createStringError("Unexpected trailing characters in command");

    return RedirectStderrToStdoutCommand();
  }
};

/// Command telling the daemon to exit.
struct ExitCommand {
  static constexpr StringRef Prefix = "exit";

  static Expected<ExitCommand> parse(StringRef Remaining) {
    Remaining = Remaining.trim();

    if (!Remaining.empty())
      return createStringError("Unexpected trailing characters in command");

    return ExitCommand();
  }
};

/// State to be configured for the next invocation.
struct InvocationOptions {
  void reset() { *this = InvocationOptions(); }

  /// String which tool shall pretend is the contents of stdin.
  std::vector<std::string> Args;
  StandardInputSource InputSource = StandardInputSource::fromString("");
  bool RedirectStderrToStdout = false;
};

/// This class implements the daemon driver functionality.
class DaemonDriver {
public:
  DaemonDriver(LLVMTool &Tool, const DaemonCommandLineOptions &Options)
      : Tool(Tool), StatusPipeOS(createStatusPipeOS(Options.StatusPipe)),
        OriginalWorkingDirectory(std::filesystem::current_path()),
        WorkingDirectoryChanged(false) {};

  int run() {
    // Ensure stdin is in binary mode to prevent newline translation on Windows
    // - this not only breaks binary input but also muddles the number of
    // characters read.
    if (const std::error_code EC = sys::ChangeStdinToBinary())
      exitWithError(Twine("Couldn't switch stdin to binary mode: ") +
                    EC.message());

    // Inform the user that the daemon is ready to receive commands.
    respondReady();

    while (!feof(stdin)) {
      const ErrorOr<std::string> Command = readNextLine(stdin);
      if (!Command) {
        exitWithError(Twine("Error reading standard input: ") +
                      Command.getError().message());
      }

      // The command is parsed from left to right, with surrounding whitespace
      // ignored.
      StringRef Remaining = StringRef(*Command).trim();

      if (Remaining.empty() || Remaining.starts_with(';')) {
        // Empty command or comment. These are supported so that tests can be
        // made more readable.
        continue;
      }

      if (tryParseCommand<RunCommand>(Remaining)) {
        const int ExitCode = runTool();
        respondReturned(ExitCode);
        NextInvocation.reset();
        resetWorkingDirectory();
      } else if (const std::optional<ArgCommand> Parsed =
                     tryParseCommand<ArgCommand>(Remaining)) {
        NextInvocation.Args.push_back(
            readStringFromStdin(Parsed->ExpectedLength));
      } else if (const std::optional<InputStringCommand> Parsed =
                     tryParseCommand<InputStringCommand>(Remaining)) {
        NextInvocation.InputSource = StandardInputSource::fromString(
            readStringFromStdin(Parsed->ExpectedLength));
      } else if (const std::optional<InputFileCommand> Parsed =
                     tryParseCommand<InputFileCommand>(Remaining)) {
        NextInvocation.InputSource =
            StandardInputSource::fromFile(std::string(Parsed->Path));
      } else if (const std::optional<ChangeDirectoryCommand> Parsed =
                     tryParseCommand<ChangeDirectoryCommand>(Remaining)) {
        changeWorkingDirectory(Parsed->Path);
      } else if (tryParseCommand<RedirectStderrToStdoutCommand>(Remaining)) {
        NextInvocation.RedirectStderrToStdout = true;
      } else if (tryParseCommand<ExitCommand>(Remaining)) {
        break;
      } else {
        exitWithError("Unexpected command: " + *Command);
      }
    }

    return 0;
  }

private:
  static constexpr StringRef ResponseReady = "ready";
  static constexpr StringRef ResponseReturned = "returned";
  static constexpr StringRef ResponseError = "error";

  static std::unique_ptr<raw_ostream>
  createStatusPipeOS(StringRef StatusPipeString) {
    // `StatusPipeString` may be:
    // - "path:{file path}"
    // - "fd:{file descriptor}"
    // - "handle:{Windows file handle}" (Windows-only)
    constexpr StringRef ErrorContext = "Parsing option 'daemon-status-pipe': ";

    if (StatusPipeString.consume_front("path:")) {
      std::error_code EC;
      auto Writer =
          std::make_unique<raw_fd_ostream>(StatusPipeString.trim(), EC);
      if (EC) {
        reportInitError(ErrorContext + "Couldn't open file '" +
                        StatusPipeString + "': " + EC.message());
      }
      Writer->SetUnbuffered();
      return Writer;
    }

    int Fd;
    if (StatusPipeString.consume_front("fd:")) {
      const bool Err = StatusPipeString.consumeInteger(10, Fd);
      if (Err) {
        reportInitError(ErrorContext + "expected integer "
                                       "after 'fd:'.");
      }
    } else if (StatusPipeString.consume_front("handle:")) {
#ifdef _WIN32
      int Handle;
      bool Err = StatusPipeString.consumeInteger(10, Handle);
      if (Err) {
        reportInitError(ErrorContext +
                        "Parsing option 'daemon-status-pipe': expected integer "
                        "after 'handle:'.");
      }

      Fd = _open_osfhandle(Handle, 0);
#else
      reportInitError(ErrorContext + "'handle' may only "
                                     "be specified on Windows");
#endif
    } else {
      reportInitError(ErrorContext + "Unexpected value : '" + StatusPipeString +
                      "'");
    }

    // Only close the status pipe if it is not a standard stream.
    const bool ShouldClose = Fd != STDOUT_FILENO && Fd != STDERR_FILENO;
    return std::make_unique<raw_fd_ostream>(Fd, ShouldClose,
                                            /*unbuffered=*/true);
  }

  template <typename Command>
  std::optional<Command> tryParseCommand(StringRef Remaining) const {
    if (!Remaining.consume_front(Command::Prefix))
      return {};

    Expected<Command> CommandOrError = Command::parse(Remaining);
    if (!CommandOrError)
      exitWithError(CommandOrError.takeError());

    return *CommandOrError;
  }

  template <typename E> [[noreturn]] void exitWithError(const E &Err) const {
    *StatusPipeOS << ResponseError << " " << Err << "\n";
    StatusPipeOS->flush();
    std::exit(StatusCommandError);
  }

  void respondReady() const {
    *StatusPipeOS << ResponseReady << "\n";
    StatusPipeOS->flush();
  }

  void respondReturned(const int ExitCode) const {
    *StatusPipeOS << ResponseReturned << ' ' << ExitCode << "\n";
    StatusPipeOS->flush();
  }

  int runTool() const {
    // Convert arguments to C strings, so that they can be passed via
    // `argc`.
    SmallVector<char *, 16> ArgsCStr;
    ArgsCStr.reserve(NextInvocation.Args.size());
    for (const std::string &Arg : NextInvocation.Args) {
      // NB: Since C++11, `std::string` is required to have a null terminator
      // after the data.
      ArgsCStr.push_back(const_cast<char *>(Arg.data()));
    }

    std::unique_ptr<ScopedFileRedirect> StderrRedirect;
    if (NextInvocation.RedirectStderrToStdout)
      StderrRedirect =
          std::make_unique<ScopedFileRedirect>(STDERR_FILENO, STDOUT_FILENO);

    const int ExitCode =
        Tool.run(ArgsCStr.size(), ArgsCStr.data(), NextInvocation.InputSource);

    Tool.resetState();

    // Important to flush `stdout` and `stderr` after `resetState()`, in case
    // `resetState()` had any output (for example if an error was encountered).
    outs().flush();
    errs().flush();

    return ExitCode;
  }

  std::string readStringFromStdin(const size_t Len) const {
    // Read `Len` bytes into `Dest`.
    std::string Dest;
    Dest.resize(Len);
    const size_t Read = fread(Dest.data(), sizeof(char), Len, stdin);

    // Make sure the expected number of bytes was read.
    if (Read != Len)
      exitWithError("Missing bytes: expected " + Twine(Len) + " got " +
                    Twine(Read));

    return Dest;
  }

  void changeWorkingDirectory(const StringRef Path) {
    std::filesystem::path FsPath(Path.str());
    if (!std::filesystem::is_directory(FsPath))
      exitWithError(Twine("cd: '") + Path + "' is not a directory.");

    std::filesystem::current_path(FsPath);
    WorkingDirectoryChanged = true;
  }

  void resetWorkingDirectory() {
    if (!WorkingDirectoryChanged)
      return;

    std::filesystem::current_path(OriginalWorkingDirectory);
    WorkingDirectoryChanged = false;
  }

  LLVMTool &Tool;
  InvocationOptions NextInvocation;
  std::unique_ptr<raw_ostream> StatusPipeOS;
  std::filesystem::path OriginalWorkingDirectory;
  bool WorkingDirectoryChanged;
};

static int runDaemonMode(LLVMTool &Tool, int Argc, char **Argv) {
  // Parse daemon command line options.
  const DaemonCommandLineOptions &Options =
      initializeDaemonCommandLineOptions();
  cl::ParseCommandLineOptions(Argc, Argv);

  return DaemonDriver(Tool, Options).run();
}
} // namespace

LLVM_ABI int llvm::runWithDaemonSupport(LLVMTool &Tool, int Argc, char **Argv) {
  if (detectDaemonArg(Argc, Argv))
    return runDaemonMode(Tool, Argc, Argv);

  return Tool.run(Argc, Argv, StandardInputSource::fromStdin());
}
