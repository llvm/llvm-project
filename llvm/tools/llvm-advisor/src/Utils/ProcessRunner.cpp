#include "ProcessRunner.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

namespace llvm {
namespace advisor {

Expected<ProcessRunner::ProcessResult>
ProcessRunner::run(const std::string &program,
                   const std::vector<std::string> &args, int timeoutSeconds) {

  auto programPath = sys::findProgramByName(program);
  if (!programPath) {
    return createStringError(programPath.getError(),
                             "Tool not found: " + program);
  }

  std::vector<StringRef> execArgs;
  execArgs.push_back(program);
  for (const auto &arg : args) {
    execArgs.push_back(arg);
  }

  SmallString<128> stdoutPath, stderrPath;
  sys::fs::createTemporaryFile("stdout", "tmp", stdoutPath);
  sys::fs::createTemporaryFile("stderr", "tmp", stderrPath);

  std::optional<StringRef> redirects[] = {
      std::nullopt,          // stdin
      StringRef(stdoutPath), // stdout
      StringRef(stderrPath)  // stderr
  };

  int exitCode = sys::ExecuteAndWait(*programPath, execArgs, std::nullopt,
                                     redirects, timeoutSeconds);

  ProcessResult result;
  result.exitCode = exitCode;
  // TODO: Collect information about compilation time
  result.executionTime = 0; // not tracking time

  auto stdoutBuffer = MemoryBuffer::getFile(stdoutPath);
  if (stdoutBuffer) {
    result.stdout = (*stdoutBuffer)->getBuffer().str();
  }

  auto stderrBuffer = MemoryBuffer::getFile(stderrPath);
  if (stderrBuffer) {
    result.stderr = (*stderrBuffer)->getBuffer().str();
  }

  sys::fs::remove(stdoutPath);
  sys::fs::remove(stderrPath);

  return result;
}

Expected<ProcessRunner::ProcessResult> ProcessRunner::runWithEnv(
    const std::string &program, const std::vector<std::string> &args,
    const std::vector<std::string> &env, int timeoutSeconds) {

  // For simplicity, just use the regular run method
  // Environment variables can be added later if needed
  return run(program, args, timeoutSeconds);
}

} // namespace advisor
} // namespace llvm
