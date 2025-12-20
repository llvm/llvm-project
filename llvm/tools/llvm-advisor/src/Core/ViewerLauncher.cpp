//===---------------- ViewerLauncher.cpp - LLVM Advisor ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ViewerLauncher.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::string> ViewerLauncher::findPythonExecutable() {
  std::vector<std::string> candidates = {"python3", "python"};

  for (const auto &candidate : candidates) {
    if (auto path = sys::findProgramByName(candidate)) {
      return *path;
    }
  }

  return createStringError(
      std::make_error_code(std::errc::no_such_file_or_directory),
      "Python executable not found. Please install Python 3.");
}

Expected<std::string> ViewerLauncher::getViewerScript() {
  SmallString<256> scriptPath;

  // Try to find the server script relative to the executable
  auto mainExecutable = sys::fs::getMainExecutable(nullptr, nullptr);
  if (mainExecutable.empty()) {
    return createStringError(
        std::make_error_code(std::errc::no_such_file_or_directory),
        "Cannot determine executable path");
  }

  // Try: relative to binary (development/build tree)
  sys::path::append(scriptPath, sys::path::parent_path(mainExecutable));
  sys::path::append(scriptPath, "..");
  sys::path::append(scriptPath, "tools");
  sys::path::append(scriptPath, "webserver");
  sys::path::append(scriptPath, "server.py");

  if (sys::fs::exists(scriptPath)) {
    return std::string(scriptPath.str());
  }

  // Try: relative to binary (same directory as executable)
  scriptPath.clear();
  sys::path::append(scriptPath, sys::path::parent_path(mainExecutable));
  sys::path::append(scriptPath, "tools");
  sys::path::append(scriptPath, "webserver");
  sys::path::append(scriptPath, "server.py");

  if (sys::fs::exists(scriptPath)) {
    return std::string(scriptPath.str());
  }

  // Try: installed location
  scriptPath.clear();
  sys::path::append(scriptPath, sys::path::parent_path(mainExecutable));
  sys::path::append(scriptPath, "..");
  sys::path::append(scriptPath, "share");
  sys::path::append(scriptPath, "llvm-advisor");
  sys::path::append(scriptPath, "tools");
  sys::path::append(scriptPath, "webserver");
  sys::path::append(scriptPath, "server.py");

  if (sys::fs::exists(scriptPath)) {
    return std::string(scriptPath.str());
  }

  return createStringError(
      std::make_error_code(std::errc::no_such_file_or_directory),
      "Web server script not found. Please ensure tools/webserver/server.py "
      "exists.");
}

Expected<int> ViewerLauncher::launch(const std::string &outputDir, int port) {
  auto pythonOrErr = findPythonExecutable();
  if (!pythonOrErr) return pythonOrErr.takeError();

  auto scriptOrErr = getViewerScript();
  if (!scriptOrErr) return scriptOrErr.takeError();

  // Verify output directory exists and has data
  if (!sys::fs::exists(outputDir))
    return createStringError(
        std::make_error_code(std::errc::no_such_file_or_directory),
        "Output directory does not exist: " + outputDir);

  std::vector<StringRef> args = {*pythonOrErr, *scriptOrErr,
                                 "--data-dir", outputDir,
                                 "--port",     std::to_string(port)};

  // Execute the Python web server
  int result = sys::ExecuteAndWait(*pythonOrErr, args);

  if (result != 0)
    return createStringError(std::make_error_code(std::errc::io_error),
                             "Web server failed with exit code: " +
                                 std::to_string(result));

  return result;
}
