//===---------------- ViewerLauncher.cpp - LLVM Advisor ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ViewerLauncher.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::string> ViewerLauncher::findPythonExecutable() {
  std::vector<std::string> Candidates = {"python3", "python"};

  for (const auto &Candidate : Candidates) {
    if (auto Path = sys::findProgramByName(Candidate))
      return *Path;
  }

  return createStringError(
      std::make_error_code(std::errc::no_such_file_or_directory),
      "Python executable not found. Please install Python 3.");
}

Expected<std::string> ViewerLauncher::getViewerScript() {
  SmallString<256> ScriptPath;

  // Try to find the server script relative to the executable
  auto MainExecutable = sys::fs::getMainExecutable(nullptr, nullptr);
  if (MainExecutable.empty()) {
    return createStringError(
        std::make_error_code(std::errc::no_such_file_or_directory),
        "Cannot determine executable path");
  }

  // Try: relative to binary (development/build tree)
  sys::path::append(ScriptPath, sys::path::parent_path(MainExecutable));
  sys::path::append(ScriptPath, "..");
  sys::path::append(ScriptPath, "tools");
  sys::path::append(ScriptPath, "webserver");
  sys::path::append(ScriptPath, "server.py");

  if (sys::fs::exists(ScriptPath))
    return std::string(ScriptPath.str());

  // Try: relative to binary (same directory as executable)
  ScriptPath.clear();
  sys::path::append(ScriptPath, sys::path::parent_path(MainExecutable));
  sys::path::append(ScriptPath, "tools");
  sys::path::append(ScriptPath, "webserver");
  sys::path::append(ScriptPath, "server.py");

  if (sys::fs::exists(ScriptPath))
    return std::string(ScriptPath.str());

  // Try: installed location
  ScriptPath.clear();
  sys::path::append(ScriptPath, sys::path::parent_path(MainExecutable));
  sys::path::append(ScriptPath, "..");
  sys::path::append(ScriptPath, "share");
  sys::path::append(ScriptPath, "llvm-advisor");
  sys::path::append(ScriptPath, "tools");
  sys::path::append(ScriptPath, "webserver");
  sys::path::append(ScriptPath, "server.py");

  if (sys::fs::exists(ScriptPath))
    return std::string(ScriptPath.str());

  return createStringError(
      std::make_error_code(std::errc::no_such_file_or_directory),
      "Web server script not found. Please ensure tools/webserver/server.py "
      "exists.");
}

Expected<int> ViewerLauncher::launch(const std::string &OutputDir, int Port) {
  auto PythonOrErr = findPythonExecutable();
  if (!PythonOrErr)
    return PythonOrErr.takeError();

  auto ScriptOrErr = getViewerScript();
  if (!ScriptOrErr)
    return ScriptOrErr.takeError();

  // Verify output directory exists and has data
  if (!sys::fs::exists(OutputDir))
    return createStringError(
        std::make_error_code(std::errc::no_such_file_or_directory),
        "Output directory does not exist: " + OutputDir);

  std::vector<std::string> OwnedArgs = {*PythonOrErr, *ScriptOrErr,
                                        "--data-dir", OutputDir,
                                        "--port",     std::to_string(Port)};
  llvm::SmallVector<StringRef, 8> Args;
  Args.reserve(OwnedArgs.size());
  for (const auto &Arg : OwnedArgs)
    Args.push_back(Arg);

  // Execute the Python web server
  int Result = sys::ExecuteAndWait(*PythonOrErr, Args);

  if (Result != 0)
    return createStringError(std::make_error_code(std::errc::io_error),
                             "Web server failed with exit code: " +
                                 std::to_string(Result));

  return Result;
}
