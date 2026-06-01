//===- DTLTODistributionDriver.cpp - DTLTO Distribution Driver ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This file implements the Integrated Distributed ThinLTO driver that prepares
// the compilation job descriptions and invokes the external distributor.
//
//===----------------------------------------------------------------------===//

#include "llvm/DTLTO/DTLTO.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace llvm;

// Generates a JSON file describing the backend compilations, for the
// distributor.
Error lto::DistributionDriver::emitJson() {
  using json::Array;
  std::error_code EC;
  raw_fd_ostream OS(DistributorJsonFile, EC);
  if (EC)
    return createStringError(EC, "Error while creating Json file");

  json::OStream JOS(OS);
  JOS.object([&]() {
    // Information common to all jobs.
    JOS.attributeObject("common", [&]() {
      JOS.attribute("linker_output", Params.LinkerOutputFile);

      JOS.attributeArray("args", [&]() {
        JOS.value(Params.RemoteCompiler);

        // Forward any supplied prepend options.
        if (!Params.RemoteCompilerPrependArgs.empty())
          for (auto &A : Params.RemoteCompilerPrependArgs)
            JOS.value(A);

        JOS.value("-c");

        JOS.value(std::string("--target=") + Params.TargetTriple.str());

        for (const auto &A : Params.CodegenOptions)
          JOS.value(A);
      });

      JOS.attribute("inputs", Array(Params.CommonInputs));
    });

    // Per-compilation-job information.
    JOS.attributeArray("jobs", [&]() {
      for (const auto &J : Jobs) {
        assert(J.Task != 0);
        if (J.Cached) {
          continue;
        }

        SmallVector<StringRef, 2> Inputs;
        SmallVector<StringRef, 1> Outputs;

        JOS.object([&]() {
          JOS.attributeArray("args", [&]() {
            JOS.value(J.ModuleID);
            Inputs.push_back(J.ModuleID);

            JOS.value(
                std::string("-fthinlto-index=" + J.SummaryIndexPath.str()));
            Inputs.push_back(J.SummaryIndexPath);

            JOS.value("-o");
            JOS.value(J.NativeObjectPath);
            Outputs.push_back(J.NativeObjectPath);
          });

          // Add the bitcode files from which imports will be made. These do
          // not explicitly appear on the backend compilation command lines
          // but are recorded in the summary index shards.
          append_range(Inputs, J.ImportsFiles);
          JOS.attribute("inputs", Array(Inputs));

          JOS.attribute("outputs", Array(Outputs));
        });
      }
    });
  });

  return Error::success();
}

// Saves JSON file on a filesystem.
Error lto::DistributionDriver::saveJson() {
  DistributorJsonFile = sys::path::parent_path(Params.LinkerOutputFile);
  TimeTraceScope TimeScope("Emit DTLTO JSON");
  sys::path::append(DistributorJsonFile,
                    sys::path::stem(Params.LinkerOutputFile) + "." +
                        itostr(sys::Process::getProcessId()) +
                        ".dist-file.json");
  if (Error E = emitJson())
    return make_error<StringError>(
        BCError + "failed to generate distributor JSON script: " +
            DistributorJsonFile,
        errorToErrorCode(std::move(E)));

  // Add JSON file to the cleanup files list.
  if (!SaveTemps)
    AddToCleanup(DistributorJsonFile);
  return Error::success();
}

// Invokes the distributor to compile uncached ThinLTO modules remotely.
Error lto::DistributionDriver::operator()() {
  if (Error E = saveJson())
    return E;

  TimeTraceScope TimeScope("Execute DTLTO distributor", Params.DistributorPath);
  SmallVector<StringRef, 3> Args = {Params.DistributorPath};
  append_range(Args, Params.DistributorArgs);
  Args.push_back(DistributorJsonFile);
  std::string ErrMsg;
  if (sys::ExecuteAndWait(Args[0], Args,
                          /*Env=*/std::nullopt, /*Redirects=*/{},
                          /*SecondsToWait=*/0, /*MemoryLimit=*/0, &ErrMsg)) {
    return make_error<StringError>(
        BCError + "distributor execution failed" +
            (!ErrMsg.empty() ? ": " + ErrMsg + Twine(".") : Twine(".")),
        inconvertibleErrorCode());
  }
  return Error::success();
}
