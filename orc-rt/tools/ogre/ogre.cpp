//===- ogre.cpp - ORC Generic Runtime Environment -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The ORC Generic Runtime Environment (OGRE) is intended as a canonical
// "blank executor".
//
//===----------------------------------------------------------------------===//

#include "orc-rt-internal/tools/OptionParser.h"
#include "orc-rt/bedrock/ConnectionSpec.h"
#include "orc-rt/bedrock/ConnectorRegistry.h"
#include "orc-rt/bedrock/NativeDylibManager.h"
#include "orc-rt/bedrock/Session.h"
#include "orc-rt/bedrock/SimpleNativeMemoryMap.h"
#include "orc-rt/bedrock/SocketConnector.h"
#include "orc-rt/bedrock/ThreadPoolRunner.h"
#include "orc-rt/bedrock/sps/AllSPSCI.h"

#include "orc-rt-c/support/Logging.h"

#include <cstdio>
#include <cstring>
#include <future>
#include <variant>

using namespace orc_rt;

struct Options {
  ConnectionSpec ConnSpec;
  bool Verbose = false;
};

/// Parse and handle options. Return the parsed options struct on success, or an
/// int error code to return from main otherwise.
static std::variant<Options, int> parseArgs(int argc, char *argv[]) noexcept {
  Options O;
  bool ShowHelp = false;

  OptionParser P;
  P.addFlag("verbose", "Print verbose output", false, O.Verbose, 'v')
      .addFlag("help", "Display this help message", false, ShowHelp, 'h');

  auto PrintHelp = [&](Error Err) -> int {
    bool Failed = !!Err;
    if (Err)
      fprintf(stderr, "error: %s\n", toString(std::move(Err)).c_str());
    const char *ProgName = argc != 0 ? argv[0] : "ogre";
    fprintf(stderr, "%s", P.formatHelp(ProgName).c_str());
    return Failed;
  };

  // ogre options end at the first '--', or at argc if no '--' is present.
  int OgreOptsEnd;
  for (OgreOptsEnd = 0; OgreOptsEnd != argc; ++OgreOptsEnd)
    if (strcmp(argv[OgreOptsEnd], "--") == 0)
      break;

  if (auto Err = P.parseAsMainArgs(OgreOptsEnd, argv))
    return PrintHelp(std::move(Err));

  if (ShowHelp)
    return PrintHelp(Error::success());

  if (P.positionals().size() != 1)
    return PrintHelp(
        make_error<StringError>("expected one positional argument"));

  if (auto ConnSpec = ConnectionSpec::parse(P.positionals().front()))
    O.ConnSpec = std::move(*ConnSpec);
  else
    return PrintHelp(ConnSpec.takeError());

  return O;
}

static void reportError(Session &S, Error Err) noexcept {
#if ORC_RT_LOG_ENABLED(Error)
  Session::logErrors(S, std::move(Err));
#else
  fprintf(stderr, "Session %p error: %s\n", &S,
          toString(std::move(Err)).c_str());
#endif // ORC_RT_LOG_ENABLED(Error)
}

static Expected<Session::DispatchFn> makeDispatcher() noexcept {
  return [R = std::make_unique<ThreadPoolRunner>(4)](Session::Task T) {
    (*R)(std::move(T));
  };
}

/// Adds the services a host executor provides, publishing their entry points
/// in BI for the controller.
static Error addHostServices(Session &S, BootstrapInfo &BI) noexcept {
  if (auto Err = sps_ci::addAll(BI.symbols()))
    return Err;

  if (auto Err = S.tryCreateService<SimpleNativeMemoryMap>(S, BI.symbols())
                     .takeError())
    return Err;

  if (auto Err =
          S.tryCreateService<NativeDylibManager>(S, BI.symbols()).takeError())
    return Err;

  return Error::success();
}

static Expected<std::unique_ptr<Session>>
makeSession(const Options &Opts) noexcept {
  auto EPI = ExecutorProcessInfo::Detect();
  if (!EPI)
    return EPI.takeError();

  if (Opts.Verbose) {
    fprintf(stderr,
            "executor info: triple = \"%s\", page-size = %zu, "
            "cpu-features = \"%s\"\n",
            EPI->targetTriple().c_str(), EPI->pageSize(),
            EPI->targetCPUFeatures().c_str());
  }

  auto D = makeDispatcher();
  if (!D)
    return D.takeError();

  auto S =
      std::make_unique<Session>(std::move(*EPI), std::move(*D), reportError);

  auto BI = BootstrapInfo::CreateDefault(*S);
  if (!BI)
    return BI.takeError();

  if (auto Err = addHostServices(*S, *BI))
    return Err;

  ConnectorRegistry Connectors;
  if (auto Err = registerSocketConnector(Connectors))
    return Err;

  if (auto Err = Connectors.connect(Opts.ConnSpec, *S, std::move(*BI)))
    return Err;

  return S;
}

static Expected<int> runOgre(const Options &Opts) noexcept {
  auto S = makeSession(Opts);
  if (!S)
    return S.takeError();

  // makeSession has already connected, so the Session may have detached by
  // now. That's fine: addOnDetach runs the callback immediately if so, so the
  // wait below cannot miss the detach.
  std::promise<void> StopP;
  std::future<void> StopF = StopP.get_future();
  (*S)->addOnDetach(
      [StopP = std::move(StopP)]() mutable noexcept { StopP.set_value(); });

  // Wait for detach.
  StopF.get();
  return 0;
}

int main(int argc, char *argv[]) {
  auto OptsOrResult = parseArgs(argc, argv);

  // If parseArgs returned an int then we should just exit with that return
  // code.
  if (int *Result = std::get_if<int>(&OptsOrResult))
    return *Result;

  if (auto *Opts = std::get_if<Options>(&OptsOrResult)) {
    if (auto Result = runOgre(*Opts))
      return *Result;
    else {
      fprintf(stderr, "error: %s\n", toString(Result.takeError()).c_str());
      return 1;
    }
  }

  ORC_RT_UNREACHABLE("OptsOrResult held unexpected type?");
}
