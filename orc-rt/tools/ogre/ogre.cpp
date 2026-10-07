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

/// Command-line options for ogre.
struct Options {
  ConnectionSpec ConnSpec;
  bool Verbose = false;
};

/// Parse and handle options. Return the parsed options struct on success, or an
/// int error code to return from main otherwise. The single positional argument
/// is a ConnectionSpec describing how to reach the controller.
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

/// The Session's error reporter: receives errors that have no caller to be
/// returned to (e.g. failures in asynchronous work), and logs them.
static void reportError(Session &S, Error Err) noexcept {
#if ORC_RT_LOG_ENABLED(Error)
  Session::logErrors(S, std::move(Err));
#else
  fprintf(stderr, "Session %p error: %s\n", &S,
          toString(std::move(Err)).c_str());
#endif // ORC_RT_LOG_ENABLED(Error)
}

/// Creates the Session's dispatcher, which runs incoming calls from the
/// controller (and other Session tasks) on a thread pool. An executor without
/// threads could run each task inline instead.
static Expected<Session::DispatchFn> makeDispatcher() noexcept {
  return [R = std::make_unique<ThreadPoolRunner>(4)](Session::Task T) {
    (*R)(std::move(T));
  };
}

/// Adds the services a host executor provides (JIT memory management and dylib
/// loading), publishing their entry points in BI for the controller.
static Error addHostServices(Session &S, BootstrapInfo &BI) noexcept {

  // Add controller interfaces for calling functions, reading/writing memory,
  // registering metadata, etc.
  if (auto Err = sps_ci::addAll(BI.symbols()))
    return Err;

  // Add SimpleNativeMemoryMap service to manage JIT'd memory. Acting as a
  // service means that SimpleNativeMemoryMap is notified when the Session
  // disconnects and is shut down, so it can free allocated resources.
  if (auto Err = S.tryCreateService<SimpleNativeMemoryMap>(S, BI.symbols())
                     .takeError())
    return Err;

  // Adds NativeDylibManager, through which the controller can load and search
  // dylibs.
  if (auto Err =
          S.tryCreateService<NativeDylibManager>(S, BI.symbols()).takeError())
    return Err;

  return Error::success();
}

/// Creates a Session for this process, adds the host services to it, and
/// connects it to the controller described by Opts.ConnSpec.
static Expected<std::unique_ptr<Session>>
makeSession(const Options &Opts) noexcept {
  // ExecutorProcessInfo describes the executing process: target triple, page
  // size, CPU features, etc.
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

  // The dispatcher runs incoming calls from the controller.
  auto D = makeDispatcher();
  if (!D)
    return D.takeError();

  // The Session is the root of the JIT'd program: it owns the program's memory
  // and lifecycle, as well as the executor's services and the connection to the
  // controller. Errors with nowhere else to go are passed to reportError.
  auto S =
      std::make_unique<Session>(std::move(*EPI), std::move(*D), reportError);

  // The BootstrapInfo struct defines the data sent over to the controller when
  // it connects: the executor process info, bootstrap symbols (entry points the
  // controller can call), and the bootstrap value map. CreateDefault adds the
  // Session's own symbol, the SPS controller-interface functions, and the
  // CPU-features value.
  auto BI = BootstrapInfo::CreateDefault(*S);
  if (!BI)
    return BI.takeError();

  // Services add their entry points to the bootstrap symbols, so they must be
  // created before connecting.
  if (auto Err = addHostServices(*S, *BI))
    return Err;

  // Connectors establish the connection to the controller for a particular
  // transport. Registering only the socket connector means ogre can only be
  // reached over a socket.
  ConnectorRegistry Connectors;
  if (auto Err = registerSocketConnector(Connectors))
    return Err;

  // Connect to the controller described by the ConnectionSpec, handing over the
  // bootstrap info. From here on the controller can call into this process.
  if (auto Err = Connectors.connect(Opts.ConnSpec, *S, std::move(*BI)))
    return Err;

  return S;
}

/// Runs ogre: creates and connects a Session, then waits for the controller to
/// detach before exiting.
static Expected<int> runOgre(const Options &Opts) noexcept {
  // Create the Session and connect it to the controller.
  auto S = makeSession(Opts);
  if (!S)
    return S.takeError();

  // Arrange to be notified when the Session detaches from the controller.
  // makeSession has already connected, so the Session may have detached by
  // now. That's fine: addOnDetach runs the callback immediately if so, so the
  // wait below cannot miss the detach.
  std::promise<void> StopP;
  std::future<void> StopF = StopP.get_future();
  (*S)->addOnDetach(
      [StopP = std::move(StopP)]() mutable noexcept { StopP.set_value(); });

  // Wait for detach. The main thread has nothing else to do: the dispatcher
  // runs the controller's calls. Returning destroys the Session.
  StopF.get();
  return 0;
}

/// Parses the options, then runs ogre, reporting any error.
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
