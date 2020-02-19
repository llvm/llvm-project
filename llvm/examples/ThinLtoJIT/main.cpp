#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

#include "ThinLtoJIT.h"

#include <string>
#include <vector>

using namespace llvm;

static cl::list<std::string>
    InputFiles(cl::Positional, cl::OneOrMore,
               cl::desc("<bitcode files or global index>"));

static cl::list<std::string> InputArgs("args", cl::Positional,
                                       cl::desc("<program arguments>..."),
                                       cl::ZeroOrMore, cl::PositionalEatsArgs);

static cl::opt<unsigned> CompileThreads("compile-threads", cl::Optional,
                                        cl::desc("Number of compile threads"),
                                        cl::init(4));

static cl::opt<unsigned> LoadThreads("load-threads", cl::Optional,
                                     cl::desc("Number of module load threads"),
                                     cl::init(8));

static cl::opt<unsigned>
    LookaheadLevels("lookahead", cl::Optional,
                    cl::desc("Calls to look ahead of execution"), cl::init(4));

static cl::opt<unsigned> DiscoveryFlagsBucketSize(
    "discovery-flag-bucket-size", cl::Optional,
    cl::desc("Flags per bucket (rounds up to memory pages)"), cl::init(4096));

static cl::opt<orc::ThinLtoJIT::ExplicitMemoryBarrier>
    MemFence("mem-fence",
             cl::desc("Control memory fences for cache synchronization"),
             cl::init(orc::ThinLtoJIT::NeverFence),
             cl::values(clEnumValN(orc::ThinLtoJIT::NeverFence, "never",
                                   "No use of memory fences"),
                        clEnumValN(orc::ThinLtoJIT::FenceStaticCode, "static",
                                   "Use of memory fences in static code only"),
                        clEnumValN(orc::ThinLtoJIT::FenceJITedCode, "jited",
                                   "Install memory fences in JITed code only"),
                        clEnumValN(orc::ThinLtoJIT::AlwaysFence, "always",
                                   "Always use of memory fences")));

static cl::opt<bool>
    AllowNudge("allow-nudge",
               cl::desc("Allow the symbol generator to nudge symbols into "
                        "discovery even though they haven't been reached"),
               cl::init(false));

static cl::opt<bool> PrintStats("print-stats",
                                cl::desc("Print module stats on shutdown"),
                                cl::init(false));

int main(int argc, char *argv[]) {
  InitLLVM X(argc, argv);
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  cl::ParseCommandLineOptions(argc, argv, "ThinLtoJIT");

  Error Err = Error::success();
  auto atLeastOne = [](unsigned N) { return std::max(1u, N); };

  orc::ThinLtoJIT Jit(InputFiles, "main", atLeastOne(LookaheadLevels),
                      atLeastOne(CompileThreads), atLeastOne(LoadThreads),
                      DiscoveryFlagsBucketSize, MemFence, AllowNudge,
                      PrintStats, Err);
  if (Err) {
    logAllUnhandledErrors(std::move(Err), errs(), "[ThinLtoJIT] ");
    exit(1);
  }

  ExitOnError ExitOnErr;
  ExitOnErr.setBanner("[ThinLtoJIT] ");

  return ExitOnErr(Jit.main(InputArgs));
}
