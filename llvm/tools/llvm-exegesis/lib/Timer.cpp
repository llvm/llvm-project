#include "Timer.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {
namespace exegesis {

bool TimerIsEnabled = false;

const char TimerGroupName[] = "llvm-exegesis";
const char TimerGroupDescription[] = "Time passes in each exegesis phase";

cl::opt<bool, true> EnableTimer("time-phases", cl::location(TimerIsEnabled),
                                cl::desc(TimerGroupDescription));

} // namespace exegesis
} // namespace llvm
