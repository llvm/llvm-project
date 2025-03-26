#include "PE.h"

using namespace llvm;

PEFrameLowering::PEFrameLowering()
    : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, Align(16), 0) {}

// ...existing code...