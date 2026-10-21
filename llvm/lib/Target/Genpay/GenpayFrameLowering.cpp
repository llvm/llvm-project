#include "GenpayFrameLowering.h"

using namespace llvm;

GenpayFrameLowering::GenpayFrameLowering()
    : TargetFrameLowering(StackGrowsDown, Align(8), 0) {}
