#ifndef LLVM_TRANSFORMS_YK_CONTROLPOINT_H
#define LLVM_TRANSFORMS_YK_CONTROLPOINT_H

#include "llvm/Pass.h"

// The name of the "dummy function" that the user puts in their interpreter
// implementation and that we will replace with our own code.
#define YK_DUMMY_CONTROL_POINT "yk_mt_control_point"

// The name of the new control point replacing the user's dummy control point.
#define YK_NEW_CONTROL_POINT "__ykrt_control_point"

// The name of the function which reconstructs the stackframe and jumps to the
// right instruction in AOT from where to continue.
#define YK_RECONSTRUCT_FRAMES "__ykrt_reconstruct_frames"

// The name of the patchpoint intrinsic we use for the control point.
#define CP_PPNAME "llvm.experimental.patchpoint.void"

namespace llvm {
ModulePass *createYkControlPointPass();
} // namespace llvm

#endif
