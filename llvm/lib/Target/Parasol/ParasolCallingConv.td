//=== ParasolCallingConv.td - Calling Conventions for Parasol --*- tablegen -*-===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
// This describes the calling conventions for Parasol architecture.
//===----------------------------------------------------------------------===//

def CC_Parasol : CallingConv<[
  // Handles byval parameters. Tell LLVM what the size and alignment
  // requirements of each stack slot are
  CCIfByVal<CCPassByVal<4, 4>>,

  // Promote i8/i16 to i32
  // CCIfType<[i1, i8, i16], CCPromoteToType<i32>>,

  // The first 32 arguments are passed in by register P0, P1, ...
  CCIfType<[i8, i16, i32], CCAssignToReg<[X10, X11, X12, X13, X14, X15, X16, X17]>>,


  // SUNSCREEN TODO: Try removing and seeing if it still works. Ideally we
  // should have no stack so we don't need this.
  // Integer values get stored in stack slots that are 4 bytes in size and
  // 4-byte aligned
  CCIfType<[i8, i16, i32], CCAssignToStack<4, 4>>
]>;

// BeyondRISC C Return Convention
def Parasol_CRetConv : CallingConv<[
  // CCIfType<[i1, i8, i16], CCPromoteToType<i32>>,

  // i32 are returned in registers X0, X1
  CCIfType<[i8, i16, i32], CCAssignToReg<[X10, X11]>>,

  // Integer values get stored in stack slots that are 4 bytes in size and
  // 4-byte aligned
  CCIfType<[i8, i16, i32], CCAssignToStack<4, 4>>
]>;

def Parasol_CalleeSavedRegs : CalleeSavedRegs<(add X8, X9)>;
