//==-- Next32CallingConv.td - Next32 Calling Conventions *- tablegen -*-======//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This describes the calling conventions for the Next32 architecture.
//
//===----------------------------------------------------------------------===//

def CC_Next32 : CallingConv<[
  CCIfByVal<CCPassIndirect<i64>>,

  CCIfType<[i8], CCPromoteToType<i32>>, CCIfType<[i16], CCPromoteToType<i32>>,
  CCIfType<[i64], CCPromoteToType<i32>>,

  CCIfType<[i32], CCCustom<"CC_Next32RegisterAllocator">>,

  CCIfSplit<CCAssignToStack<4, 8>>, CCAssignToStack<4, 4>
]>;

def CC_Next32VarArg
    : CallingConv<[CCIfSplit<CCAssignToStack<4, 8>>, CCAssignToStack<4, 4>]>;
