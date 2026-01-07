//===- HexagonCallingConv.td ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// We cannot use the standard CCIfArgVarArg class here since in Hexagon there
// exists a special case for a musl environment.  In a musl environment VarArgs
// are treated like non VarArgs.  I.e., in a musl enviroment unnamed arguments
// can also be passed in registers.  The CCIfArgVarArg class only checks each
// individual argument, but not whether State.isVarArg() is true.  We also have
// to check State.isVarArg() which is determined by the TreatAsVarArg argument.
class CCIfStateVarArgAndArgVarArg<CCAction A>
  : CCIf<"State.isVarArg() && ArgFlags.isVarArg()", A>;

def CC_HexagonStack: CallingConv<[
  CCIfType<[i32,v2i16,v4i8],
    CCAssignToStack<4,4>>,
  CCIfType<[i64,v2i32,v4i16,v8i8],
    CCAssignToStack<8,8>>
]>;

def CC_Hexagon_Legacy: CallingConv<[
  CCIfType<[i1,i8,i16],
    CCPromoteToType<i32>>,
  CCIfType<[bf16],
    CCBitConvertToType<i32>>,
  CCIfType<[f32],
    CCBitConvertToType<i32>>,
  CCIfType<[f64],
    CCBitConvertToType<i64>>,

  CCIfByVal<
    CCPassByVal<8,8>>,
  CCIfStateVarArgAndArgVarArg<
    CCDelegateTo<CC_HexagonStack>>,

  // Pass split values in pairs, allocate odd register if necessary.
  CCIfType<[i32],
    CCIfSplit<
      CCCustom<"CC_SkipOdd">>>,

  CCIfType<[i32,v2i16,v4i8],
    CCAssignToReg<[R0,R1,R2,R3,R4,R5]>>,
  // Make sure to allocate any skipped 32-bit register, so it does not get
  // allocated to a subsequent 32-bit value.
  CCIfType<[i64,v2i32,v4i16,v8i8],
    CCCustom<"CC_SkipOdd">>,
  CCIfType<[i64,v2i32,v4i16,v8i8],
    CCAssignToReg<[D0,D1,D2]>>,

  CCDelegateTo<CC_HexagonStack>
]>;

def CC_Hexagon: CallingConv<[
  CCIfType<[i1,i8,i16],
    CCPromoteToType<i32>>,
  CCIfType<[bf16],
    CCBitConvertToType<i32>>,
  CCIfType<[f32],
    CCBitConvertToType<i32>>,
  CCIfType<[f64],
    CCBitConvertToType<i64>>,

  CCIfByVal<
    CCPassByVal<8,1>>,
  CCIfStateVarArgAndArgVarArg<
    CCDelegateTo<CC_HexagonStack>>,

  // Pass split values in pairs, allocate odd register if necessary.
  CCIfType<[i32],
    CCIfSplit<
      CCCustom<"CC_SkipOdd">>>,
  CCIfType<[v2i1],  CCPromoteToType<v2i32>>,
  CCIfType<[v4i1],  CCPromoteToType<v4i16>>,
  CCIfType<[v8i1],  CCPromoteToType<v8i8>>,

  CCIfType<[i32,v2i16,v4i8],
    CCAssignToReg<[R0,R1,R2,R3,R4,R5]>>,
  // Make sure to allocate any skipped 32-bit register, so it does not get
  // allocated to a subsequent 32-bit value.
  CCIfType<[i64,v2i32,v4i16,v8i8],
    CCCustom<"CC_SkipOdd">>,
  CCIfType<[i64,v2i32,v4i16,v8i8],
    CCAssignToReg<[D0,D1,D2]>>,

  CCDelegateTo<CC_HexagonStack>
]>;

def RetCC_Hexagon: CallingConv<[
  CCIfType<[i1,i8,i16],
    CCPromoteToType<i32>>,
  CCIfType<[bf16],
    CCBitConvertToType<i32>>,
  CCIfType<[f32],
    CCBitConvertToType<i32>>,
  CCIfType<[f64],
    CCBitConvertToType<i64>>,

  // Small structures are returned in a pair of registers, (which is
  // always r1:0). In such case, what is returned are two i32 values
  // without any additional information (in ArgFlags) stating that
  // they are parts of a structure. Because of that there is no way
  // to differentiate that situation from an attempt to return two
  // values, so always assign R0 and R1.
  CCIfSplit<
    CCAssignToReg<[R0,R1]>>,
  CCIfType<[i32,v2i16,v4i8],
    CCAssignToReg<[R0,R1]>>,
  CCIfType<[i64,v2i32,v4i16,v8i8],
    CCAssignToReg<[D0]>>
]>;


class CCIfHvx64<CCAction A>
  : CCIf<"State.getMachineFunction().getSubtarget<HexagonSubtarget>()"
         ".useHVX64BOps()", A>;

class CCIfHvx128<CCAction A>
  : CCIf<"State.getMachineFunction().getSubtarget<HexagonSubtarget>()"
         ".useHVX128BOps()", A>;

def CC_Hexagon_HVX: CallingConv<[
  // HVX 64-byte mode

  CCIfHvx64<
        CCIfType<[v16i1], CCPromoteToType<v16i32>>>,
  CCIfHvx64<
        CCIfType<[v32i1], CCPromoteToType<v32i16>>>,
  CCIfHvx64<
        CCIfType<[v64i1], CCPromoteToType<v64i8>>>,

  CCIfHvx64<
    CCIfType<[v16i32,v32i16,v64i8],
      CCAssignToReg<[V0,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15]>>>,
  CCIfHvx64<
    CCIfType<[v32i32,v64i16,v128i8],
      CCAssignToReg<[W0,W1,W2,W3,W4,W5,W6,W7]>>>,
  CCIfHvx64<
    CCIfType<[v16i32,v32i16,v64i8],
      CCAssignToStack<64,64>>>,
  CCIfHvx64<
    CCIfType<[v32i32,v64i16,v128i8],
      CCAssignToStack<128,64>>>,

  // HVX 128-byte mode

  CCIfHvx128<
        CCIfType<[v32i1], CCPromoteToType<v32i32>>>,
  CCIfHvx128<
        CCIfType<[v64i1], CCPromoteToType<v64i16>>>,
  CCIfHvx128<
        CCIfType<[v128i1], CCPromoteToType<v128i8>>>,

  CCIfHvx128<
    CCIfType<[v32i32,v64i16,v128i8,v32f32,v64f16,v64bf16],
      CCAssignToReg<[V0,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15]>>>,
  CCIfHvx128<
    CCIfType<[v64i32,v128i16,v256i8,v64f32,v128f16,v128bf16],
      CCAssignToReg<[W0,W1,W2,W3,W4,W5,W6,W7]>>>,
  CCIfHvx128<
    CCIfType<[v32i32,v64i16,v128i8,v32f32,v64f16,v64bf16],
      CCAssignToStack<128,128>>>,
  CCIfHvx128<
    CCIfType<[v64i32,v128i16,v256i8,v64f32,v128f16,v64bf16],
      CCAssignToStack<256,128>>>,

  CCDelegateTo<CC_Hexagon>
]>;

def RetCC_Hexagon_HVX: CallingConv<[
  // HVX 64-byte mode
  CCIfHvx64<
    CCIfType<[v16i32,v32i16,v64i8],
      CCAssignToReg<[V0]>>>,
  CCIfHvx64<
    CCIfType<[v32i32,v64i16,v128i8],
      CCAssignToReg<[W0]>>>,

  // HVX 128-byte mode
  CCIfHvx128<
    CCIfType<[v32i32,v64i16,v128i8,v32f32,v64f16,v64bf16],
      CCAssignToReg<[V0]>>>,
  CCIfHvx128<
    CCIfType<[v64i32,v128i16,v256i8,v64f32,v128f16,v128bf16],
      CCAssignToReg<[W0]>>>,

  CCDelegateTo<RetCC_Hexagon>
]>;

