//===-- HexagonVarargsCallingConvention.h - Calling Conventions -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the functions that assign locations to outgoing function
// arguments. Adapted from the target independent version but this handles
// calls to varargs functions
//
//===----------------------------------------------------------------------===//
//




static bool RetCC_Hexagon32_VarArgs(unsigned ValNo, EVT ValVT,
                                    EVT LocVT, CCValAssign::LocInfo LocInfo,
                                    ISD::ArgFlagsTy ArgFlags,
                                    Hexagon_CCState &State,
                                    int NonVarArgsParams,
                                    int CurrentParam,
                                    bool ForceMem);


static bool CC_Hexagon32_VarArgs(unsigned ValNo, EVT ValVT,
                                 EVT LocVT, CCValAssign::LocInfo LocInfo,
                                 ISD::ArgFlagsTy ArgFlags,
                                 Hexagon_CCState &State,
                                 int NonVarArgsParams,
                                 int CurrentParam,
                                 bool ForceMem) {
  unsigned ByValSize = 0;
  if (ArgFlags.isByVal() &&
      ((ByValSize = ArgFlags.getByValSize()) >
       (MVT(MVT::i64).getSizeInBits() / 8))) {
    ForceMem = true;
  }


  // Only assign registers for named (non varargs) arguments
  if ( !ForceMem && ((NonVarArgsParams == -1) || (CurrentParam <=
                                                  NonVarArgsParams))) {

    if (LocVT == MVT::i32 ||
        LocVT == MVT::i16 ||
        LocVT == MVT::i8 ||
        LocVT == MVT::f32) {
      static const unsigned RegList1[] = {
        Hexagon::R0, Hexagon::R1, Hexagon::R2, Hexagon::R3, Hexagon::R4,
        Hexagon::R5
      };
      if (unsigned Reg = State.AllocateReg(RegList1, 6)) {
        State.addLoc(CCValAssign::getReg(ValNo, ValVT.getSimpleVT(), Reg,
                                         LocVT.getSimpleVT(), LocInfo));
        return false;
      }
    }

    if (LocVT == MVT::i64 ||
        LocVT == MVT::f64) {
      static const unsigned RegList2[] = {
        Hexagon::D0, Hexagon::D1, Hexagon::D2
      };
      if (unsigned Reg = State.AllocateReg(RegList2, 3)) {
        State.addLoc(CCValAssign::getReg(ValNo, ValVT.getSimpleVT(), Reg,
                                         LocVT.getSimpleVT(), LocInfo));
        return false;
      }
    }
  }

  const Type* ArgTy = LocVT.getTypeForEVT(State.getContext());
  unsigned Alignment =
    State.getTarget().getDataLayout()->getABITypeAlignment(ArgTy);
  unsigned Size =
    State.getTarget().getDataLayout()->getTypeSizeInBits(ArgTy) / 8;

  // If it's passed by value, then we need the size of the aggregate not of
  // the pointer.
  if (ArgFlags.isByVal()) {
    Size = ByValSize;

    // Hexagon_TODO: Get the alignment of the contained type here.
    Alignment = 8;
  }

  unsigned Offset3 = State.AllocateStack(Size, Alignment);
  State.addLoc(CCValAssign::getMem(ValNo, ValVT.getSimpleVT(), Offset3,
                                   LocVT.getSimpleVT(), LocInfo));
  return false;
}


static bool RetCC_Hexagon32_VarArgs(unsigned ValNo, EVT ValVT,
                                    EVT LocVT, CCValAssign::LocInfo LocInfo,
                                    ISD::ArgFlagsTy ArgFlags,
                                    Hexagon_CCState &State,
                                    int NonVarArgsParams,
                                    int CurrentParam,
                                    bool ForceMem) {

  if (LocVT == MVT::i32 ||
      LocVT == MVT::f32) {
    static const unsigned RegList1[] = {
      Hexagon::R0, Hexagon::R1, Hexagon::R2, Hexagon::R3, Hexagon::R4,
      Hexagon::R5
    };
    if (unsigned Reg = State.AllocateReg(RegList1, 6)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT.getSimpleVT(), Reg,
                                       LocVT.getSimpleVT(), LocInfo));
      return false;
    }
  }

  if (LocVT == MVT::i64 ||
      LocVT == MVT::f64) {
    static const unsigned RegList2[] = {
      Hexagon::D0, Hexagon::D1, Hexagon::D2
    };
    if (unsigned Reg = State.AllocateReg(RegList2, 3)) {
      State.addLoc(CCValAssign::getReg(ValNo, ValVT.getSimpleVT(), Reg,
                                       LocVT.getSimpleVT(), LocInfo));
      return false;
    }
  }

  const Type* ArgTy = LocVT.getTypeForEVT(State.getContext());
  unsigned Alignment =
    State.getTarget().getDataLayout()->getABITypeAlignment(ArgTy);
  unsigned Size =
    State.getTarget().getDataLayout()->getTypeSizeInBits(ArgTy) / 8;

  unsigned Offset3 = State.AllocateStack(Size, Alignment);
  State.addLoc(CCValAssign::getMem(ValNo, ValVT.getSimpleVT(), Offset3,
                                   LocVT.getSimpleVT(), LocInfo));
  return false;
}
