//=- PISACombine.td - Define PISA Combine Rules --------------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Target/GlobalISel/Combine.td"

def extra_commute_iconstant_to_rhs : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_SMIN, G_SMAX, G_UMIN, G_UMAX, G_UADDO, G_SADDO,
                           G_UMULO, G_SMULO, G_UMULH, G_SMULH, G_UADDSAT,
                           G_SADDSAT, G_SMULFIX, G_UMULFIX, G_SMULFIXSAT,
                           G_UMULFIXSAT):$root, [{
    return getIConstantVRegVal(${root}->getOperand(1).getReg(), MRI).has_value() &&
        !getIConstantVRegVal(${root}->getOperand(2).getReg(), MRI).has_value();
  }]),
  (apply [{
    Observer.changingInstr(*${root});
    Register LHSReg = ${root}->getOperand(1).getReg();
    Register RHSReg = ${root}->getOperand(2).getReg();
    ${root}->getOperand(1).setReg(RHSReg);
    ${root}->getOperand(2).setReg(LHSReg);
    Observer.changedInstr(*${root});
  }])
>;

def extra_commute_fconstant_to_rhs : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_FMINNUM, G_FMAXNUM, G_FMINNUM_IEEE,
                           G_FMAXNUM_IEEE, G_FMINIMUM, G_FMAXIMUM,
                           G_FADD, G_FMUL):$root, [{
    return getFConstantVRegValWithLookThrough(${root}->getOperand(1).getReg(), MRI).has_value() &&
           !getFConstantVRegValWithLookThrough(${root}->getOperand(2).getReg(), MRI).has_value();
  }]),
  (apply [{
    Observer.changingInstr(*${root});
    Register LHSReg = ${root}->getOperand(1).getReg();
    Register RHSReg = ${root}->getOperand(2).getReg();
    ${root}->getOperand(1).setReg(RHSReg);
    ${root}->getOperand(2).setReg(LHSReg);
    Observer.changedInstr(*${root});
  }])
>;

def extra_commute_intrinsic_iconstant_to_rhs : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_INTRINSIC):$root, [{
    auto IntrinsicID = cast<GIntrinsic>(*${root}).getIntrinsicID();
    auto isFloatArg = false;
    switch (IntrinsicID) {
      default:
        return false;
      case Intrinsic::pisa_fadd:
      case Intrinsic::pisa_fmul:
        isFloatArg = true;
        break;
    }
    return isFloatArg ?
          (getFConstantVRegValWithLookThrough(${root}->getOperand(2).getReg(), MRI).has_value() &&
            !getFConstantVRegValWithLookThrough(${root}->getOperand(3).getReg(), MRI).has_value()) :
          (getIConstantVRegVal(${root}->getOperand(2).getReg(), MRI).has_value() &&
            !getIConstantVRegVal(${root}->getOperand(3).getReg(), MRI).has_value());
  }]),
  (apply [{
    Observer.changingInstr(*${root});
    Register LHSReg = ${root}->getOperand(2).getReg();
    Register RHSReg = ${root}->getOperand(3).getReg();
    ${root}->getOperand(2).setReg(RHSReg);
    ${root}->getOperand(3).setReg(LHSReg);
    Observer.changedInstr(*${root});
  }])
>;

// AND rd1, rs1, C1
//   AND rd2, rd1, C1 - remove copy
def extra_remove_redundant_constant : GICombineRule<
  (defs root:$root, register_matchinfo:$matchinfo),
  (match (wip_match_opcode G_AND, G_OR):$root,
    [{
      auto C1 = getIConstantVRegValWithLookThrough(${root}->getOperand(2).getReg(), MRI);
      if (C1.has_value()) {
        auto *DefMI = getDefIgnoringCopies(${root}->getOperand(1).getReg(), MRI);
        if (DefMI && (${root}->getOpcode() == DefMI->getOpcode())) {
          auto C2 = getIConstantVRegValWithLookThrough(DefMI->getOperand(2).getReg(), MRI);
          if (C2.has_value()) {
            ${matchinfo} = DefMI->getOperand(0).getReg();
            return C1->Value == C2->Value;
          }
        }
      }
      return false;
    }]),
  (apply
    [{
      Helper.replaceSingleDefInstWithReg(*${root}, ${matchinfo});
    }])
>;

// G_[F]CONSTANT i16 x
// G_[F]CONSTANT i16 y
// G_[F]CONSTANT i16 z
// G_[F]CONSTANT i16 w
// G_BUILD_VECTOR x, y, z, w
// => G_CONSTANT i64 [wzyx]
// => G_BITCAST
def build_vector_with_constants : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_BUILD_VECTOR):$root,
    [{
      auto Type = MRI.getType(${root}->getOperand(0).getReg());
      if (Type.isPointerVector())
        return false;
      auto Size = Type.getSizeInBits();
      if ((Size <= 64) && isPowerOf2_32(Size)) {
        for (unsigned I=1; I < ${root}->getNumOperands(); I++) {
          auto C = getAnyConstantVRegValWithLookThrough(${root}->getOperand(I).getReg(), MRI);
          if (!C.has_value()) {
              return false;
          }
        }
        return true;
      }
      return false;
    }]),
  (apply
    [{
      applyBuildVectorWithConstants(*${root});
    }])
>;

// i56 = G_LOAD x
// i32 = G_TRUNC i56
// => i32 = G_LOAD x
def truncated_load : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_TRUNC):$root,
    [{
      auto *DefMI = getDefIgnoringCopies(${root}->getOperand(1).getReg(), MRI);
      auto Type = MRI.getType(DefMI->getOperand(0).getReg());
      auto Size = Type.getSizeInBits();
      auto IsVector = Type.isVector();
      return !isPowerOf2_32(Size) && !IsVector && (DefMI->getOpcode() == TargetOpcode::G_LOAD);
    }]),
  (apply
    [{
      applyTruncatedLoad(*${root});
    }])
>;

// i24 x = G_TRUNC i32
// G_STORE i24 x
// => <4 x i8> y = BITCAST x
// <3 x i8> z = SHUFFLE_VECTOR y, <0,1,2>
// G_STORE <3 x i8> z
def truncated_store : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_STORE):$root,
    [{
      return matchTruncatedStore(*${root});
    }]),
  (apply
    [{
      applyTruncatedStore(*${root});
    }])
>;

// i24 = G_LOAD x
// i32 = G_ZEXT i24
// =>
// i32 loadRes = G_LOAD x
// i32 = AND loadRes, 0x00FFFFFF
def extended_load : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_ZEXT):$root,
    [{
      return matchExtendedLoad(*${root});
    }]),
  (apply
    [{
      applyExtendedLoad(*${root});
    }])
>;

// i24 = G_LOAD x
// => 
// %3:_(s24) = G_ZEXTLOAD %0:reg32b(p0)
// %7:_(s32) = G_CONSTANT i32 2
// %6:reg32b(p0) = G_PTR_ADD %0:reg32b, %7:_(s32)
// %9:_(s24) = G_ZEXTLOAD %6:reg32b(p0)
// %11:_(s24) = G_CONSTANT i24 16
// %10:_(s24) = G_SHL %9:_, %11:_(s24)
// %12:_(s24) = G_OR %3:_, %10:_
// Similar transformation is done for G_STORE.
def expand_nonpowerof2_load_store : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_LOAD, G_STORE):$root,
    [{
      return matchExpandNonPowerOf2LoadStore(*${root});
    }]),
  (apply
    [{
      applyExpandNonPowerOf2LoadStore(*${root});
    }])
>;

// If the value we store resulted from a G_LOAD that the rule above expanded,
// then we can use the individual load values directly instead of merging into
// one integer, and then splitting it again. Size modifications between the
// loads and store (i.e. SEXT/ZEXT/TRUNC) are also supported by
// truncating/extending the relevant load results (or, in the case of
// truncation, ignoring some loads entirely)
//
// Below is a simple example where we simply load and store an i56:
//
//  bb.1.entry:
//    %0:reg32b(p0) = functionParameter_32b 0
// -> %3:_(s56) = G_ZEXTLOAD %0:reg32b(p0) ::
//      (load (s32) from %ir.dst)
//    %4:_(s56) = G_CONSTANT i56 0
//    %7:_(s32) = G_CONSTANT i32 4
//    %6:reg32b(p0) = G_PTR_ADD %0:reg32b, %7:_(s32)
// -> %9:_(s56) = G_ZEXTLOAD %6:reg32b(p0) ::
//      (load (s16) from %ir.dst + 4, align 4)
//    %11:_(s56) = G_CONSTANT i56 32
//    %10:_(s56) = G_SHL %9:_, %11:_(s56)
//    %12:_(s56) = G_OR %3:_, %10:_
//    %14:_(s32) = G_CONSTANT i32 6
//    %13:reg32b(p0) = G_PTR_ADD %0:reg32b, %14:_(s32)
// -> %16:_(s56) = G_ZEXTLOAD %13:reg32b(p0) ::
//      (load (s8) from %ir.dst + 6, align 2, basealign 4)
//    %18:_(s56) = G_CONSTANT i56 48
//    %17:_(s56) = G_SHL %16:_, %18:_(s56)
//    %19:_(s56) = G_OR %12:_, %17:_
//    G_STORE %19:_(s56), %0:reg32b(p0) ::
//      (store (s56) into %ir.dst, align 4)
//    ret
def loadVector_matchinfo : GIDefMatchData<"SmallVector<std::pair<MachineInstr *, unsigned>, 8>">;
def MI_matchinfo : GIDefMatchData<"MachineInstr *">;
def simplify_nonpowerof2_load_store_chain : GICombineRule<
  (defs root:$root, loadVector_matchinfo:$matchinfo, MI_matchinfo:$MI_matchinfo),
  (match (wip_match_opcode G_STORE):$root,
    [{
      return matchSimplifyNonPowerOf2LoadStoreChain(*${root}, ${matchinfo}, ${MI_matchinfo});
    }]),
  (apply
    [{
      applySimplifyNonPowerOf2LoadStoreChain(*${root}, ${matchinfo}, ${MI_matchinfo});
    }])
>;

// i1 C = G_CMP ne i? A, 0
// i? S = G_SELECT i1 C, i? LHS, i? RHS
// => S = sel.? LHS, RHS, A
// ... or ... 
// i1 C = G_CMP eq i? A, 0
// i? S = G_SELECT i1 C, i? LHS, i? RHS
// => S = sel.? RHS, LHS, A
def compare_select_reg : GICombineRule<
  (defs root:$SelectMI),
  (match (wip_match_opcode G_SELECT):$SelectMI,
    [{
      return matchCompareSelect(${SelectMI});
    }]),
  (apply
    [{
      applyCompareSelect(${SelectMI});
    }])
>;

// arcp G_FDIV %0, %1
// => 
// %rcp = arcp G_INTRINSIC intrinsic(@llvm.pisa.frcp), %1
// %result = arcp G_FMUL %0, %rcp
def fdiv_to_rcp_fmul : GICombineRule<
  (defs root:$root, build_fn_matchinfo:$matchinfo),
  (match (wip_match_opcode G_FDIV):$root,
    [{ return matchFDivToRcpFMul(*${root}, ${matchinfo}); }]),
  (apply [{ Helper.applyBuildFn(*${root}, ${matchinfo}); }])
>;

//%2:registers(s32) = G_SHL %0:reg32b, 2
//%4:registers(s32) = G_ADD %2:registers, %3:registers
// => %4:registers(s32) = G_PISA_SMAD %0:reg32b, 4, %3:registers
def shl_add_to_mad : GICombineRule<
  (defs root:$root, build_fn_matchinfo:$matchinfo),
  (match (wip_match_opcode G_ADD):$root,
    [{ return matchShlAddToMad(*${root}, ${matchinfo}); }]),
  (apply [{ Helper.applyBuildFn(*${root}, ${matchinfo}); }])>;


// i32 %lo = G_TRUNC i32 a 16
// i32 %shift = G_LSHR i32 a 16
// %hi = G_TRUNC i32 %shift to i16
// => %1 = G_BITCAST i32 a to <2 x 16>
// => %hi = G_EXTRACT_VECTOR_ELT <2 x 16> %1, 1
// => %lo = G_EXTRACT_VECTOR_ELT <2 x 16> %1, 0 
def trunc_shift_to_vector_extract : GICombineRule<
  (defs root:$TruncMI),
  (match (wip_match_opcode G_TRUNC):$TruncMI,
    [{ return matchTruncatedShift(*${TruncMI}); }]),
  (apply
    [{ applyTruncatedShift(*${TruncMI}); }])
>;

// %5 = G_CONSTANT i16 x
// %6 = G_CONSTANT i16 y
// %7 = G_AND %5:_, %6:_
// %2 = G_TRUNC %7 to i8
// => %2 = COPY (i8)(x & y)
def const_and_trunc_to_copy : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_TRUNC):$root,
    [{
      auto *AndMI = getDefIgnoringCopies(${root}->getOperand(1).getReg(), MRI);
      if (AndMI->getOpcode() == TargetOpcode::G_AND) {
        auto C1 = getIConstantVRegValWithLookThrough(AndMI->getOperand(1).getReg(), MRI);
        auto C2 = getIConstantVRegValWithLookThrough(AndMI->getOperand(2).getReg(), MRI);
        if (C1.has_value() && C2.has_value()) {
          auto Value = C1->Value & C2->Value;
          return (Value.getZExtValue() >> MRI.getType(${root}->getOperand(0).getReg()).getSizeInBits()) == 0;
        }
      }
      return false;
    }]),
  (apply
    [{ 
      auto *AndMI = getDefIgnoringCopies(${root}->getOperand(1).getReg(), MRI);
      auto C1 = getIConstantVRegValWithLookThrough(AndMI->getOperand(1).getReg(), MRI);
      auto C2 = getIConstantVRegValWithLookThrough(AndMI->getOperand(2).getReg(), MRI);
      auto Value = (C1->Value & C2->Value).trunc(MRI.getType(${root}->getOperand(0).getReg()).getSizeInBits());
      Helper.replaceInstWithConstant(*${root}, Value);
    }])
>;

// G_AND (G_ZEXT x), C -> G_ZEXT (G_AND x, trunc(C)).
// CodeGenPrepare can form the widened mask, but keeping the mask narrow lets
// address folding see the original arithmetic shape.
def zext_and_to_and_zext : GICombineRule<
  (defs root:$root, build_fn_matchinfo:$matchinfo),
  (match (wip_match_opcode G_AND):$root,
    [{ return matchZExtAndToAndZExt(*${root}, ${matchinfo}); }]),
  (apply [{ Helper.applyBuildFn(*${root}, ${matchinfo}); }])>;

// %114:_(<5 x s32>) = G_EXTRACT_SUBVECTOR %113:_(<8 x s32>), 0
// %65:_(s32) = G_EXTRACT_VECTOR_ELT %114:_(<5 x s32>), %66:_(s32)
// => %65:_(s32) = G_EXTRACT_VECTOR_ELT %113:_(<8 x s32>), %66:_(s32)

def extract_subvector_to_extract_elt : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_EXTRACT_VECTOR_ELT):$root,
    [{
      auto *SubvectorMI = getDefIgnoringCopies(${root}->getOperand(1).getReg(), MRI);
      if (!SubvectorMI) {
        return false;
      }
      if (SubvectorMI->getOpcode() != TargetOpcode::G_EXTRACT_SUBVECTOR) {
        return false;
      }
      if (SubvectorMI->getOperand(2).isReg()){
       // Skip to avoid needing to generate add instruction for index adjustment.
       return false;
      }
      auto C = getIConstantVRegValWithLookThrough(${root}->getOperand(2).getReg(), MRI);
      if (!C.has_value()){
        if (SubvectorMI->getOperand(2).getImm() != 0){
        // Skip to avoid needing to generate add instruction for index adjustment.
          return false;
         }
      }
      LLVM_DEBUG(dbgs() << "  Success: extract_subvector_to_extract_elt rule matches!\n");
      return true;
    }]),
  (apply
    [{
      auto *SubvectorMI = getDefIgnoringCopies(${root}->getOperand(1).getReg(), MRI);
      Register SrcReg = SubvectorMI->getOperand(1).getReg();
      Register IdxReg = ${root}->getOperand(2).getReg();
      Register DstReg = ${root}->getOperand(0).getReg();
      auto C = getIConstantVRegValWithLookThrough(${root}->getOperand(2).getReg(), MRI);
      if (C.has_value() && SubvectorMI->getOperand(2).isImm()){
        uint64_t NewIdx = C->Value.getZExtValue() + SubvectorMI->getOperand(2).getImm();
        B.buildExtractVectorElementConstant(DstReg, SrcReg, NewIdx);
      }
      else {
         B.buildExtractVectorElement(DstReg, SrcReg, IdxReg);
      }
      ${root}->eraseFromParent();
    }])
>;

// %28:_(<8 x i32>) = G_BUILD_VECTOR %91:_(i32), %92:_(i32), %93:_(i32), %94:_(i32), %95:_(i32), %96:_(i32), %97:_(i32), %98:_(i32)
// %37:_(<4 x i32>) = G_EXTRACT_SUBVECTOR %28:_(<8 x i32>), 0
// => %37:_(<4 x i32>) = G_BUILD_VECTOR %91:_(i32), %92:_(i32), %93:_(i32), %94:_(i32)
def extract_subvector_build_vector_matchdata : GIDefMatchData<"SmallVector<Register, 8>">;
def extract_subvector_build_vector : GICombineRule<
  (defs root:$root, extract_subvector_build_vector_matchdata:$matchinfo),
  (match (wip_match_opcode G_EXTRACT_SUBVECTOR):$root,
    [{ return matchExtractSubvectorBuildVector(*${root}, ${matchinfo}); }]),
  (apply
    [{ applyExtractSubvectorBuildVector(*${root}, ${matchinfo}); }])
>;

// %29:_(<64 x s32>) = IMPLICIT_DEF
// %30:_(<64 x s32>) = G_INSERT_SUBVECTOR %29:_, %7:_(<8 x s32>), 0
// %31:_(<64 x s32>) = G_INSERT_SUBVECTOR %30:_, %10:_(<8 x s32>), 8
// %32:_(<64 x s32>) = G_INSERT_SUBVECTOR %31:_, %13:_(<8 x s32>), 16
// %33:_(<64 x s32>) = G_INSERT_SUBVECTOR %32:_, %16:_(<8 x s32>), 24
// %34:_(<64 x s32>) = G_INSERT_SUBVECTOR %33:_, %19:_(<8 x s32>), 32
// %35:_(<64 x s32>) = G_INSERT_SUBVECTOR %34:_, %22:_(<8 x s32>), 40
// %36:_(<64 x s32>) = G_INSERT_SUBVECTOR %35:_, %25:_(<8 x s32>), 48
// %37:_(<64 x s32>) = G_INSERT_SUBVECTOR %36:_, %28:_(<8 x s32>), 56
// %4:_(<32 x s32>) = G_EXTRACT_SUBVECTOR %37:_(<64 x s32>), 32
// =>
// %29:_(<32 x s32>) = IMPLICIT_DEF
// %34:_(<32 x s32>) = G_INSERT_SUBVECTOR %33:_, %19:_(<8 x s32>), 0
// %35:_(<32 x s32>) = G_INSERT_SUBVECTOR %34:_, %22:_(<8 x s32>), 8
// %36:_(<32 x s32>) = G_INSERT_SUBVECTOR %35:_, %25:_(<8 x s32>), 16
// %37:_(<32 x s32>) = G_INSERT_SUBVECTOR %36:_, %28:_(<8 x s32>), 24
// %4:_(<32 x s32>) = COPY %37:_(<32 x s32>)
def extract_subvector_partial_matchdata : GIDefMatchData<"SmallVector<MachineInstr *, 8>">;
def extract_subvector_partial : GICombineRule<
  (defs root:$root, extract_subvector_partial_matchdata:$matchinfo),
  (match (wip_match_opcode G_EXTRACT_SUBVECTOR):$root,
    [{ return matchExtractSubvectorPartial(*${root}, ${matchinfo}); }]),
  (apply
    [{ applyExtractSubvectorPartial(*${root}, ${matchinfo}); }])
>;

// A G_BUILD_VECTOR whose elements are consecutive lanes of a single wider
// source vector (extracted via G_UNMERGE_VALUES at lanes base, base+1) is a
// sub-vector slice of that source. Rewrite it to G_EXTRACT_SUBVECTOR so ISel
// emits a direct composite sub-register COPY (.xy / .zw) that register
// coalescing folds, instead of an element-wise gather. This is what lets the
// .zw high half become copy-free (the .xy low half already coalesced).
// %70,%71,%72,%73 = G_UNMERGE_VALUES %0(<4 x s16>)
// %23:_(<2 x s16>) = G_BUILD_VECTOR %72, %73          ; lanes 2,3 of %0
// => %23:_(<2 x s16>) = G_EXTRACT_SUBVECTOR %0, 2
def build_vector_from_unmerge_lanes_matchdata : GIDefMatchData<"std::tuple<Register, int64_t>">;
def build_vector_from_unmerge_lanes : GICombineRule<
  (defs root:$root, build_vector_from_unmerge_lanes_matchdata:$matchinfo),
  (match (wip_match_opcode G_BUILD_VECTOR):$root,
    [{ return matchBuildVectorFromUnmergeLanes(*${root}, ${matchinfo}); }]),
  (apply
    [{ applyBuildVectorFromUnmergeLanes(*${root}, ${matchinfo}); }])
>;

// A G_BUILD_VECTOR that concatenates whole sub-vectors (each source's lanes
// appear in order) is a G_CONCAT_VECTORS. Rewrite it so ISel writes each
// source into its composite sub-register slice (.xy / .zw) directly, instead
// of an element-wise rebuild.
// %56,%57 = G_UNMERGE_VALUES %14(<2 x s16>) ; %36,%37 = G_UNMERGE_VALUES %29
// %2:_(<4 x s16>) = G_BUILD_VECTOR %56, %57, %36, %37
// => %2:_(<4 x s16>) = G_CONCAT_VECTORS %14, %29
def build_vector_concat_subvectors_matchdata : GIDefMatchData<"SmallVector<Register, 4>">;
def build_vector_concat_subvectors : GICombineRule<
  (defs root:$root, build_vector_concat_subvectors_matchdata:$matchinfo),
  (match (wip_match_opcode G_BUILD_VECTOR):$root,
    [{ return matchBuildVectorConcatSubvectors(*${root}, ${matchinfo}); }]),
  (apply
    [{ applyBuildVectorConcatSubvectors(*${root}, ${matchinfo}); }])
>;

// %6 = G_CONSTANT i32 3
// %2 = G_BUILD_VECTOR %23, %25, %27, %29, %30
// %3 = G_EXTRACT_VECTOR_ELT %2, %6
// => %3 = COPY %29
def build_vector_extract_to_copy : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_EXTRACT_VECTOR_ELT):$root,
    [{
      auto *BuildMI = getDefIgnoringCopies(${root}->getOperand(1).getReg(), MRI);
      if (BuildMI->getOpcode() == TargetOpcode::G_BUILD_VECTOR) {
        auto C = getIConstantVRegValWithLookThrough(${root}->getOperand(2).getReg(), MRI);
        return C.has_value();
      }
      return false;
    }]),
  (apply
    [{ 
      auto [BuildMI, RegIdx] = PISA::getDefIgnoringBitcasts(${root}->getOperand(1).getReg(), MRI);
      auto C = getIConstantVRegValWithLookThrough(${root}->getOperand(2).getReg(), MRI);
      auto DstReg = ${root}->getOperand(0).getReg();
      auto OpndReg = BuildMI->getOperand(C->Value.getZExtValue() + 1).getReg();
      if (DstReg != OpndReg)
        B.buildCopy(DstReg, OpndReg);
      ${root}->eraseFromParent();
    }])
>;

// s16 %1 = G_CONSTANT i16 1
// s32 %2 = G_CONSTANT i32 2
// s16 = G_SHL %1, %2
// => s16 = G_CONSTANT i16 4
// note: occurs when dealing with vectors of i1
def shifts_of_constants : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_SHL, G_LSHR, G_ASHR):$root,
    [{
      auto *Src0MI = getDefIgnoringCopies(${root}->getOperand(1).getReg(), MRI);
      auto *Src1MI = getDefIgnoringCopies(${root}->getOperand(2).getReg(), MRI);
      return (Src0MI->getOpcode() == TargetOpcode::G_CONSTANT)
        &&  (Src1MI->getOpcode() == TargetOpcode::G_CONSTANT);
    }]),
  (apply
    [{
      applyShiftOfConstants(${root});
    }])
>;

// %5:_(s32) = G_CONSTANT i32 3
// %4:_(s32) = G_INTRINSIC intrinsic(@llvm.pisa.lane.id)
// %6:_(s32) = nuw nsw G_SHL %4:_, %5:_(s32)
// %7:_(s64) = nneg G_ZEXT %6:_(s32)
// %31:_(s64) = G_CONSTANT i64 2
// %9:_(s64) = nuw nsw G_SHL %7:_, %31:_(s64)
// => %4:_(s32) = G_INTRINSIC intrinsic(@llvm.pisa.lane.id)
//    %5:_(s32) = G_CONSTANT i32 5
//    %6:_(s32) = nuw nsw G_SHL %4:_, %5:_(s32)
//    %7:_(s64) = nneg G_ZEXT %6:_(s32)
def lane_id_shl_chain : GICombineRule<
  (defs root:$root, build_fn_matchinfo:$matchinfo),
  (match (wip_match_opcode G_SHL):$root,
    [{ return matchLaneIdLeftShiftChain(*${root}, ${matchinfo}); }]),
  (apply [{ Helper.applyBuildFn(*${root}, ${matchinfo}); }])>;

// A(16) = G_EXTRACT_VECTOR_ELT ARG(32), 0
// B(16) = G_EXTRACT_VECTOR_ELT ARG(32), 1
// C(<2x16>) = G_BUILD_VECTOR A, B
// D(32) = G_BITCAST C(<2x16>)
// => D(32) = COPY ARG(32)
def remove_redundant_moves_pre : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_BITCAST):$root,
    [{ return matchRedundantMovesPre(*${root}); }]),
  (apply
    [{ applyRedundantMovesPre(*${root}); }])
>;
// A(<2x16>) = G_BITCAST ARG(32)
// B(16), C(16) = G_UNMERGE_VALUES A(<2x16)
// D(<2x16>) = G_BUILD_VECTOR B, C
// E(32) = G_BITCAST D(<2x16>)
// => E(32) = COPY ARG(32)
def remove_redundant_moves_post : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_BITCAST):$root,
    [{ return matchRedundantMovesPost(*${root}); }]),
  (apply
    [{ applyRedundantMovesPost(*${root}); }])
>;

// %4:_(<32 x s8>) = G_BITCAST %3:_(<4 x s64>)
// %5:_(<8 x s32>) = G_BITCAST %4:_(<32 x s8>)
// => %5:_(<8 x s32>) = G_BITCAST %3:_(<4 x s64>)
def redundant_bitcasts : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_BITCAST):$root,
    [{
      auto DstReg = ${root}->getOperand(0).getReg();
      auto SrcReg = ${root}->getOperand(1).getReg();
      auto *SrcMI = getDefIgnoringCopies(SrcReg, MRI);
      // do not remove casts between integers and floating point
      if (MRI.getType(DstReg).getScalarType().isFloat() != MRI.getType(SrcReg).getScalarType().isFloat())
        return false;
      // favor <? x s32> as we can use extract/insert ops
      return (SrcMI->getOpcode() == TargetOpcode::G_BITCAST)
        && (MRI.getType(DstReg).getScalarSizeInBits() == 32);
    }]),
  (apply
    [{
      auto DstReg = ${root}->getOperand(0).getReg();
      auto SrcReg = ${root}->getOperand(1).getReg();
      auto *SrcMI = getDefIgnoringCopies(SrcReg, MRI);
      auto OrigSrcReg = SrcMI->getOperand(1).getReg();
      if (MRI.getType(DstReg) == MRI.getType(OrigSrcReg))
        B.buildCopy(DstReg, OrigSrcReg);
      else
        B.buildBitcast(DstReg, OrigSrcReg);
      ${root}->eraseFromParent();
    }])
>;

// A(s16) = G_EXTRACT_VECTOR_ELT ARG(<N x s16), 0
// B(s16) = G_EXTRACT_VECTOR_ELT ARG(<N x s16), 1
// ...
// X(s16) = G_EXTRACT_VECTOR_ELT ARG(<N x s16>), N-1
// Y(<N x s16>) = G_BUILD_VECTOR A, B, ..., X
// => Y(<N x s16>) = COPY ARG(<N x s16>)
def extract_all_build_vector : GICombineRule<
  (defs root:$root, register_matchinfo:$matchinfo),
  (match (wip_match_opcode G_BUILD_VECTOR):$root,
    [{ return matchExtractAllToBuildVector(*${root}, ${matchinfo}); }]),
  (apply
    [{ applyExtractAllToBuildVector(*${root}, ${matchinfo}); }])
>;

// Fold int2ptr(ptr2int(x)) -> x
// - LLVM version does not consider difference
//   in size between PTR and integer
def p2i_to_i2p_fixed: GICombineRule<
  (defs root:$dst, register_matchinfo:$matchinfo),
  (match (G_PTRTOINT $int, $ptr),
         (G_INTTOPTR $dst, $int):$root,
    [{
      auto PTy = MRI.getType(${ptr}.getReg());
      auto TTy = MRI.getType(${int}.getReg());
      auto DTy = MRI.getType(${dst}.getReg());
      if (PTy.getScalarSizeInBits() != TTy.getScalarSizeInBits())
        return false;
      if (TTy.getScalarSizeInBits() != DTy.getScalarSizeInBits())
        return false;
      // A COPY between pointers of different address spaces is illegal
      if (PTy != DTy)
        return false;
      ${matchinfo} = ${ptr}.getReg();
      return true;
    }]),
  (apply
    [{
      B.buildCopy(${dst}, ${ptr});
      ${root}->eraseFromParent();
    }])
>;

def or_and_to_bfi : GICombineRule<
  (defs root:$root, build_fn_matchinfo:$matchinfo),
  (match (wip_match_opcode G_OR):$root,
    [{ return matchOrAndToBfi(*${root}, ${matchinfo}); }]),
  (apply [{ Helper.applyBuildFn(*${root}, ${matchinfo}); }])>;

// s32 %0 = G_CONSTANT i32 32
// s32 %1 = G_SUB i32 %0, %width
// s32 %2 = G_SUB i32 %1, %offset
// s32 %3 = G_SHL i32 %data, %2
// s32 %4 = G_LSHR i32 %3, %1
// => bfe %data %width %offset
def shift_sub_to_bfe : GICombineRule<
  (defs root:$root, build_fn_matchinfo:$matchinfo),
  (match (wip_match_opcode G_LSHR, G_ASHR):$root,
    [{ return matchShiftSubToBfe(*${root}, ${matchinfo}); }]),
  (apply [{ Helper.applyBuildFn(*${root}, ${matchinfo}); }])>;

// s32 %maskedwidth = G_AND %width, 63
// s32 %maskedoffset = G_AND %offset, 31
// bfe %data %maskedwidth %maskedoffset
// =>
// bfe %data %width %offset
def and_bitfield_to_bitfield : GICombineRule<
  (defs root:$root, build_fn_matchinfo:$matchinfo),
  (match (wip_match_opcode G_INTRINSIC, G_UBFX, G_SBFX):$root,
    [{ return matchAndBitfieldToBitfield(*${root}, ${matchinfo}); }]),
  (apply [{ Helper.applyBuildFn(*${root}, ${matchinfo}); }])>;

// This transforms an unoptimal pattern for fcmp + zext i1->i8
//  %14:_(<2 x s16>) = G_BITCAST %CmpRes:_(s32)
//  %20:_(s16), %21:_(s16) = G_UNMERGE_VALUES %14:_(<2 x s16>)
//  %22:_(s8) = G_TRUNC %20:_(s16)
//  %23:_(s8) = G_TRUNC %21:_(s16)
//  %17:_(<2 x s8>) = G_BUILD_VECTOR %22:_(s8), %23:_(s8)
//  %28:_(s16) = G_CONSTANT i16 257
//  %19:_(<2 x s8>) = G_BITCAST %28:_(s16)
//  %24:_(s16) = G_BITCAST %17:_(<2 x s8>)
//  %25:_(s16) = G_BITCAST %19:_(<2 x s8>)
//  %26:_(s16) = G_AND %24:_, %25:_
//  %res:_(<2 x s8>) = G_BITCAST %26:_(s16)
// => bfe %1, %CmpRes, 1, 0
//    bfe %2, %CmpRes, 1, 16
//    bfi %3, %1, %2, 8
//    trunc.16.32 %res, %3
def v2_i1_zext_to_bfe_bfi : GICombineRule<
  (defs root:$root, register_matchinfo:$matchinfo),
  (match (wip_match_opcode G_BITCAST):$root,
    [{ return matchV2i1ZextToBfeBfi(*${root}, ${matchinfo}); }]),
  (apply [{ applyV2i1ZextToBfeBfi(*${root}, ${matchinfo}); }])>;

// frcp(sqrt(x))       -> frsqrt(x)
// sqrt(frcp(x))       -> frsqrt(x)
// frcp(fabs(sqrt(x))) -> fabs(frsqrt(x))
def rcp_sqrt_to_rsqrt : GICombineRule<
  (defs root:$root, build_fn_matchinfo:$matchinfo),
  (match (wip_match_opcode G_INTRINSIC, G_FSQRT, G_FDIV):$root,
    [{ return matchRcpSqrtToRsqrt(*${root}, ${matchinfo}); }]),
  (apply [{ Helper.applyBuildFn(*${root}, ${matchinfo}); }])>;

// s32 %floor = G_FFLOOR %x
// s32 %dst   = G_FSUB %x, %floor
// => s32 %dst = frc %x
// The plain G_FSUB fold requires contract on both the fsub and the floor. Also
// matches the pisa.fsub intrinsic form, but only for round.towardzero with no
// saturation.
def sub_floor_to_frc : GICombineRule<
  (defs root:$root, build_fn_matchinfo:$matchinfo),
  (match (wip_match_opcode G_FSUB, G_INTRINSIC):$root,
    [{ return matchSubFloorToFrc(*${root}, ${matchinfo}); }]),
  (apply [{ Helper.applyBuildFn(*${root}, ${matchinfo}); }])>;

// sequence repeated multiple times ...
// %39:_(s8) = G_EXTRACT_VECTOR_ELT %4:_(<32 x s8>), [0..31]
// %41:_(s32) = G_ZEXT %39:_(s8)
// %42:_(s32) = nuw nsw G_SHL %41:_, [0,8,16,24]
// %43:_(s32) = disjoint G_OR %38:_, %42:_
// followed by ...
// %157:_(<8 x s32>) = G_BUILD_VECTOR %43:_(s32), ...
// => %157:_(<8 x s32>) = G_BITCAST %4:_(<32 x s8>)
def extract_insert_to_bitcast : GICombineRule<
  (defs root:$root, register_matchinfo:$matchinfo),
  (match (wip_match_opcode G_BUILD_VECTOR):$root,
    [{ return matchExtractInsertToBitcast(*${root}, ${matchinfo}); }]),
  (apply
    [{ applyExtractInsertToBitcast(*${root}, ${matchinfo}); }])
>;

// %374:_(s16) = G_CONSTANT i16 1
// %375:_(s16) = G_AND %321:_, %374:_
// %421:_(s16) = G_CONSTANT i16 0
// %422:_(s16) = G_CONSTANT i16 1
// %405:_(s16) = G_PISA_SELECT %375:_, %422:_, %421:_
// => %374:_(s16) = G_CONSTANT i16 1
// => %375:_(s16) = G_AND %321:_, %374:_
// => %405:_(s16) = COPY %375
def and_select : GICombineRule<
  (defs root:$root, register_matchinfo:$matchinfo),
  (match (wip_match_opcode G_PISA_SELECT):$root,
    [{ return matchAndSelect(*${root}, ${matchinfo}); }]),
  (apply
    [{ applyAndSelect(*${root}, ${matchinfo}); }])
>;

// %23:_(<4 x s8>) = G_BITCAST %1:_(s32)
//  %2:_(s8) = G_EXTRACT_VECTOR_ELT %23:_(<4 x s8>), %25:_(s32)
//  %7:_(s32) = G_LSHR %1:_, %6:_(s32)
//  %8:_(s8) = G_TRUNC %7:_(s32)
//  %12:_(s32) = G_LSHR %1:_, %11:_(s32)
//  %13:_(s8) = G_TRUNC %12:_(s32)
//  %24:_(s32) = G_CONSTANT i32 3
//  %18:_(s8) = G_EXTRACT_VECTOR_ELT %23:_(<4 x s8>), %24:_(s32)
//  %19:_(<4 x s8>) = G_BUILD_VECTOR %2:_(s8), %8:_(s8), %13:_(s8), %18:_(s8)
// => %19:_(<4 x s8>) = G_BITCAST %1:_(s32)
def extract_build_vector_to_bitcast : GICombineRule<
  (defs root:$root, register_matchinfo:$matchinfo),
  (match (wip_match_opcode G_BUILD_VECTOR):$root,
    [{ return matchExtractBuildVectorToBitcast(*${root}, ${matchinfo}); }]),
  (apply
    [{ applyExtractBuildVectorToBitcast(*${root}, ${matchinfo}); }])
>;

// Reduce predicates in s32.
// %12:_(s1) = G_CONSTANT i1 true
// %3:_(<3 x s1>) = G_FCMP floatpred(oeq), %0:regv3_32b(<3 x s32>), %1:_
// %4:_(s1) = G_EXTRACT_VECTOR_ELT %3:_(<3 x s1>), %5:_(s32)
// %6:_(s1) = G_EXTRACT_VECTOR_ELT %3:_(<3 x s1>), %7:_(s32)
// %8:_(s1) = G_EXTRACT_VECTOR_ELT %3:_(<3 x s1>), %9:_(s32)
// %10:_(s1) = G_AND %4:_, %6:_
// %11:_(s1) = G_AND %10:_, %8:_
// %13:_(s1) = G_ICMP intpred(eq), %11:_(s1), %12:_
// => %3:_(<3 x s1>) = G_FCMP floatpred(oeq), %0:regv3_32b(<3 x s32>), %1:_
//    %18:_(<3 x s32>) = G_SEXT %3:_(<3 x s1>)
//    %20:_(s32) = G_CONSTANT i32 0
//    %19:_(s32) = G_EXTRACT_VECTOR_ELT %18:_(<3 x s32>), %20:_(s32)
//    %23:_(s32) = G_CONSTANT i32 1
//    %21:_(s32) = G_EXTRACT_VECTOR_ELT %18:_(<3 x s32>), %23:_(s32)
//    %22:_(s32) = G_AND %19:_, %21:_
//    %26:_(s32) = G_CONSTANT i32 2
//    %24:_(s32) = G_EXTRACT_VECTOR_ELT %18:_(<3 x s32>), %26:_(s32)
//    %25:_(s32) = G_AND %22:_, %24:_
//    %27:_(s32) = G_CONSTANT i32 -1
//    %13:_(s1) = G_ICMP intpred(eq), %25:_(s32), %27:_
// Supports patterns with sext too:
// %13:_(s8) = G_CONSTANT i8 -1
// %3:_(<3 x s1>) = G_FCMP floatpred(oeq), %0:regv3_32b(<3 x s32>), %1:_
// %4:_(<3 x s8>) = G_SEXT %3:_(<3 x s1>)
// %5:_(s8) = G_EXTRACT_VECTOR_ELT %4:_(<3 x s8>), %6:_(s32)
// %7:_(s8) = G_EXTRACT_VECTOR_ELT %4:_(<3 x s8>), %8:_(s32)
// %9:_(s8) = G_EXTRACT_VECTOR_ELT %4:_(<3 x s8>), %10:_(s32)
// %11:_(s8) = G_AND %5:_, %7:_
// %12:_(s8) = G_AND %11:_, %9:_
// %14:_(s1) = G_ICMP intpred(eq), %12:_(s8), %13:_
def reduce_predicates : GICombineRule<
  (defs root:$root, build_fn_matchinfo:$matchinfo),
  (match (wip_match_opcode G_AND, G_OR, G_XOR):$root,
    [{ return matchReducePredicates(*${root}, ${matchinfo}); }]),
  (apply [{ Helper.applyBuildFn(*${root}, ${matchinfo}); }])>;

// %23:_(s32) = G_CONSTANT i32 8
// %17:_(s8) = G_TRUNC %1:reg32b(s32)
// %24:_(s32) = G_LSHR %1:reg32b, %23:_(s32)
// %25:_(s8) = G_TRUNC %24:_(s32)
// %28:_(s8) = G_TRUNC %2:reg32b(s32)
// %31:_(s32) = G_LSHR %2:reg32b, %23:_(s32)
// %32:_(s8) = G_TRUNC %31:_(s32)
///%149:_(<4 x s8>) = G_BUILD_VECTOR %17:_(s8), %25:_(s8), %28:_(s8), %32:_(s8)
// %141:_(s32) = G_BITCAST %149:_(<4 x s8>)
// => %A = G_AND %1, 0xFFFF
// => %B = G_SHL %2, 16
// => %141 = G_OR %A, %B
def reg2_matchdata : GIDefMatchData<"Reg2MatchInfo">;
def build_reg_from_2 : GICombineRule<
  (defs root:$root, reg2_matchdata:$matchinfo),
  (match (wip_match_opcode G_BITCAST):$root,
    [{ return matchBuildRegFrom2(*${root}, ${matchinfo}); }]),
  (apply
    [{ applyBuildRegFrom2(*${root}, ${matchinfo}); }])
>;

// %9:_(s1) = G_FCMP floatpred(oge), %1:reg32b(s32), %2:reg32b
// %22:_(s32) = G_CONSTANT i32 0
// %23:_(s32) = G_CONSTANT i32 -1
// %10:_(s32) = G_SELECT %9:_(s1), %23:_, %22:_
// %11:_(s32) = G_CONSTANT i32 -1
// %12:_(s32) = G_XOR %10:_, %11:_
// %13:_(s32) = G_CONSTANT i32 -1
// %6:_(s1) = G_ICMP intpred(eq), %12:_(s32), %13:_
// G_BRCOND %6:_(s1), %bb.4
// G_BR %bb.3
// => %9:_(s1) = G_FCMP floatpred(oge), %1:reg32b(s32), %2:reg32b
// => G_BRCOND %6:_(s1), %bb.3
// => G_BR %bb.4
def opt_fcmp_brcond_by_inverting_cond_matchdata : GIDefMatchData<"std::tuple<MachineInstr *, Register>">;
def opt_fcmp_brcond_by_inverting_cond : GICombineRule<
  (defs root:$root, opt_fcmp_brcond_by_inverting_cond_matchdata:$matchinfo),
  (match (wip_match_opcode G_BR):$root,
         [{ return matchFCmpInvertedCond(*${root}, ${matchinfo}); }]),
  (apply [{ applyFCmpInvertedCond(*${root}, ${matchinfo}); }])
>;

def cmp_and_all_ones : GICombineRule<
  (defs root:$root, register_matchinfo:$matchinfo),
  (match (wip_match_opcode G_ICMP):$root,
    [{ return matchCmpAndAllOnes(*${root}, VT, ${matchinfo}); }]),
  (apply
    [{ applyCmpAndAllOnes(*${root}, ${matchinfo}); }])
>;

def sink_trunc_matchdata : GIDefMatchData<"std::tuple<Register, int64_t>">;
def sink_trunc : GICombineRule<
  (defs root:$root, sink_trunc_matchdata:$matchinfo),
  (match (wip_match_opcode G_AND, G_OR, G_XOR):$root,
    [{ return matchSinkTrunc(*${root}, ${matchinfo}); }]),
  (apply
    [{ applySinkTrunc(*${root}, ${matchinfo}); }])
>;

def trunc_trunc : GICombineRule<
  (defs root:$root, register_matchinfo:$matchinfo),
  (match (wip_match_opcode G_TRUNC):$root,
    [{ return matchTruncTrunc(*${root}, ${matchinfo}); }]),
  (apply
    [{ applyTruncTrunc(*${root}, ${matchinfo}); }])
>;

// %1:registers(s32) = G_ABS %0:reg32b
// %4:registers(s32) = G_INTRINSIC_CONVERGENT intrinsic(@llvm.pisa.ired),
// IRedOp::UMAX, %1:registers(s32), %2:reg32b(s32), %3:reg32b(s32)
// => %1:registers(s32) = G_INTRINSIC_CONVERGENT intrinsic(@llvm.pisa.ired),
// IRedOp::ABSMAX, %0:registers(s32), %2:reg32b(s32), %3:reg32b(s32)
def abs_redmax_to_redabsmax : GICombineRule<
  (defs root:$root, build_fn_matchinfo:$matchinfo),
  (match (wip_match_opcode G_INTRINSIC_CONVERGENT):$root,
    [{ return matchAbsRedMaxToRedAbsMax(*${root}, ${matchinfo}); }]),
  (apply [{ Helper.applyBuildFn(*${root}, ${matchinfo}); }])>;

// %9:_(s1) = G_CONSTANT i1 true
// %8:_(s1) = G_ICMP intpred(eq), %15:_(s32), %22:_
// %10:_(s1) = G_ICMP intpred(eq), %8:_(s1), %9:_
// %11:_(s32) = G_SEXT %10:_(s1)
// => %8:_(s1) = G_ICMP intpred(eq), %15:_(s32), %22:_
//    %11:_(s32) = G_SEXT %8:_(s1)
// or
// %12:_(s1) = G_CONSTANT i1 true
// %11:_(s1) = G_AND %10:_, %35:_
// %13:_(s1) = G_ICMP intpred(eq), %11:_(s1), %12:_
// %14:_(s32) = G_SEXT %13:_(s1)
// => %11:_(s1) = G_AND %10:_, %35:_
//    %14:_(s32) = G_SEXT %11:_(s1)
def cmp_int1 : GICombineRule<
  (defs root:$root, build_fn_matchinfo:$matchinfo),
  (match (wip_match_opcode G_ICMP):$root,
    [{ return matchCmpInt1(*${root}, ${matchinfo}); }]),
  (apply [{ Helper.applyBuildFn(*${root}, ${matchinfo}); }])>;

// %1:_(i1) = G_TRUNC %0:_(wide)
// %2:_(N) = G_SELECT %1:_(i1), N 1, N 0
// => %3:_(wide) = G_AND %0, 1
//    %2:_(N) = G_TRUNC %3
// Avoids introducing an illegal i1 compare when lowering bool-to-int.
def select_trunc_one_zero : GICombineRule<
  (defs root:$root, build_fn_matchinfo:$matchinfo),
  (match (wip_match_opcode G_SELECT):$root,
    [{ return matchSelectTruncOneZero(*${root}, ${matchinfo}); }]),
  (apply [{ Helper.applyBuildFn(*${root}, ${matchinfo}); }])>;

// %45:_(s32) = G_CONSTANT i32 -1
// %60:_(s32) = G_CONSTANT i32 0
// %46:_(s32) = G_SELECT %10:_(s1), %45:_, %60:_
// %47:_(s32) = G_SELECT %35:_(s1), %45:_, %60:_
// %48:_(s32) = G_AND %46:_, %47:_
// %59:_(s32) = G_CONSTANT i32 31
// %58:_(s32) = G_SHL %48:_, %59:_(s32)
// %14:_(s32) = G_ASHR %58:_, %59:_(s32)
// => %14:_(s32) = G_AND %46:_, %47:_
def shift_true_false : GICombineRule<
  (defs root:$root, build_fn_matchinfo:$matchinfo),
  (match (wip_match_opcode G_ASHR):$root,
    [{ return matchShiftTrueFalse(*${root}, ${matchinfo}); }]),
  (apply [{ Helper.applyBuildFn(*${root}, ${matchinfo}); }])>;

// %54:_(s8), %55:_(s8), %56:_(s8), %57:_(s8) = G_UNMERGE_VALUES %21:_(<4 x s8>)
// %39:_(s16) = G_ANYEXT %54:_(s8)
// %40:_(s16) = G_ANYEXT %55:_(s8)
// %41:_(s16) = G_ADD %39:_, %40:_
// %36:_(s16) = G_ANYEXT %56:_(s8)
// %37:_(s16) = G_ANYEXT %57:_(s8)
// %38:_(s16) = G_ADD %36:_, %37:_
// %35:_(s16) = G_ADD %41:_, %38:_
// => %40:_(s32) = G_INTRINSIC intrinsic(@llvm.pisa.dp4a.uu), 0, %21, 0x01010101, 0
//    %35:_(s16) = G_TRUNC %40:_(s32)
def add_i8_reduction : GICombineRule<
  (defs root:$root, build_fn_matchinfo:$matchinfo),
  (match (wip_match_opcode G_ADD):$root,
    [{ return matchAddInt8Reduction(*${root}, ${matchinfo}); }]),
  (apply [{ Helper.applyBuildFn(*${root}, ${matchinfo}); }])>;


// %370:_(i16), %371:_(i16) = G_UNMERGE_VALUES %366:_(<2 x i16>)
// %378:_(f16) = G_BITCAST %370:_(i16)
// %379:_(f16) = G_BITCAST %371:_(i16)
// %37:_(<2 x f16>) = G_BUILD_VECTOR %378:_(f16), %379:_(f16)
// => %37:_(<2 x f16>) = G_BITCAST %366:_(<2 x i16>)
def unmerge_bitcast_buildvector_to_bitcast : GICombineRule<
  (defs root:$root, register_matchinfo:$matchinfo),
  (match (wip_match_opcode G_BUILD_VECTOR):$root,
    [{ return matchUnmergeBitcastBuildVectorToBitcast(*${root}, ${matchinfo}); }]),
  (apply
    [{ applyUnmergeBitcastBuildVectorToBitcast(*${root}, ${matchinfo}); }])>;

// G_FENCE ordering_A, scope_X
// ... pure computation (no memory ops) ...
// G_FENCE ordering_B, scope_X
// => coalesce into a single G_FENCE with the merged AtomicOrdering
//    (e.g. release+acquire -> acq_rel, anything+seq_cst -> seq_cst)
def redundant_fence : GICombineRule<
  (defs root:$root, MI_matchinfo:$matchinfo),
  (match (wip_match_opcode G_FENCE):$root,
    [{ return matchRedundantFence(*${root}, ${matchinfo}); }]),
  (apply
    [{ applyRedundantFence(*${root}, ${matchinfo}); }])
>;

// G_FENCE ordering, scope_X-addrspace_A
// ... pure computation (no memory ops) ...
// G_FENCE ordering, scope_X-addrspace_B
// => G_FENCE ordering, scope_X-generic
// Merges adjacent fences with same ordering and same base scope but different
// address spaces into a single generic fence.
def merge_adjacent_fences : GICombineRule<
  (defs root:$root, MI_matchinfo:$matchinfo),
  (match (wip_match_opcode G_FENCE):$root,
    [{ return matchMergeAdjacentFences(*${root}, ${matchinfo}); }]),
  (apply
    [{ applyMergeAdjacentFences(*${root}, ${matchinfo}); }])
>;

// p2i_to_i2p do not account for pointer-size or address-space difference
def pisa_identity_combines : GICombineGroup<
  !listremove(identity_combines.Rules, [p2i_to_i2p])>;
// avoid G_SELECT %2:_(i1), -1, 0 => G_SEXT %2:_(i1)
def pisa_select_combines : GICombineGroup<
  !listremove(select_combines.Rules, [match_selects])>;
// avoid G_BUILD_VECTOR => G_CONCAT_VECTOR
// avoid x,y<dead>=unmerge(z) => x=trunc(z): causes G_TRUNC on wide types
def pisa_merge_combines : GICombineGroup<
  !listremove(merge_combines.Rules, [combine_build_unmerge, unmerge_dead_to_trunc])>;
// avoid G_BUILD_VECTOR(undef,undef) => G_IMPLICIT_DEF
// avoid G_BITCAST/G_TRUNC(undef) => G_IMPLICIT_DEF: post-legalizer must not
// introduce G_IMPLICIT_DEF for types the legalizer already scalarized
// (e.g., <2 x i32> resulting from a bitcast of a legal i64 undef)
def pisa_undef_combines : GICombineGroup<
  !listremove(undef_combines.Rules, [propagate_undef_all_ops,
                                     unary_undef_to_undef])>;
// narrow_binop_* are conflicting with hoist_logic_op_with_same_opcode_hands, remove them to avoid loop.
def pisa_cast_combines: GICombineGroup<
  !listremove(cast_combines.Rules, [
    narrow_binop_and,
    narrow_binop_or,
    narrow_binop_xor])>;

// Rules removed from both pre- and post-legalizer combiners.
def pisa_removed_combines : GICombineGroup<[
  identity_combines, // use pisa_identity_combines
  select_combines, // use pisa_select_combines
  cast_combines, // use pisa_cast_combines
  merge_combines, // use pisa_merge_combines
  undef_combines, // use pisa_undef_combines
  combines_for_extload, // prevents post-legalizer re-folding of G_LOAD+G_SEXT → G_SEXTLOAD
  div_rem_to_divrem, // G_[US]DIVREM not supported
  intrem_combines, // use [su]div.rem instruction
  opt_brcond_by_inverting_cond,
  trunc_shift,
  fold_binop_into_select, // breaks sext(fcmp) AND patterns needed for bfn matching
  freeze_combines // push_freeze_to_prevent_poison_from_propagating can push G_FREEZE through G_SELECT creating illegal G_FREEZE(i1) after legalization; exclude entire group and re-add freeze_of_non_undef_non_poison explicitly below
]>;

// Additional rules removed only from the post-legalizer combiner.
// - intdiv_combines(sdiv_by_const) emits G_SMULH/G_UMULH
def pisa_post_removed_combines : GICombineGroup<[
  commute_shift,
  intdiv_combines
]>;

// Fold min/max with an identity-element constant:
//   umin(x, UINT_MAX) -> x,  umax(x, 0)       -> x
//   smin(x, INT_MAX)  -> x,  smax(x, INT_MIN)  -> x
// The legalizer introduces such patterns when expanding G_VECREDUCE_U/SMIN/MAX
// (padding lanes are filled with the identity element).
def minmax_identity_fold : GICombineRule<
  (defs root:$root, register_matchinfo:$matchinfo),
  (match (wip_match_opcode G_UMIN, G_UMAX, G_SMIN, G_SMAX):$root,
    [{ return matchMinMaxIdentityFold(*${root}, ${matchinfo}); }]),
  (apply [{ Helper.replaceSingleDefInstWithReg(*${root}, ${matchinfo}); }])
>;

// Fix G_SHL/G_LSHR/G_ASHR shift amounts to i32 post-legalization.
// PISA only legalizes shifts with i32 shift amounts ({I16,I32}, {I32,I32},
// {I64,I32}), but post-legalizer rules such as mul_to_shl may introduce a
// new shift whose amount has the value type (e.g. i64).  This rule truncates
// or zero-extends such shift amounts back to i32.
def pisa_fix_shift_amt : GICombineRule<
  (defs root:$root),
  (match (wip_match_opcode G_SHL, G_LSHR, G_ASHR):$root,
    [{ return matchFixIllegalShiftAmt(*${root}); }]),
  (apply [{ applyFixIllegalShiftAmt(*${root}); }])>;

let CombineAllMethodName = "tryCombineAllImpl" in {

def PISAPreLegalizerCombiner: GICombiner<
  "PISAPreLegalizerCombinerImpl",
  !listremove(
    !listconcat(all_combines.Rules, [
      pisa_identity_combines, pisa_cast_combines, pisa_select_combines, pisa_merge_combines, pisa_undef_combines,
      truncated_load, truncated_store, trunc_shift_to_vector_extract,
      extended_load, simplify_nonpowerof2_load_store_chain, expand_nonpowerof2_load_store,
      extract_insert_to_bitcast, redundant_bitcasts, remove_redundant_moves_pre,
      rcp_sqrt_to_rsqrt, sub_floor_to_frc, extract_build_vector_to_bitcast,
      reduce_predicates, cmp_int1, select_trunc_one_zero,
      // match_selects must run in the pre-legalizer (after select_trunc_one_zero
      // so trunc-based patterns take priority) to convert
      // G_SELECT(i1, true, x) → G_OR(i1, x) via tryFoldBoolSelectToLogic.
      // After legalization this OR becomes a real G_OR that PISABFNMatcher uses
      // to produce bfn.0xa8.  It is excluded from pisa_select_combines
      // (post-legalizer only) to prevent select(i1, wide -1, wide 0) → sext(i1).
      match_selects,
      lane_id_shl_chain, zext_and_to_and_zext,
      freeze_of_non_undef_non_poison]),
    pisa_removed_combines.Rules)>;

def PISAPostLegalizerCombiner: GICombiner<
  "PISAPostLegalizerCombinerImpl",
  !listremove(
    !listconcat(all_combines.Rules, [
      pisa_identity_combines, pisa_cast_combines, pisa_select_combines, pisa_merge_combines, pisa_undef_combines,
      extra_commute_iconstant_to_rhs, extra_commute_fconstant_to_rhs, extra_commute_intrinsic_iconstant_to_rhs,
      extra_remove_redundant_constant, build_vector_with_constants, compare_select_reg, 
      shl_add_to_mad, fdiv_to_rcp_fmul, const_and_trunc_to_copy,
      build_vector_extract_to_copy, extract_subvector_to_extract_elt, extract_subvector_build_vector, extract_subvector_partial, build_vector_from_unmerge_lanes, build_vector_concat_subvectors, shifts_of_constants,
      remove_redundant_moves_post, p2i_to_i2p_fixed, extract_all_build_vector, unmerge_bitcast_buildvector_to_bitcast, 
      redundant_bitcasts, or_and_to_bfi, shift_sub_to_bfe, v2_i1_zext_to_bfe_bfi, and_bitfield_to_bitfield,
      and_select, build_reg_from_2, opt_fcmp_brcond_by_inverting_cond,
      cmp_and_all_ones, sink_trunc, trunc_trunc, abs_redmax_to_redabsmax,
      shift_true_false, add_i8_reduction, redundant_fence,
      freeze_of_non_undef_non_poison, pisa_fix_shift_amt, minmax_identity_fold,
      merge_adjacent_fences]),
    !listconcat(pisa_removed_combines.Rules, pisa_post_removed_combines.Rules))>;
}
