//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

let Namespace = "PISA" in {

  class SpecialRegister <string name> : Register<name> {
    string Name = NAME;
  }

  class PISARegisterClass<list<ValueType> regTypes, int alignment, dag regList>
    : RegisterClass<"PISA", regTypes, alignment, regList> {}

  // .reg style registers. See the PISA specification for register variables:
  // https://intel.github.io/pisa/variables.html#register-variables
  class GeneralRegisterClass<list<ValueType> regTypes, int alignment, dag regList> :
    PISARegisterClass<regTypes, alignment, regList> {
    // Keep flags in sync with RegType in RegEncoder
    let TSFlags{1-0} = 1;
  }

  // .pred style registers. See the PISA specification for predicate variables:
  // https://intel.github.io/pisa/variables.html#predicate-variables
  class PredRegisterClass<list<ValueType> regTypes, int alignment, dag regList> :
    PISARegisterClass<regTypes, alignment, regList> {
    // Keep flags in sync with RegType in RegEncoder
    let TSFlags{1-0} = 2;
  }

  def DummyReg : Register<"%null">;

  multiclass VectorSpecialRegSubRegs<int n, string name> {
    def X : SpecialRegister<name#".x">;
    if !ge(n, 2) then {
      def Y : SpecialRegister<name#".y">;
    }
    if !ge(n, 3) then {
      def Z : SpecialRegister<name#".z">;
    }
    if !ge(n, 4) then {
      def W : SpecialRegister<name#".w">;
    }
  }

  // Registers to describe Vector-subregister relationships
  class VecReg<list<Register> subregs> : RegisterWithSubRegs<"", subregs>;

  foreach Index = 0-3 in { // .x, .y, .z, .w
    def sub8_#Index  : SubRegIndex<8,  !shl(Index, 3)>;
    def sub16_#Index : SubRegIndex<16, !shl(Index, 4)>;
    def sub32_#Index : SubRegIndex<32, !shl(Index, 5)>;
    def sub64_#Index : SubRegIndex<64, !shl(Index, 6)>;
  }
  foreach Index = 0-1 in { // .xy, .zw
    defvar Part = !if(!eq(Index, 0), "xy", "zw");
    def sub8_#Part  : SubRegIndex<!shl(8,  1), !shl(Index, 4)>;
    def sub16_#Part : SubRegIndex<!shl(16, 1), !shl(Index, 5)>;
    def sub32_#Part : SubRegIndex<!shl(32, 1), !shl(Index, 6)>;
    def sub64_#Part : SubRegIndex<!shl(64, 1), !shl(Index, 7)>;
  }
  foreach Index = 4-31 in { // large vector support
    def sub32_#Index : SubRegIndex<32, !shl(Index, 5)>;
  }

  // Special registers
  defset list<SpecialRegister> LocalIdSubRegs = {
    defm SpecialReg_LocalId : VectorSpecialRegSubRegs<3, "%localid">;
  }
  defset list<SpecialRegister> LocalSizeSubRegs = {
    defm SpecialReg_LocalSize : VectorSpecialRegSubRegs<3, "%localsize">;
  }
  defset list<SpecialRegister> EnqueuedLocalSizeSubRegs = {
    defm SpecialReg_EnqueuedLocalSize : VectorSpecialRegSubRegs<3, "%enqueuedlocalsize">;
  }
  defset list<SpecialRegister> GlobalOffsetSubRegs = {
    defm SpecialReg_GlobalOffset : VectorSpecialRegSubRegs<3, "%globaloffset">;
  }
  defset list<SpecialRegister> GlobalSizeSubRegs = {
    defm SpecialReg_GlobalSize : VectorSpecialRegSubRegs<3, "%globalsize">;
  }
  defset list<SpecialRegister> GroupIdSubRegs = {
    defm SpecialReg_GroupId : VectorSpecialRegSubRegs<3, "%groupid">;
  }
  defset list<SpecialRegister> GroupCountSubRegs = {
    defm SpecialReg_GroupCount : VectorSpecialRegSubRegs<3, "%groupcount">;
  }

  def SpecialReg_ActiveMask    : SpecialRegister<"%activemask">;
  def SpecialReg_LaneId        : SpecialRegister<"%laneid">;
  def SpecialReg_SubgroupSize  : SpecialRegister<"%subgroupsize">;
  def SpecialReg_WorkDim       : SpecialRegister<"%workdim">;



  //
  // Scalar registers
  //
  foreach Index = 0-3 in {
    def Reg8b_#Index  : Register<"">;
    def Reg16b_#Index : Register<"">;
    def Reg32b_#Index : Register<"">;
    def Reg64b_#Index : Register<"">;
    def Reg128b_#Index : Register<"">;
  }
  foreach Index = 4-63 in {
    def Reg32b_#Index : Register<"">;
  }

  // Predicate physical registers (used only so the Pred register class has
  // non-zero allocatable registers, PISA never allocates physical registers).
  foreach Index = 0-3 in {
    def Pred_#Index : Register<"">;
  }

  def Pred : PredRegisterClass<[i1], 8, (add DummyReg, (sequence "Pred_%u", 0, 3))>;
  def Reg8b : GeneralRegisterClass<[i8], 8, (add (sequence "Reg8b_%u", 0, 3))>;
  def Reg16b : GeneralRegisterClass<[i16, f16, bf16], 16, (add (sequence "Reg16b_%u", 0, 3))>;
  def Reg32b : GeneralRegisterClass<[i32, f32], 32,
    (add GroupIdSubRegs, GroupCountSubRegs, SpecialReg_SubgroupSize,
         LocalIdSubRegs, LocalSizeSubRegs, SpecialReg_ActiveMask,
         EnqueuedLocalSizeSubRegs, SpecialReg_LaneId, SpecialReg_WorkDim,
         DummyReg,
         (sequence "Reg32b_%u", 0, 63))>;
  def Reg64b : GeneralRegisterClass<[i64, f64], 64,
    (add
      GlobalOffsetSubRegs, GlobalSizeSubRegs,
      (sequence "Reg64b_%u", 0, 3))>;
  def Reg128b : GeneralRegisterClass<[i128], 128, (add (sequence "Reg128b_%u", 0, 3))>;

  //
  // Vector registers
  //
  // 8b subregisters
  def Reg8bx2 : VecReg<[Reg8b_0, Reg8b_1]> {
    let SubRegIndices = [sub8_0, sub8_1];
    let CoveredBySubRegs = 1;
  }
  def Reg8bx3 : VecReg<[Reg8b_0, Reg8b_1, Reg8b_2]> {
    let SubRegIndices = [sub8_0, sub8_1, sub8_2];
    let CoveredBySubRegs = 1;
  }
  def Reg8bx2H : VecReg<[Reg8b_2, Reg8b_3]> {
    let SubRegIndices = [sub8_2, sub8_3];
    let CoveredBySubRegs = 1;
  }
  def Reg8bx4 : VecReg<[Reg8bx2, Reg8bx2H]> {
    let SubRegIndices = [sub8_xy, sub8_zw];
    let CoveredBySubRegs = 1;
  }

  // 8b vectors
  def RegV2_8b : GeneralRegisterClass<[v2i8], 8, (add Reg8bx2)>;
  def RegV3_8b : GeneralRegisterClass<[v3i8], 8, (add Reg8bx3)>;
  def RegV4_8b : GeneralRegisterClass<[v4i8], 8, (add Reg8bx4)>;
  let isAllocatable = 0 in {
    def RegV2_8bH : GeneralRegisterClass<[v2i8], 8, (add Reg8bx2H)>;
  }

  // 16b subregisters
  def Reg16bx2 : VecReg<[Reg16b_0, Reg16b_1]> {
    let SubRegIndices = [sub16_0, sub16_1];
    let CoveredBySubRegs = 1;
  }
  def Reg16bx3 : VecReg<[Reg16b_0, Reg16b_1, Reg16b_2]> {
    let SubRegIndices = [sub16_0, sub16_1, sub16_2];
    let CoveredBySubRegs = 1;
  }
  def Reg16bx2H : VecReg<[Reg16b_2, Reg16b_3]> {
    let SubRegIndices = [sub16_2, sub16_3];
    let CoveredBySubRegs = 1;
  }
  def Reg16bx4 : VecReg<[Reg16bx2, Reg16bx2H]> {
    let SubRegIndices = [sub16_xy, sub16_zw];
    let CoveredBySubRegs = 1;
  }

  // 16b vectors
  def RegV2_16b : GeneralRegisterClass<[v2i16, v2f16, v2bf16], 16, (add Reg16bx2)>;
  def RegV3_16b : GeneralRegisterClass<[v3i16, v3f16], 16, (add Reg16bx3)>;
  def RegV4_16b : GeneralRegisterClass<[v4i16, v4f16], 16, (add Reg16bx4)>;
  let isAllocatable = 0 in {
    def RegV2_16bH : GeneralRegisterClass<[v2i16, v2f16, v2bf16], 16, (add Reg16bx2H)>;
  }

  // 32b subregisters
  def Reg32bx2 : VecReg<[Reg32b_0, Reg32b_1]> {
    let SubRegIndices = [sub32_0, sub32_1];
    let CoveredBySubRegs = 1;
  }
  def Reg32bx3 : VecReg<[Reg32b_0, Reg32b_1, Reg32b_2]> {
    let SubRegIndices = [sub32_0, sub32_1, sub32_2];
    let CoveredBySubRegs = 1;
  }
  def Reg32bx2H : VecReg<[Reg32b_2, Reg32b_3]> {
    let SubRegIndices = [sub32_2, sub32_3];
    let CoveredBySubRegs = 1;
  }
  def Reg32bx4 : VecReg<[Reg32bx2, Reg32bx2H]> {
    let SubRegIndices = [sub32_xy, sub32_zw];
    let CoveredBySubRegs = 1;
  }

  // 32b vectors
  def RegV2_32b : GeneralRegisterClass<[v2i32, v2f32], 32, (add Reg32bx2)>;
  def RegV3_32b : GeneralRegisterClass<[v3i32, v3f32], 32, (add Reg32bx3)>;
  def RegV4_32b : GeneralRegisterClass<[v4i32, v4f32], 32, (add Reg32bx4)>;
  let isAllocatable = 0 in {
    def RegV2_32bH : GeneralRegisterClass<[v2i32, v2f32], 32, (add Reg32bx2H)>;
  }

  // large vector support
  foreach length = {5-8,16,32} in {
    // 32b vector registers
    // Define one VecReg for each 32b vector length
    def "Reg32bx"#length : VecReg<!foreach(i, !range(length), !cast<Register>("Reg32b_"#i))> {
      let SubRegIndices = !foreach(i, !range(length), !cast<SubRegIndex>("sub32_"#i));
      let CoveredBySubRegs = 1;
    }
    // 32b vector register classes
    // Define one GeneralRegisterClass for each 32b vector length
    def "RegV"#length#"_32b" : GeneralRegisterClass<[!cast<ValueType>("v"#length#"i32"), !cast<ValueType>("v"#length#"f32")], 32, (add !cast<VecReg>("Reg32bx"#length))>;
  }

  // v64 register: defined without per-element sub-register indices because
  // LLVM's LaneBitmask (64 bits) cannot represent 64 individual sub-register
  // lanes plus the existing sub-register indices. All element access uses
  // extract/insert instructions rather than sub-register operations.
  def Reg32bx64 : Register<"">;
  def RegV64_32b : GeneralRegisterClass<[v64i32, v64f32], 32, (add Reg32bx64)>;

  // 64b subregisters
  def Reg64bx2 : VecReg<[Reg64b_0, Reg64b_1]> {
    let SubRegIndices = [sub64_0, sub64_1];
    let CoveredBySubRegs = 1;
  }
  def Reg64bx3 : VecReg<[Reg64b_0, Reg64b_1, Reg64b_2]> {
    let SubRegIndices = [sub64_0, sub64_1, sub64_2];
    let CoveredBySubRegs = 1;
  }
  def Reg64bx2H : VecReg<[Reg64b_2, Reg64b_3]> {
    let SubRegIndices = [sub64_2, sub64_3];
    let CoveredBySubRegs = 1;
  }
  def Reg64bx4 : VecReg<[Reg64bx2, Reg64bx2H]> {
    let SubRegIndices = [sub64_xy, sub64_zw];
    let CoveredBySubRegs = 1;
  }

  // 64b vectors
  def RegV2_64b : GeneralRegisterClass<[v2i64, v2f64], 64, (add Reg64bx2)>;
  def RegV3_64b : GeneralRegisterClass<[v3i64, v3f64], 64, (add Reg64bx3)>;
  def RegV4_64b : GeneralRegisterClass<[v4i64, v4f64], 64, (add Reg64bx4)>;
  let isAllocatable = 0 in {
    def RegV2_64bH : GeneralRegisterClass<[v2i64, v2f64], 64, (add Reg64bx2H)>;
  }
}

def SpecialRegNames : GenericTable {
  let FilterClass = "SpecialRegister";
  let CppTypeName = "SpecialRegEntry";
  let Fields = ["Name"];
}

def lookupSpecialRegByName : SearchIndex {
  let Table = SpecialRegNames;
  let Key = ["Name"];
}
