//===---- X86InstrACE.td - ACE Instruction Set Extension --*- tablegen -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the instructions that make up the ACE (AI Compute
// Extensions) instruction set.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ACE multiclasses
//
// Most ACE operations need three definitions: the instruction itself, a pseudo
// for the user-facing intrinsic that takes an immediate tile ID, and a pseudo
// for the internal intrinsic that takes a shape and a tile register so the
// register allocator can assign TMMs. The multiclasses below emit all of them
// from one mnemonic, which also names the intrinsics.
//===----------------------------------------------------------------------===//

// BSRMOVH and BSRMOVL move half of the block scale register, which is wider
// than a vector register, in either direction.
multiclass ACE_BSRMOV_HALF<string Half, Prefix P> {
  defvar Insert = !cast<Intrinsic>("int_x86_acev1_bsr0mov"#Half#"insert");
  defvar Extract = !cast<Intrinsic>("int_x86_acev1_bsr0mov"#Half#"extract");

  let SchedRW = [WriteSystem], OpPrefix = P in {
    let Defs = [BSR0], Uses = [BSR0] in {
      def rr_set : I<0x95, MRM0r, (outs),
                     (ins VR512:$src),
                     !strconcat("bsrmov", Half, "\t{$src, %bsr0|bsr0, $src}"),
                     [(Insert (v64i8 VR512:$src))]>,
                     EVEX, T_MAP6, REX_W, EVEX_V512;

      let mayLoad = 1 in
      def rm_set : I<0x95, MRM0m, (outs),
                     (ins f512mem:$src),
                     !strconcat("bsrmov", Half, "\t{$src, %bsr0|bsr0, $src}"),
                     []>,
                     EVEX, T_MAP6, REX_W, EVEX_V512, EVEX_CD8<64, CD8VF>;
    }

    let Uses = [BSR0] in {
      def rr_get : I<0x95, MRM0r, (outs VR512:$dst),
                     (ins),
                     !strconcat("bsrmov", Half, "\t{%bsr0, $dst|$dst, bsr0}"),
                     [(set (v64i8 VR512:$dst), (Extract))]>,
                     EVEX, T_MAP6, EVEX_V512;

      let mayStore = 1 in
      def mr_get : I<0x95, MRM0m, (outs),
                     (ins f512mem:$dst),
                     !strconcat("bsrmov", Half, "\t{%bsr0, $dst|$dst, bsr0}"),
                     [(store (v64i8 (Extract)), addr:$dst)]>,
                     EVEX, T_MAP6, EVEX_V512, EVEX_CD8<64, CD8VF>;
    }
  }
}

// TILEMOVCOL and TILEMOVROW in the tile-destination direction write one column
// or row and keep the rest, so $dst is a source too. Only the register index
// form has an internal twin; the shape operands carry the index.
multiclass ACE_TILEMOVE<bits<8> OpcImm, bits<8> OpcReg, string Opstr,
                        string IntName> {
  let Constraints = "$src0 = $dst", SchedRW = [WriteSystem] in {
    def tri : Ii8<OpcImm, MRMSrcReg, (outs TILE:$dst),
                  (ins TILE:$src0, VR512:$src, u8imm:$idx),
                  !strconcat(Opstr, "\t{$idx, $src, $dst|$dst, $src, $idx}"),
                  []>,
                  EVEX, PD, TA, REX_W, EVEX_V512;

    def tre : I<OpcReg, MRMSrcReg4VOp3, (outs TILE:$dst),
                (ins TILE:$src0, VR512:$src, GR32:$idx),
                !strconcat(Opstr, "\t{$idx, $src, $dst|$dst, $src, $idx}"),
                []>,
                EVEX, PD, T8, REX_W, VVVV, EVEX_V512;
  }

  let usesCustomInserter = 1, hasNoSchedulingInfo = 1 in {
    // The index is i32 so that an out-of-range one truncates, as it does for
    // AMX; only its low four bits name the row, so truncating keeps those.
    def "P"#NAME#"tri" : PseudoI<(outs),
                                 (ins u8imm:$dst, VR512:$src, i32u8imm:$idx),
                                 []>;

    def "P"#NAME#"tre" : PseudoI<(outs),
                                 (ins u8imm:$dst, VR512:$src, GR32:$idx), []>;
  }

  let isPseudo = true, hasNoSchedulingInfo = 1, usesCustomInserter = 1,
      Constraints = "$src0 = $dst" in {
    // One intrinsic covers both index forms, as for the row reads.
    def "P"#NAME#"triV"
        : PseudoI<(outs TILE:$dst),
                  (ins GR16:$row, GR16:$col, TILE:$src0, VR512:$src,
                       i32u8imm:$idx),
                  [(set TILE:$dst,
                    (!cast<Intrinsic>("int_x86_acev1_"#IntName#"_internal")
                     GR16:$row, GR16:$col, TILE:$src0, VR512:$src,
                     imm:$idx))]>;

    def "P"#NAME#"treV"
        : PseudoI<(outs TILE:$dst),
                  (ins GR16:$row, GR16:$col, TILE:$src0, VR512:$src,
                       GR32:$idx),
                  [(set TILE:$dst,
                    (!cast<Intrinsic>("int_x86_acev1_"#IntName#"_internal")
                     GR16:$row, GR16:$col, TILE:$src0, VR512:$src,
                     GR32:$idx))]>;
  }
}

// Outer products accumulate into the destination tile, so $src1 is tied to it.
multiclass ACE_TOP<bits<8> Opc, string Opstr, Prefix P> {
  let Constraints = "$src1 = $dst", SchedRW = [WriteSystem], OpPrefix = P in
  def trr : I<Opc, MRMSrcReg4VOp3, (outs TILE:$dst),
              (ins TILE:$src1, VR512:$src2, VR512:$src3),
              !strconcat(Opstr, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
              []>,
              EVEX, VVVV, T8, EVEX_V512;

  let usesCustomInserter = 1, hasNoSchedulingInfo = 1 in
  def "P"#NAME#"trr"
      : PseudoI<(outs), (ins u8imm:$dst, VR512:$src1, VR512:$src2),
                [(!cast<Intrinsic>("int_x86_acev1_"#Opstr)
                  timm:$dst, VR512:$src1, VR512:$src2)]>;

  let isPseudo = true, hasNoSchedulingInfo = 1, usesCustomInserter = 1,
      Constraints = "$src1 = $dst" in
  def "P"#NAME#"trrV"
      : PseudoI<(outs TILE:$dst),
                (ins GR16:$row, GR16:$col,
                     TILE:$src1, VR512:$src2, VR512:$src3),
                [(set TILE:$dst,
                  (!cast<Intrinsic>("int_x86_acev1_"#Opstr#"_internal")
                   GR16:$row, GR16:$col,
                   TILE:$src1, VR512:$src2, VR512:$src3))]>;
}

// The mixed-precision outer products take an immediate selecting the element
// formats and read the scales from the block scale register.
multiclass ACE_TOP4MX<bits<8> Opc, string Opstr, Prefix P> {
  // 4VOp3 takes the EVEX.vvvv source last, so the operand lists read in order.
  let Constraints = "$src1 = $dst", SchedRW = [WriteSystem], Uses = [BSR0],
      OpPrefix = P in
  def trri : Ii8<Opc, MRMSrcReg4VOp3, (outs TILE:$dst),
                 (ins TILE:$src1, VR512:$src2, VR512:$src3, u8imm:$imm),
                 !strconcat(Opstr,
                     "\t{$imm, $src3, $src2, $dst|$dst, $src2, $src3, $imm}"),
                 []>,
                 EVEX, VVVV, TA, EVEX_V512;

  let usesCustomInserter = 1, hasNoSchedulingInfo = 1 in
  def "P"#NAME#"trri"
      : PseudoI<(outs),
                (ins u8imm:$dst, VR512:$src2, VR512:$src3, u8imm:$imm),
                [(!cast<Intrinsic>("int_x86_acev1_"#Opstr)
                  timm:$dst, VR512:$src2, VR512:$src3, timm:$imm)]>;

  let isPseudo = true, hasNoSchedulingInfo = 1, usesCustomInserter = 1,
      Constraints = "$src1 = $dst", Uses = [BSR0] in
  def "P"#NAME#"trriV"
      : PseudoI<(outs TILE:$dst),
                (ins GR16:$row, GR16:$col, u8imm:$imm,
                     TILE:$src1, VR512:$src2, VR512:$src3),
                [(set TILE:$dst,
                  (!cast<Intrinsic>("int_x86_acev1_"#Opstr#"_internal")
                   GR16:$row, GR16:$col, timm:$imm,
                   TILE:$src1, VR512:$src2, VR512:$src3))]>;
}

// The row reads have no ACE-specific encoding - they are the AMX instructions
// under palette 2 - so only pseudos are needed here. The opcode is what
// records the ACE programming model.
multiclass ACE_ROWREAD<string Opstr> {
  let SchedRW = [WriteSystem], usesCustomInserter = 1 in {
    def rte : PseudoI<(outs VR512:$dst), (ins u8imm:$src1, GR32:$src2),
                      [(set VR512:$dst,
                        (!cast<Intrinsic>("int_x86_acev1_"#Opstr)
                         timm:$src1, GR32:$src2))]>;

    // One ACE intrinsic covers both index forms, so a constant index matches
    // the immediate encoding here and anything else falls back to the register
    // one.
    def rti : PseudoI<(outs VR512:$dst), (ins u8imm:$src1, i32u8imm:$src2),
                      [(set VR512:$dst,
                        (!cast<Intrinsic>("int_x86_acev1_"#Opstr)
                         timm:$src1, imm:$src2))]>;
  }

  let isPseudo = true, hasNoSchedulingInfo = 1, SchedRW = [WriteSystem],
      usesCustomInserter = 1 in {
    def rtiV : PseudoI<(outs VR512:$dst),
                       (ins GR16:$src1, GR16:$src2, TILE:$src3, i32u6imm:$src4),
                       [(set VR512:$dst,
                         (!cast<Intrinsic>("int_x86_acev1_"#Opstr#"_internal")
                          GR16:$src1, GR16:$src2, TILE:$src3, imm:$src4))]>;

    def rteV : PseudoI<(outs VR512:$dst),
                       (ins GR16:$src1, GR16:$src2, TILE:$src3, GR32:$src4),
                       [(set VR512:$dst,
                         (!cast<Intrinsic>("int_x86_acev1_"#Opstr#"_internal")
                          GR16:$src1, GR16:$src2, TILE:$src3, GR32:$src4))]>;
  }
}

//===----------------------------------------------------------------------===//
// ACE instructions

let Predicates = [HasACEV1, In64BitMode] in {

// BSRINIT - Initialize Block Scale Register
let SchedRW = [WriteSystem], hasSideEffects = 1, Defs = [BSR0] in {
  def BSRINIT : I<0x49, MRM_C0, (outs), (ins),
                  "bsrinit\t{%bsr0|bsr0}", []>,
                  VEX, XD, T8, REX_W;
}

// BSRMOVF - Move Full to BSR
let SchedRW = [WriteSystem], Defs = [BSR0] in {
  def BSRMOVFrr : I<0x95, MRM0r, (outs),
                    (ins VR512:$src1, VR512:$src2),
                    "bsrmovf\t{$src2, $src1, %bsr0|bsr0, $src1, $src2}",
                    [(int_x86_acev1_bsr0movf (v64i8 VR512:$src1), (v64i8 VR512:$src2))]>,
                    EVEX, T_MAP6, REX_W, EVEX_V512, VVVV;

  let mayLoad = 1 in
  def BSRMOVFrm : I<0x95, MRM0m, (outs),
                    (ins VR512:$src1, f512mem:$src2),
                    "bsrmovf\t{$src2, $src1, %bsr0|bsr0, $src1, $src2}", []>,
                    EVEX, T_MAP6, REX_W, EVEX_V512, EVEX_CD8<64, CD8VF>, VVVV;
}

// BSRMOVH - Move Half to/from BSR
defm BSRMOVH : ACE_BSRMOV_HALF<"h", XD>;

// BSRMOVL - Move Low to/from BSR
defm BSRMOVL : ACE_BSRMOV_HALF<"l", XS>;

// TILEMOVCOL - Move Column from Vector to Tile
defm TILEMOVCOL : ACE_TILEMOVE<0x2F, 0x4B, "tilemovcol", "tilemovcolinsert">;

// TILEMOVROW - Move Row from Vector to Tile
defm TILEMOVROW : ACE_TILEMOVE<0x07, 0x4A, "tilemovrow", "tilemovrowinsert">;

// Outer Product Instructions
// TOP2BF16PS - Outer Product BF16 to Single Precision
defm TOP2BF16PS : ACE_TOP<0x5C, "top2bf16ps", XS>;

// TOP4 Instructions - integer outer products
defm TOP4BUUD : ACE_TOP<0x5E, "top4buud", PS>;
defm TOP4BUSD : ACE_TOP<0x5E, "top4busd", PD>;
defm TOP4BSSD : ACE_TOP<0x5E, "top4bssd", XD>;
defm TOP4BSUD : ACE_TOP<0x5E, "top4bsud", XS>;

// TOP4MX Instructions - Mixed Precision Outer Products with Immediate
defm TOP4MXHF8PS : ACE_TOP4MX<0x8D, "top4mxhf8ps", PD>;
defm TOP4MXBHF8PS : ACE_TOP4MX<0x8D, "top4mxbhf8ps", XD>;
defm TOP4MXHBF8PS : ACE_TOP4MX<0x8D, "top4mxhbf8ps", XS>;
defm TOP4MXBF8PS : ACE_TOP4MX<0x8D, "top4mxbf8ps", PS>;
defm TOP4MXBSSPS : ACE_TOP4MX<0x8F, "top4mxbssps", XD>;

//===----------------------------------------------------------------------===//
// ACE Pseudo Instructions
// These are used by intrinsics and expanded via custom inserter
//===----------------------------------------------------------------------===//

// Pseudo instruction for a user-facing intrinsic with no instruction operands.
let usesCustomInserter = 1, hasNoSchedulingInfo = 1 in
def PBSRINIT : PseudoI<(outs), (ins), [(int_x86_acev1_bsr0init)]>;

// ACE twins of the AMX row reads, so the ACE model reaches MIR. The ACE read
// direction of TILEMOVROW has an intrinsic name of its own.
defm ACE_PTCVTROWD2PS : ACE_ROWREAD<"tcvtrowd2ps">;
defm ACE_PTCVTROWPS2BF16H : ACE_ROWREAD<"tcvtrowps2bf16h">;
defm ACE_PTCVTROWPS2BF16L : ACE_ROWREAD<"tcvtrowps2bf16l">;
defm ACE_PTCVTROWPS2PHH : ACE_ROWREAD<"tcvtrowps2phh">;
defm ACE_PTCVTROWPS2PHL : ACE_ROWREAD<"tcvtrowps2phl">;
defm ACE_PTILEMOVROW : ACE_ROWREAD<"tilemovrowextract">;

// ACE twin of the AMX tile-zero user-facing pseudo.
let SchedRW = [WriteSystem], usesCustomInserter = 1 in
def ACE_PTILEZERO : PseudoI<(outs), (ins u8imm:$src),
                            [(int_x86_acev1_tilezero timm:$src)]>;

// ACE twin of the AMX tile-zero RA pseudo, so the ACE model reaches MIR.
let isPseudo = true, isReMaterializable = 1, isAsCheapAsAMove = 1,
    canFoldAsLoad = 1, hasNoSchedulingInfo = 1, usesCustomInserter = 1 in
def ACE_PTILEZEROV : PseudoI<(outs TILE:$dst), (ins GR16:$src1, GR16:$src2),
                             [(set TILE:$dst, (int_x86_acev1_tilezero_internal
                               GR16:$src1, GR16:$src2))]>;

// Pseudo instructions for tile spill and reload (register allocation)
let isPseudo = true, hasNoSchedulingInfo = 1 in {
  let mayStore = 1 in
  def ACE_TILESPILL : PseudoI<(outs), (ins opaquemem:$dst, TILE:$src), []>;

  let mayLoad = 1 in {
    def ACE_TILERELOAD : PseudoI<(outs TILE:$dst), (ins opaquemem:$src), []>;

    // Reload a PHI tile; the dead shape keeps this a tile def for the config.
    def ACE_TILERELOADV : PseudoI<(outs TILE:$dst),
                                  (ins GR16:$src1, GR16:$src2,
                                       opaquemem:$src3), []>;
  }
}

} // end HasACEV1, In64BitMode
