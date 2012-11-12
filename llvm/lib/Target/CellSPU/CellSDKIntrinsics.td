//===-- CellSDKIntrinsics.td - Cell SDK Intrinsics ---------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

///--==-- Arithmetic ops intrinsics --==--
def CellSDKah:
    RR_Int_v8i16<0b00010011000, "ah", IntegerOp, int_spu_si_ah>;
def CellSDKahi:
    RI10_Int_v8i16<0b00010011000, "ahi", IntegerOp, int_spu_si_ahi>;
def CellSDKa:
    RR_Int_v4i32<0b00000011000, "a", IntegerOp, int_spu_si_a>;
def CellSDKai:
    RI10_Int_v4i32<0b00111000, "ai", IntegerOp, int_spu_si_ai>;
def CellSDKsfh:
    RR_Int_v8i16<0b00010010000, "sfh", IntegerOp, int_spu_si_sfh>;
def CellSDKsfhi:
    RI10_Int_v8i16<0b10110000, "sfhi", IntegerOp, int_spu_si_sfhi>;
def CellSDKsf:
    RR_Int_v4i32<0b00000010000, "sf", IntegerOp, int_spu_si_sf>;
def CellSDKsfi:
    RI10_Int_v4i32<0b00110000, "sfi", IntegerOp, int_spu_si_sfi>;
def CellSDKaddx:
    RR_Int_v4i32<0b00000010110, "addx", IntegerOp, int_spu_si_addx>;
def CellSDKcg:
    RR_Int_v4i32<0b0100001100, "cg", IntegerOp, int_spu_si_cg>;
def CellSDKcgx:
    RR_Int_v4i32<0b01000010110, "cgx", IntegerOp, int_spu_si_cgx>;
def CellSDKsfx:
    RR_Int_v4i32<0b10000010110, "sfx", IntegerOp, int_spu_si_sfx>;
def CellSDKbg:
    RR_Int_v4i32<0b01000010000, "bg", IntegerOp, int_spu_si_bg>;
def CellSDKbgx:
    RR_Int_v4i32<0b11000010110, "bgx", IntegerOp, int_spu_si_bgx>;

def CellSDKmpy:
    RRForm<0b00100011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "mpy $rT, $rA, $rB", IntegerMulDiv,
      [(set (v4i32 VECREG:$rT), (int_spu_si_mpy (v8i16 VECREG:$rA),
                                                (v8i16 VECREG:$rB)))]>;

def CellSDKmpyu:
    RRForm<0b00110011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "mpyu $rT, $rA, $rB", IntegerMulDiv,
      [(set (v4i32 VECREG:$rT), (int_spu_si_mpyu (v8i16 VECREG:$rA),
                                                 (v8i16 VECREG:$rB)))] >;

def CellSDKmpyi:
    RI10Form<0b00101110, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
      "mpyi $rT, $rA, $val", IntegerMulDiv,
      [(set (v4i32 VECREG:$rT), (int_spu_si_mpyi (v8i16 VECREG:$rA),
                                                 i16ImmSExt10:$val))]>;

def CellSDKmpyui:
    RI10Form<0b10101110, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
      "mpyui $rT, $rA, $val", IntegerMulDiv,
      [(set (v4i32 VECREG:$rT), (int_spu_si_mpyui (v8i16 VECREG:$rA),
                                                  i16ImmSExt10:$val))]>;

def CellSDKmpya:
    RRRForm<0b0011, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
      "mpya $rT, $rA, $rB, $rC", IntegerMulDiv,
      [(set (v4i32 VECREG:$rT), (int_spu_si_mpya (v8i16 VECREG:$rA),
                                                 (v8i16 VECREG:$rB),
                                                 (v8i16 VECREG:$rC)))]>;

def CellSDKmpyh:
    RRForm<0b10100011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "mpyh $rT, $rA, $rB", IntegerMulDiv,
      [(set (v4i32 VECREG:$rT), (int_spu_si_mpyh (v4i32 VECREG:$rA),
                                                 (v8i16 VECREG:$rB)))]>;

def CellSDKmpys:
    RRForm<0b11100011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "mpys $rT, $rA, $rB", IntegerMulDiv,
      [(set (v4i32 VECREG:$rT), (int_spu_si_mpys (v8i16 VECREG:$rA),
                                                 (v8i16 VECREG:$rB)))]>;

def CellSDKmpyhh:
    RRForm<0b01100011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "mpyhh $rT, $rA, $rB", IntegerMulDiv,
      [(set (v4i32 VECREG:$rT), (int_spu_si_mpyhh (v8i16 VECREG:$rA),
                                                  (v8i16 VECREG:$rB)))]>;

def CellSDKmpyhha:
    RRForm<0b01100010110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "mpyhha $rT, $rA, $rB", IntegerMulDiv,
      [(set (v4i32 VECREG:$rT), (int_spu_si_mpyhha (v8i16 VECREG:$rA),
                                                   (v8i16 VECREG:$rB)))]>;

// Not sure how to match a (set $rT, (add $rT (mpyhh $rA, $rB)))... so leave
// as an intrinsic for the time being
def CellSDKmpyhhu:
    RRForm<0b01110011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "mpyhhu $rT, $rA, $rB", IntegerMulDiv,
      [(set (v4i32 VECREG:$rT), (int_spu_si_mpyhhu (v8i16 VECREG:$rA),
                                                   (v8i16 VECREG:$rB)))]>;

def CellSDKmpyhhau:
    RRForm<0b01110010110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
      "mpyhhau $rT, $rA, $rB", IntegerMulDiv,
      [(set (v4i32 VECREG:$rT), (int_spu_si_mpyhhau (v8i16 VECREG:$rA),
                                                    (v8i16 VECREG:$rB)))]>;

def CellSDKand:
        RRForm<0b1000011000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
          "and\t $rT, $rA, $rB", IntegerOp,
          [(set (v4i32 VECREG:$rT),
                (int_spu_si_and (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))]>;

def CellSDKandc:
        RRForm<0b10000011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
          "andc\t $rT, $rA, $rB", IntegerOp,
          [(set (v4i32 VECREG:$rT),
                (int_spu_si_andc (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))]>;

def CellSDKandbi:
     RI10Form<0b01101000, (outs VECREG:$rT), (ins VECREG:$rA, u10imm_i8:$val),
       "andbi\t $rT, $rA, $val", BranchResolv,
       [(set (v16i8 VECREG:$rT),
             (int_spu_si_andbi (v16i8 VECREG:$rA), immU8:$val))]>;

def CellSDKandhi:
     RI10Form<0b10101000, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
           "andhi\t $rT, $rA, $val", BranchResolv,
       [(set (v8i16 VECREG:$rT),
             (int_spu_si_andhi (v8i16 VECREG:$rA), i16ImmSExt10:$val))]>;

def CellSDKandi:
     RI10Form<0b00101000, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
           "andi\t $rT, $rA, $val", BranchResolv,
       [(set (v4i32 VECREG:$rT),
             (int_spu_si_andi (v4i32 VECREG:$rA), i32ImmSExt10:$val))]>;

def CellSDKor:
        RRForm<0b10000010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
          "or\t $rT, $rA, $rB", IntegerOp,
          [(set (v4i32 VECREG:$rT),
                (int_spu_si_or (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))]>;

def CellSDKorc:
        RRForm<0b10010011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
          "addc\t $rT, $rA, $rB", IntegerOp,
          [(set (v4i32 VECREG:$rT),
                (int_spu_si_orc (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))]>;

def CellSDKorbi:
     RI10Form<0b01100000, (outs VECREG:$rT), (ins VECREG:$rA, u10imm_i8:$val),
       "orbi\t $rT, $rA, $val", BranchResolv,
       [(set (v16i8 VECREG:$rT),
             (int_spu_si_orbi (v16i8 VECREG:$rA), immU8:$val))]>;

def CellSDKorhi:
     RI10Form<0b10100000, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
           "orhi\t $rT, $rA, $val", BranchResolv,
       [(set (v8i16 VECREG:$rT),
             (int_spu_si_orhi (v8i16 VECREG:$rA), i16ImmSExt10:$val))]>;

def CellSDKori:
     RI10Form<0b00100000, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
           "ori\t $rT, $rA, $val", BranchResolv,
       [(set (v4i32 VECREG:$rT),
             (int_spu_si_ori (v4i32 VECREG:$rA), i32ImmSExt10:$val))]>;

def CellSDKxor:
        RRForm<0b10000010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
          "xor\t $rT, $rA, $rB", IntegerOp,
          [(set (v4i32 VECREG:$rT), 
                (int_spu_si_xor (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))]>;

def CellSDKxorbi:
     RI10Form<0b01100000, (outs VECREG:$rT), (ins VECREG:$rA, u10imm_i8:$val),
       "xorbi\t $rT, $rA, $val", BranchResolv,
       [(set (v16i8 VECREG:$rT), (int_spu_si_xorbi (v16i8 VECREG:$rA), immU8:$val))]>;

def CellSDKxorhi:
     RI10Form<0b10100000, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
       "xorhi\t $rT, $rA, $val", BranchResolv,
       [(set (v8i16 VECREG:$rT), 
             (int_spu_si_xorhi (v8i16 VECREG:$rA), i16ImmSExt10:$val))]>;

def CellSDKxori:
     RI10Form<0b00100000, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
           "xori\t $rT, $rA, $val", BranchResolv,
       [(set (v4i32 VECREG:$rT), 
             (int_spu_si_xori (v4i32 VECREG:$rA), i32ImmSExt10:$val))]>;

def CellSDKnor:
        RRForm<0b10000010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
          "nor\t $rT, $rA, $rB", IntegerOp,
          [(set (v4i32 VECREG:$rT), 
                (int_spu_si_nor (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))]>;

def CellSDKnand:
        RRForm<0b10000010000, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
          "nand\t $rT, $rA, $rB", IntegerOp,
          [(set (v4i32 VECREG:$rT), 
                (int_spu_si_nand (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))]>;

//===----------------------------------------------------------------------===//
// Shift/rotate intrinsics:
//===----------------------------------------------------------------------===//

def CellSDKshli:
  Pat<(int_spu_si_shli (v4i32 VECREG:$rA), uimm7:$val),
      (SHLIv4i32 VECREG:$rA, (TO_IMM32 imm:$val))>;

def CellSDKshlqbi:
  Pat<(int_spu_si_shlqbi VECREG:$rA, R32C:$rB),
      (SHLQBIv16i8 VECREG:$rA, R32C:$rB)>;

def CellSDKshlqii:
  Pat<(int_spu_si_shlqbii VECREG:$rA, uimm7:$val),
      (SHLQBIIv16i8 VECREG:$rA, (TO_IMM32 imm:$val))>;

def CellSDKshlqby:
  Pat<(int_spu_si_shlqby VECREG:$rA, R32C:$rB),
      (SHLQBYv16i8 VECREG:$rA, R32C:$rB)>;

def CellSDKshlqbyi:
  Pat<(int_spu_si_shlqbyi VECREG:$rA, uimm7:$val),
      (SHLQBYIv16i8 VECREG:$rA, (TO_IMM32 imm:$val))>;

          
//===----------------------------------------------------------------------===//
// Branch/compare intrinsics:
//===----------------------------------------------------------------------===//

def CellSDKceq:
  RRForm<0b00000011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
        "ceq\t $rT, $rA, $rB", BranchResolv,
        [(set (v4i32 VECREG:$rT), 
              (int_spu_si_ceq (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))]>;

def CellSDKceqi:
  RI10Form<0b00111110, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
        "ceqi\t $rT, $rA, $val", BranchResolv,
    [(set (v4i32 VECREG:$rT), 
          (int_spu_si_ceqi (v4i32 VECREG:$rA), i32ImmSExt10:$val))]>;

def CellSDKceqb:
  RRForm<0b00001011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
        "ceqb\t $rT, $rA, $rB", BranchResolv,
        [(set (v16i8 VECREG:$rT), 
              (int_spu_si_ceqb (v16i8 VECREG:$rA), (v16i8 VECREG:$rB)))]>;

def CellSDKceqbi:
  RI10Form<0b01111110, (outs VECREG:$rT), (ins VECREG:$rA, u10imm_i8:$val),
        "ceqbi\t $rT, $rA, $val", BranchResolv,
    [(set (v16i8 VECREG:$rT), (int_spu_si_ceqbi (v16i8 VECREG:$rA), immU8:$val))]>;

def CellSDKceqh:
  RRForm<0b00010011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
        "ceqh\t $rT, $rA, $rB", BranchResolv,
        [(set (v8i16 VECREG:$rT), 
              (int_spu_si_ceqh (v8i16 VECREG:$rA), (v8i16 VECREG:$rB)))]>;

def CellSDKceqhi:
  RI10Form<0b10111110, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
        "ceqhi\t $rT, $rA, $val", BranchResolv,
    [(set (v8i16 VECREG:$rT), 
          (int_spu_si_ceqhi (v8i16 VECREG:$rA), i16ImmSExt10:$val))]>;
def CellSDKcgth:
  RRForm<0b00010011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "cgth\t $rT, $rA, $rB", BranchResolv,
        [(set (v8i16 VECREG:$rT),
              (int_spu_si_cgth (v8i16 VECREG:$rA), (v8i16 VECREG:$rB)))]>;

def CellSDKcgthi:
  RI10Form<0b10111110, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
    "cgthi\t $rT, $rA, $val", BranchResolv,
        [(set (v8i16 VECREG:$rT), 
              (int_spu_si_cgthi (v8i16 VECREG:$rA), i16ImmSExt10:$val))]>;

def CellSDKcgt:
  RRForm<0b00000010010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "cgt\t $rT, $rA, $rB", BranchResolv,
        [(set (v4i32 VECREG:$rT), 
              (int_spu_si_cgt (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))]>;

def CellSDKcgti:
  RI10Form<0b00110010, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
    "cgti\t $rT, $rA, $val", BranchResolv,
        [(set (v4i32 VECREG:$rT), 
              (int_spu_si_cgti (v4i32 VECREG:$rA), i32ImmSExt10:$val))]>;

def CellSDKcgtb:
  RRForm<0b00001010010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "cgtb\t $rT, $rA, $rB", BranchResolv,
        [(set (v16i8 VECREG:$rT), 
              (int_spu_si_cgtb (v16i8 VECREG:$rA), (v16i8 VECREG:$rB)))]>;

def CellSDKcgtbi:
  RI10Form<0b01110010, (outs VECREG:$rT), (ins VECREG:$rA, u10imm_i8:$val),
    "cgtbi\t $rT, $rA, $val", BranchResolv,
        [(set (v16i8 VECREG:$rT), (int_spu_si_cgtbi (v16i8 VECREG:$rA), immU8:$val))]>;

def CellSDKclgth:
  RRForm<0b00010011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "clgth\t $rT, $rA, $rB", BranchResolv,
        [(set (v8i16 VECREG:$rT), 
              (int_spu_si_clgth (v8i16 VECREG:$rA), (v8i16 VECREG:$rB)))]>;

def CellSDKclgthi:
  RI10Form<0b10111010, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
    "clgthi\t $rT, $rA, $val", BranchResolv,
        [(set (v8i16 VECREG:$rT), 
              (int_spu_si_clgthi (v8i16 VECREG:$rA), i16ImmSExt10:$val))]>;

def CellSDKclgt:
  RRForm<0b00000011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "clgt\t $rT, $rA, $rB", BranchResolv,
        [(set (v4i32 VECREG:$rT), 
              (int_spu_si_clgt (v4i32 VECREG:$rA), (v4i32 VECREG:$rB)))]>;

def CellSDKclgti:
  RI10Form<0b00111010, (outs VECREG:$rT), (ins VECREG:$rA, s10imm:$val),
    "clgti\t $rT, $rA, $val", BranchResolv,
        [(set (v4i32 VECREG:$rT), 
              (int_spu_si_clgti (v4i32 VECREG:$rA), i32ImmSExt10:$val))]>;

def CellSDKclgtb:
  RRForm<0b00001011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "clgtb\t $rT, $rA, $rB", BranchResolv,
    [(set (v16i8 VECREG:$rT),
          (int_spu_si_clgtb (v16i8 VECREG:$rA), (v16i8 VECREG:$rB)))]>;

def CellSDKclgtbi:
  RI10Form<0b01111010, (outs VECREG:$rT), (ins VECREG:$rA, u10imm_i8:$val),
    "clgtbi\t $rT, $rA, $val", BranchResolv,
    [(set (v16i8 VECREG:$rT),
          (int_spu_si_clgtbi (v16i8 VECREG:$rA), immU8:$val))]>;

//===----------------------------------------------------------------------===//
// Floating-point intrinsics:
//===----------------------------------------------------------------------===//

def CellSDKfa:
  RRForm<0b00100011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "fa\t $rT, $rA, $rB", SPrecFP,
        [(set (v4f32 VECREG:$rT), (int_spu_si_fa (v4f32 VECREG:$rA),
                                                 (v4f32 VECREG:$rB)))]>;

def CellSDKfs:
  RRForm<0b10100011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "fs\t $rT, $rA, $rB", SPrecFP,
        [(set (v4f32 VECREG:$rT), (int_spu_si_fs (v4f32 VECREG:$rA),
                                                 (v4f32 VECREG:$rB)))]>;

def CellSDKfm:
  RRForm<0b01100011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "fm\t $rT, $rA, $rB", SPrecFP,
        [(set (v4f32 VECREG:$rT), (int_spu_si_fm (v4f32 VECREG:$rA),
                                                 (v4f32 VECREG:$rB)))]>;

def CellSDKfceq:
  RRForm<0b01000011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "fceq\t $rT, $rA, $rB", SPrecFP,
        [(set (v4f32 VECREG:$rT), (int_spu_si_fceq (v4f32 VECREG:$rA),
                                                   (v4f32 VECREG:$rB)))]>;

def CellSDKfcgt:
  RRForm<0b01000011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "fcgt\t $rT, $rA, $rB", SPrecFP,
        [(set (v4f32 VECREG:$rT), (int_spu_si_fcgt (v4f32 VECREG:$rA),
                                                   (v4f32 VECREG:$rB)))]>;

def CellSDKfcmeq:
  RRForm<0b01010011110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "fcmeq\t $rT, $rA, $rB", SPrecFP,
        [(set (v4f32 VECREG:$rT), (int_spu_si_fcmeq (v4f32 VECREG:$rA),
                                                    (v4f32 VECREG:$rB)))]>;

def CellSDKfcmgt:
  RRForm<0b01010011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "fcmgt\t $rT, $rA, $rB", SPrecFP,
        [(set (v4f32 VECREG:$rT), (int_spu_si_fcmgt (v4f32 VECREG:$rA),
                                                    (v4f32 VECREG:$rB)))]>;

def CellSDKfma:
  RRRForm<0b0111, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
    "fma\t $rT, $rA, $rB, $rC", SPrecFP,
        [(set (v4f32 VECREG:$rT), (int_spu_si_fma (v4f32 VECREG:$rA),
                                                  (v4f32 VECREG:$rB),
                                                  (v4f32 VECREG:$rC)))]>;

def CellSDKfnms:
  RRRForm<0b1011, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
    "fnms\t $rT, $rA, $rB, $rC", SPrecFP,
        [(set (v4f32 VECREG:$rT), (int_spu_si_fnms (v4f32 VECREG:$rA),
                                                   (v4f32 VECREG:$rB),
                                                   (v4f32 VECREG:$rC)))]>;

def CellSDKfms:
  RRRForm<0b1111, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB, VECREG:$rC),
    "fms\t $rT, $rA, $rB, $rC", SPrecFP,
        [(set (v4f32 VECREG:$rT), (int_spu_si_fms (v4f32 VECREG:$rA),
                                                  (v4f32 VECREG:$rB),
                                                  (v4f32 VECREG:$rC)))]>;

//===----------------------------------------------------------------------===//
// Double precision floating-point intrinsics:
//===----------------------------------------------------------------------===//

def CellSDKdfa:
  RRForm<0b00110011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "dfa\t $rT, $rA, $rB", DPrecFP,
        [(set (v2f64 VECREG:$rT), (int_spu_si_dfa (v2f64 VECREG:$rA),
                                                  (v2f64 VECREG:$rB)))]>;

def CellSDKdfs:
  RRForm<0b10110011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "dfs\t $rT, $rA, $rB", DPrecFP,
        [(set (v2f64 VECREG:$rT), (int_spu_si_dfs (v2f64 VECREG:$rA),
                                                  (v2f64 VECREG:$rB)))]>;

def CellSDKdfm:
  RRForm<0b01110011010, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "dfm\t $rT, $rA, $rB", DPrecFP,
        [(set (v2f64 VECREG:$rT), (int_spu_si_dfm (v2f64 VECREG:$rA),
                                                  (v2f64 VECREG:$rB)))]>;

def CellSDKdfma:
  RRForm<0b00111010110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "dfma\t $rT, $rA, $rB", DPrecFP,
        [(set (v2f64 VECREG:$rT), (int_spu_si_dfma (v2f64 VECREG:$rA),
                                                   (v2f64 VECREG:$rB)))]>;

def CellSDKdfnma:
  RRForm<0b11111010110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "dfnma\t $rT, $rA, $rB", DPrecFP,
        [(set (v2f64 VECREG:$rT), (int_spu_si_dfnma (v2f64 VECREG:$rA),
                                                    (v2f64 VECREG:$rB)))]>;

def CellSDKdfnms:
  RRForm<0b01111010110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "dfnms\t $rT, $rA, $rB", DPrecFP,
        [(set (v2f64 VECREG:$rT), (int_spu_si_dfnms (v2f64 VECREG:$rA),
                                                    (v2f64 VECREG:$rB)))]>;

def CellSDKdfms:
  RRForm<0b10111010110, (outs VECREG:$rT), (ins VECREG:$rA, VECREG:$rB),
    "dfms\t $rT, $rA, $rB", DPrecFP,
        [(set (v2f64 VECREG:$rT), (int_spu_si_dfms (v2f64 VECREG:$rA),
                                                   (v2f64 VECREG:$rB)))]>;
