const ManualMapEntry ManualMapSet[] = {
    // Part1: These following records are for manually mapping instructions that
    // do not match by their encoding.
    { "ADD16ri_DB",         "ADD16mi",         TB_NO_REVERSE  },
    { "ADD16ri8_DB",        "ADD16mi8",        TB_NO_REVERSE  },
    { "ADD16rr_DB",         "ADD16mr",         TB_NO_REVERSE  },
    { "ADD32ri_DB",         "ADD32mi",         TB_NO_REVERSE  },
    { "ADD32ri8_DB",        "ADD32mi8",        TB_NO_REVERSE  },
    { "ADD32rr_DB",         "ADD32mr",         TB_NO_REVERSE  },
    { "ADD64ri32_DB",       "ADD64mi32",       TB_NO_REVERSE  },
    { "ADD64ri8_DB",        "ADD64mi8",        TB_NO_REVERSE  },
    { "ADD64rr_DB",         "ADD64mr",         TB_NO_REVERSE  },
    { "ADD8ri_DB",          "ADD8mi",          TB_NO_REVERSE  },
    { "ADD8rr_DB",          "ADD8mr",          TB_NO_REVERSE  },
    { "ADD16rr_DB",         "ADD16rm",         TB_NO_REVERSE  },
    { "ADD32rr_DB",         "ADD32rm",         TB_NO_REVERSE  },
    { "ADD64rr_DB",         "ADD64rm",         TB_NO_REVERSE  },
    { "ADD8rr_DB",          "ADD8rm",          TB_NO_REVERSE  },
    { "MMX_MOVD64from64rr", "MMX_MOVQ64mr",    TB_FOLDED_STORE },
    { "MMX_MOVD64grr",      "MMX_MOVD64mr",    TB_FOLDED_STORE },
    { "MOV64toSDrr",        "MOV64mr",         TB_FOLDED_STORE | TB_NO_REVERSE },
    { "MOVDI2SSrr",         "MOV32mr",         TB_FOLDED_STORE | TB_NO_REVERSE },
    { "MOVPQIto64rr",       "MOVPQI2QImr",     TB_FOLDED_STORE | TB_NO_REVERSE },
    { "MOVSDto64rr",        "MOVSDmr",         TB_FOLDED_STORE | TB_NO_REVERSE },
    { "MOVSS2DIrr",         "MOVSSmr",         TB_FOLDED_STORE },
    { "MOVLHPSrr",          "MOVHPSrm",        TB_NO_REVERSE  },
    { "PUSH16r",            "PUSH16rmm",       TB_FOLDED_LOAD },
    { "PUSH32r",            "PUSH32rmm",       TB_FOLDED_LOAD },
    { "PUSH64r",            "PUSH64rmm",       TB_FOLDED_LOAD },
    { "TAILJMPr",           "TAILJMPm",        TB_FOLDED_LOAD },
    { "TAILJMPr64",         "TAILJMPm64",      TB_FOLDED_LOAD },
    { "TAILJMPr64_REX",     "TAILJMPm64_REX",  TB_FOLDED_LOAD },
    { "TCRETURNri",         "TCRETURNmi",      TB_FOLDED_LOAD | TB_NO_FORWARD },
    { "TCRETURNri64",       "TCRETURNmi64",    TB_FOLDED_LOAD | TB_NO_FORWARD },
    { "VMOVLHPSZrr",        "VMOVHPSZ128rm",   TB_NO_REVERSE  },
    { "VMOVLHPSrr",         "VMOVHPSrm",       TB_NO_REVERSE  },
    { "VMOV64toSDZrr",      "MOV64mr",         TB_FOLDED_STORE | TB_NO_REVERSE },
    { "VMOV64toSDrr",       "MOV64mr",         TB_FOLDED_STORE | TB_NO_REVERSE },
    { "VMOVDI2SSZrr",       "MOV32mr",         TB_FOLDED_STORE | TB_NO_REVERSE },
    { "VMOVDI2SSrr",        "MOV32mr",         TB_FOLDED_STORE | TB_NO_REVERSE },
    { "VMOVPQIto64Zrr",     "VMOVPQI2QIZmr",   TB_FOLDED_STORE | TB_NO_REVERSE },
    { "VMOVPQIto64rr",      "VMOVPQI2QImr",    TB_FOLDED_STORE | TB_NO_REVERSE },
    { "VMOVSDto64Zrr",      "VMOVSDZmr",       TB_FOLDED_STORE | TB_NO_REVERSE },
    { "VMOVSDto64rr",       "VMOVSDmr",        TB_FOLDED_STORE | TB_NO_REVERSE },
    { "VMOVSS2DIZrr",       "VMOVSSZmr",       TB_FOLDED_STORE },
    { "VMOVSS2DIrr",        "VMOVSSmr",        TB_FOLDED_STORE },
    { "MMX_MOVD64to64rr",   "MMX_MOVQ64rm",    0 },
    { "MOV64toPQIrr",       "MOVQI2PQIrm",     TB_NO_REVERSE },
    { "MOV64toSDrr",        "MOVSDrm_alt",     TB_NO_REVERSE },
    { "MOVDI2SSrr",         "MOVSSrm_alt",     0 },
    { "VMOV64toPQIZrr",     "VMOVQI2PQIZrm",    TB_NO_REVERSE },
    { "VMOV64toPQIrr",      "VMOVQI2PQIrm",     TB_NO_REVERSE },
    { "VMOV64toSDZrr",      "VMOVSDZrm_alt",   TB_NO_REVERSE },
    { "VMOV64toSDrr",       "VMOVSDrm_alt",    TB_NO_REVERSE },
    { "VMOVDI2SSZrr",       "VMOVSSZrm_alt",   0 },
    { "VMOVDI2SSrr",        "VMOVSSrm_alt",    0 },
    { "MOVSDrr",            "MOVLPDrm",         TB_NO_REVERSE },
    { "VMOVSDZrr",          "VMOVLPDZ128rm",    TB_NO_REVERSE },
    { "VMOVSDrr",           "VMOVLPDrm",        TB_NO_REVERSE },

    // Part2: These following records are for manually mapping instructions that
    // have same opcode.
    // INSERTPSrm has no count_s while INSERTPSrr has count_s.
    // count_s is to indicate which element in dst vector is inserted.
    // if count_s!=0, we can't fold INSERTPSrr into INSERTPSrm
    //
    // the following folding can happen when count_s==0
    // load xmm0, m32
    // insertpsrr xmm1, xmm0, imm
    // =>
    // insertpsrm xmm1, m32, imm
    { "INSERTPSrr",         "INSERTPSrm",      TB_NO_REVERSE | TB_NO_FORWARD },
    { "UD1Lr",              "UD1Lm",           TB_NO_REVERSE | TB_NO_FORWARD },
    { "UD1Qr",              "UD1Qm",           TB_NO_REVERSE | TB_NO_FORWARD },
    { "UD1Wr",              "UD1Wm",           TB_NO_REVERSE | TB_NO_FORWARD },
    // Remove {"MMX_MOVQ64rr", "MMX_MOVQ64mr"} since it will create duplicate in
    // unfolding table due to the {"MMX_MOVD64from64rr", "MMX_MOVQ64mr"}
    { "MMX_MOVQ64rr",       "MMX_MOVQ64mr",    TB_NO_FORWARD | TB_NO_REVERSE },
    // Remove {"MMX_MOVQ64rr", "MMX_MOVQ64rm"} since it will create duplicate in
    // unfolding table due to the {"MMX_MOVD64from64rr", "MMX_MOVQ64rm"}
    { "MMX_MOVQ64rr",       "MMX_MOVQ64rm",    TB_NO_FORWARD | TB_NO_REVERSE },
};

