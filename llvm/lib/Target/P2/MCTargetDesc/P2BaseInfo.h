//===-- P2BaseInfo.h - Top level definitions for P2 MC ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone helper functions and enum definitions for
// the P2 target useful for the compiler back-end and the MC libraries.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_TARGET_P2_MCTARGETDESC_P2BASEINFO_H
#define LLVM_LIB_TARGET_P2_MCTARGETDESC_P2BASEINFO_H

#include "P2MCTargetDesc.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

    namespace P2 {
        /*
        def _ret_           : P2Effect<"_ret_", 0b0000>;
        def if_nc_and_nz    : P2Effect<"if_nc_and_nz", 0b0001>;
        def if_nc_and_z     : P2Effect<"if_nc_and_z", 0b0010>;
        def if_nc           : P2Effect<"if_nc", 0b0011>;
        def if_c_and_nz     : P2Effect<"if_c_and_nz", 0b0100>;
        def if_nz           : P2Effect<"if_nz", 0b0101>;
        def if_c_ne_z       : P2Effect<"if_c_ne_z", 0b0110>;
        def if_nc_or_nz     : P2Effect<"if_nc_or_nz", 0b0111>;
        def if_c_and_z      : P2Effect<"if_c_and_z", 0b1000>;
        def if_c_eq_z       : P2Effect<"if_c_eq_z", 0b1001>;
        def if_z            : P2Effect<"if_z", 0b1010>;
        def if_nc_or_z      : P2Effect<"if_nc_or_z", 0b1011>;
        def if_c            : P2Effect<"if_c", 0b1100>;
        def if_c_or_nz      : P2Effect<"if_c_or_nz", 0b1101>;
        def if_c_or_z       : P2Effect<"if_c_or_z", 0b1110>;
        def always          : P2Effect<"", 0b1111>;
        */

        // special immediates for rd/wrbyte/word/long that will modify PTRx
        enum name {
            PTRA_POSTINC = 0x161,
            PTRA_PREDEC = 0x15f
        };

        enum {
            _RET_ = 0,
            IF_NC_AND_NZ,
            IF_NC_AND_Z,
            IF_NC,
            IF_C_AND_NZ,
            IF_NZ,
            IF_C_NE_Z,
            IF_NC_OR_NZ,
            IF_C_AND_Z,
            IF_C_EQ_Z,
            IF_Z,
            IF_NC_OR_Z,
            IF_C,
            IF_C_OR_NZ,
            IF_C_OR_Z,
            ALWAYS
        };

        enum {
            P2Inst,
            P2InstCZIDS,
            P2Inst3NIDS,
            P2Inst2NIDS,
            P2Inst1NIDS,
            P2InstIDS,
            P2InstZIDS,
            P2InstCIDS,
            P2InstLIDS,
            P2InstIS,
            P2InstCLIDS,
            P2InstLD,
            P2InstCLD,
            P2InstCZD,
            P2InstCZ,
            P2InstCZLD,
            P2InstD,
            P2InstRA,
            P2InstWRA,
            P2InstN
        };
    }

}

#endif