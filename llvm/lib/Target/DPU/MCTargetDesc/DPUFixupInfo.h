//===-- DPUFixupInfo.h - DPU Fixup Information ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DPUFIXUPINFO_H
#define LLVM_DPUFIXUPINFO_H

#include "DPUFixupKinds.h"

namespace llvm {
namespace DPU {

static inline DPU::Fixups findFixupForOperand(unsigned OpNum, unsigned InstOpcode) {
	switch (InstOpcode) {
		default:
			llvm_unreachable("unknown instruction");
		case DPU::LSLXrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::SUBC_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::TELLri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::NEGrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::NANDrric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ANDzrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28;
			}
		case DPU::XORzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_SH_SLzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::OR_Srric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ANDrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::CALLrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSL_ADD_Urrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::LSLzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::NAND_Srric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ADD_Urri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM32_DUS_RB;
			}
		case DPU::SUB_Srirci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::AND_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUB_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ANDN_Srrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR1Xrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::JGESrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::CLS_Urr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ORNrrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::CMPB4rrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::CMPB4zrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_UL_UH_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::TIME_CFG_Urr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::RSUBCrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSR1zrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ROL_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUBCrirci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::RSUB_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::NXORrrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::SUBC_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SDesir:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_RB;
			}
		case DPU::NANDrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SL_SHrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SH_ULrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NORzric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28;
			}
		case DPU::LSLXzri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::NAND_Srrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSLXrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::CLOzr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::CLOrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::CLO_Srrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ROLzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SH_ULzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::NOR_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::RSUBC_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::JLEUrii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM27_PC;
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR1Xzric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::NOR_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LHS_Serri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::CLSrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSRzrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ROL_ADD_Urrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::SBSrii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM8_STR;
			}
		case DPU::SHrir:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24_RB;
			}
		case DPU::ASRrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ROL_Urri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::LSLzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NXOR_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::AND_Srrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUB_Srirc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::SDMArri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM8_DMA;
			}
		case DPU::ORN_Srrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::CLO_Urrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSLXzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SHrii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM16_STR;
			}
		case DPU::AND_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADDCzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::NXORzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::BOOTri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
			}
		case DPU::NOR_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SDSrii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM16_STR;
			}
		case DPU::ROLrric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::SD_IDrii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM16_STR;
			}
		case DPU::AND_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSR1_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL1rrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SATS_Urrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL1rric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::NXORrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ORN_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ASRrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SH_UH_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::FAULTi:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 0:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::RSUB_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ROR_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::CLOzrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SL_SHrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUBrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ROL_Srric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::CMPB4_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::CLO_Urrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SHerir:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_RB;
			}
		case DPU::LHU_Urri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::SUBrirci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::CLZ_Urrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ANDN_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::CLZrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ORNrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SH_SLrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::HASH_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NANDrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ORNzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::TIME_CFGzr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::CAO_Srrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ASR_Srrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR_Srrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUBSrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM17_24;
			}
		case DPU::LSR1rrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ROLzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ROLrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::CLZ_Srrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SWSrii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM16_STR;
			}
		case DPU::ADD_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ORNrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::RSUBC_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL1zrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ANDN_Srric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LHU_Uerri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::SWesii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM16_STR;
			}
		case DPU::NORrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADD_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::EXTUHzrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL1zri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::LW_Urri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::EXTSHzrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::HASH_Urrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::MUL_UH_ULzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ASR_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADDC_Urrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::CLR_RUNrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR_ADD_Urrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::ORrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ADDzric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM27;
			}
		case DPU::LSLX_Urrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::CMPB4rrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ROR_Urri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::SWrir:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24_RB;
			}
		case DPU::EXTSB_Srrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LBU_Urri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::NANDzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL1_Srrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SDerir:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_RB;
			}
		case DPU::LWSrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::MUL_SH_SHrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::CALLrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SL_SH_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NXORzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::XORzric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28;
			}
		case DPU::RSUB_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NAND_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ROR_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ORNzrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28_PC_OPC8;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LWerri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::TIME_CFG_Srrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::EXTSHrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NXOR_Srrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_SL_SHrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSR1_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSRX_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_UL_UL_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ANDzrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28_PC_OPC8;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ASR_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL1X_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSRX_Urrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ASRzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSRX_Srri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::MUL_UL_UHzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUBC_Urirf:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::EXTUBrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUBC_Srirci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ROL_ADD_Urrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 5:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::CMPB4rrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADDC_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::EXTSH_Srr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::CLS_Srr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::CALLri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::MUL_SH_UL_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_SH_SH_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::CMPB4_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SWerii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM16_STR;
			}
		case DPU::LSL_SUBrrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 5:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL_SUBzrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::LHS_Srri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::CLOrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ASRzric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::ROL_Srri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::STOP:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUBzirc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 0:
					return DPU::Fixups::FIXUP_DPU_IMM27;
			}
		case DPU::LSR1_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::EXTUHrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::CLZrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ROL_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSRXzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_UL_UHrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSRXzric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::TIME_Sr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSRX_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::EXTSBrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSRX_Srrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::HASH_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::CLZ_Srrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSRXrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUBC_Srirc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::NXOR_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SH_ULzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SH_SHzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_STEPrrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::JZri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::BOOTrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::CLOrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LBUerri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ORrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::RORrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::CMPB4zrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::HASH_Urric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::JGTSrii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM27_PC;
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR1X_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL1Xrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSR1Xrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MOVErici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::CLS_Srrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL1rrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ORN_Srrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::RORrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_SH_SLzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::XOR_Srric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSRX_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LBUSrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ANDzri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM32_ZERO_RB;
			}
		case DPU::LSR1Xzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NAND_Urrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ASR_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::CLSzrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR1Xzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSLX_Srri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::NXORrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::JUMPi:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 0:
					return DPU::Fixups::FIXUP_DPU_IMM28;
			}
		case DPU::LW_Srri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSR_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SWSrir:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24_RB;
			}
		case DPU::ASRrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::CLZzrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SATSzr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_UH_UHrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_SL_SLrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSLrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::LSR_Urric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::CMPB4_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUBzric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM27;
			}
		case DPU::AND_Srric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ORzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SL_ULrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL1zrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::EXTUH_Urrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SDrir:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24_RB;
			}
		case DPU::SD_IDerii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM16_STR;
			}
		case DPU::SUBzirci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 0:
					return DPU::Fixups::FIXUP_DPU_IMM27_PC;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUBCzirci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 0:
					return DPU::Fixups::FIXUP_DPU_IMM27_PC;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::JLESrii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM27_PC;
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ADDCzrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM27;
			}
		case DPU::EXTUBzr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADDrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::XOR_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL1Xzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::JLEUrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ORrric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::NANDzric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28;
			}
		case DPU::LSL_Urrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL1_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSR1X_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::AND_Urric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LHUrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSL1X_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADDzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::DIV_STEPrrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::LDMArri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM8_DMA;
			}
		case DPU::ASRzri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::MUL_SH_UHrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::OR_Urri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM32_DUS_RB;
			}
		case DPU::NXORrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSL_ADD_Srrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::NXORzrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28_PC_OPC8;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ADDC_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL_ADD_Srrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 5:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ADDC_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::RORrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SW_IDrii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM16_STR;
			}
		case DPU::MUL_UL_ULzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_SH_UHzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL_ADDzrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::AND_Srri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM32_DUS_RB;
			}
		case DPU::CAOrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NANDzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUBsss:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ROL_ADDzrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_STEPrrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 5:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_UH_UHzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUBCzir:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 0:
					return DPU::Fixups::FIXUP_DPU_IMM32_ZERO_RB;
			}
		case DPU::LBS_Serri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ADDCrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::XORrrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSL1X_Urrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NAND_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::EXTSB_Srrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::JLTUrii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM27_PC;
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NANDzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::JUMPri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28;
			}
		case DPU::LSR1X_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADDC_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::XORzri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM32_ZERO_RB;
			}
		case DPU::HASHrric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ORN_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSLX_Srrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_UH_UL_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR1X_Urri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::ROR_Srri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::HASHrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::JEQrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSRXrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SL_SL_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::AND_Urki:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM32;
			}
		case DPU::SATS_Urrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ORrrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::MUL_SL_ULzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LDSrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LDMAIrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM8_DMA;
			}
		case DPU::ASR_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SHesii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM16_STR;
			}
		case DPU::LSR1Xrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ADDzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SDSrir:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24_RB;
			}
		case DPU::LSR_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SH_SH_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::JUMPr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::EXTSH_Srrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_UL_ULzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SATSzrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::AND_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::HASHzric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28;
			}
		case DPU::LSLrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ORNrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR1rrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR1rrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::NOR_Srrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::CLS_Urrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::EXTUH_Urrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LBUrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::MUL_SL_ULrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADDCzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::TIME_Ur:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::NXORrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ROR_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUB_Srrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::EXTUB_Urrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADDC_Srrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ROLzri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::XOR_Srrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSL_SUB_Srrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 5:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::CLO_Srr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LBSerri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::SUBrirc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::SUBC_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SH_SL_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LHSSrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::SUBCrric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::NANDrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR1X_Srri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::HASHrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::OR_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUBCzric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM27;
			}
		case DPU::ORN_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SWrii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM16_STR;
			}
		case DPU::LSR1_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUB_Urirc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::SUBCzrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM27_PC;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ASR_Urri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::LSLXzric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::AND_Srki:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM32;
			}
		case DPU::LDrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::CLZ_Urr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUBC_Urric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::MUL_UH_UL_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADD_Urrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::NANDrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_SH_SL_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ANDzric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28;
			}
		case DPU::ACQUIRErici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM16_ATM;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ANDN_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUBCzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSR1X_Srrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::TIMEr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL1_Srri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::ADDC_Srrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::XOR_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::EXTUHrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::RSUBCrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUB_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::RORrric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::HASHrrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::NAND_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::JLTUrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ANDN_Srrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::RSUBzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUBzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ROR_Srric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::MUL_SL_ULzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSRrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ASRrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::RSUBrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL1_Urri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::MUL_UL_ULzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MOVDrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSLzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::TIME_Urci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_SH_UHzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::NXOR_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_UH_ULrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ASR_Urrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSLrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSLX_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::HASHrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::CLS_Srrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_UL_UL_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::NANDrrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ORN_Urrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUBC_Urirc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSL1_Urrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR1X_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADDC_Urric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSL1Xzri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::LW_Serri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::OR_Urrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSR1X_Urrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSRX_Srric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::LSL_Urri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::CMPB4_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUBssi:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM17_24;
			}
		case DPU::LHUSrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ANDNzrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28_PC_OPC8;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ROLrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LHUerri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ADDzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADDSrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM17_24;
			}
		case DPU::NAND_Urrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL_ADDzrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::RSUBCzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR1X_Srric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::SUBCrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::TIMEz:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL1zrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::JGTSrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NAND_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LWersi:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::MUL_SL_SH_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::RORrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::RORzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL_SUB_Urrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 5:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL_ADDrrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::ROL_Urric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::LSL1X_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SWAPDrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SD_IDri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
			}
		case DPU::MOVE_Uri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM32;
			}
		case DPU::LBSSrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::MUL_SH_SLzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::HASH_Srrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::SUBzir:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 0:
					return DPU::Fixups::FIXUP_DPU_IMM32_ZERO_RB;
			}
		case DPU::LSL1zrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::OR_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SL_UHrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUBC_Urrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ADDC_Srric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSRzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SL_UHzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ORN_Srric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::HASH_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUB_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ROL_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ORNrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::SUB_Srirf:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ROL_ADDrrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 5:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::TIME_CFG_Srr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUBCzrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM27;
			}
		case DPU::ANDNzrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28;
			}
		case DPU::LHSersi:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSR1_Srrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR_ADD_Srrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 5:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::OR_Urrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSRrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::HASH_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSR_Srric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::NORrrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSL_ADDrrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 5:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_UH_UHzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ADD_Srri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM32_DUS_RB;
			}
		case DPU::LBUersi:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSLXzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ANDzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LBS_Srri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSL1rri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::MUL_UH_ULrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ADDCrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM32;
			}
		case DPU::LSL_Srri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::LSRzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUBCrir:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM32;
			}
		case DPU::NXORzric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28;
			}
		case DPU::NXOR_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ORzri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM32_ZERO_RB;
			}
		case DPU::RORzric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::MUL_UL_ULrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSR1_Srri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::LSR1_Urrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::CAOzr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ANDrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NOR_Srric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::CLZ_Srr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::XOR_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::AND_Urrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL1Xrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::CLSzr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::JNEQrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_SH_SH_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUBCzirf:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 0:
					return DPU::Fixups::FIXUP_DPU_IMM27;
			}
		case DPU::MUL_UH_ULzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUB_Urirci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_SL_SLzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_SL_UH_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::CLSrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSLX_Urri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::XORrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL1Xzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADD_Srric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::MUL_UL_UL_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::NAND_Srrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ANDNzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SBrii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM8_STR;
			}
		case DPU::LSRzri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::ORzrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28;
			}
		case DPU::LSRX_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::HASHrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::EXTUHzrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::EXTUBrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUB_Urirf:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::DIV_STEPrrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 5:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NORrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::OR_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::EXTSHrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ORzrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28_PC_OPC8;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SDMA_DBGrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM8_DMA;
			}
		case DPU::CAO_Srrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSRXzrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ADDCzric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM27;
			}
		case DPU::LSRXrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::NORzrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28;
			}
		case DPU::READ_RUNrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSR1zrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSR1X_Urric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::ADD_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::TIME_CFGzrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSLrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSRXrric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::ROL_ADDzrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::TIMErci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LHUersi:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSLX_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ANDNrric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::EXTUH_Urr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SH_IDerii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM16_STR;
			}
		case DPU::SDrii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM16_STR;
			}
		case DPU::EXTUB_Urr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SH_UHrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::HASH_Srric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::EXTSHzrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUBCzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SL_SHzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ORNzric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28;
			}
		case DPU::LSL1X_Srric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::LDMA_DBGrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM8_DMA;
			}
		case DPU::MUL_SL_SHzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NXOR_Urrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::JGTUrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SB_IDri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
			}
		case DPU::ORN_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ANDN_Urrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUBC_Srirf:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::SUBCrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADDCrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_SL_SLzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSR1zrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SH_ULrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::CALLrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::NAND_Urric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::CLZrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::RORzri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::LW_Uerri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::MUL_UH_UHrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::OR_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::HASHzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL1X_Srri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::NORrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ORzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR1X_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NXORrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSRXzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::RSUB_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SL_UH_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ORNzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NOR_Urric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSR_Urri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::ADDC_Urrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ADD_Urrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::XORrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM32;
			}
		case DPU::SATS_Srr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::RSUBCzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SWAPDrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ASR_Srri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::ROL_ADD_Srrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::RSUBC_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::XOR_Srrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL1Xrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSRXzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADD_Srrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ROL_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NORzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUBCrirc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSLzric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::SUBrir:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM32;
			}
		case DPU::LSR_ADDrrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 5:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LDersi:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::NOP:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SH_IDrii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM16_STR;
			}
		case DPU::LSR1Xrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ROLrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::NORzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::JGEUrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::CAO_Urr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::XOR_Urrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::TIME_CFGrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUBCrirf:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::SUBC_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR1Xrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::XORrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_SH_SL_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::TIME_CFGr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSR1Xrric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::NXORzrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28;
			}
		case DPU::LSL_SUB_Srrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::RSUBzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MOVE_Srr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSLzri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::EXTUBzrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUBrric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ADDrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM32;
			}
		case DPU::LHSrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::JEQrii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM27_PC;
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MOVDrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SBesii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM8_STR;
			}
		case DPU::CLZzrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::JGEUrii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM27_PC;
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ORNzrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28;
			}
		case DPU::LSL1Xzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::HASH_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SH_SLrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::EXTUHrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR_Urrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR1_Urri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::SUBC_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::RSUBCzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::CLSrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SH_UH_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::CAOrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ROL_Srrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ASR_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSLXzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL1X_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::CLZ_Urrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LWrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::SATS_Urr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ORNzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::RSUBzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ANDrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_SL_SL_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::HASHrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NXOR_Srric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::XORzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL1_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SH_SHzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SW_IDri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
			}
		case DPU::LSL1X_Urric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::MUL_SL_ULrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSLrric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::SUBzirf:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 0:
					return DPU::Fixups::FIXUP_DPU_IMM27;
			}
		case DPU::SB_IDerii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM8_STR;
			}
		case DPU::MUL_SL_SL_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SATSrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSLXzrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::RORrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSR1_Srric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::EXTSBrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::XOR_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SBerir:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_RB;
			}
		case DPU::ADDCrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUBC_Srrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUBC_Srrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ANDN_Urrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::CLSzrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::RESUMEri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
			}
		case DPU::ADDCzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL1_Urric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::LSR1rri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::EXTUBrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSR1X_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SATS_Srrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSRX_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_UL_UHzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ROLrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SL_UHrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL1_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::XORzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSR_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::CAOzrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::NOR_Urrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::EXTSBzrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR_ADDzrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SHSrii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM16_STR;
			}
		case DPU::HASHzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_UL_UHzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::CLZzr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ROL_Urrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUBCrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SHesir:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_RB;
			}
		case DPU::CLS_Urrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUBCrrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::SBSrir:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24_RB;
			}
		case DPU::MUL_SH_ULzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::AND_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSLX_Srric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::LSR_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SH_UHrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ANDNzric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28;
			}
		case DPU::RSUBrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL1X_Srrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL_SUB_Urrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::RSUBC_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::JNEQrii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM27_PC;
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LHSerri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSLXrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_SL_UH_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SHSrir:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24_RB;
			}
		case DPU::SUBrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SATS_Srrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::HASH_Urrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSRX_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUB_Srrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::OR_Srrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_UH_UL_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ANDNrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ANDN_Urric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSLzrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_UH_UHzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::TIME_Srci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SATSrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SDesii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM16_STR;
			}
		case DPU::EXTUHzr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::CAO_Urrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::RORzrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ANDNrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ROR_Urrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR_ADD_Urrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 5:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ROLzric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::OR_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ROR_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADDCrric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::SUB_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_UH_UHrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SL_UL_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::CMPB4_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADD_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::TIME_CFGrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_UL_UHrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SWerir:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_RB;
			}
		case DPU::JLTSrii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM27_PC;
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ROR_Srrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::EXTSH_Srrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ANDzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUBC_Urrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::RESUMErici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL1zric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::NXOR_Urric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::MUL_SL_SH_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADDCzri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM32_ZERO_RB;
			}
		case DPU::LSR_Srri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::ADDrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NOR_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SB_IDrii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM8_STR;
			}
		case DPU::ROR_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSLrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL1X_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADD_Srrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::EXTSB_Srr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSLXrric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::ANDrrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSR_ADDrrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::MOVE_Sri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM32;
			}
		case DPU::ROLzrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSLXrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSRX_Urric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::MUL_SH_UL_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUBC_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSLX_Urric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::CLO_Urr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSR_ADDzrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::ASRrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::SUB_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUBCzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ANDN_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ORN_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADDSrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::OR_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ORrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM32;
			}
		case DPU::ANDNrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SL_UL_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::BKP:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_UL_UH_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::XOR_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NOR_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUBrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUB_Urric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSR1Xzri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::JLTSrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL1Xrric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::CMPB4zrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSRrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::OR_Srrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::HASH_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_SH_SHrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSLX_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL1rrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ORN_Urric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::CALLzri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28;
			}
		case DPU::RSUBC_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::AND_Srrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ADDC_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSR1rrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::NXOR_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ANDzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ANDNrrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSRXzri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::XOR_Urrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ASRzrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NXORrric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSRrric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::SUBC_Urirci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL1Xrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSR1Xzrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ANDrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_UH_ULrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ORN_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSR_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::TIMEzci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NANDzrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28;
			}
		case DPU::ADDrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::XORzrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28;
			}
		case DPU::ADDCrrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ANDNzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_UL_UHrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUBzrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM27_PC;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSRXrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ANDN_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR1rric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::ASR_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::NEGrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_UL_ULrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::HASHzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL1Xzric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::NOR_Urrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::SUBCrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SBrir:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24_RB;
			}
		case DPU::MUL_SL_SLzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::CLOzrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ASRrric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::SUB_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NORrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ROL_ADDrrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::LBU_Uerri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::NXOR_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ORzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::JNZri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSLX_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::CAOzrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::CMPB4_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::RSUBrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADDzri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM32_ZERO_RB;
			}
		case DPU::STOPci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::RELEASErici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM16_ATM;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::CAOrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MOVE_Urr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::EXTUBzrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADDssi:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM17_24;
			}
		case DPU::LSRrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NORrric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ADDCzrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM27_PC;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ORNrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::CAO_Urrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::RSUBCrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ADD_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSRX_Urri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::RSUB_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SH_IDri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
			}
		case DPU::CAO_Srr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LBSersi:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::MUL_UL_UH_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_SH_SLrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ASR_Urric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::ROLrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::MUL_SH_UHzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LDerri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ADD_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_SH_ULrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SL_SLrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::JGESrii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM27_PC;
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSRzric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::NANDrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::MUL_UL_ULrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::XORzrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28_PC_OPC8;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NOR_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::NORzrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28_PC_OPC8;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SWesir:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_RB;
			}
		case DPU::NXOR_Urrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NOTrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ANDrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM32;
			}
		case DPU::ANDNrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ORNrric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::SUBSrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::NXOR_Srrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSL_ADD_Urrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 5:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::CLO_Srrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADDrrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ANDN_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SATSrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SH_SHrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ROL_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL1Xzrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUBrrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::MOVErrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR1_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::OR_Urric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ADDC_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ANDNzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR1zri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::LSL1Xrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::TIME_CFG_Urrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_SH_SHzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::RORzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ORN_Urrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::CALLzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ASRzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL1_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ASR_Srric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::LSL_SUBrrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::SUBzrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM27;
			}
		case DPU::ADD_Urric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::SUB_Urrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ROR_Urric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::MUL_SL_UHrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::NAND_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::XORrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SDerii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM16_STR;
			}
		case DPU::ASRzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SBesir:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_RB;
			}
		case DPU::MUL_SL_SLrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL1_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUBrirf:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::HASHzrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28;
			}
		case DPU::ANDNrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SW_IDerii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM16_STR;
			}
		case DPU::ORzric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28;
			}
		case DPU::NXORzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ROLzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::AND_Urri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM32_DUS_RB;
			}
		case DPU::LSL_SUBzrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ADDCrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::NOR_Srrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUB_Urrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::SUBC_Srric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::ADDrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SHerii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM16_STR;
			}
		case DPU::NORzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::HASH_Srrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL_Srrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSLX_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL1_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL1rrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_SL_UHzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SL_SHzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::RSUBC_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSR1zric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::LSRzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::JGTUrii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM27_PC;
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR_ADD_Srrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::LSLX_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUBzrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL_Urric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::AND_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::EXTSHzr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADDzrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM27_PC;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR1_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSRrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SBerii:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM12_STR;
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM8_STR;
			}
		case DPU::LSLXrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::NORrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24_PC;
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_SL_ULzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::EXTSBzrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::EXTSBzr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_UH_ULzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUB_Srric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSL_Srric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::ROR_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::EXTUB_Urrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::OR_Srri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM32_DUS_RB;
			}
		case DPU::XOR_Urric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::EXTSHrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_UH_UH_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL1X_Urri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::ADDzrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM27;
			}
		case DPU::ANDrric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::MOVEri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM32;
			}
		case DPU::ADDrric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::HASHzrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28_PC_OPC8;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::ORrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::NANDzrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM28_PC_OPC8;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::SUBrrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ORrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::EXTSBrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSL1X_Srrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_SL_UL_Srrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_UH_UH_Urrrc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::AND_Urrif:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::MUL_SL_UHzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::LSRXrrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSL1_Srric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::SUBzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LBSrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::LSR1zrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 1:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR1_Urric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM5;
			}
		case DPU::ROL_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::MUL_SH_UH_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ANDN_Urrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::NAND_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SUBCzirc:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 0:
					return DPU::Fixups::FIXUP_DPU_IMM27;
			}
		case DPU::ROL_ADD_Srrrici:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_IMM5;
				case 5:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::RORzrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 3:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MOVErr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::XORrric:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_IMM24;
			}
		case DPU::MUL_UH_UH_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::RSUB_Urrrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 4:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::MUL_SH_UL_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::SATSzrci:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::JLESrri:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
				case 2:
					return DPU::Fixups::FIXUP_DPU_PC;
			}
		case DPU::LSR1Xzrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::XOR_Srrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::ADDsss:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
		case DPU::XORrrr:
			switch (OpNum) {
				default:
					llvm_unreachable("invalid operand number");
			}
	}
}
}
}
#endif // LLVM_DPUFIXUPINFO_H
