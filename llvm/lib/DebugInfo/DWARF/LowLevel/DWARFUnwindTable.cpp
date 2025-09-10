//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/LowLevel/DWARFUnwindTable.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <optional>

using namespace llvm;
using namespace dwarf;

UnwindLocation UnwindLocation::createUnspecified() { return {Unspecified}; }

UnwindLocation UnwindLocation::createUndefined() { return {Undefined}; }

UnwindLocation UnwindLocation::createSame() { return {Same}; }

UnwindLocation UnwindLocation::createIsConstant(int32_t Value) {
  return {Constant, InvalidRegisterNumber, Value, std::nullopt, false};
}

UnwindLocation UnwindLocation::createIsCFAPlusOffset(int32_t Offset) {
  return {CFAPlusOffset, InvalidRegisterNumber, Offset, std::nullopt, false};
}

UnwindLocation UnwindLocation::createAtCFAPlusOffset(int32_t Offset) {
  return {CFAPlusOffset, InvalidRegisterNumber, Offset, std::nullopt, true};
}

UnwindLocation
UnwindLocation::createIsRegisterPlusOffset(uint32_t RegNum, int32_t Offset,
                                           std::optional<uint32_t> AddrSpace) {
  return {RegPlusOffset, RegNum, Offset, AddrSpace, false};
}

UnwindLocation
UnwindLocation::createAtRegisterPlusOffset(uint32_t RegNum, int32_t Offset,
                                           std::optional<uint32_t> AddrSpace) {
  return {RegPlusOffset, RegNum, Offset, AddrSpace, true};
}

UnwindLocation UnwindLocation::createIsDWARFExpression(DWARFExpression Expr) {
  return {Expr, false};
}

UnwindLocation UnwindLocation::createAtDWARFExpression(DWARFExpression Expr) {
  return {Expr, true};
}

bool UnwindLocation::operator==(const UnwindLocation &RHS) const {
  if (Kind != RHS.Kind)
    return false;
  switch (Kind) {
  case Unspecified:
  case Undefined:
  case Same:
    return true;
  case CFAPlusOffset:
    return Offset == RHS.Offset && Dereference == RHS.Dereference;
  case RegPlusOffset:
    return RegNum == RHS.RegNum && Offset == RHS.Offset &&
           Dereference == RHS.Dereference;
  case DWARFExpr:
    return *Expr == *RHS.Expr && Dereference == RHS.Dereference;
  case Constant:
    return Offset == RHS.Offset;
  }
  return false;
}

Expected<UnwindTable::RowContainer>
llvm::dwarf::parseRows(const CFIProgram &CFIP, UnwindRow &Row,
                       const RegisterLocations *InitialLocs) {
  // All the unwinding rows parsed during processing of the CFI program.
  UnwindTable::RowContainer Rows;

  // State consists of CFA value and register locations.
  std::vector<std::pair<UnwindLocation, RegisterLocations>> States;
  for (const CFIProgram::Instruction &Inst : CFIP) {
    switch (Inst.Opcode) {
    case dwarf::DW_CFA_set_loc: {
      // The DW_CFA_set_loc instruction takes a single operand that
      // represents a target address. The required action is to create a new
      // table row using the specified address as the location. All other
      // values in the new row are initially identical to the current row.
      // The new location value is always greater than the current one. If
      // the segment_size field of this FDE's CIE is non- zero, the initial
      // location is preceded by a segment selector of the given length
      llvm::Expected<uint64_t> NewAddress = Inst.getOperandAsUnsigned(CFIP, 0);
      if (!NewAddress)
        return NewAddress.takeError();
      if (*NewAddress <= Row.getAddress())
        return createStringError(
            errc::invalid_argument,
            "%s with adrress 0x%" PRIx64 " which must be greater than the "
            "current row address 0x%" PRIx64,
            CFIP.callFrameString(Inst.Opcode).str().c_str(), *NewAddress,
            Row.getAddress());
      Rows.push_back(Row);
      Row.setAddress(*NewAddress);
      break;
    }

    case dwarf::DW_CFA_advance_loc:
    case dwarf::DW_CFA_advance_loc1:
    case dwarf::DW_CFA_advance_loc2:
    case dwarf::DW_CFA_advance_loc4: {
      // The DW_CFA_advance instruction takes a single operand that
      // represents a constant delta. The required action is to create a new
      // table row with a location value that is computed by taking the
      // current entry’s location value and adding the value of delta *
      // code_alignment_factor. All other values in the new row are initially
      // identical to the current row.
      Rows.push_back(Row);
      llvm::Expected<uint64_t> Offset = Inst.getOperandAsUnsigned(CFIP, 0);
      if (!Offset)
        return Offset.takeError();
      Row.slideAddress(*Offset);
      break;
    }

    case dwarf::DW_CFA_restore:
    case dwarf::DW_CFA_restore_extended: {
      // The DW_CFA_restore instruction takes a single operand (encoded with
      // the opcode) that represents a register number. The required action
      // is to change the rule for the indicated register to the rule
      // assigned it by the initial_instructions in the CIE.
      if (InitialLocs == nullptr)
        return createStringError(
            errc::invalid_argument, "%s encountered while parsing a CIE",
            CFIP.callFrameString(Inst.Opcode).str().c_str());
      llvm::Expected<uint64_t> RegNum = Inst.getOperandAsUnsigned(CFIP, 0);
      if (!RegNum)
        return RegNum.takeError();
      if (std::optional<UnwindLocation> O =
              InitialLocs->getRegisterLocation(*RegNum))
        Row.getRegisterLocations().setRegisterLocation(*RegNum, *O);
      else
        Row.getRegisterLocations().removeRegisterLocation(*RegNum);
      break;
    }

    case dwarf::DW_CFA_offset:
    case dwarf::DW_CFA_offset_extended:
    case dwarf::DW_CFA_offset_extended_sf: {
      llvm::Expected<uint64_t> RegNum = Inst.getOperandAsUnsigned(CFIP, 0);
      if (!RegNum)
        return RegNum.takeError();
      llvm::Expected<int64_t> Offset = Inst.getOperandAsSigned(CFIP, 1);
      if (!Offset)
        return Offset.takeError();
      Row.getRegisterLocations().setRegisterLocation(
          *RegNum, UnwindLocation::createAtCFAPlusOffset(*Offset));
      break;
    }

    case dwarf::DW_CFA_nop:
      break;

    case dwarf::DW_CFA_remember_state:
      States.push_back(
          std::make_pair(Row.getCFAValue(), Row.getRegisterLocations()));
      break;

    case dwarf::DW_CFA_restore_state:
      if (States.empty())
        return createStringError(errc::invalid_argument,
                                 "DW_CFA_restore_state without a matching "
                                 "previous DW_CFA_remember_state");
      Row.getCFAValue() = States.back().first;
      Row.getRegisterLocations() = States.back().second;
      States.pop_back();
      break;

    case dwarf::DW_CFA_GNU_window_save:
      switch (CFIP.triple()) {
      case Triple::aarch64:
      case Triple::aarch64_be:
      case Triple::aarch64_32: {
        // DW_CFA_GNU_window_save is used for different things on different
        // architectures. For aarch64 it is known as
        // DW_CFA_AARCH64_negate_ra_state. The action is to toggle the
        // value of the return address state between 1 and 0. If there is
        // no rule for the AARCH64_DWARF_PAUTH_RA_STATE register, then it
        // should be initially set to 1.
        constexpr uint32_t AArch64DWARFPAuthRaState = 34;
        auto LRLoc = Row.getRegisterLocations().getRegisterLocation(
            AArch64DWARFPAuthRaState);
        if (LRLoc) {
          if (LRLoc->getLocation() == UnwindLocation::Constant) {
            // Toggle the constant value from 0 to 1 or 1 to 0.
            LRLoc->setConstant(LRLoc->getConstant() ^ 1);
            Row.getRegisterLocations().setRegisterLocation(
                AArch64DWARFPAuthRaState, *LRLoc);
          } else {
            return createStringError(
                errc::invalid_argument,
                "%s encountered when existing rule for this register is not "
                "a constant",
                CFIP.callFrameString(Inst.Opcode).str().c_str());
          }
        } else {
          Row.getRegisterLocations().setRegisterLocation(
              AArch64DWARFPAuthRaState, UnwindLocation::createIsConstant(1));
        }
        break;
      }

      case Triple::sparc:
      case Triple::sparcv9:
      case Triple::sparcel:
        for (uint32_t RegNum = 16; RegNum < 32; ++RegNum) {
          Row.getRegisterLocations().setRegisterLocation(
              RegNum, UnwindLocation::createAtCFAPlusOffset((RegNum - 16) * 8));
        }
        break;

      default: {
        return createStringError(
            errc::not_supported,
            "DW_CFA opcode %#x is not supported for architecture %s",
            Inst.Opcode, Triple::getArchTypeName(CFIP.triple()).str().c_str());

        break;
      }
      }
      break;

    case dwarf::DW_CFA_AARCH64_negate_ra_state_with_pc: {
      constexpr uint32_t AArch64DWARFPAuthRaState = 34;
      auto LRLoc = Row.getRegisterLocations().getRegisterLocation(
          AArch64DWARFPAuthRaState);
      if (LRLoc) {
        if (LRLoc->getLocation() == UnwindLocation::Constant) {
          // Toggle the constant value of bits[1:0] from 0 to 1 or 1 to 0.
          LRLoc->setConstant(LRLoc->getConstant() ^ 0x3);
        } else {
          return createStringError(
              errc::invalid_argument,
              "%s encountered when existing rule for this register is not "
              "a constant",
              CFIP.callFrameString(Inst.Opcode).str().c_str());
        }
      } else {
        Row.getRegisterLocations().setRegisterLocation(
            AArch64DWARFPAuthRaState, UnwindLocation::createIsConstant(0x3));
      }
      break;
    }

    case dwarf::DW_CFA_undefined: {
      llvm::Expected<uint64_t> RegNum = Inst.getOperandAsUnsigned(CFIP, 0);
      if (!RegNum)
        return RegNum.takeError();
      Row.getRegisterLocations().setRegisterLocation(
          *RegNum, UnwindLocation::createUndefined());
      break;
    }

    case dwarf::DW_CFA_same_value: {
      llvm::Expected<uint64_t> RegNum = Inst.getOperandAsUnsigned(CFIP, 0);
      if (!RegNum)
        return RegNum.takeError();
      Row.getRegisterLocations().setRegisterLocation(
          *RegNum, UnwindLocation::createSame());
      break;
    }

    case dwarf::DW_CFA_GNU_args_size:
      break;

    case dwarf::DW_CFA_register: {
      llvm::Expected<uint64_t> RegNum = Inst.getOperandAsUnsigned(CFIP, 0);
      if (!RegNum)
        return RegNum.takeError();
      llvm::Expected<uint64_t> NewRegNum = Inst.getOperandAsUnsigned(CFIP, 1);
      if (!NewRegNum)
        return NewRegNum.takeError();
      Row.getRegisterLocations().setRegisterLocation(
          *RegNum, UnwindLocation::createIsRegisterPlusOffset(*NewRegNum, 0));
      break;
    }

    case dwarf::DW_CFA_val_offset:
    case dwarf::DW_CFA_val_offset_sf: {
      llvm::Expected<uint64_t> RegNum = Inst.getOperandAsUnsigned(CFIP, 0);
      if (!RegNum)
        return RegNum.takeError();
      llvm::Expected<int64_t> Offset = Inst.getOperandAsSigned(CFIP, 1);
      if (!Offset)
        return Offset.takeError();
      Row.getRegisterLocations().setRegisterLocation(
          *RegNum, UnwindLocation::createIsCFAPlusOffset(*Offset));
      break;
    }

    case dwarf::DW_CFA_expression: {
      llvm::Expected<uint64_t> RegNum = Inst.getOperandAsUnsigned(CFIP, 0);
      if (!RegNum)
        return RegNum.takeError();
      Row.getRegisterLocations().setRegisterLocation(
          *RegNum, UnwindLocation::createAtDWARFExpression(*Inst.Expression));
      break;
    }

    case dwarf::DW_CFA_val_expression: {
      llvm::Expected<uint64_t> RegNum = Inst.getOperandAsUnsigned(CFIP, 0);
      if (!RegNum)
        return RegNum.takeError();
      Row.getRegisterLocations().setRegisterLocation(
          *RegNum, UnwindLocation::createIsDWARFExpression(*Inst.Expression));
      break;
    }

    case dwarf::DW_CFA_def_cfa_register: {
      llvm::Expected<uint64_t> RegNum = Inst.getOperandAsUnsigned(CFIP, 0);
      if (!RegNum)
        return RegNum.takeError();
      if (Row.getCFAValue().getLocation() != UnwindLocation::RegPlusOffset)
        Row.getCFAValue() =
            UnwindLocation::createIsRegisterPlusOffset(*RegNum, 0);
      else
        Row.getCFAValue().setRegister(*RegNum);
      break;
    }

    case dwarf::DW_CFA_def_cfa_offset:
    case dwarf::DW_CFA_def_cfa_offset_sf: {
      llvm::Expected<int64_t> Offset = Inst.getOperandAsSigned(CFIP, 0);
      if (!Offset)
        return Offset.takeError();
      if (Row.getCFAValue().getLocation() != UnwindLocation::RegPlusOffset) {
        return createStringError(
            errc::invalid_argument,
            "%s found when CFA rule was not RegPlusOffset",
            CFIP.callFrameString(Inst.Opcode).str().c_str());
      }
      Row.getCFAValue().setOffset(*Offset);
      break;
    }

    case dwarf::DW_CFA_def_cfa:
    case dwarf::DW_CFA_def_cfa_sf: {
      llvm::Expected<uint64_t> RegNum = Inst.getOperandAsUnsigned(CFIP, 0);
      if (!RegNum)
        return RegNum.takeError();
      llvm::Expected<int64_t> Offset = Inst.getOperandAsSigned(CFIP, 1);
      if (!Offset)
        return Offset.takeError();
      Row.getCFAValue() =
          UnwindLocation::createIsRegisterPlusOffset(*RegNum, *Offset);
      break;
    }

    case dwarf::DW_CFA_LLVM_def_aspace_cfa:
    case dwarf::DW_CFA_LLVM_def_aspace_cfa_sf: {
      llvm::Expected<uint64_t> RegNum = Inst.getOperandAsUnsigned(CFIP, 0);
      if (!RegNum)
        return RegNum.takeError();
      llvm::Expected<int64_t> Offset = Inst.getOperandAsSigned(CFIP, 1);
      if (!Offset)
        return Offset.takeError();
      llvm::Expected<uint32_t> CFAAddrSpace =
          Inst.getOperandAsUnsigned(CFIP, 2);
      if (!CFAAddrSpace)
        return CFAAddrSpace.takeError();
      Row.getCFAValue() = UnwindLocation::createIsRegisterPlusOffset(
          *RegNum, *Offset, *CFAAddrSpace);
      break;
    }

    case dwarf::DW_CFA_def_cfa_expression:
      Row.getCFAValue() =
          UnwindLocation::createIsDWARFExpression(*Inst.Expression);
      break;
    }
  }
  return Rows;
}
