//==-- llvm/CodeGen/GlobalISel/RegisterBankInfo.h ----------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This file declares the API for the register bank info.
/// This API is responsible for handling the register banks.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_REGBANKINFO_H
#define LLVM_CODEGEN_GLOBALISEL_REGBANKINFO_H

#include <memory> // For unique_ptr.

namespace llvm {
class RegisterBank;
class TargetRegisterInfo;

/// Holds all the information related to register banks.
class RegisterBankInfo {
protected:
  /// Hold the set of supported register banks.
  std::unique_ptr<RegisterBank[]> RegBanks;
  /// Total number of register banks.
  unsigned NumRegBanks;

  /// Create a RegisterBankInfo that can accomodate up to \p NumRegBanks
  /// RegisterBank instances.
  ///
  /// \note For the verify method to succeed all the \p NumRegBanks
  /// must be initialized by createRegisterBank and updated with
  /// addRegBankCoverage RegisterBank.
  RegisterBankInfo(unsigned NumRegBanks);

  virtual ~RegisterBankInfo();

  /// Create a new register bank with the given parameter and add it
  /// to RegBanks.
  /// \pre \p ID must not already be used.
  /// \pre \p ID < NumRegBanks.
  void createRegisterBank(unsigned ID, const char *Name);

  /// Add \p RC to the set of register class that the register bank
  /// identified \p ID covers.
  /// This method transitively adds all the sub classes and the subreg-classes
  /// of \p RC to the set of covered register classes.
  /// It also adjusts the size of the register bank to reflect the maximal
  /// size of a value that can be hold into that register bank.
  ///
  /// \note This method does *not* add the super classes of \p RC.
  /// The rationale is if \p ID covers the registers of \p RC, that
  /// does not necessarily mean that \p ID covers the set of registers
  /// of RC's superclasses.
  /// This method does *not* add the superreg classes as well for consistents.
  /// The expected use is to add the coverage top-down with respect to the
  /// register hierarchy.
  ///
  /// \todo TableGen should just generate the BitSet vector for us.
  void addRegBankCoverage(unsigned ID, const TargetRegisterClass &RC,
                          const TargetRegisterInfo &TRI);

  /// Get the register bank identified by \p ID.
  RegisterBank &getRegBank(unsigned ID) {
    assert(ID < getNumRegBanks() && "Accessing an unknown register bank");
    return RegBanks[ID];
  }

public:
  /// Get the register bank identified by \p ID.
  const RegisterBank &getRegBank(unsigned ID) const {
    return const_cast<RegisterBankInfo *>(this)->getRegBank(ID);
  }

  /// Get the total number of register banks.
  unsigned getNumRegBanks() const { return NumRegBanks; }

  /// Get the cost of a copy from \p B to \p A, or put differently,
  /// get the cost of A = COPY B.
  virtual unsigned copyCost(const RegisterBank &A,
                            const RegisterBank &B) const {
    return 0;
  }

  void verify(const TargetRegisterInfo &TRI) const;
};
} // End namespace llvm.

#endif
