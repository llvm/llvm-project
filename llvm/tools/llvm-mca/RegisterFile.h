//===--------------------- RegisterFile.h -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines a register mapping file class.  This class is responsible
/// for managing hardware register files and the tracking of data dependencies
/// between registers.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_REGISTER_FILE_H
#define LLVM_TOOLS_LLVM_MCA_REGISTER_FILE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSchedule.h"

namespace mca {

class ReadState;
class WriteState;

/// Manages hardware register files, and tracks register definitions for
/// register renaming purposes.
class RegisterFile {
  const llvm::MCRegisterInfo &MRI;

  // Each register file is associated with an instance of RegisterMappingTracker.
  // A RegisterMappingTracker keeps track of the number of physical registers
  // which have been dynamically allocated by the simulator.
  struct RegisterMappingTracker {
    // The total number of physical registers that are available in this
    // register file for register renaming purpouses.  A value of zero for this
    // field means: this register file has an unbounded number of physical
    // registers.
    const unsigned NumPhysRegs;
    // Number of physical registers that are currently in use.
    unsigned NumUsedPhysRegs;

    RegisterMappingTracker(unsigned NumPhysRegisters)
        : NumPhysRegs(NumPhysRegisters), NumUsedPhysRegs(0) {}
  };

  // A vector of register file descriptors.  This set always contains at least
  // one entry. Entry at index #0 is reserved.  That entry describes a register
  // file with an unbounded number of physical registers that "sees" all the
  // hardware registers declared by the target (i.e. all the register
  // definitions in the target specific `XYZRegisterInfo.td` - where `XYZ` is
  // the target name).
  // 
  // Users can limit the number of physical registers that are available in
  // regsiter file #0 specifying command line flag `-register-file-size=<uint>`.
  llvm::SmallVector<RegisterMappingTracker, 4> RegisterFiles;

  // This pair is used to identify the owner of a register, as well as
  // the "register cost". Register cost is defined as the number of physical
  // registers required to allocate a user register.
  // For example: on X86 BtVer2, a YMM register consumes 2 128-bit physical
  // registers. So, the cost of allocating a YMM register in BtVer2 is 2.
  using IndexPlusCostPairTy = std::pair<unsigned, unsigned>;

  // RegisterMapping objects are mainly used to track physical register
  // definitions. There is a RegisterMapping for every register defined by the
  // Target. For each register, a RegisterMapping pair contains a descriptor of
  // the last register write (in the form of a WriteState object), as well as a
  // IndexPlusCostPairTy to quickly identify owning register files.
  //
  // This implementation does not allow overlapping register files. The only
  // register file that is allowed to overlap with other register files is
  // register file #0. If we exclude register #0, every register is "owned" by
  // at most one register file.
  using RegisterMapping = std::pair<WriteState *, IndexPlusCostPairTy>;

  // This map contains one entry for each register defined by the target.
  std::vector<RegisterMapping> RegisterMappings;

  // This method creates a new register file descriptor.
  // The new register file owns all of the registers declared by register
  // classes in the 'RegisterClasses' set.
  //
  // Processor models allow the definition of RegisterFile(s) via tablegen. For
  // example, this is a tablegen definition for a x86 register file for
  // XMM[0-15] and YMM[0-15], that allows up to 60 renames (each rename costs 1
  // physical register).
  //
  //    def FPRegisterFile : RegisterFile<60, [VR128RegClass, VR256RegClass]>
  //
  // Here FPRegisterFile contains all the registers defined by register class
  // VR128RegClass and VR256RegClass. FPRegisterFile implements 60
  // registers which can be used for register renaming purpose.
  void
  addRegisterFile(llvm::ArrayRef<llvm::MCRegisterCostEntry> RegisterClasses,
                  unsigned NumPhysRegs);

  // Consumes physical registers in each register file specified by the
  // `IndexPlusCostPairTy`. This method is called from `addRegisterMapping()`.
  void allocatePhysRegs(IndexPlusCostPairTy IPC,
                        llvm::MutableArrayRef<unsigned> UsedPhysRegs);

  // Releases previously allocated physical registers from the register file(s)
  // referenced by the IndexPlusCostPairTy object. This method is called from
  // `invalidateRegisterMapping()`.
  void freePhysRegs(IndexPlusCostPairTy IPC,
                    llvm::MutableArrayRef<unsigned> FreedPhysRegs);

  // Create an instance of RegisterMappingTracker for every register file
  // specified by the processor model.
  // If no register file is specified, then this method creates a default
  // register file with an unbounded number of physical registers.
  void initialize(const llvm::MCSchedModel &SM, unsigned NumRegs);

public:
  RegisterFile(const llvm::MCSchedModel &SM, const llvm::MCRegisterInfo &mri,
               unsigned NumRegs = 0)
      : MRI(mri), RegisterMappings(mri.getNumRegs(), {nullptr, {0, 0}}) {
    initialize(SM, NumRegs);
  }

  // This method updates the register mappings inserting a new register
  // definition. This method is also responsible for updating the number of
  // allocated physical registers in each register file modified by the write.
  // No physical regiser is allocated when flag ShouldAllocatePhysRegs is set.
  void addRegisterWrite(WriteState &WS,
                        llvm::MutableArrayRef<unsigned> UsedPhysRegs,
                        bool ShouldAllocatePhysRegs = true);

  // Removes write \param WS from the register mappings.
  // Physical registers may be released to reflect this update.
  void removeRegisterWrite(const WriteState &WS,
                           llvm::MutableArrayRef<unsigned> FreedPhysRegs,
                           bool ShouldFreePhysRegs = true);

  // Checks if there are enough physical registers in the register files.
  // Returns a "response mask" where each bit represents the response from a
  // different register file.  A mask of all zeroes means that all register
  // files are available.  Otherwise, the mask can be used to identify which
  // register file was busy.  This sematic allows us classify dispatch dispatch
  // stalls caused by the lack of register file resources.
  unsigned isAvailable(llvm::ArrayRef<unsigned> Regs) const;
  void collectWrites(llvm::SmallVectorImpl<WriteState *> &Writes,
                     unsigned RegID) const;
  void updateOnRead(ReadState &RS, unsigned RegID);

  unsigned getNumRegisterFiles() const { return RegisterFiles.size(); }

#ifndef NDEBUG
  void dump() const;
#endif
};

} // namespace mca

#endif // LLVM_TOOLS_LLVM_MCA_REGISTER_FILE_H
