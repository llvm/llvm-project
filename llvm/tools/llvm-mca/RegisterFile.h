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

/// Manages hardware register files, and tracks data dependencies
/// between registers.
class RegisterFile {
  const llvm::MCRegisterInfo &MRI;

  // Each register file is described by an instance of RegisterMappingTracker.
  // RegisterMappingTracker tracks the number of register mappings dynamically
  // allocated during the execution.
  struct RegisterMappingTracker {
    // Total number of register mappings that are available for register
    // renaming. A value of zero for this field means: this register file has
    // an unbounded number of registers.
    const unsigned TotalMappings;
    // Number of mappings that are currently in use.
    unsigned NumUsedMappings;

    RegisterMappingTracker(unsigned NumMappings)
        : TotalMappings(NumMappings), NumUsedMappings(0) {}
  };

  // This is where information related to the various register files is kept.
  // This set always contains at least one register file at index #0. That
  // register file "sees" all the physical registers declared by the target, and
  // (by default) it allows an unbounded number of mappings.
  // Users can limit the number of mappings that can be created by register file
  // #0 through the command line flag `-register-file-size`.
  llvm::SmallVector<RegisterMappingTracker, 4> RegisterFiles;

  // This pair is used to identify the owner of a physical register, as well as
  // the cost of using that register file.
  using IndexPlusCostPairTy = std::pair<unsigned, unsigned>;

  // RegisterMapping objects are mainly used to track physical register
  // definitions. A WriteState object describes a register definition, and it is
  // used to track RAW dependencies (see Instruction.h).  A RegisterMapping
  // object also specifies the set of register files.  The mapping between
  // physreg and register files is done using a "register file mask".
  //
  // A register file index identifies a user defined register file.
  // There is one index per RegisterMappingTracker, and index #0 is reserved to
  // the default unified register file.
  //
  // This implementation does not allow overlapping register files. The only
  // register file that is allowed to overlap with other register files is
  // register file #0.
  using RegisterMapping = std::pair<WriteState *, IndexPlusCostPairTy>;

  // This map contains one entry for each physical register defined by the
  // processor scheduling model.
  std::vector<RegisterMapping> RegisterMappings;

  // This method creates a new RegisterMappingTracker for a register file that
  // contains all the physical registers specified by the register classes in
  // the 'RegisterClasses' set.
  //
  // The long term goal is to let scheduling models optionally describe register
  // files via tablegen definitions. This is still a work in progress.
  // For example, here is how a tablegen definition for a x86 FP register file
  // that features AVX might look like:
  //
  //    def FPRegisterFile : RegisterFile<[VR128RegClass, VR256RegClass], 60>
  //
  // Here FPRegisterFile contains all the registers defined by register class
  // VR128RegClass and VR256RegClass. FPRegisterFile implements 60
  // registers which can be used for register renaming purpose.
  //
  // The list of register classes is then converted by the tablegen backend into
  // a list of register class indices. That list, along with the number of
  // available mappings, is then used to create a new RegisterMappingTracker.
  void
  addRegisterFile(llvm::ArrayRef<llvm::MCRegisterCostEntry> RegisterClasses,
                  unsigned NumPhysRegs);

  // Allocates register mappings in register file specified by the
  // IndexPlusCostPairTy object. This method is called from addRegisterMapping.
  void allocatePhysRegs(IndexPlusCostPairTy IPC,
                        llvm::MutableArrayRef<unsigned> UsedPhysRegs);

  // Removes a previously allocated mapping from the register file referenced
  // by the IndexPlusCostPairTy object. This method is called from
  // invalidateRegisterMapping.
  void freePhysRegs(IndexPlusCostPairTy IPC,
                    llvm::MutableArrayRef<unsigned> FreedPhysRegs);

  // Create an instance of RegisterMappingTracker for every register file
  // specified by the processor model.
  // If no register file is specified, then this method creates a single
  // register file with an unbounded number of registers.
  void initialize(const llvm::MCSchedModel &SM, unsigned NumRegs);

public:
  RegisterFile(const llvm::MCSchedModel &SM, const llvm::MCRegisterInfo &mri,
               unsigned NumRegs = 0)
      : MRI(mri), RegisterMappings(mri.getNumRegs(), {nullptr, {0, 0}}) {
    initialize(SM, NumRegs);
  }

  // This method updates the data dependency graph by inserting a new register
  // definition. This method is also responsible for updating the number of used
  // physical registers in the register file(s). The number of physical
  // registers is updated only if flag ShouldAllocatePhysRegs is set.
  void addRegisterWrite(WriteState &WS,
                        llvm::MutableArrayRef<unsigned> UsedPhysRegs,
                        bool ShouldAllocatePhysRegs = true);

  // Updates the data dependency graph by removing a write. It also updates the
  // internal state of the register file(s) by freeing physical registers.
  // The number of physical registers is updated only if flag ShouldFreePhysRegs
  // is set.
  void removeRegisterWrite(const WriteState &WS,
                           llvm::MutableArrayRef<unsigned> FreedPhysRegs,
                           bool ShouldFreePhysRegs = true);

  // Checks if there are enough microarchitectural registers in the register
  // files.  Returns a "response mask" where each bit is the response from a
  // RegisterMappingTracker.
  // For example: if all register files are available, then the response mask
  // is a bitmask of all zeroes. If Instead register file #1 is not available,
  // then the response mask is 0b10.
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
