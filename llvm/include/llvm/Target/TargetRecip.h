//===--------------------- llvm/Target/TargetRecip.h ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class is used to customize machine-specific reciprocal estimate code
// generation in a target-independent way.
// If a target does not support operations in this specification, then code
// generation will default to using supported operations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETRECIP_H
#define LLVM_TARGET_TARGETRECIP_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace llvm {

class StringRef;

struct TargetRecip {
public:
  TargetRecip();

  /// Parse a comma-separated string of reciprocal settings to set values in
  /// this struct.
  void set(StringRef &Args);

  /// Set enablement and refinement steps for a particular reciprocal operation.
  /// Use "all" to give all operations the same values.
  void set(StringRef Key, bool Enable, unsigned RefSteps);

  /// Return true if the reciprocal operation has been enabled.
  bool isEnabled(StringRef Key) const;

  /// Return the number of iterations necessary to refine the
  /// the result of a machine instruction for the given reciprocal operation.
  unsigned getRefinementSteps(StringRef Key) const;

  bool operator==(const TargetRecip &Other) const;

private:
  // TODO: We should be able to use special values (enums) to simplify this into
  // just an int, but we have to be careful because the user is allowed to
  // specify "default" as a setting and just change the refinement step count.
  struct RecipParams {
    bool Enabled;
    int8_t RefinementSteps;

    RecipParams() : Enabled(false), RefinementSteps(0) {}
  };

  std::map<StringRef, RecipParams> RecipMap;
  typedef std::map<StringRef, RecipParams>::iterator RecipIter;
  typedef std::map<StringRef, RecipParams>::const_iterator ConstRecipIter;

  bool parseGlobalParams(const std::string &Arg);
  void parseIndividualParams(const std::vector<std::string> &Args);
};

} // end namespace llvm

#endif // LLVM_TARGET_TARGETRECIP_H
