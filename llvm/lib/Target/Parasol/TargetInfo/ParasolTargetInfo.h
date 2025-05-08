//===---- ParasolTargetInfo.h - Parasol Target Implementation -------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PARASOL_TARGETINFO_PARASOLTARGETINFO_H
#define LLVM_LIB_TARGET_PARASOL_TARGETINFO_PARASOLTARGETINFO_H

namespace llvm {

class Target;

Target &getTheParasolTarget();

} // namespace llvm

#endif // LLVM_LIB_TARGET_PARASOL_TARGETINFO_PARASOLTARGETINFO_H
