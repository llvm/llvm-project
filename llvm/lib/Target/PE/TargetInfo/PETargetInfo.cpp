#include "TargetInfo/PETargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/Triple.h"
using namespace llvm;

namespace llvm {
  static Target PETarget;

Target &getPETarget() {
  return PETarget;
}

} // end namespace llvm

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializePETargetInfo() {
  RegisterTarget<Triple::pe,/*HasJIT=*/true> X(
    getPETarget(),  // 必须返回 Target 实例
    "pe",           // 目标短名称（命令行参数名）
    "PE Architecture (32-bit)", // 描述
    "PE"            // 父组件名（可选，与 CMake 中的 ADD_TO_COMPONENT 对应）
  );
}