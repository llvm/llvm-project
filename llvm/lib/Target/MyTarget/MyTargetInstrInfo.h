// MyTargetInstrInfo.h
#ifndef LLVM_LIB_TARGET_MyTarget_INSTRINFO_H
#define LLVM_LIB_TARGET_MyTarget_INSTRINFO_H

#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "MyTargetGenInstrInfo.inc"   // ← 這就是由 HelloWorld.td 生成的

#endif
