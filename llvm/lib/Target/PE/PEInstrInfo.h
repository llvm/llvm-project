/* --- PEInstrInfo.h --- */

/* ------------------------------------------
Author: 高宇翔
Date: 4/1/2025
------------------------------------------ */

#ifndef PEINSTRINFO_H
#define PEINSTRINFO_H

#include "llvm/CodeGen/TargetInstrInfo.h"

//使用CodeGen生成的源文件
#define GET_INSTRINFO_HEADER
#include "PEGenInstrInfo.inc"
namespace llvm{
class PEInstrInfo : public PEGenInstrInfo{
public:

    explicit PEInstrInfo();
    ~PEInstrInfo();

private:

};
}

#endif // PEINSTRINFO_H
