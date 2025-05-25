#ifndef PEMCTARGETDESC_H
#define PEMCTARGETDESC_H

//获取codegen生成文件，方便其他文件引用

#define GET_REGINFO_ENUM
#include "PEGenRegisterInfo.inc"

#define GET_INSTRINFO_ENUM
#include "PEGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "PEGenSubtargetInfo.inc"

#endif //PEMCTARGETDESC_H