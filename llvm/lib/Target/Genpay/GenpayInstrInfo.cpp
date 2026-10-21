#include "GenpayInstrInfo.h"

using namespace llvm;

// Provide the definitions of the tablegen-generated base class ctor/dtor.
#define GET_INSTRINFO_CTOR_DTOR
#include "GenpayGenInstrInfo.inc"

GenpayInstrInfo::GenpayInstrInfo() = default;
