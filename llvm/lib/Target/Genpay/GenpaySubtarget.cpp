#include "GenpaySubtarget.h"

using namespace llvm;

// Construct with empty feature/scheduling tables: a single "generic" subtarget
// with the generic scheduling model. (When you add subtarget features later,
// switch to a tablegen-generated GenpayGenSubtargetInfo.)
GenpaySubtarget::GenpaySubtarget(const Triple &TT, StringRef CPU, StringRef FS,
                                 const TargetMachine &TM,
                                 const TargetOptions &Options,
                                 CodeModel::Model CM, CodeGenOptLevel OL)
    : TargetSubtargetInfo(TT, CPU, /*TuneCPU=*/CPU, FS,
                          /*PN=*/ArrayRef<StringRef>(),
                          /*PF=*/ArrayRef<SubtargetFeatureKV>(),
                          /*PD=*/ArrayRef<SubtargetSubTypeKV>(),
                          /*WPR=*/nullptr, /*WL=*/nullptr, /*RA=*/nullptr,
                          /*IS=*/nullptr, /*OC=*/nullptr, /*FP=*/nullptr),
      InstrInfo(), RegInfo(), TLInfo(TM, *this), FrameLowering() {}
