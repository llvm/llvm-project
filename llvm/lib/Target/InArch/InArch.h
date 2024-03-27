#ifndef LLVM_LIB_TARGET_InArch_InArch_H
#define LLVM_LIB_TARGET_InArch_InArch_H

#include "llvm/Support/raw_ostream.h"

#define INARCH_DUMP(Color)                                                        \
  {                                                                            \
    llvm::errs().changeColor(Color)                                            \
        << __func__ << "\n\t\t" << __FILE__ << ":" << __LINE__ << "\n";        \
    llvm::errs().changeColor(llvm::raw_ostream::WHITE);                        \
  }
// #define INARCH_DUMP(Color) {}

#define INARCH_DUMP_RED INARCH_DUMP(llvm::raw_ostream::RED)
#define INARCH_DUMP_GREEN INARCH_DUMP(llvm::raw_ostream::GREEN)
#define INARCH_DUMP_YELLOW INARCH_DUMP(llvm::raw_ostream::YELLOW)
#define INARCH_DUMP_CYAN INARCH_DUMP(llvm::raw_ostream::CYAN)
#define INARCH_DUMP_MAGENTA INARCH_DUMP(llvm::raw_ostream::MAGENTA)

#endif // LLVM_LIB_TARGET_InArch_InArch_H