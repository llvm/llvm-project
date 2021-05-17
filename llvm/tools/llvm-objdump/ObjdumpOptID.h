#ifndef LLVM_TOOLS_LLVM_OBJDUMP_OBJDUMP_OPT_ID_H
#define LLVM_TOOLS_LLVM_OBJDUMP_OBJDUMP_OPT_ID_H

enum ObjdumpOptID {
  OBJDUMP_INVALID = 0, // This is not an option ID.
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  OBJDUMP_##ID,
#include "ObjdumpOpts.inc"
#undef OPTION
};

#endif // LLVM_TOOLS_LLVM_OBJDUMP_OBJDUMP_OPT_ID_H
