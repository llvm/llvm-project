#define M1 Value1
#include "dwarfdump-macro.h"
#define M2(x, y)   ((x)+(y)* Value2)

// Built with GCC
// $ mkdir -p /tmp/dbginfo
// $ cp dwarfdump-macro.cc /tmp/dbginfo
// $ cp dwarfdump-macro.h /tmp/dbginfo
// $ cp dwarfdump-macro-cmd.h /tmp/dbginfo
// $ cd /tmp/dbginfo
// $ g++ -c -g3 -O0 -DM3=Value3 -include dwarfdump-macro-cmd.h dwarfdump-macro.cc -o <output>
