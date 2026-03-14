#include "dwarfdump-test4-decl.h"
int c(){a();}

// Built with gcc 4.6.3
// $ mkdir -p /tmp/dbginfo
// $ cp dwarfdump-test4-*.* /tmp/dbginfo
// $ cd /tmp/dbginfo
// $ gcc -fPIC -shared -g dwarfdump-test4-part*.cc -o <output>
