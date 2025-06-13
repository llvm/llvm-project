// This is a truncated version of an SB API file
// used to test framework-header-fix.py to make sure the includes are correctly fixed
// up for the LLDB.framework.

// Local includes must be changed to framework level includes.
// e.g. #include "lldb/API/SBDefines.h" -> #include <LLDB/SBDefines.h>
#include "lldb/API/SBDefines.h"
#include "lldb/API/SBModule.h"

// Any include guards specified at the command line must be removed.
#ifndef SWIG
int a = 10
#endif
