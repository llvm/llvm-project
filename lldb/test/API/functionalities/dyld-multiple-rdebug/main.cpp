#include "library_file.h"
#include <link.h>
#include <stdio.h>
// Make a duplicate "_r_debug" symbol that is visible. This is the global
// variable name that the dynamic loader uses to communicate changes in shared
// libraries that get loaded and unloaded. LLDB finds the address of this
// variable by reading the DT_DEBUG entry from the .dynamic section of the main
// executable.
// What will happen is the dynamic loader will use the "_r_debug" symbol from
// itself until the a.out executable gets loaded. At this point the new
// "_r_debug" symbol will take precedence over the orignal "_r_debug" symbol
// from the dynamic loader and the copy below will get updated with shared
// library state changes while the version that LLDB checks in the dynamic
// loader stays the same for ever after this.
//
// When our DYLDRendezvous.cpp tries to check the state in the _r_debug
// structure, it will continue to get the last eAdd as the state before the
// switch in symbol resolution.
//
// Before a fix in LLDB, this would mean that we wouldn't ever load any shared
// libraries since DYLDRendezvous was waiting to see a eAdd state followed by a
// eConsistent state which would trigger the adding of shared libraries, but we
// would never see this change because the local copy below is actually what
// would get updated. Now if DYLDRendezvous detects two eAdd states in a row,
// it will load the shared libraries instead of doing nothing and a log message
// will be printed out if "log enable lldb dyld" is active.
r_debug _r_debug;

int main() {
  library_function(); // Break here
  return 0;
}
