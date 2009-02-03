//===-- llvm/lib/Debugger/ - LLVM Debugger interfaces ---------------------===//

This directory contains the implementation of the LLVM debugger backend.  This
directory builds into a library which can be used by various debugger 
front-ends to debug LLVM programs.  The current command line LLVM debugger, 
llvm-db is currently the only client of this library, but others could be 
built, to provide a GUI front-end for example.

