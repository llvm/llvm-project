test/Regression/Archive
=======================

This directory contains various tests of llvm-ar and llvm-ranlib to ensure 
compatibility reading other ar(1) formats. It also provides a basic
functionality test for these tools.

There are four archives stored in CVS with these tests: 

GNU.a    - constructed on Linux with GNU ar
MacOSX.a - constructed on Mac OS X with its native BSD4.4 ar
SVR4.a   - constructed on Solaris with /usr/ccs/bin/ar
xpg4.a   - constructed on Solaris with /usr/xpg4/bin/ar

Each type of test is run on each of these archive files.  These archives each 
contain four members:

oddlen - a member with an odd lengthed name and content
evenlen - a member with an even lengthed name and content
IsNAN.o - a Linux native binary
very_long_bytecode_file_name.bc - LLVM bytecode file with really long name

These files test different aspects of the archiver that should cause failures
in llvm-ar if regressions are introduced.
