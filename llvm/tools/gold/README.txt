This directory contains a plugin that is designed to work with binutils
gold linker. At present time, this is not the default linker in
binutils, and the default build of gold does not support plugins.

Obtaining binutils:

  cvs -z 9 -d :pserver:anoncvs@sourceware.org:/cvs/src login
  {enter "anoncvs" as the password}
  cvs -z 9 -d :pserver:anoncvs@sourceware.org:/cvs/src co binutils

This will create a src/ directory. Make a build/ directory and from
there configure binutils with "../src/configure --enable-gold --enable-plugins".
Then build binutils with "make all-gold".

To build the LLVMgold plugin, configure LLVM with the option
--with-binutils-include=/path/to/binutils/src/include/ --enable-pic. To use the
plugin, run "ld-new --plugin /path/to/LLVMgold.so".
Without PIC libLTO and LLVMgold are not being built (because they would fail
link on x86-64 with a relocation error: PIC and non-PIC can't be combined).
As an alternative to passing --enable-pic, you can use 'make ENABLE_PIC=1' in
your entire LLVM build.
