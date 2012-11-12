# Compile with:
#   ARGS="-shared -nostdlib -Wl,--version-script=elfver.script"
#   clang $ARGS -m32 elfver.S -lc -o elf-versioning-test.i386
#   clang $ARGS -m64 elfver.S -lc -o elf-versioning-test.x86_64

# Also, strip off non-dynamic symbols:
#   strip elf-versioning-test.i386
#   strip elf-versioning-test.x86_64

#ifdef __i386__
.symver _puts, puts@GLIBC_2.0
#else
.symver _puts, puts@GLIBC_2.2.5
#endif
call _puts@PLT

.symver foo1, foo@VER1
.globl foo1
.type foo1, @function
foo1:
  ret

.symver foo2, foo@@VER2
.globl foo2
.type foo2, @function
foo2:
  ret

.globl unversioned_define
.type unversioned_define, @function
unversioned_define:
