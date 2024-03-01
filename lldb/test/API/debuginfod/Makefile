C_SOURCES := main.c
LDFLAGS := -Wl,--build-id

MAKE_DWO := YES
SPLIT_DEBUG_SYMBOLS := YES
KEEP_FULL_DEBUG_BINARY := YES
MERGE_DWOS := YES

# From shell, I can do this stuff pretty easily:
# OBJCOPY --dump-section .note.gnu.build-id=<output.36.bytes.bin> <input.bin>
# hexdump -s 16 -e '"%02x"' output.bin
# Or this in a single command:
# objcopy --dump-section=.note.gnu.build-id=/dev/stdout adfinder.stripped | xxd -s 16 -g 0 -p
# I think I need to add capabilities in the Makefile.rules to strip the binary (not an -OKD binary)
# and to create a DWP file

include Makefile.rules


#!/bin/sh

# Testing 5 different scenarios:
# 1 - A stripped binary with it's corresponding unstripped binary:
# 2 - A stripped binary with a corresponding --only-keep-debug symbols file
# 3 - A split binary with it's corresponding DWP file
# 4 - A stripped, split binary with an unstripped binary and a DWP file
# 5 - A stripped, split binary with an --only-keep-debug symbols file and a DWP file

##mkdir -p gen
##mkdir -p run

# First, compile & link the binaries (normal and split)

##${builddir}/bin/clang -g -o gen/bin-normal.o -O0 -c main.c
##ld -nostdlib gen/bin-normal.o --build-id=sha1 -o gen/bin-normal
##${builddir}/bin/clang -g -gsplit-dwarf -o gen/bin-split.o -O0 -c main.c
##ld -nostdlib gen/bin-split.o --build-id=sha1 -o gen/bin-split

# Next, create the file variations we need

# Variation 1: -g, stripped
##${builddir}/bin/llvm-objcopy --strip-debug gen/bin-normal gen/bin-stripped
# Variation 2: -g, stripped, --only-keep-debug symbols
##${builddir}/bin/llvm-objcopy --only-keep-debug gen/bin-normal gen/sym-stripped
# Variation 3: -gsplit-dwarf: .dwp
##${builddir}/bin/llvm-dwp -e gen/bin-split -o gen/bin-split.dwp
# Variation 4: -gsplit-dwarf: stripped, .dwp
##${builddir}/bin/llvm-objcopy --strip-debug gen/bin-split gen/bin-split-stripped
# Variation 5: -gsplit-dwarf: stripped, --only-keep-debug + .dwp
##${builddir}/bin/llvm-objcopy --only-keep-debug gen/bin-split gen/sym-split

# Finally, produce the .yaml files for testing

# Scenario 1:
# target:  bin-stripped
# Scenario 1a:
# symbols: bin-normal (hosted as debuginfo)
# Scenario 1b:
# symbols: bin-normal (hosted as executable)
##${builddir}/bin/obj2yaml gen/bin-stripped -o bin-stripped.yaml
##${builddir}/bin/obj2yaml gen/bin-normal -o bin-normal.yaml
# @ testing time: yaml2obj bin-stripped.yaml -o ${out}/bin-stripped
# @ testing time: yaml2obj bin-normal.yaml -o ${out}/bin-normal


# Scenario 2:
# target:  bin-stripped-okd
# Scenario 2a:
# symbols: sym-stripped (hosted as debuginfo)
# Scenario 2b:
# symbols: sym-stripped (hosted as executable)
##${builddir}/bin/obj2yaml gen/sym-stripped -o sym-stripped.yaml
# To produce a correct .gnu.debuglink, you have to do it at test generation time.
# The section includes a CRC that yaml2obj doesn't properly produce.
# @ testing time: yaml2obj sym-stripped.yaml -o ${out}/sym-stripped
# @ testing time: llvm-objcopy bin-stripped --add-gnu-debuglink=${out}/sym-stripped ${out}/bin-stripped-okd

# Scenario 3:
# target: bin-split
# DWP:    bin-split.dwp (hosted as debuginfo)
##${builddir}/bin/obj2yaml gen/bin-split -o bin-split.yaml
##${builddir}/bin/obj2yaml gen/bin-split.dwp -o bin-split-dwp.yaml
# @ testing time: yaml2obj bin-split.yaml -o ${out}/bin-split
# @ testing time: yaml2obj bin-split-dwp.yaml -o ${out}/bin-split.dwp

# Scenario 4:
# target:  bin-split-stripped
# symbols: bin-split (hosted as executable)
# DWP      bin-split.dwp (hosted as debuginfo)
# This doesn't work from a file system "as is".
# I believe you can set the symbol file manually to the bin-split file.
# TODO: Need to check for the -no-locator test to see what the name of
# the .dwp is expected to be.
##${builddir}/bin/obj2yaml gen/bin-split-stripped -o bin-split-stripped.yaml
# bin-split and bin-split.dwp already generated in Scenario 3
# @ testing time: yaml2obj bin-split-stripped.yaml -o ${out}/bin-split-stripped

# Scenario 5:
# target:  bin-split-stripped-okd
# symbols: sym-split (hosted as executable)
# DWP:     bin-split.dwp (hosted as debuginfo)
##${builddir}/bin/obj2yaml gen/sym-split -o sym-split.yaml
# @ testing time: yaml2obj main-split-nodbg.yaml -o gen/main-split-nodbg.tmp
# @ testing time: yaml2obj main-split-dbg.yaml -o run/main-split-dbg
# @ testing time: llvm-objcopy gen/main-split-nodbg.tmp --add-gnu-debuglink=run/main-split-dbg run/main-split-nodbg
