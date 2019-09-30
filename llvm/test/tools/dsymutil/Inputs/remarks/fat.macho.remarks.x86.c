/* This file was used to generate:
 * - fat.macho.remarks.x86.o
 * - fat.macho.remarks.x86.opt.bitstream
 * - fat.macho.remarks.x86
 */

/* for ARCH in x86_64 x86_64h i386
 * do
 *   clang -gline-tables-only -c fat.macho.remarks.x86.c -fsave-optimization-record=bitstream -foptimization-record-file=fat.macho.remarks."$ARCH".opt.bitstream -mllvm -remarks-section -arch "$ARCH"
 * done
 * lipo -create -output fat.macho.remarks.x86.o fat.macho.remarks.x86_64.o fat.macho.remarks.x86_64h.o fat.macho.remarks.i386.o
 * clang -gline-tables-only fat.macho.remarks.x86.o -arch x86_64 -arch x86_64h -arch i386 -o fat.macho.remarks.x86
 */
int main(void) { return 0; }
