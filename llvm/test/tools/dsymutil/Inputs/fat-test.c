/* Compile with:
   clang -c -g -arch x86_64h -arch x86_64 -arch i386 fat-test.c
   libtool -static -o libfat-test.a fat-test.o

   to generate a dylib instead:
   clang -arch ...  -arch ... -arch ...  -dynamiclib fat-test.o -o fat-test.dylib

   To reduce the size of the fat binary:
   lipo -thin i386 -o fat-test.i386.o fat-test.o 
   lipo -thin x86_64 -o fat-test.x86_64.o fat-test.o 
   lipo -thin x86_64h -o fat-test.x86_64h.o fat-test.o 
   lipo -create -arch x86_64h fat-test.x86_64h.o -arch x86_64 fat-test.x86_64.o -arch i386 fat-test.i386.o -o fat-test.o -segalign i386 8 -segalign x86_64 8 -segalign x86_64h 8
 */
#ifdef __x86_64h__
int x86_64h_var;
#elif defined(__x86_64__)
int x86_64_var;
#elif defined(__i386__)
int i386_var;
#elif defined(__ARM_ARCH_7S__)
int armv7s_var;
#elif defined(__ARM_ARCH_7A__)
int armv7_var;
#elif defined(__ARM64_ARCH_8__)
int arm64_var;
#else
#error "Unknown architecture"
#endif
