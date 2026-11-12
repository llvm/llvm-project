/* Compile with:
   clang -arch armv7 -arch armv7m -arch armv7em -arch x86_64 -arch x86_64h -c
*/

#ifdef __x86_64h__
void x86_64h_function() {}
#elif defined(__x86_64__)
void x86_64_function() {}
#elif defined(__ARM_ARCH_7EM__)
void armv7em_function() {}
#elif defined(__ARM_ARCH_7M__)
void armv7m_function() {}
#elif defined(__ARM_ARCH_7A__)
void armv7_function() {}
#endif
