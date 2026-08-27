// A minimal shared library for NativeDylibManager tests.

#if defined(_WIN32)
#define TEST_EXPORT __declspec(dllexport)
#else
#define TEST_EXPORT __attribute__((visibility("default")))
#endif

extern "C" TEST_EXPORT int NativeDylibManagerTestFunc() { return 42; }
extern "C" TEST_EXPORT int NativeDylibManagerTestFunc2() { return 7; }
