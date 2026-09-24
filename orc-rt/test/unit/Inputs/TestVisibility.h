#ifndef ORC_RT_TEST_UNIT_INPUTS_TESTVISIBILITY_H
#define ORC_RT_TEST_UNIT_INPUTS_TESTVISIBILITY_H

#if defined(_WIN32)
#if defined(ORC_RT_TEST_DLL_EXPORTS)
#define TEST_EXPORT __declspec(dllexport)
#else
#define TEST_EXPORT __declspec(dllimport)
#endif
#else
#define TEST_EXPORT __attribute__((visibility("default")))
#endif

#endif // ORC_RT_TEST_UNIT_INPUTS_TESTVISIBILITY_H
