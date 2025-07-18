#ifndef MATHTEST_TYPEEXTRAS_HPP
#define MATHTEST_TYPEEXTRAS_HPP

namespace mathtest {

#ifdef __FLT16_MAX__
#define MATHTEST_HAS_FLOAT16
typedef _Float16 float16;
#endif // __FLT16_MAX__
} // namespace mathtest

#endif // MATHTEST_TYPEEXTRAS_HPP
