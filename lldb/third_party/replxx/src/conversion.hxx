#ifndef REPLXX_CONVERSION_HXX_INCLUDED
#define REPLXX_CONVERSION_HXX_INCLUDED 1

#include "ConvertUTF.h"

#ifdef __has_include
#if __has_include( <version> )
#include <version>
#endif
#endif

#if ! ( defined( __cpp_lib_char8_t ) || ( defined( __clang_major__ ) && ( __clang_major__ >= 8 ) && ( __cplusplus > 201703L ) ) )
namespace replxx {
typedef unsigned char char8_t;
}
#endif

namespace replxx {

ConversionResult copyString8to32( char32_t* dst, int dstSize, int& dstCount, char const* src );
ConversionResult copyString8to32( char32_t* dst, int dstSize, int& dstCount, char8_t const* src );
int copyString32to8( char* dst, int dstSize, char32_t const* src, int srcSize );

namespace locale {
extern bool is8BitEncoding;
}

}

#endif
