#include <algorithm>
#include <string>
#include <cstring>
#include <cctype>
#include <locale.h>

#include "conversion.hxx"

#ifdef _WIN32
#define strdup _strdup
#endif

using namespace std;

namespace replxx {

namespace locale {

void to_lower( std::string& s_ ) {
	transform( s_.begin(), s_.end(), s_.begin(), static_cast<int(*)(int)>( &tolower ) );
}

bool is_8bit_encoding( void ) {
	bool is8BitEncoding( false );
	string origLC( setlocale( LC_CTYPE, nullptr ) );
	string lc( origLC );
	to_lower( lc );
	if ( lc == "c" ) {
		setlocale( LC_CTYPE, "" );
	}
	lc = setlocale( LC_CTYPE, nullptr );
	setlocale( LC_CTYPE, origLC.c_str() );
	to_lower( lc );
	if ( lc.find( "8859" ) != std::string::npos ) {
		is8BitEncoding = true;
	}
	return ( is8BitEncoding );
}

bool is8BitEncoding( is_8bit_encoding() );

}

ConversionResult copyString8to32(char32_t* dst, int dstSize, int& dstCount, const char* src) {
	ConversionResult res = ConversionResult::conversionOK;
	if ( ! locale::is8BitEncoding ) {
		const UTF8* sourceStart = reinterpret_cast<const UTF8*>(src);
		const UTF8* sourceEnd = sourceStart + strlen(src);
		UTF32* targetStart = reinterpret_cast<UTF32*>(dst);
		UTF32* targetEnd = targetStart + dstSize;

		res = ConvertUTF8toUTF32(
				&sourceStart, sourceEnd, &targetStart, targetEnd, lenientConversion);

		if (res == conversionOK) {
			dstCount = static_cast<int>( targetStart - reinterpret_cast<UTF32*>( dst ) );

			if (dstCount < dstSize) {
				*targetStart = 0;
			}
		}
	} else {
		for ( dstCount = 0; ( dstCount < dstSize ) && src[dstCount]; ++ dstCount ) {
			dst[dstCount] = src[dstCount];
		}
	}
	return res;
}

ConversionResult copyString8to32(char32_t* dst, int dstSize, int& dstCount, const char8_t* src) {
	return copyString8to32(
		dst, dstSize, dstCount, reinterpret_cast<const char*>(src)
	);
}

int copyString32to8( char* dst, int dstSize, const char32_t* src, int srcSize ) {
	int resCount( 0 );
	if ( ! locale::is8BitEncoding ) {
		const UTF32* sourceStart = reinterpret_cast<const UTF32*>(src);
		const UTF32* sourceEnd = sourceStart + srcSize;
		UTF8* targetStart = reinterpret_cast<UTF8*>(dst);
		UTF8* targetEnd = targetStart + dstSize;

		ConversionResult res = ConvertUTF32toUTF8(
			&sourceStart, sourceEnd, &targetStart, targetEnd, lenientConversion
		);

		if ( res == conversionOK ) {
			resCount = static_cast<int>( targetStart - reinterpret_cast<UTF8*>( dst ) );
			if ( resCount < dstSize ) {
				*targetStart = 0;
			}
		}
	} else {
		int i( 0 );
		for ( i = 0; ( i < dstSize ) && ( i < srcSize ) && src[i]; ++ i ) {
			dst[i] = static_cast<char>( src[i] );
		}
		resCount = i;
		if ( i < dstSize ) {
			dst[i] = 0;
		}
	}
	return ( resCount );
}

}

