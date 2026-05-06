#ifndef REPLXX_UTF8STRING_HXX_INCLUDED
#define REPLXX_UTF8STRING_HXX_INCLUDED

#include <memory>

#include "unicodestring.hxx"

namespace replxx {

class Utf8String {
private:
	typedef std::unique_ptr<char[]> buffer_t;
	buffer_t _data;
	int _bufSize;
	int _len;
public:
	Utf8String( void )
		: _data()
		, _bufSize( 0 )
		, _len( 0 ) {
	}
	explicit Utf8String( UnicodeString const& src )
		: _data()
		, _bufSize( 0 )
		, _len( 0 ) {
		assign( src, src.length() );
	}

	Utf8String( UnicodeString const& src_, int len_ )
		: _data()
		, _bufSize( 0 )
		, _len( 0 ) {
		assign( src_, len_ );
	}

	void assign( UnicodeString const& str_ ) {
		assign( str_, str_.length() );
	}

	void assign( UnicodeString const& str_, int len_ ) {
		assign( str_.get(), len_ );
	}

	void assign( char32_t const* str_, int len_ ) {
		int len( len_ * 4 );
		realloc( len );
		_len = copyString32to8( _data.get(), len, str_, len_ );
	}

	void assign( std::string const& str_ ) {
		realloc( static_cast<int>( str_.length() ) );
		strncpy( _data.get(), str_.c_str(), str_.length() );
		_len = static_cast<int>( str_.length() );
	}

	void assign( Utf8String const& other_ ) {
		realloc( other_._len );
		strncpy( _data.get(), other_._data.get(), other_._len );
		_len = other_._len;
	}

	char const* get() const {
		return _data.get();
	}

	int size( void ) const {
		return ( _len );
	}

	bool operator != ( Utf8String const& other_ ) {
		return (
			( other_._len != _len )
			|| (
				(  _len != 0 )
				&& ( memcmp( other_._data.get(), _data.get(), _len ) != 0 )
			)
		);
	}

private:
	void realloc( int reqLen ) {
		if ( ( reqLen + 1 ) > _bufSize ) {
			_bufSize = 1;
			while ( ( reqLen + 1 ) > _bufSize ) {
				_bufSize *= 2;
			}
			_data.reset( new char[_bufSize] );
			memset( _data.get(), 0, _bufSize );
		}
		_data[reqLen] = 0;
		return;
	}
	Utf8String(const Utf8String&) = delete;
	Utf8String& operator=(const Utf8String&) = delete;
};

}

#endif

