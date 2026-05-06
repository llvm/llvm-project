#ifndef REPLXX_UNICODESTRING_HXX_INCLUDED
#define REPLXX_UNICODESTRING_HXX_INCLUDED

#include <vector>
#include <cstring>
#include <cwctype>
#include <cassert>

#include "conversion.hxx"

namespace replxx {

inline bool case_sensitive_equal( char32_t l, char32_t r ) {
	return l == r;
}

inline bool case_insensitive_equal( char32_t l, char32_t r ) {
	return towlower( static_cast<wint_t>( l ) ) == towlower( static_cast<wint_t>( r ) );
}

class UnicodeString {
public:
	typedef std::vector<char32_t> data_buffer_t;
	typedef data_buffer_t::const_iterator const_iterator;
	typedef data_buffer_t::iterator iterator;
private:
	data_buffer_t _data;
public:
	UnicodeString()
		: _data() {
	}

	explicit UnicodeString( std::string const& src )
		: _data() {
		assign( src );
	}

	explicit UnicodeString( UnicodeString const& other, int offset, int len = -1 )
		: _data() {
		_data.insert(
			_data.end(),
			other._data.begin() + offset,
			len > 0 ? other._data.begin() + offset + len : other._data.end()
		);
	}

	explicit UnicodeString( char const* src )
		: _data() {
		assign( src );
	}

	explicit UnicodeString( char8_t const* src )
		: UnicodeString( reinterpret_cast<const char*>( src ) ) {
	}

	explicit UnicodeString( char32_t const* src )
		: _data() {
		int len( 0 );
		while ( src[len] != 0 ) {
			++ len;
		}
		_data.assign( src, src + len );
	}

	explicit UnicodeString( char32_t const* src, int len )
		: _data() {
		_data.assign( src, src + len );
	}

	explicit UnicodeString( int len )
		: _data() {
		_data.resize( len );
	}

	UnicodeString& assign( std::string const& str_ ) {
		_data.resize( static_cast<int>( str_.length() ) );
		int len( 0 );
		copyString8to32( _data.data(), static_cast<int>( str_.length() ), len, str_.c_str() );
		_data.resize( len );
		return *this;
	}

	UnicodeString& assign( char const* str_ ) {
		int byteCount( static_cast<int>( strlen( str_ ) ) );
		_data.resize( byteCount );
		int len( 0 );
		copyString8to32( _data.data(), byteCount, len, str_ );
		_data.resize( len );
		return *this;
	}

	UnicodeString& assign( UnicodeString const& other_ ) {
		_data = other_._data;
		return *this;
	}

	explicit UnicodeString( UnicodeString const& ) = default;
	UnicodeString& operator = ( UnicodeString const& ) = default;
	UnicodeString( UnicodeString&& ) = default;
	UnicodeString& operator = ( UnicodeString&& ) = default;
	bool operator == ( UnicodeString const& other_ ) const {
		return ( _data == other_._data );
	}

	bool operator != ( UnicodeString const& other_ ) const {
		return ( _data != other_._data );
	}

	bool operator < ( UnicodeString const& other_ ) const {
		return std::lexicographical_compare(begin(), end(), other_.begin(), other_.end());
	}

	UnicodeString& append( UnicodeString const& other ) {
		_data.insert( _data.end(), other._data.begin(), other._data.end() );
		return *this;
	}

	void push_back( char32_t c_ ) {
		_data.push_back( c_ );
	}

	UnicodeString& append( char32_t const* src, int len ) {
		_data.insert( _data.end(), src, src + len );
		return *this;
	}

	UnicodeString& insert( int pos_, UnicodeString const& str_, int offset_, int len_ ) {
		_data.insert( _data.begin() + pos_, str_._data.begin() + offset_, str_._data.begin() + offset_ + len_ );
		return *this;
	}

	UnicodeString& insert( int pos_, char32_t c_ ) {
		_data.insert( _data.begin() + pos_, c_ );
		return *this;
	}

	UnicodeString& erase( int pos_ ) {
		_data.erase( _data.begin() + pos_ );
		return *this;
	}

	UnicodeString& erase( int pos_, int len_ ) {
		_data.erase( _data.begin() + pos_, _data.begin() + pos_ + len_ );
		return *this;
	}

	char32_t const* get() const {
		return _data.data();
	}

	char32_t* get() {
		return _data.data();
	}

	int length() const {
		return static_cast<int>( _data.size() );
	}

	void clear( void ) {
		_data.clear();
	}

	const char32_t& operator[]( int pos ) const {
		assert( ( pos >= 0 ) && ( pos < static_cast<int>( _data.size() ) ) );
		return _data[pos];
	}

	char32_t& operator[]( int pos ) {
		assert( ( pos >= 0 ) && ( pos < static_cast<int>( _data.size() ) ) );
		return _data[pos];
	}

	bool starts_with( data_buffer_t::const_iterator first_, data_buffer_t::const_iterator last_ ) const {
		return (
			( std::distance( first_, last_ ) <= length() )
			&& ( std::equal( first_, last_, _data.begin() ) )
		);
	}

	template <class BinaryPredicate>
	bool starts_with( data_buffer_t::const_iterator first_, data_buffer_t::const_iterator last_, BinaryPredicate&& pred ) const {
		return (
			( std::distance( first_, last_ ) <= length() )
			&& ( std::equal( first_, last_, _data.begin(), std::forward<BinaryPredicate>( pred ) ) )
		);
	}

	bool ends_with( data_buffer_t::const_iterator first_, data_buffer_t::const_iterator last_ ) const {
		int len( static_cast<int>( std::distance( first_, last_ ) ) );
		return (
			( len <= length() )
			&& ( std::equal( first_, last_, _data.end() - len ) )
		);
	}

	bool is_empty( void ) const {
		return ( _data.size() == 0 );
	}

	void swap( UnicodeString& other_ ) {
		_data.swap( other_._data );
	}

	const_iterator begin( void ) const {
		return ( _data.begin() );
	}

	const_iterator end( void ) const {
		return ( _data.end() );
	}

	iterator begin( void ) {
		return ( _data.begin() );
	}

	iterator end( void ) {
		return ( _data.end() );
	}

	char32_t back( void ) const {
		return ( _data.back() );
	}
};

}

#endif

