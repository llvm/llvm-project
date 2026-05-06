#include <algorithm>
#include <memory>
#include <fstream>
#include <ostream>
#include <istream>
#include <cstring>

#ifndef _WIN32

#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

#endif /* _WIN32 */

#include "replxx.hxx"
#include "history.hxx"

using namespace std;

namespace replxx {

namespace {
void delete_ReplxxHistoryScanImpl( Replxx::HistoryScanImpl* impl_ ) {
	delete impl_;
}
static int const ETB = 0x17;
}

static int const REPLXX_DEFAULT_HISTORY_MAX_LEN( 1000 );

Replxx::HistoryScan::HistoryScan( impl_t impl_ )
	: _impl( std::move( impl_ ) ) {
}

bool Replxx::HistoryScan::next( void ) {
	return ( _impl->next() );
}

Replxx::HistoryScanImpl::HistoryScanImpl( History::entries_t const& entries_ )
	: _entries( entries_ )
	, _it( _entries.end() )
	, _utf8Cache()
	, _entryCache( std::string(), std::string() )
	, _cacheValid( false ) {
}

Replxx::HistoryEntry const& Replxx::HistoryScan::get( void ) const {
	return ( _impl->get() );
}

bool Replxx::HistoryScanImpl::next( void ) {
	if ( _it == _entries.end() ) {
		_it = _entries.begin();
	} else {
		++ _it;
	}
	_cacheValid = false;
	return ( _it != _entries.end() );
}

Replxx::HistoryEntry const& Replxx::HistoryScanImpl::get( void ) const {
	if ( _cacheValid ) {
		return ( _entryCache );
	}
	_utf8Cache.assign( _it->text() );
	_entryCache = Replxx::HistoryEntry( _it->timestamp(), _utf8Cache.get() );
	_cacheValid = true;
	return ( _entryCache );
}

Replxx::HistoryScan::impl_t History::scan( void ) const {
	return ( Replxx::HistoryScan::impl_t( new Replxx::HistoryScanImpl( _entries ), delete_ReplxxHistoryScanImpl ) );
}

History::History( void )
	: _entries()
	, _maxSize( REPLXX_DEFAULT_HISTORY_MAX_LEN )
	, _current( _entries.begin() )
	, _yankPos( _entries.end() )
	, _previous( _entries.begin() )
	, _recallMostRecent( false )
	, _unique( true ) {
}

void History::add( UnicodeString const& line, std::string const& when ) {
	if ( _maxSize <= 0 ) {
		return;
	}
	if ( ! _entries.empty() && ( line == _entries.back().text() ) ) {
		_entries.back() = Entry( now_ms_str(), line );
		return;
	}
	remove_duplicate( line );
	trim_to_max_size();
	_entries.emplace_back( when, line );
	_locations.insert( make_pair( line, last() ) );
	if ( _current == _entries.end() ) {
		_current = last();
	}
	_yankPos = _entries.end();
}

#ifndef _WIN32
class FileLock {
	std::string _path;
	int _lockFd;
public:
	FileLock( std::string const& name_ )
		: _path( name_ + ".lock" )
		, _lockFd( ::open( _path.c_str(), O_CREAT | O_RDWR, 0600 ) ) {
		static_cast<void>( ::lockf( _lockFd, F_LOCK, 0 ) == 0 );
	}
	~FileLock( void ) {
		static_cast<void>( ::lockf( _lockFd, F_ULOCK, 0 ) == 0 );
		::close( _lockFd );
		::unlink( _path.c_str() );
		return;
	}
};
#endif

bool History::save( std::string const& filename, bool sync_ ) {
#ifndef _WIN32
	mode_t old_umask = umask( S_IXUSR | S_IRWXG | S_IRWXO );
	FileLock fileLock( filename );
#endif
	entries_t entries;
	locations_t locations;
	if ( ! sync_ ) {
		entries.swap( _entries );
		locations.swap( _locations );
		_entries = entries;
		reset_iters();
	}
	/* scope for ifstream object auto-close */ {
		ifstream histFile( filename );
		if ( histFile ) {
			do_load( histFile );
		}
	}
	sort();
	remove_duplicates();
	trim_to_max_size();
	ofstream histFile( filename );
	if ( ! histFile ) {
		return ( false );
	}
#ifndef _WIN32
	umask( old_umask );
	chmod( filename.c_str(), S_IRUSR | S_IWUSR );
#endif
	save( histFile );
	if ( ! sync_ ) {
		_entries = std::move( entries );
		_locations = std::move( locations );
	}
	reset_iters();
	return ( true );
}

void History::save( std::ostream& histFile ) {
	Utf8String utf8;
	UnicodeString us;
	for ( Entry& h : _entries ) {
		h.reset_scratch();
		if ( ! h.text().is_empty() ) {
			us.assign( h.text() );
			std::replace( us.begin(), us.end(), char32_t( '\n' ), char32_t( ETB ) );
			utf8.assign( us );
			histFile << "### " << h.timestamp() << "\n" << utf8.get() << endl;
		}
	}
}

namespace {

bool is_timestamp( std::string const& s ) {
	static char const TIMESTAMP_PATTERN[] = "### dddd-dd-dd dd:dd:dd.ddd";
	static int const TIMESTAMP_LENGTH( sizeof ( TIMESTAMP_PATTERN ) - 1 );
	if ( s.length() != TIMESTAMP_LENGTH ) {
		return ( false );
	}
	for ( int i( 0 ); i < TIMESTAMP_LENGTH; ++ i ) {
		if ( TIMESTAMP_PATTERN[i] == 'd' ) {
			if ( ! isdigit( s[i] ) ) {
				return ( false );
			}
		} else if ( s[i] != TIMESTAMP_PATTERN[i] ) {
			return ( false );
		}
	}
	return ( true );
}

}

void History::do_load( std::istream& histFile ) {
	string line;
	string when( "0000-00-00 00:00:00.000" );
	UnicodeString us;
	while ( getline( histFile, line ).good() ) {
		string::size_type eol( line.find_first_of( "\r\n" ) );
		if ( eol != string::npos ) {
			line.erase( eol );
		}
		if ( is_timestamp( line ) ) {
			when.assign( line, 4, std::string::npos );
			continue;
		}
		if ( ! line.empty() ) {
			us.assign( line );
			std::replace( us.begin(), us.end(), char32_t( ETB ), char32_t( '\n' ) );
			_entries.emplace_back( when, us );
		}
	}
}

bool History::load( std::string const& filename ) {
	ifstream histFile( filename );
	if ( ! histFile ) {
		clear();
		return false;
	}
	load(histFile);
	return true;
}

void History::load( std::istream& histFile ) {
	clear();
	do_load( histFile );
	sort();
	remove_duplicates();
	trim_to_max_size();
	_previous = _current = last();
	_yankPos = _entries.end();
}

void History::sort( void ) {
	typedef std::vector<Entry> sortable_entries_t;
	_locations.clear();
	sortable_entries_t sortableEntries( _entries.begin(), _entries.end() );
	std::stable_sort( sortableEntries.begin(), sortableEntries.end() );
	_entries.clear();
	_entries.insert( _entries.begin(), sortableEntries.begin(), sortableEntries.end() );
}

void History::clear( void ) {
	_locations.clear();
	_entries.clear();
	_current = _entries.begin();
	_recallMostRecent = false;
}

void History::set_max_size( int size_ ) {
	if ( size_ >= 0 ) {
		_maxSize = size_;
		trim_to_max_size();
	}
}

void History::reset_yank_iterator( void ) {
	_yankPos = _entries.end();
}

bool History::next_yank_position( void ) {
	bool resetYankSize( false );
	if ( _yankPos == _entries.end() ) {
		resetYankSize = true;
	}
	if ( ( _yankPos != _entries.begin() ) && ( _yankPos != _entries.end() ) ) {
		-- _yankPos;
	} else {
		_yankPos = moved( _entries.end(), -2 );
	}
	return ( resetYankSize );
}

bool History::move( bool up_ ) {
	bool doRecall( _recallMostRecent && ! up_ );
	if ( doRecall ) {
		_current = _previous; // emulate Windows down-arrow
	}
	_recallMostRecent = false;
	return ( doRecall || move( _current, up_ ? -1 : 1 ) );
}

void History::jump( bool start_, bool reset_ ) {
	if ( start_ ) {
		_current = _entries.begin();
	} else {
		_current = last();
	}
	if ( reset_ ) {
		_recallMostRecent = false;
	}
}

void History::save_pos( void ) {
	_previous = _current;
}

void History::restore_pos( void ) {
	_current = _previous;
}

bool History::common_prefix_search( UnicodeString const& prefix_, int prefixSize_, bool back_, bool ignoreCase ) {
	int step( back_ ? -1 : 1 );
	entries_t::iterator it( moved( _current, step, true ) );
	bool lowerCaseContext( std::none_of( prefix_.begin(), prefix_.end(), []( char32_t x ) { return iswupper( static_cast<wint_t>( x ) ); } ) );
	while ( it != _current ) {
		if ( it->text().starts_with( prefix_.begin(), prefix_.begin() + prefixSize_, ignoreCase && lowerCaseContext ? case_insensitive_equal : case_sensitive_equal ) ) {
			_current = it;
			commit_index();
			return ( true );
		}
		move( it, step, true );
	}
	return ( false );
}

bool History::move( entries_t::iterator& it_, int by_, bool wrapped_ ) {
	if ( by_ > 0 ) {
		for ( int i( 0 ); i < by_; ++ i ) {
			++ it_;
			if ( it_ != _entries.end() ) {
			} else if ( wrapped_ ) {
				it_ = _entries.begin();
			} else {
				-- it_;
				return ( false );
			}
		}
	} else {
		for ( int i( 0 ); i > by_; -- i ) {
			if ( it_ != _entries.begin() ) {
				-- it_;
			} else if ( wrapped_ ) {
				it_ = last();
			} else {
				return ( false );
			}
		}
	}
	return ( true );
}

History::entries_t::iterator History::moved( entries_t::iterator it_, int by_, bool wrapped_ ) {
	move( it_, by_, wrapped_ );
	return ( it_ );
}

void History::erase( entries_t::iterator it_ ) {
	bool invalidated( it_ == _current );
	_locations.erase( it_->text() );
	it_ = _entries.erase( it_ );
	if ( invalidated ) {
		_current = it_;
	}
	if ( ( _current == _entries.end() ) && ! _entries.empty() ) {
		-- _current;
	}
	_yankPos = _entries.end();
	_previous = _current;
}

void History::trim_to_max_size( void ) {
	while ( size() > _maxSize ) {
		erase( _entries.begin() );
	}
}

void History::remove_duplicate( UnicodeString const& line_ ) {
	if ( ! _unique ) {
		return;
	}
	locations_t::iterator it( _locations.find( line_ ) );
	if ( it == _locations.end() ) {
		return;
	}
	erase( it->second );
}

void History::remove_duplicates( void ) {
	if ( ! _unique ) {
		return;
	}
	_locations.clear();
	typedef std::pair<locations_t::iterator, bool> locations_insertion_result_t;
	for ( entries_t::iterator it( _entries.begin() ), end( _entries.end() ); it != end; ++ it ) {
		it->reset_scratch();
		locations_insertion_result_t locationsInsertionResult( _locations.insert( make_pair( it->text(), it ) ) );
		if ( ! locationsInsertionResult.second ) {
			_entries.erase( locationsInsertionResult.first->second );
			locationsInsertionResult.first->second = it;
		}
	}
}

void History::update_last( UnicodeString const& line_ ) {
	if ( _unique ) {
		_locations.erase( _entries.back().text() );
		remove_duplicate( line_ );
		_locations.insert( make_pair( line_, last() ) );
	}
	_entries.back() = Entry( now_ms_str(), line_ );
}

void History::drop_last( void ) {
	reset_current_scratch();
	erase( last() );
}

bool History::is_last( void ) {
	return ( _current == last() );
}

History::entries_t::iterator History::last( void ) {
	return ( moved( _entries.end(), -1 ) );
}

void History::reset_iters( void ) {
	_previous = _current = last();
	_yankPos = _entries.end();
}

}

