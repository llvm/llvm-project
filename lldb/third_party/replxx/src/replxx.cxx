/*
 * Copyright (c) 2017-2018, Marcin Konarski (amok at codestation.org)
 * Copyright (c) 2010, Salvatore Sanfilippo <antirez at gmail dot com>
 * Copyright (c) 2010, Pieter Noordhuis <pcnoordhuis at gmail dot com>
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *	 * Redistributions of source code must retain the above copyright notice,
 *		 this list of conditions and the following disclaimer.
 *	 * Redistributions in binary form must reproduce the above copyright
 *		 notice, this list of conditions and the following disclaimer in the
 *		 documentation and/or other materials provided with the distribution.
 *	 * Neither the name of Redis nor the names of its contributors may be used
 *		 to endorse or promote products derived from this software without
 *		 specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * line editing lib needs to be 20,000 lines of C code.
 *
 * You can find the latest source code at:
 *
 *	 http://github.com/antirez/linenoise
 *
 * Does a number of crazy assumptions that happen to be true in 99.9999% of
 * the 2010 UNIX computers around.
 *
 * References:
 * - http://invisible-island.net/xterm/ctlseqs/ctlseqs.html
 * - http://www.3waylabs.com/nw/WWW/products/wizcon/vt220.html
 *
 * Todo list:
 * - Switch to gets() if $TERM is something we can't support.
 * - Filter bogus Ctrl+<char> combinations.
 * - Win32 support
 *
 * Bloat:
 * - Completion?
 * - History search like Ctrl+r in readline?
 *
 * List of escape sequences used by this program, we do everything just
 * with three sequences. In order to be so cheap we may have some
 * flickering effect with some slow terminal, but the lesser sequences
 * the more compatible.
 *
 * CHA (Cursor Horizontal Absolute)
 *		Sequence: ESC [ n G
 *		Effect: moves cursor to column n (1 based)
 *
 * EL (Erase Line)
 *		Sequence: ESC [ n K
 *		Effect: if n is 0 or missing, clear from cursor to end of line
 *		Effect: if n is 1, clear from beginning of line to cursor
 *		Effect: if n is 2, clear entire line
 *
 * CUF (Cursor Forward)
 *		Sequence: ESC [ n C
 *		Effect: moves cursor forward of n chars
 *
 * The following are used to clear the screen: ESC [ H ESC [ 2 J
 * This is actually composed of two sequences:
 *
 * cursorhome
 *		Sequence: ESC [ H
 *		Effect: moves the cursor to upper left corner
 *
 * ED2 (Clear entire screen)
 *		Sequence: ESC [ 2 J
 *		Effect: clear the whole screen
 *
 */

#include <algorithm>
#include <cstdarg>

#ifdef _WIN32

#include <io.h>
#define STDIN_FILENO 0

#else /* _WIN32 */

#include <signal.h>
#include <unistd.h>
#include <sys/stat.h>

#endif /* _WIN32 */

#include "replxx.h"
#include "replxx.hxx"
#include "replxx_impl.hxx"
#include "history.hxx"

static_assert(
	static_cast<int>( replxx::Replxx::ACTION::SEND_EOF ) == static_cast<int>( REPLXX_ACTION_SEND_EOF ),
	"C and C++ `ACTION` APIs are missaligned!"
);

static_assert(
	static_cast<int>( replxx::Replxx::KEY::PASTE_FINISH ) == static_cast<int>( REPLXX_KEY_PASTE_FINISH ),
	"C and C++ `KEY` APIs are missaligned!"
);

using namespace std;
using namespace std::placeholders;
using namespace replxx;

namespace replxx {

namespace {
void delete_ReplxxImpl( Replxx::ReplxxImpl* impl_ ) {
	delete impl_;
}
}

Replxx::Replxx( void )
	: _impl( new Replxx::ReplxxImpl( nullptr, nullptr, nullptr ), delete_ReplxxImpl ) {
}

void Replxx::set_completion_callback( completion_callback_t const& fn ) {
	_impl->set_completion_callback( fn );
}

void Replxx::set_modify_callback( modify_callback_t const& fn ) {
	_impl->set_modify_callback( fn );
}

void Replxx::set_highlighter_callback( highlighter_callback_t const& fn ) {
	_impl->set_highlighter_callback( fn );
}

void Replxx::set_hint_callback( hint_callback_t const& fn ) {
	_impl->set_hint_callback( fn );
}

char const* Replxx::input( std::string const& prompt ) {
	return ( _impl->input( prompt ) );
}

void Replxx::history_add( std::string const& line ) {
	_impl->history_add( line );
}

bool Replxx::history_sync( std::string const& filename ) {
	return ( _impl->history_sync( filename ) );
}

bool Replxx::history_save( std::string const& filename ) {
	return ( _impl->history_save( filename ) );
}

void Replxx::history_save( std::ostream& out ) {
	_impl->history_save( out );
}

bool Replxx::history_load( std::string const& filename ) {
	return ( _impl->history_load( filename ) );
}

void Replxx::history_load( std::istream& in ) {
	_impl->history_load( in );
}

void Replxx::history_clear( void ) {
	_impl->history_clear();
}

int Replxx::history_size( void ) const {
	return ( _impl->history_size() );
}

Replxx::HistoryScan Replxx::history_scan( void ) const {
	return ( _impl->history_scan() );
}

void Replxx::set_preload_buffer( std::string const& preloadText ) {
	_impl->set_preload_buffer( preloadText );
}

void Replxx::set_word_break_characters( char const* wordBreakers ) {
	_impl->set_word_break_characters( wordBreakers );
}

void Replxx::set_max_hint_rows( int count ) {
	_impl->set_max_hint_rows( count );
}

void Replxx::set_hint_delay( int milliseconds ) {
	_impl->set_hint_delay( milliseconds );
}

void Replxx::set_completion_count_cutoff( int count ) {
	_impl->set_completion_count_cutoff( count );
}

void Replxx::set_double_tab_completion( bool val ) {
	_impl->set_double_tab_completion( val );
}

void Replxx::set_complete_on_empty( bool val ) {
	_impl->set_complete_on_empty( val );
}

void Replxx::set_beep_on_ambiguous_completion( bool val ) {
	_impl->set_beep_on_ambiguous_completion( val );
}

void Replxx::set_immediate_completion( bool val ) {
	_impl->set_immediate_completion( val );
}

void Replxx::set_unique_history( bool val ) {
	_impl->set_unique_history( val );
}

void Replxx::set_no_color( bool val ) {
	_impl->set_no_color( val );
}

void Replxx::set_indent_multiline( bool val ) {
	_impl->set_indent_multiline( val );
}

void Replxx::set_max_history_size( int len ) {
	_impl->set_max_history_size( len );
}

void Replxx::clear_screen( void ) {
	_impl->clear_screen( 0 );
}

void Replxx::emulate_key_press( char32_t keyPress_ ) {
	_impl->emulate_key_press( keyPress_ );
}

Replxx::ACTION_RESULT Replxx::invoke( ACTION action_, char32_t keyPress_ ) {
	return ( _impl->invoke( action_, keyPress_ ) );
}

void Replxx::bind_key( char32_t keyPress_, key_press_handler_t handler_ ) {
	_impl->bind_key( keyPress_, handler_ );
}

void Replxx::bind_key_internal( char32_t keyPress_, char const* actionName_ ) {
	_impl->bind_key_internal( keyPress_, actionName_ );
}

Replxx::State Replxx::get_state( void ) const {
	return ( _impl->get_state() );
}

void Replxx::set_state( Replxx::State const& state_ ) {
	_impl->set_state( state_ );
}

void Replxx::set_ignore_case( bool val ) {
	_impl->set_ignore_case( val );
}

int Replxx::install_window_change_handler( void ) {
	return ( _impl->install_window_change_handler() );
}

void Replxx::enable_bracketed_paste( void ) {
	_impl->enable_bracketed_paste();
}

void Replxx::disable_bracketed_paste( void ) {
	_impl->disable_bracketed_paste();
}

void Replxx::print( char const* format_, ... ) {
	::std::va_list ap;
	va_start( ap, format_ );
	int size = static_cast<int>( vsnprintf( nullptr, 0, format_, ap ) );
	va_end( ap );
	va_start( ap, format_ );
	unique_ptr<char[]> buf( new char[size + 1] );
	vsnprintf( buf.get(), static_cast<size_t>( size + 1 ), format_, ap );
	va_end( ap );
	return ( _impl->print( buf.get(), size ) );
}

void Replxx::write( char const* str, int length ) {
	return ( _impl->print( str, length ) );
}

void Replxx::set_prompt( std::string prompt ) {
	return ( _impl->set_prompt( std::move( prompt ) ) );
}

namespace color {

Replxx::Color operator | ( Replxx::Color color1_, Replxx::Color color2_ ) {
	return static_cast<Replxx::Color>( static_cast<int unsigned>( color1_ ) | static_cast<int unsigned>( color2_ ) );
}

Replxx::Color bg( Replxx::Color color_ ) {
	return static_cast<Replxx::Color>( ( ( static_cast<int unsigned>( color_ ) & 0xFFu ) << 8 ) | color::BACKGROUND_COLOR_SET );
}

Replxx::Color bold( Replxx::Color color_ ) {
	return static_cast<Replxx::Color>( static_cast<int unsigned>( color_ ) | color::BOLD );
}

Replxx::Color underline( Replxx::Color color_ ) {
	return static_cast<Replxx::Color>( static_cast<int unsigned>( color_ ) | color::UNDERLINE );
}

Replxx::Color grayscale( int level_ ) {
	assert( ( level_ >= 0 ) && ( level_ < 24 ) );
	return static_cast<Replxx::Color>( abs( level_ ) % 24 + static_cast<int unsigned>( color::GRAYSCALE ) );
}

Replxx::Color rgb666( int red_, int green_, int blue_ ) {
	assert( ( red_ >= 0 ) && ( red_ < 6 ) && ( green_ >= 0 ) && ( green_ < 6 ) && ( blue_ >= 0 ) && ( blue_ < 6 ) );
	return static_cast<Replxx::Color>(
		( abs( red_ ) % 6 ) * 36
		+ ( abs( green_ ) % 6 ) * 6
		+ ( abs( blue_ ) % 6 )
		+ static_cast<int unsigned>( color::RGB666 )
	);
}

}

}

::Replxx* replxx_init() {
	typedef ::Replxx* replxx_data_t;
	return ( reinterpret_cast<replxx_data_t>( new replxx::Replxx::ReplxxImpl( nullptr, nullptr, nullptr ) ) );
}

void replxx_end( ::Replxx* replxx_ ) {
	delete reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ );
}

void replxx_clear_screen( ::Replxx* replxx_ ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->clear_screen( 0 );
}

void replxx_emulate_key_press( ::Replxx* replxx_, int unsigned keyPress_ ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->emulate_key_press( keyPress_ );
}

ReplxxActionResult replxx_invoke( ::Replxx* replxx_, ReplxxAction action_, int unsigned keyPress_ ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	return ( static_cast<ReplxxActionResult>( replxx->invoke( static_cast<replxx::Replxx::ACTION>( action_ ), keyPress_ ) ) );
}

replxx::Replxx::ACTION_RESULT key_press_handler_forwarder( key_press_handler_t handler_, char32_t code_, void* userData_ ) {
	return ( static_cast<replxx::Replxx::ACTION_RESULT>( handler_( code_, userData_ ) ) );
}

void replxx_bind_key( ::Replxx* replxx_, int code_, key_press_handler_t handler_, void* userData_ ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->bind_key( code_, std::bind( key_press_handler_forwarder, handler_, _1, userData_ ) );
}

int replxx_bind_key_internal( ::Replxx* replxx_, int code_, char const* actionName_ ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	try {
		replxx->bind_key_internal( code_, actionName_ );
	} catch ( ... ) {
		return ( -1 );
	}
	return ( 0 );
}

void replxx_get_state( ::Replxx* replxx_, ReplxxState* state ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx::Replxx::State s( replxx->get_state() );
	state->text = s.text();
	state->cursorPosition = s.cursor_position();
}

void replxx_set_state( ::Replxx* replxx_, ReplxxState* state ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->set_state( replxx::Replxx::State( state->text, state->cursorPosition ) );
}

void replxx_set_ignore_case( ::Replxx* replxx_, int val ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->set_ignore_case( val );
}

/**
 * replxx_set_preload_buffer provides text to be inserted into the command buffer
 *
 * the provided text will be processed to be usable and will be used to preload
 * the input buffer on the next call to replxx_input()
 *
 * @param preloadText text to begin with on the next call to replxx_input()
 */
void replxx_set_preload_buffer(::Replxx* replxx_, const char* preloadText) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->set_preload_buffer( preloadText ? preloadText : "" );
}

/**
 * replxx_input is a readline replacement.
 *
 * call it with a prompt to display and it will return a line of input from the
 * user
 *
 * @param prompt text of prompt to display to the user
 * @return the returned string is managed by replxx library
 * and it must NOT be freed in the client.
 */
char const* replxx_input( ::Replxx* replxx_, const char* prompt ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	return ( replxx->input( prompt ) );
}

int replxx_print( ::Replxx* replxx_, char const* format_, ... ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	::std::va_list ap;
	va_start( ap, format_ );
	int size = static_cast<int>( vsnprintf( nullptr, 0, format_, ap ) );
	va_end( ap );
	va_start( ap, format_ );
	unique_ptr<char[]> buf( new char[size + 1] );
	vsnprintf( buf.get(), static_cast<size_t>( size + 1 ), format_, ap );
	va_end( ap );
	try {
		replxx->print( buf.get(), size );
	} catch ( ... ) {
		return ( -1 );
	}
	return ( size );
}

int replxx_write( ::Replxx* replxx_, char const* str, int length ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	try {
		replxx->print( str, length );
	} catch ( ... ) {
		return ( -1 );
	}
	return static_cast<int>( length );
}

void replxx_set_prompt( ::Replxx* replxx_, const char* prompt ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->set_prompt( prompt );
}

struct replxx_completions {
	replxx::Replxx::completions_t data;
};

struct replxx_hints {
	replxx::Replxx::hints_t data;
};

void modify_fwd( replxx_modify_callback_t fn, std::string& line_, int& cursorPosition_, void* userData_ ) {
#ifdef _WIN32
#define strdup _strdup
#endif
	char* s( strdup( line_.c_str() ) );
#undef strdup
	fn( &s, &cursorPosition_, userData_ );
	line_ = s;
	free( s );
	return;
}

void replxx_set_modify_callback(::Replxx* replxx_, replxx_modify_callback_t* fn, void* userData) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->set_modify_callback( std::bind( &modify_fwd, fn, _1, _2, userData ) );
}

replxx::Replxx::completions_t completions_fwd( replxx_completion_callback_t fn, std::string const& input_, int& contextLen_, void* userData ) {
	replxx_completions completions;
	fn( input_.c_str(), &completions, &contextLen_, userData );
	return ( completions.data );
}

/* Register a callback function to be called for tab-completion. */
void replxx_set_completion_callback(::Replxx* replxx_, replxx_completion_callback_t* fn, void* userData) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->set_completion_callback( std::bind( &completions_fwd, fn, _1, _2, userData ) );
}

void highlighter_fwd( replxx_highlighter_callback_t fn, std::string const& input, replxx::Replxx::colors_t& colors, void* userData ) {
	std::vector<ReplxxColor> colorsTmp( colors.size() );
	std::transform(
		colors.begin(),
		colors.end(),
		colorsTmp.begin(),
		[]( replxx::Replxx::Color c ) {
			return ( static_cast<ReplxxColor>( c ) );
		}
	);
	fn( input.c_str(), colorsTmp.data(), static_cast<int>( colors.size() ), userData );
	std::transform(
		colorsTmp.begin(),
		colorsTmp.end(),
		colors.begin(),
		[]( ReplxxColor c ) {
			return ( static_cast<replxx::Replxx::Color>( c ) );
		}
	);
}

void replxx_set_highlighter_callback( ::Replxx* replxx_, replxx_highlighter_callback_t* fn, void* userData ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->set_highlighter_callback( std::bind( &highlighter_fwd, fn, _1, _2, userData ) );
}

replxx::Replxx::hints_t hints_fwd( replxx_hint_callback_t fn, std::string const& input_, int& contextLen_, replxx::Replxx::Color& color_, void* userData ) {
	replxx_hints hints;
	ReplxxColor c( static_cast<ReplxxColor>( color_ ) );
	fn( input_.c_str(), &hints, &contextLen_, &c, userData );
	return ( hints.data );
}

void replxx_set_hint_callback( ::Replxx* replxx_, replxx_hint_callback_t* fn, void* userData ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->set_hint_callback( std::bind( &hints_fwd, fn, _1, _2, _3, userData ) );
}

void replxx_add_hint(replxx_hints* lh, const char* str) {
	lh->data.emplace_back(str);
}

void replxx_add_completion( replxx_completions* lc, const char* str ) {
	lc->data.emplace_back( str );
}

void replxx_add_color_completion( replxx_completions* lc, const char* str, ReplxxColor color ) {
	lc->data.emplace_back( str, static_cast<replxx::Replxx::Color>( color ) );
}

void replxx_history_add( ::Replxx* replxx_, const char* line ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->history_add( line );
}

void replxx_set_max_history_size( ::Replxx* replxx_, int len ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->set_max_history_size( len );
}

void replxx_set_max_hint_rows( ::Replxx* replxx_, int count ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->set_max_hint_rows( count );
}

void replxx_set_hint_delay( ::Replxx* replxx_, int milliseconds ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->set_hint_delay( milliseconds );
}

void replxx_set_completion_count_cutoff( ::Replxx* replxx_, int count ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->set_completion_count_cutoff( count );
}

void replxx_set_word_break_characters( ::Replxx* replxx_, char const* breakChars_ ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->set_word_break_characters( breakChars_ );
}

void replxx_set_double_tab_completion( ::Replxx* replxx_, int val ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->set_double_tab_completion( val ? true : false );
}

void replxx_set_complete_on_empty( ::Replxx* replxx_, int val ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->set_complete_on_empty( val ? true : false );
}

void replxx_set_no_color( ::Replxx* replxx_, int val ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->set_no_color( val ? true : false );
}

void replxx_set_indent_multiline( ::Replxx* replxx_, int val ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->set_indent_multiline( val ? true : false );
}

void replxx_set_beep_on_ambiguous_completion( ::Replxx* replxx_, int val ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->set_beep_on_ambiguous_completion( val ? true : false );
}

void replxx_set_immediate_completion( ::Replxx* replxx_, int val ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->set_immediate_completion( val ? true : false );
}

void replxx_set_unique_history( ::Replxx* replxx_, int val ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->set_unique_history( val ? true : false );
}

void replxx_enable_bracketed_paste( ::Replxx* replxx_ ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->enable_bracketed_paste();
}

void replxx_disable_bracketed_paste( ::Replxx* replxx_ ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->disable_bracketed_paste();
}

ReplxxHistoryScan* replxx_history_scan_start( ::Replxx* replxx_ ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	return ( reinterpret_cast<ReplxxHistoryScan*>( replxx->history_scan().release() ) );
}

void replxx_history_scan_stop( ::Replxx*, ReplxxHistoryScan* historyScan_ ) {
	delete reinterpret_cast<replxx::Replxx::HistoryScanImpl*>( historyScan_ );
}

int replxx_history_scan_next( ::Replxx*, ReplxxHistoryScan* historyScan_, ReplxxHistoryEntry* historyEntry_ ) {
	replxx::Replxx::HistoryScanImpl* historyScan( reinterpret_cast<replxx::Replxx::HistoryScanImpl*>( historyScan_ ) );
	bool hasNext( historyScan->next() );
	if ( hasNext ) {
		replxx::Replxx::HistoryEntry const& historyEntry( historyScan->get() );
		historyEntry_->timestamp = historyEntry.timestamp().c_str();
		historyEntry_->text = historyEntry.text().c_str();
	}
	return ( hasNext ? 0 : -1 );
}

/* Save the history in the specified file. On success 0 is returned
 * otherwise -1 is returned. */
int replxx_history_sync( ::Replxx* replxx_, const char* filename ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	return ( replxx->history_sync( filename ) ? 0 : -1 );
}

/* Save the history in the specified file. On success 0 is returned
 * otherwise -1 is returned. */
int replxx_history_save( ::Replxx* replxx_, const char* filename ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	return ( replxx->history_save( filename ) ? 0 : -1 );
}

/* Load the history from the specified file. If the file does not exist
 * zero is returned and no operation is performed.
 *
 * If the file exists and the operation succeeded 0 is returned, otherwise
 * on error -1 is returned. */
int replxx_history_load( ::Replxx* replxx_, const char* filename ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	return ( replxx->history_load( filename ) ? 0 : -1 );
}

void replxx_history_clear( ::Replxx* replxx_ ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	replxx->history_clear();
}

int replxx_history_size( ::Replxx* replxx_ ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	return ( replxx->history_size() );
}

/* This special mode is used by replxx in order to print scan codes
 * on screen for debugging / development purposes. It is implemented
 * by the replxx-c-api-example program using the --keycodes option. */
#ifdef __REPLXX_DEBUG__
void replxx_debug_dump_print_codes(void) {
	char quit[4];

	printf(
			"replxx key codes debugging mode.\n"
			"Press keys to see scan codes. Type 'quit' at any time to exit.\n");
	if (enableRawMode() == -1) return;
	memset(quit, ' ', 4);
	while (1) {
		char c;
		int nread;

#if _WIN32
		nread = _read(STDIN_FILENO, &c, 1);
#else
		nread = read(STDIN_FILENO, &c, 1);
#endif
		if (nread <= 0) continue;
		memmove(quit, quit + 1, sizeof(quit) - 1); /* shift string to left. */
		quit[sizeof(quit) - 1] = c; /* Insert current char on the right. */
		if (memcmp(quit, "quit", sizeof(quit)) == 0) break;

		printf("'%c' %02x (%d) (type quit to exit)\n", isprint(c) ? c : '?', (int)c,
					 (int)c);
		printf("\r"); /* Go left edge manually, we are in raw mode. */
		fflush(stdout);
	}
	disableRawMode();
}
#endif // __REPLXX_DEBUG__

int replxx_install_window_change_handler( ::Replxx* replxx_ ) {
	replxx::Replxx::ReplxxImpl* replxx( reinterpret_cast<replxx::Replxx::ReplxxImpl*>( replxx_ ) );
	return ( replxx->install_window_change_handler() );
}

using namespace replxx::color;
ReplxxColor replxx_color_combine( ReplxxColor color1_, ReplxxColor color2_ ) {
	return static_cast<ReplxxColor>( static_cast<replxx::Replxx::Color>( color1_ ) | static_cast<replxx::Replxx::Color>( color2_ ) );
}

ReplxxColor replxx_color_bg( ReplxxColor color_ ) {
	return static_cast<ReplxxColor>( color::bg( static_cast<replxx::Replxx::Color>( color_ ) ) );
}

ReplxxColor replxx_color_bold( ReplxxColor color_ ) {
	return static_cast<ReplxxColor>( color::bold( static_cast<replxx::Replxx::Color>( color_ ) ) );
}

ReplxxColor replxx_color_underline( ReplxxColor color_ ) {
	return static_cast<ReplxxColor>( color::underline( static_cast<replxx::Replxx::Color>( color_ ) ) );
}

ReplxxColor replxx_color_grayscale( int level_ ) {
	return static_cast<ReplxxColor>( color::grayscale( level_ ) );
}

ReplxxColor replxx_color_rgb666( int r_, int g_, int b_ ) {
	return static_cast<ReplxxColor>( color::rgb666( r_, g_, b_ ) );
}

