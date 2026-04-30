#include <algorithm>
#include <memory>
#include <cerrno>
#include <iostream>
#include <chrono>
#include <cassert>

#ifdef _WIN32

#include <windows.h>
#include <io.h>
#if _MSC_VER < 1900
#define snprintf _snprintf // Microsoft headers use underscores in some names
#endif
#define strcasecmp _stricmp
#define write _write
#define STDIN_FILENO 0

#else /* _WIN32 */

#include <unistd.h>
#include <signal.h>

#endif /* _WIN32 */

#ifdef _WIN32
#include "windows.hxx"
#endif

#include "replxx_impl.hxx"
#include "utf8string.hxx"
#include "prompt.hxx"
#include "util.hxx"
#include "terminal.hxx"
#include "history.hxx"
#include "replxx.hxx"

using namespace std;
using namespace replxx::color;

namespace replxx {

namespace {

namespace action_names {

char const INSERT_CHARACTER[]                  = "insert_character";
char const NEW_LINE[]                          = "new_line";
char const MOVE_CURSOR_TO_BEGINING_OF_LINE[]   = "move_cursor_to_begining_of_line";
char const MOVE_CURSOR_TO_END_OF_LINE[]        = "move_cursor_to_end_of_line";
char const MOVE_CURSOR_LEFT[]                  = "move_cursor_left";
char const MOVE_CURSOR_RIGHT[]                 = "move_cursor_right";
char const MOVE_CURSOR_ONE_WORD_LEFT[]         = "move_cursor_one_word_left";
char const MOVE_CURSOR_ONE_WORD_RIGHT[]        = "move_cursor_one_word_right";
char const MOVE_CURSOR_ONE_SUBWORD_LEFT[]      = "move_cursor_one_subword_left";
char const MOVE_CURSOR_ONE_SUBWORD_RIGHT[]     = "move_cursor_one_subword_right";
char const KILL_TO_WHITESPACE_ON_LEFT[]        = "kill_to_whitespace_on_left";
char const KILL_TO_END_OF_WORD[]               = "kill_to_end_of_word";
char const KILL_TO_END_OF_SUBWORD[]            = "kill_to_end_of_subword";
char const KILL_TO_BEGINING_OF_WORD[]          = "kill_to_begining_of_word";
char const KILL_TO_BEGINING_OF_SUBWORD[]       = "kill_to_begining_of_subword";
char const KILL_TO_BEGINING_OF_LINE[]          = "kill_to_begining_of_line";
char const KILL_TO_END_OF_LINE[]               = "kill_to_end_of_line";
char const YANK[]                              = "yank";
char const YANK_CYCLE[]                        = "yank_cycle";
char const YANK_LAST_ARG[]                     = "yank_last_arg";
char const CAPITALIZE_WORD[]                   = "capitalize_word";
char const LOWERCASE_WORD[]                    = "lowercase_word";
char const UPPERCASE_WORD[]                    = "uppercase_word";
char const CAPITALIZE_SUBWORD[]                = "capitalize_subword";
char const LOWERCASE_SUBWORD[]                 = "lowercase_subword";
char const UPPERCASE_SUBWORD[]                 = "uppercase_subword";
char const TRANSPOSE_CHARACTERS[]              = "transpose_characters";
char const ABORT_LINE[]                        = "abort_line";
char const SEND_EOF[]                          = "send_eof";
char const TOGGLE_OVERWRITE_MODE[]             = "toggle_overwrite_mode";
char const DELETE_CHARACTER_UNDER_CURSOR[]     = "delete_character_under_cursor";
char const DELETE_CHARACTER_LEFT_OF_CURSOR[]   = "delete_character_left_of_cursor";
char const COMMIT_LINE[]                       = "commit_line";
char const CLEAR_SCREEN[]                      = "clear_screen";
char const COMPLETE_NEXT[]                     = "complete_next";
char const COMPLETE_PREVIOUS[]                 = "complete_previous";
char const HISTORY_NEXT[]                      = "history_next";
char const HISTORY_PREVIOUS[]                  = "history_previous";
char const LINE_NEXT[]                         = "line_next";
char const LINE_PREVIOUS[]                     = "line_previous";
char const HISTORY_LAST[]                      = "history_last";
char const HISTORY_FIRST[]                     = "history_first";
char const HISTORY_RESTORE[]                   = "history_restore";
char const HISTORY_RESTORE_CURRENT[]           = "history_restore_current";
char const HINT_PREVIOUS[]                     = "hint_previous";
char const HINT_NEXT[]                         = "hint_next";
char const VERBATIM_INSERT[]                   = "verbatim_insert";
char const SUSPEND[]                           = "suspend";
char const COMPLETE_LINE[]                     = "complete_line";
char const HISTORY_INCREMENTAL_SEARCH[]        = "history_incremental_search";
char const HISTORY_SEEDED_INCREMENTAL_SEARCH[] = "history_seeded_incremental_search";
char const HISTORY_COMMON_PREFIX_SEARCH[]      = "history_common_prefix_search";
}

static int const REPLXX_MAX_HINT_ROWS( 4 );
/*
 * All whitespaces and all non-alphanumerical characters from ASCII range
 * with an exception of an underscore ('_').
 */
char const defaultWordBreakChars[] = " \t\v\f\a\b\r\n`~!@#$%^&*()-=+[{]}\\|;:'\",<.>/?";
/*
 * All whitespaces and all non-alphanumerical characters from ASCII range
 */
char const defaultSubwordBreakChars[] = " \t\v\f\a\b\r\n`~!@#$%^&*()-=+[{]}\\|;:'\",<.>/?_";
static const char* unsupported_term[] = {"dumb", "cons25", "emacs", NULL};

static bool isUnsupportedTerm(void) {
	char* term = getenv("TERM");
	if (term == NULL) {
		return false;
	}
	for (int j = 0; unsupported_term[j]; ++j) {
		if (!strcasecmp(term, unsupported_term[j])) {
			return true;
		}
	}
	return false;
}

int long long RAPID_REFRESH_MS = 1;
int long long RAPID_REFRESH_US = RAPID_REFRESH_MS * 1000;

inline int long long now_us( void ) {
	return ( std::chrono::duration_cast<std::chrono::microseconds>( std::chrono::high_resolution_clock::now().time_since_epoch() ).count() );
}

class IOModeGuard {
	Terminal& _terminal;
public:
	IOModeGuard( Terminal& terminal_ )
		: _terminal( terminal_ ) {
	}
	~IOModeGuard( void ) {
		try {
			_terminal.reset_raw_mode();
		} catch ( ... ) {
		}
	}
};

}

Replxx::ReplxxImpl::ReplxxImpl( FILE*, FILE*, FILE* )
	: _utf8Buffer()
	, _data()
	, _pos( 0 )
	, _display()
	, _displayInputLength( 0 )
	, _hint()
	, _prefix( 0 )
	, _hintSelection( -1 )
	, _history()
	, _killRing()
	, _lastRefreshTime( now_us() )
	, _refreshSkipped( false )
	, _lastYankSize( 0 )
	, _maxHintRows( REPLXX_MAX_HINT_ROWS )
	, _hintDelay( 0 )
	, _wordBreakChars( defaultWordBreakChars )
	, _subwordBreakChars( defaultSubwordBreakChars )
	, _completionCountCutoff( 100 )
	, _overwrite( false )
	, _doubleTabCompletion( false )
	, _completeOnEmpty( true )
	, _beepOnAmbiguousCompletion( false )
	, _immediateCompletion( true )
	, _bracketedPaste( false )
	, _noColor( false )
	, _indentMultiline( true )
	, _namedActions()
	, _keyPressHandlers()
	, _terminal()
	, _currentThread()
	, _prompt( _terminal )
	, _completionCallback( nullptr )
	, _highlighterCallback( nullptr )
	, _hintCallback( nullptr )
	, _keyPresses()
	, _messages()
	, _asyncPrompt()
	, _updatePrompt( false )
	, _completions()
	, _completionContextLength( 0 )
	, _completionSelection( -1 )
	, _preloadedBuffer()
	, _errorMessage()
	, _previousSearchText()
	, _modifiedState( false )
	, _hintColor( Replxx::Color::GRAY )
	, _hintsCache()
	, _hintContextLenght( -1 )
	, _hintSeed()
	, _hasNewlines( false )
	, _oldPos( 0 )
	, _moveCursor( false )
	, _ignoreCase( false )
	, _mutex() {
	using namespace std::placeholders;
	_namedActions[action_names::INSERT_CHARACTER]                  = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::INSERT_CHARACTER,                  _1 );
	_namedActions[action_names::NEW_LINE]                          = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::NEW_LINE,                          _1 );
	_namedActions[action_names::MOVE_CURSOR_TO_BEGINING_OF_LINE]   = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::MOVE_CURSOR_TO_BEGINING_OF_LINE,   _1 );
	_namedActions[action_names::MOVE_CURSOR_TO_END_OF_LINE]        = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::MOVE_CURSOR_TO_END_OF_LINE,        _1 );
	_namedActions[action_names::MOVE_CURSOR_LEFT]                  = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::MOVE_CURSOR_LEFT,                  _1 );
	_namedActions[action_names::MOVE_CURSOR_RIGHT]                 = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::MOVE_CURSOR_RIGHT,                 _1 );
	_namedActions[action_names::MOVE_CURSOR_ONE_WORD_LEFT]         = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::MOVE_CURSOR_ONE_WORD_LEFT,         _1 );
	_namedActions[action_names::MOVE_CURSOR_ONE_WORD_RIGHT]        = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::MOVE_CURSOR_ONE_WORD_RIGHT,        _1 );
	_namedActions[action_names::MOVE_CURSOR_ONE_SUBWORD_LEFT]      = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::MOVE_CURSOR_ONE_SUBWORD_LEFT,      _1 );
	_namedActions[action_names::MOVE_CURSOR_ONE_SUBWORD_RIGHT]     = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::MOVE_CURSOR_ONE_SUBWORD_RIGHT,     _1 );
	_namedActions[action_names::KILL_TO_WHITESPACE_ON_LEFT]        = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::KILL_TO_WHITESPACE_ON_LEFT,        _1 );
	_namedActions[action_names::KILL_TO_END_OF_WORD]               = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::KILL_TO_END_OF_WORD,               _1 );
	_namedActions[action_names::KILL_TO_BEGINING_OF_WORD]          = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::KILL_TO_BEGINING_OF_WORD,          _1 );
	_namedActions[action_names::KILL_TO_END_OF_SUBWORD]            = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::KILL_TO_END_OF_SUBWORD,            _1 );
	_namedActions[action_names::KILL_TO_BEGINING_OF_SUBWORD]       = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::KILL_TO_BEGINING_OF_SUBWORD,       _1 );
	_namedActions[action_names::KILL_TO_BEGINING_OF_LINE]          = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::KILL_TO_BEGINING_OF_LINE,          _1 );
	_namedActions[action_names::KILL_TO_END_OF_LINE]               = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::KILL_TO_END_OF_LINE,               _1 );
	_namedActions[action_names::YANK]                              = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::YANK,                              _1 );
	_namedActions[action_names::YANK_CYCLE]                        = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::YANK_CYCLE,                        _1 );
	_namedActions[action_names::YANK_LAST_ARG]                     = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::YANK_LAST_ARG,                     _1 );
	_namedActions[action_names::CAPITALIZE_WORD]                   = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::CAPITALIZE_WORD,                   _1 );
	_namedActions[action_names::LOWERCASE_WORD]                    = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::LOWERCASE_WORD,                    _1 );
	_namedActions[action_names::UPPERCASE_WORD]                    = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::UPPERCASE_WORD,                    _1 );
	_namedActions[action_names::CAPITALIZE_SUBWORD]                = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::CAPITALIZE_SUBWORD,                _1 );
	_namedActions[action_names::LOWERCASE_SUBWORD]                 = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::LOWERCASE_SUBWORD,                 _1 );
	_namedActions[action_names::UPPERCASE_SUBWORD]                 = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::UPPERCASE_SUBWORD,                 _1 );
	_namedActions[action_names::TRANSPOSE_CHARACTERS]              = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::TRANSPOSE_CHARACTERS,              _1 );
	_namedActions[action_names::ABORT_LINE]                        = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::ABORT_LINE,                        _1 );
	_namedActions[action_names::SEND_EOF]                          = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::SEND_EOF,                          _1 );
	_namedActions[action_names::TOGGLE_OVERWRITE_MODE]             = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::TOGGLE_OVERWRITE_MODE,             _1 );
	_namedActions[action_names::DELETE_CHARACTER_UNDER_CURSOR]     = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::DELETE_CHARACTER_UNDER_CURSOR,     _1 );
	_namedActions[action_names::DELETE_CHARACTER_LEFT_OF_CURSOR]   = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::DELETE_CHARACTER_LEFT_OF_CURSOR,   _1 );
	_namedActions[action_names::COMMIT_LINE]                       = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::COMMIT_LINE,                       _1 );
	_namedActions[action_names::CLEAR_SCREEN]                      = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::CLEAR_SCREEN,                      _1 );
	_namedActions[action_names::COMPLETE_NEXT]                     = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::COMPLETE_NEXT,                     _1 );
	_namedActions[action_names::COMPLETE_PREVIOUS]                 = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::COMPLETE_PREVIOUS,                 _1 );
	_namedActions[action_names::LINE_NEXT]                         = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::LINE_NEXT,                         _1 );
	_namedActions[action_names::LINE_PREVIOUS]                     = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::LINE_PREVIOUS,                     _1 );
	_namedActions[action_names::HISTORY_NEXT]                      = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::HISTORY_NEXT,                      _1 );
	_namedActions[action_names::HISTORY_PREVIOUS]                  = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::HISTORY_PREVIOUS,                  _1 );
	_namedActions[action_names::HISTORY_LAST]                      = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::HISTORY_LAST,                      _1 );
	_namedActions[action_names::HISTORY_FIRST]                     = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::HISTORY_FIRST,                     _1 );
	_namedActions[action_names::HISTORY_RESTORE]                   = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::HISTORY_RESTORE,                   _1 );
	_namedActions[action_names::HISTORY_RESTORE_CURRENT]           = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::HISTORY_RESTORE_CURRENT,           _1 );
	_namedActions[action_names::HINT_PREVIOUS]                     = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::HINT_PREVIOUS,                     _1 );
	_namedActions[action_names::HINT_NEXT]                         = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::HINT_NEXT,                         _1 );
#ifndef _WIN32
	_namedActions[action_names::VERBATIM_INSERT]                   = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::VERBATIM_INSERT,                   _1 );
	_namedActions[action_names::SUSPEND]                           = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::SUSPEND,                           _1 );
#else
	_namedActions[action_names::VERBATIM_INSERT] = _namedActions[action_names::SUSPEND] = Replxx::key_press_handler_t();
#endif
	_namedActions[action_names::COMPLETE_LINE]                     = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::COMPLETE_LINE,                     _1 );
	_namedActions[action_names::HISTORY_INCREMENTAL_SEARCH]        = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::HISTORY_INCREMENTAL_SEARCH,        _1 );
	_namedActions[action_names::HISTORY_SEEDED_INCREMENTAL_SEARCH] = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::HISTORY_SEEDED_INCREMENTAL_SEARCH, _1 );
	_namedActions[action_names::HISTORY_COMMON_PREFIX_SEARCH]      = std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::HISTORY_COMMON_PREFIX_SEARCH,      _1 );

	bind_key( Replxx::KEY::control( 'A' ),                 _namedActions.at( action_names::MOVE_CURSOR_TO_BEGINING_OF_LINE ) );
	bind_key( Replxx::KEY::HOME + 0,                       _namedActions.at( action_names::MOVE_CURSOR_TO_BEGINING_OF_LINE ) );
	bind_key( Replxx::KEY::control( 'E' ),                 _namedActions.at( action_names::MOVE_CURSOR_TO_END_OF_LINE ) );
	bind_key( Replxx::KEY::END + 0,                        _namedActions.at( action_names::MOVE_CURSOR_TO_END_OF_LINE ) );
	bind_key( Replxx::KEY::control( 'B' ),                 _namedActions.at( action_names::MOVE_CURSOR_LEFT ) );
	bind_key( Replxx::KEY::LEFT + 0,                       _namedActions.at( action_names::MOVE_CURSOR_LEFT ) );
	bind_key( Replxx::KEY::control( 'F' ),                 _namedActions.at( action_names::MOVE_CURSOR_RIGHT ) );
	bind_key( Replxx::KEY::RIGHT + 0,                      _namedActions.at( action_names::MOVE_CURSOR_RIGHT ) );
	bind_key( Replxx::KEY::meta( 'b' ),                    _namedActions.at( action_names::MOVE_CURSOR_ONE_WORD_LEFT ) );
	bind_key( Replxx::KEY::meta( 'B' ),                    _namedActions.at( action_names::MOVE_CURSOR_ONE_SUBWORD_LEFT ) );
	bind_key( Replxx::KEY::control( Replxx::KEY::LEFT ),   _namedActions.at( action_names::MOVE_CURSOR_ONE_WORD_LEFT ) );
	bind_key( Replxx::KEY::meta( Replxx::KEY::LEFT ),      _namedActions.at( action_names::MOVE_CURSOR_ONE_WORD_LEFT ) ); // Emacs allows Meta, readline don't
	bind_key( Replxx::KEY::meta( 'f' ),                    _namedActions.at( action_names::MOVE_CURSOR_ONE_WORD_RIGHT ) );
	bind_key( Replxx::KEY::meta( 'F' ),                    _namedActions.at( action_names::MOVE_CURSOR_ONE_SUBWORD_RIGHT ) );
	bind_key( Replxx::KEY::control( Replxx::KEY::RIGHT ),  _namedActions.at( action_names::MOVE_CURSOR_ONE_WORD_RIGHT ) );
	bind_key( Replxx::KEY::meta( Replxx::KEY::RIGHT ),     _namedActions.at( action_names::MOVE_CURSOR_ONE_WORD_RIGHT ) ); // Emacs allows Meta, readline don't
	bind_key( Replxx::KEY::meta( Replxx::KEY::BACKSPACE ), _namedActions.at( action_names::KILL_TO_WHITESPACE_ON_LEFT ) );
	bind_key( Replxx::KEY::meta( 'd' ),                    _namedActions.at( action_names::KILL_TO_END_OF_WORD ) );
	bind_key( Replxx::KEY::meta( 'D' ),                    _namedActions.at( action_names::KILL_TO_END_OF_SUBWORD ) );
	bind_key( Replxx::KEY::control( 'W' ),                 _namedActions.at( action_names::KILL_TO_BEGINING_OF_WORD ) );
	bind_key( Replxx::KEY::meta( 'W' ),                    _namedActions.at( action_names::KILL_TO_BEGINING_OF_SUBWORD ) );
	bind_key( Replxx::KEY::control( 'U' ),                 _namedActions.at( action_names::KILL_TO_BEGINING_OF_LINE ) );
	bind_key( Replxx::KEY::control( 'K' ),                 _namedActions.at( action_names::KILL_TO_END_OF_LINE ) );
	bind_key( Replxx::KEY::control( 'Y' ),                 _namedActions.at( action_names::YANK ) );
	bind_key( Replxx::KEY::meta( 'y' ),                    _namedActions.at( action_names::YANK_CYCLE ) );
	bind_key( Replxx::KEY::meta( 'Y' ),                    _namedActions.at( action_names::YANK_CYCLE ) );
	bind_key( Replxx::KEY::meta( '.' ),                    _namedActions.at( action_names::YANK_LAST_ARG ) );
	bind_key( Replxx::KEY::meta( 'c' ),                    _namedActions.at( action_names::CAPITALIZE_WORD ) );
	bind_key( Replxx::KEY::meta( 'C' ),                    _namedActions.at( action_names::CAPITALIZE_SUBWORD ) );
	bind_key( Replxx::KEY::meta( 'l' ),                    _namedActions.at( action_names::LOWERCASE_WORD ) );
	bind_key( Replxx::KEY::meta( 'L' ),                    _namedActions.at( action_names::LOWERCASE_SUBWORD ) );
	bind_key( Replxx::KEY::meta( 'u' ),                    _namedActions.at( action_names::UPPERCASE_WORD ) );
	bind_key( Replxx::KEY::meta( 'U' ),                    _namedActions.at( action_names::UPPERCASE_SUBWORD ) );
	bind_key( Replxx::KEY::control( 'T' ),                 _namedActions.at( action_names::TRANSPOSE_CHARACTERS ) );
	bind_key( Replxx::KEY::control( 'C' ),                 _namedActions.at( action_names::ABORT_LINE ) );
	bind_key( Replxx::KEY::ABORT,                          _namedActions.at( action_names::ABORT_LINE ) );
	bind_key( Replxx::KEY::control( 'D' ),                 _namedActions.at( action_names::SEND_EOF ) );
	bind_key( Replxx::KEY::INSERT + 0,                     _namedActions.at( action_names::TOGGLE_OVERWRITE_MODE ) );
	bind_key( 127,                                         _namedActions.at( action_names::DELETE_CHARACTER_UNDER_CURSOR ) );
	bind_key( Replxx::KEY::DELETE + 0,                     _namedActions.at( action_names::DELETE_CHARACTER_UNDER_CURSOR ) );
	bind_key( Replxx::KEY::BACKSPACE + 0,                  _namedActions.at( action_names::DELETE_CHARACTER_LEFT_OF_CURSOR ) );
	bind_key( Replxx::KEY::control( 'J' ),                 _namedActions.at( action_names::NEW_LINE ) );
	bind_key( Replxx::KEY::meta( '\r' ),                   _namedActions.at( action_names::NEW_LINE ) );
	bind_key( Replxx::KEY::ENTER + 0,                      _namedActions.at( action_names::COMMIT_LINE ) );
	bind_key( Replxx::KEY::control( 'L' ),                 _namedActions.at( action_names::CLEAR_SCREEN ) );
	bind_key( Replxx::KEY::control( 'N' ),                 _namedActions.at( action_names::COMPLETE_NEXT ) );
	bind_key( Replxx::KEY::control( 'P' ),                 _namedActions.at( action_names::COMPLETE_PREVIOUS ) );
	bind_key( Replxx::KEY::DOWN + 0,                       _namedActions.at( action_names::LINE_NEXT ) );
	bind_key( Replxx::KEY::UP + 0,                         _namedActions.at( action_names::LINE_PREVIOUS ) );
	bind_key( Replxx::KEY::meta( Replxx::KEY::DOWN ),      _namedActions.at( action_names::HISTORY_NEXT ) );
	bind_key( Replxx::KEY::meta( Replxx::KEY::UP ),        _namedActions.at( action_names::HISTORY_PREVIOUS ) );
	bind_key( Replxx::KEY::meta( '<' ),                    _namedActions.at( action_names::HISTORY_FIRST ) );
	bind_key( Replxx::KEY::PAGE_UP + 0,                    _namedActions.at( action_names::HISTORY_FIRST ) );
	bind_key( Replxx::KEY::meta( '>' ),                    _namedActions.at( action_names::HISTORY_LAST ) );
	bind_key( Replxx::KEY::PAGE_DOWN + 0,                  _namedActions.at( action_names::HISTORY_LAST ) );
	bind_key( Replxx::KEY::control( 'G' ),                 _namedActions.at( action_names::HISTORY_RESTORE_CURRENT ) );
	bind_key( Replxx::KEY::meta( 'g' ),                    _namedActions.at( action_names::HISTORY_RESTORE ) );
	bind_key( Replxx::KEY::control( Replxx::KEY::UP ),     _namedActions.at( action_names::HINT_PREVIOUS ) );
	bind_key( Replxx::KEY::control( Replxx::KEY::DOWN ),   _namedActions.at( action_names::HINT_NEXT ) );
#ifndef _WIN32
	bind_key( Replxx::KEY::control( 'V' ),                 _namedActions.at( action_names::VERBATIM_INSERT ) );
	bind_key( Replxx::KEY::control( 'Z' ),                 _namedActions.at( action_names::SUSPEND ) );
#endif
	bind_key( Replxx::KEY::TAB + 0,                        _namedActions.at( action_names::COMPLETE_LINE ) );
	bind_key( Replxx::KEY::control( 'R' ),                 _namedActions.at( action_names::HISTORY_INCREMENTAL_SEARCH ) );
	bind_key( Replxx::KEY::control( 'S' ),                 _namedActions.at( action_names::HISTORY_INCREMENTAL_SEARCH ) );
	bind_key( Replxx::KEY::meta( 'r' ),                    _namedActions.at( action_names::HISTORY_SEEDED_INCREMENTAL_SEARCH ) );
	bind_key( Replxx::KEY::meta( 'p' ),                    _namedActions.at( action_names::HISTORY_COMMON_PREFIX_SEARCH ) );
	bind_key( Replxx::KEY::meta( 'P' ),                    _namedActions.at( action_names::HISTORY_COMMON_PREFIX_SEARCH ) );
	bind_key( Replxx::KEY::meta( 'n' ),                    _namedActions.at( action_names::HISTORY_COMMON_PREFIX_SEARCH ) );
	bind_key( Replxx::KEY::meta( 'N' ),                    _namedActions.at( action_names::HISTORY_COMMON_PREFIX_SEARCH ) );
	bind_key( Replxx::KEY::PASTE_START,                    std::bind( &ReplxxImpl::invoke, this, Replxx::ACTION::BRACKETED_PASTE, _1 ) );
}

Replxx::ReplxxImpl::~ReplxxImpl( void ) {
	disable_bracketed_paste();
}

Replxx::ACTION_RESULT Replxx::ReplxxImpl::invoke( Replxx::ACTION action_, char32_t code ) {
	switch ( action_ ) {
		case ( Replxx::ACTION::INSERT_CHARACTER ):                  return ( action( RESET_KILL_ACTION | HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::insert_character, code ) );
		case ( Replxx::ACTION::NEW_LINE ):                          return ( action( RESET_KILL_ACTION | HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::new_line, code ) );
		case ( Replxx::ACTION::DELETE_CHARACTER_UNDER_CURSOR ):     return ( action( RESET_KILL_ACTION | HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::delete_character, code ) );
		case ( Replxx::ACTION::DELETE_CHARACTER_LEFT_OF_CURSOR ):   return ( action( RESET_KILL_ACTION | HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::backspace_character, code ) );
		case ( Replxx::ACTION::KILL_TO_END_OF_LINE ):               return ( action( WANT_REFRESH | SET_KILL_ACTION | HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::kill_to_end_of_line, code ) );
		case ( Replxx::ACTION::KILL_TO_BEGINING_OF_LINE ):          return ( action( SET_KILL_ACTION | HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::kill_to_begining_of_line, code ) );
		case ( Replxx::ACTION::KILL_TO_END_OF_WORD ):               return ( action( SET_KILL_ACTION | HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::kill_word_to_right<false>, code ) );
		case ( Replxx::ACTION::KILL_TO_BEGINING_OF_WORD ):          return ( action( SET_KILL_ACTION | HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::kill_word_to_left<false>, code ) );
		case ( Replxx::ACTION::KILL_TO_END_OF_SUBWORD ):            return ( action( SET_KILL_ACTION | HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::kill_word_to_right<true>, code ) );
		case ( Replxx::ACTION::KILL_TO_BEGINING_OF_SUBWORD ):       return ( action( SET_KILL_ACTION | HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::kill_word_to_left<true>, code ) );
		case ( Replxx::ACTION::KILL_TO_WHITESPACE_ON_LEFT ):        return ( action( SET_KILL_ACTION | HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::kill_to_whitespace_to_left, code ) );
		case ( Replxx::ACTION::YANK ):                              return ( action( HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::yank, code ) );
		case ( Replxx::ACTION::YANK_CYCLE ):                        return ( action( HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::yank_cycle, code ) );
		case ( Replxx::ACTION::YANK_LAST_ARG ):                     return ( action( HISTORY_RECALL_MOST_RECENT | DONT_RESET_HIST_YANK_INDEX, &Replxx::ReplxxImpl::yank_last_arg, code ) );
		case ( Replxx::ACTION::MOVE_CURSOR_TO_BEGINING_OF_LINE ):   return ( action( MOVE_CURSOR | RESET_KILL_ACTION, &Replxx::ReplxxImpl::go_to_begining_of_line, code ) );
		case ( Replxx::ACTION::MOVE_CURSOR_TO_END_OF_LINE ):        return ( action( MOVE_CURSOR | RESET_KILL_ACTION, &Replxx::ReplxxImpl::go_to_end_of_line, code ) );
		case ( Replxx::ACTION::MOVE_CURSOR_ONE_WORD_LEFT ):         return ( action( MOVE_CURSOR | RESET_KILL_ACTION, &Replxx::ReplxxImpl::move_one_word_left<false>, code ) );
		case ( Replxx::ACTION::MOVE_CURSOR_ONE_WORD_RIGHT ):        return ( action( MOVE_CURSOR | RESET_KILL_ACTION, &Replxx::ReplxxImpl::move_one_word_right<false>, code ) );
		case ( Replxx::ACTION::MOVE_CURSOR_ONE_SUBWORD_LEFT ):      return ( action( MOVE_CURSOR | RESET_KILL_ACTION, &Replxx::ReplxxImpl::move_one_word_left<true>, code ) );
		case ( Replxx::ACTION::MOVE_CURSOR_ONE_SUBWORD_RIGHT ):     return ( action( MOVE_CURSOR | RESET_KILL_ACTION, &Replxx::ReplxxImpl::move_one_word_right<true>, code ) );
		case ( Replxx::ACTION::MOVE_CURSOR_LEFT ):                  return ( action( MOVE_CURSOR | RESET_KILL_ACTION, &Replxx::ReplxxImpl::move_one_char_left, code ) );
		case ( Replxx::ACTION::MOVE_CURSOR_RIGHT ):                 return ( action( MOVE_CURSOR | RESET_KILL_ACTION, &Replxx::ReplxxImpl::move_one_char_right, code ) );
		case ( Replxx::ACTION::LINE_NEXT ):                         return ( action( MOVE_CURSOR | RESET_KILL_ACTION, &Replxx::ReplxxImpl::line_next, code ) );
		case ( Replxx::ACTION::LINE_PREVIOUS ):                     return ( action( MOVE_CURSOR | RESET_KILL_ACTION, &Replxx::ReplxxImpl::line_previous, code ) );
		case ( Replxx::ACTION::HISTORY_NEXT ):                      return ( action( MOVE_CURSOR | RESET_KILL_ACTION, &Replxx::ReplxxImpl::history_next, code ) );
		case ( Replxx::ACTION::HISTORY_PREVIOUS ):                  return ( action( MOVE_CURSOR | RESET_KILL_ACTION, &Replxx::ReplxxImpl::history_previous, code ) );
		case ( Replxx::ACTION::HISTORY_FIRST ):                     return ( action( MOVE_CURSOR | RESET_KILL_ACTION, &Replxx::ReplxxImpl::history_first, code ) );
		case ( Replxx::ACTION::HISTORY_LAST ):                      return ( action( MOVE_CURSOR | RESET_KILL_ACTION, &Replxx::ReplxxImpl::history_last, code ) );
		case ( Replxx::ACTION::HISTORY_RESTORE_CURRENT ):           return ( action( MOVE_CURSOR | RESET_KILL_ACTION, &Replxx::ReplxxImpl::history_restore_current, code ) );
		case ( Replxx::ACTION::HISTORY_RESTORE ):                   return ( action( MOVE_CURSOR | RESET_KILL_ACTION, &Replxx::ReplxxImpl::history_restore, code ) );
		case ( Replxx::ACTION::HISTORY_INCREMENTAL_SEARCH ):        return ( action( NOOP, &Replxx::ReplxxImpl::incremental_history_search, code ) );
		case ( Replxx::ACTION::HISTORY_SEEDED_INCREMENTAL_SEARCH ): return ( action( NOOP, &Replxx::ReplxxImpl::incremental_history_search, code ) );
		case ( Replxx::ACTION::HISTORY_COMMON_PREFIX_SEARCH ):      return ( action( RESET_KILL_ACTION | DONT_RESET_PREFIX, &Replxx::ReplxxImpl::common_prefix_search, code ) );
		case ( Replxx::ACTION::HINT_NEXT ):                         return ( action( NOOP, &Replxx::ReplxxImpl::hint_next, code ) );
		case ( Replxx::ACTION::HINT_PREVIOUS ):                     return ( action( NOOP, &Replxx::ReplxxImpl::hint_previous, code ) );
		case ( Replxx::ACTION::CAPITALIZE_WORD ):                   return ( action( RESET_KILL_ACTION | HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::capitalize_word<false>, code ) );
		case ( Replxx::ACTION::LOWERCASE_WORD ):                    return ( action( RESET_KILL_ACTION | HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::lowercase_word<false>, code ) );
		case ( Replxx::ACTION::UPPERCASE_WORD ):                    return ( action( RESET_KILL_ACTION | HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::uppercase_word<false>, code ) );
		case ( Replxx::ACTION::CAPITALIZE_SUBWORD ):                return ( action( RESET_KILL_ACTION | HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::capitalize_word<true>, code ) );
		case ( Replxx::ACTION::LOWERCASE_SUBWORD ):                 return ( action( RESET_KILL_ACTION | HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::lowercase_word<true>, code ) );
		case ( Replxx::ACTION::UPPERCASE_SUBWORD ):                 return ( action( RESET_KILL_ACTION | HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::uppercase_word<true>, code ) );
		case ( Replxx::ACTION::TRANSPOSE_CHARACTERS ):              return ( action( RESET_KILL_ACTION | HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::transpose_characters, code ) );
		case ( Replxx::ACTION::TOGGLE_OVERWRITE_MODE ):             return ( action( NOOP, &Replxx::ReplxxImpl::toggle_overwrite_mode, code ) );
#ifndef _WIN32
		case ( Replxx::ACTION::VERBATIM_INSERT ):                   return ( action( WANT_REFRESH | RESET_KILL_ACTION, &Replxx::ReplxxImpl::verbatim_insert, code ) );
		case ( Replxx::ACTION::SUSPEND ):                           return ( action( WANT_REFRESH, &Replxx::ReplxxImpl::suspend, code ) );
#endif
		case ( Replxx::ACTION::CLEAR_SCREEN ):                      return ( action( NOOP, &Replxx::ReplxxImpl::clear_screen, code ) );
		case ( Replxx::ACTION::CLEAR_SELF ): clear_self_to_end_of_screen(); return ( Replxx::ACTION_RESULT::CONTINUE );
		case ( Replxx::ACTION::REPAINT ):    repaint();             return ( Replxx::ACTION_RESULT::CONTINUE );
		case ( Replxx::ACTION::COMPLETE_LINE ):                     return ( action( HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::complete_line, code ) );
		case ( Replxx::ACTION::COMPLETE_NEXT ):                     return ( action( RESET_KILL_ACTION | DONT_RESET_COMPLETIONS | HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::complete_next, code ) );
		case ( Replxx::ACTION::COMPLETE_PREVIOUS ):                 return ( action( RESET_KILL_ACTION | DONT_RESET_COMPLETIONS | HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::complete_previous, code ) );
		case ( Replxx::ACTION::COMMIT_LINE ):                       return ( action( RESET_KILL_ACTION, &Replxx::ReplxxImpl::commit_line, code ) );
		case ( Replxx::ACTION::ABORT_LINE ):                        return ( action( RESET_KILL_ACTION | HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::abort_line, code ) );
		case ( Replxx::ACTION::SEND_EOF ):                          return ( action( HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::send_eof, code ) );
		case ( Replxx::ACTION::BRACKETED_PASTE ):                   return ( action( WANT_REFRESH | RESET_KILL_ACTION | HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::bracketed_paste, code ) );
	}
	return ( Replxx::ACTION_RESULT::BAIL );
}

void Replxx::ReplxxImpl::bind_key( char32_t code_, Replxx::key_press_handler_t handler_ ) {
	_keyPressHandlers[code_] = handler_;
}

void Replxx::ReplxxImpl::bind_key_internal( char32_t code_, char const* actionName_ ) {
	named_actions_t::const_iterator it( _namedActions.find( actionName_ ) );
	if ( it == _namedActions.end() ) {
		throw std::runtime_error( std::string( "replxx: Unknown action name: " ).append( actionName_ ) );
	}
	if ( !! it->second ) {
		bind_key( code_, it->second );
	}
}

Replxx::State Replxx::ReplxxImpl::get_state( void ) const {
	_utf8Buffer.assign( _data );
	return ( Replxx::State( _utf8Buffer.get(), _pos ) );
}

void Replxx::ReplxxImpl::set_state( Replxx::State const& state_ ) {
	_data.assign( state_.text() );
	if ( state_.cursor_position() >= 0 ) {
		_pos = min( state_.cursor_position(), _data.length() );
	}
	_modifiedState = true;
}

void Replxx::ReplxxImpl::set_ignore_case( bool val ) {
	_ignoreCase = val;
}

char32_t Replxx::ReplxxImpl::read_char( HINT_ACTION hintAction_ ) {
	/* try scheduled key presses */ {
		std::lock_guard<std::mutex> l( _mutex );
		if ( !_keyPresses.empty() ) {
			char32_t keyPress( _keyPresses.front() );
			_keyPresses.pop_front();
			return ( keyPress );
		}
	}
	int hintDelay(
		_refreshSkipped
			? static_cast<int>( RAPID_REFRESH_MS * 2 )
			: ( hintAction_ != HINT_ACTION::SKIP ? _hintDelay : 0 )
	);
	while ( true ) {
		Terminal::EVENT_TYPE eventType( _terminal.wait_for_input( hintDelay ) );
		if ( eventType == Terminal::EVENT_TYPE::TIMEOUT ) {
			refresh_line( _refreshSkipped ? HINT_ACTION::REGENERATE : HINT_ACTION::REPAINT );
			hintDelay = 0;
			_refreshSkipped = false;
			continue;
		}
		if ( eventType == Terminal::EVENT_TYPE::KEY_PRESS ) {
			break;
		}
		if ( eventType == Terminal::EVENT_TYPE::RESIZE ) {
			// caught a window resize event
			// now redraw the prompt and line
			_prompt.update_screen_columns();
			// redraw the original prompt with current input
			refresh_line( HINT_ACTION::REPAINT );
			continue;
		}

		std::lock_guard<std::mutex> l( _mutex );
		_terminal.set_cursor_visible( false );
		clear_self_to_end_of_screen();

		if ( _updatePrompt ) {
			// Update the prompt after the screen has been cleared and before it is redrawn
			_updatePrompt = false;
			std::string const updated = std::move( _asyncPrompt );
			_prompt.set_text( UnicodeString( updated ) );
		}

		while ( ! _messages.empty() ) {
			string const& message( _messages.front() );
			_terminal.write8( message.data(), static_cast<int>( message.length() ) );
			_messages.pop_front();
		}
		_lastRefreshTime = 0;
		repaint();
	}
	/* try scheduled key presses */ {
		std::lock_guard<std::mutex> l( _mutex );
		if ( !_keyPresses.empty() ) {
			char32_t keyPress( _keyPresses.front() );
			_keyPresses.pop_front();
			return ( keyPress );
		}
	}
	return ( _terminal.read_char() );
}

void Replxx::ReplxxImpl::clear( void ) {
	_pos = 0;
	_prefix = 0;
	_completions.clear();
	_completionContextLength = 0;
	_completionSelection = -1;
	_data.clear();
	_hintSelection = -1;
	_hint = UnicodeString();
	_display.clear();
	_displayInputLength = 0;
}

void Replxx::ReplxxImpl::call_modify_callback( void ) {
	if ( ! _modifyCallback ) {
		return;
	}
	_utf8Buffer.assign( _data );
	std::string origLine( _utf8Buffer.get() );
	int pos( _pos );
	std::string line( origLine );
	/* IOModeGuard scope */ {
		IOModeGuard ioModeGuard( _terminal );
		_modifyCallback( line, pos );
	}
	if ( ( pos != _pos ) || ( line != origLine ) ) {
		_data.assign( line.c_str() );
		_pos = min( pos, _data.length() );
		_modifiedState = true;
	}
}

Replxx::ReplxxImpl::completions_t Replxx::ReplxxImpl::call_completer( std::string const& input, int& contextLen_ ) const {
	Replxx::completions_t completionsIntermediary(
		!! _completionCallback
			? _completionCallback( input, contextLen_ )
			: Replxx::completions_t()
	);
	completions_t completions;
	completions.reserve( completionsIntermediary.size() );
	for ( Replxx::Completion const& c : completionsIntermediary ) {
		completions.emplace_back( c );
	}
	return ( completions );
}

Replxx::ReplxxImpl::hints_t Replxx::ReplxxImpl::call_hinter( std::string const& input, int& contextLen, Replxx::Color& color ) const {
	Replxx::hints_t hintsIntermediary(
		!! _hintCallback
			? _hintCallback( input, contextLen, color )
			: Replxx::hints_t()
	);
	hints_t hints;
	hints.reserve( hintsIntermediary.size() );
	for ( std::string const& h : hintsIntermediary ) {
		hints.emplace_back( h.c_str() );
	}
	return ( hints );
}

void Replxx::ReplxxImpl::set_preload_buffer( std::string const& preloadText ) {
	_preloadedBuffer = preloadText;
	// remove characters that won't display correctly
	bool controlsStripped = false;
	int whitespaceSeen( 0 );
	for ( std::string::iterator it( _preloadedBuffer.begin() ); it != _preloadedBuffer.end(); ) {
		unsigned char c = *it;
		if ( '\r' == c ) { // silently skip CR
			_preloadedBuffer.erase( it, it + 1 );
			continue;
		}
		if ( ( '\n' == c ) || ( '\t' == c ) ) { // note newline or tab
			++ whitespaceSeen;
			++ it;
			continue;
		}
		if ( whitespaceSeen > 0 ) {
			it -= whitespaceSeen;
			*it = ' ';
			_preloadedBuffer.erase( it + 1, it + whitespaceSeen - 1 );
		}
		if ( is_control_code( c ) ) { // remove other control characters, flag for message
			controlsStripped = true;
			if ( whitespaceSeen > 0 ) {
				_preloadedBuffer.erase( it, it + 1 );
				-- it;
			} else {
				*it = ' ';
			}
		}
		whitespaceSeen = 0;
		++ it;
	}
	if ( whitespaceSeen > 0 ) {
		std::string::iterator it = _preloadedBuffer.end() - whitespaceSeen;
		*it = ' ';
		if ( whitespaceSeen > 1 ) {
			_preloadedBuffer.erase( it + 1, _preloadedBuffer.end() );
		}
	}
	_errorMessage.clear();
	if ( controlsStripped ) {
		_errorMessage.assign( " [Edited line: control characters were converted to spaces]\n" );
	}
}

char const* Replxx::ReplxxImpl::read_from_stdin( void ) {
	if ( _preloadedBuffer.empty() ) {
		getline( cin, _preloadedBuffer );
		if ( ! cin.good() ) {
			return nullptr;
		}
	}
	while ( ! _preloadedBuffer.empty() && ( ( _preloadedBuffer.back() == '\r' ) || ( _preloadedBuffer.back() == '\n' ) ) ) {
		_preloadedBuffer.pop_back();
	}
	_utf8Buffer.assign( _preloadedBuffer );
	_preloadedBuffer.clear();
	return _utf8Buffer.get();
}

void Replxx::ReplxxImpl::emulate_key_press( char32_t keyCode_ ) {
	std::lock_guard<std::mutex> l( _mutex );
	_keyPresses.push_back( keyCode_ );
	if ( ( _currentThread != std::thread::id() ) && ( _currentThread != std::this_thread::get_id() ) ) {
		_terminal.notify_event( Terminal::EVENT_TYPE::KEY_PRESS );
	}
}

char const* Replxx::ReplxxImpl::input( std::string const& prompt ) {
	try {
		errno = 0;
		if ( ! tty::in ) { // input not from a terminal, we should work with piped input, i.e. redirected stdin
			return ( read_from_stdin() );
		}
		if ( ! _errorMessage.empty() ) {
			printf( "%s", _errorMessage.c_str() );
			fflush( stdout );
			_errorMessage.clear();
		}
		if ( isUnsupportedTerm() ) {
			fprintf( stdout, "%s", prompt.c_str() );
			fflush( stdout );
			return ( read_from_stdin() );
		}
		std::unique_lock<std::mutex> l( _mutex );
		if ( _terminal.enable_raw_mode() == -1 ) {
			return nullptr;
		}

		_asyncPrompt.clear();
		_updatePrompt = false;
		_prompt.set_text( UnicodeString( prompt ) );
		_currentThread = std::this_thread::get_id();
		l.unlock();
		clear();
		if ( !_preloadedBuffer.empty() ) {
			preload_puffer( _preloadedBuffer.c_str() );
			_preloadedBuffer.clear();
		}
		if ( get_input_line() == -1 ) {
			return ( finalize_input( nullptr ) );
		}
		_terminal.write8( "\n", 1 );
		_utf8Buffer.assign( _data );
		return ( finalize_input( _utf8Buffer.get() ) );
	} catch ( std::exception const& ) {
		return ( finalize_input( nullptr ) );
	}
}

char const* Replxx::ReplxxImpl::finalize_input( char const* retVal_ ) {
	std::unique_lock<std::mutex> l( _mutex );
	while ( ! _messages.empty() ) {
		string const& message( _messages.front() );
		l.unlock();
		_terminal.write8( message.data(), static_cast<int>( message.length() ) );
		l.lock();
		_messages.pop_front();
	}
	_currentThread = std::thread::id();
	_terminal.disable_raw_mode();
	return ( retVal_ );
}

int Replxx::ReplxxImpl::install_window_change_handler( void ) {
#ifndef _WIN32
	return ( _terminal.install_window_change_handler() );
#else
	return 0;
#endif
}

void Replxx::ReplxxImpl::enable_bracketed_paste( void ) {
	if ( _bracketedPaste ) {
		return;
	}
	_terminal.enable_bracketed_paste();
	_bracketedPaste = true;
}

void Replxx::ReplxxImpl::disable_bracketed_paste( void ) {
	if ( ! _bracketedPaste ) {
		return;
	}
	_terminal.disable_bracketed_paste();
	_bracketedPaste = false;
}

void Replxx::ReplxxImpl::print( char const* str_, int size_ ) {
	std::unique_lock<std::mutex> l( _mutex );
	if ( ( _currentThread == std::thread::id() ) || ( _currentThread == std::this_thread::get_id() ) ) {
#ifndef _WIN32
		l.unlock();
#endif
		_terminal.write8( str_, size_ );
	} else {
		_messages.emplace_back( str_, size_ );
		_terminal.notify_event( Terminal::EVENT_TYPE::MESSAGE );
	}
	return;
}

void Replxx::ReplxxImpl::set_prompt( std::string prompt ) {
	std::unique_lock<std::mutex> l( _mutex );
	if ( _currentThread == std::this_thread::get_id() ) {
		_prompt.set_text( UnicodeString( prompt ) );
		l.unlock();
		clear_self_to_end_of_screen();
		repaint();
	} else if ( _currentThread != std::thread::id() ) {
		_asyncPrompt = std::move( prompt );
		_updatePrompt = true;
		_terminal.notify_event( Terminal::EVENT_TYPE::MESSAGE );
	}
}

void Replxx::ReplxxImpl::preload_puffer(const char* preloadText) {
	_data.assign( preloadText );
	_prefix = _pos = _data.length();
}

void Replxx::ReplxxImpl::set_color( Replxx::Color color_ ) {
	char const* code( ansi_color( color_ ) );
	while ( *code ) {
		_display.push_back( *code );
		++ code;
	}
}

void Replxx::ReplxxImpl::indent( void ) {
	if ( ! _indentMultiline ) {
		return;
	}
	for ( int i( 0 ); i < _prompt.indentation(); ++ i ) {
		_display.push_back( ' ' );
	}
}

void Replxx::ReplxxImpl::render( char32_t ch ) {
	if ( ch == Replxx::KEY::ESCAPE ) {
		_display.push_back( '^' );
		_display.push_back( '[' );
	} else if ( is_control_code( ch ) && ( ch != '\n' ) ) {
		_display.push_back( '^' );
		_display.push_back( control_to_human( ch ) );
	} else {
		_display.push_back( ch );
	}
	if ( ch == '\n' ) {
		_hasNewlines = true;
		indent();
	}
	return;
}

void Replxx::ReplxxImpl::render( HINT_ACTION hintAction_ ) {
	if ( hintAction_ == HINT_ACTION::TRIM ) {
		_display.erase( _display.begin() + _displayInputLength, _display.end() );
		_modifiedState = false;
		return;
	}
	if ( hintAction_ == HINT_ACTION::SKIP ) {
		return;
	}
	_hasNewlines = false;
	_display.clear();
	if ( _noColor ) {
		for ( char32_t ch : _data ) {
			render( ch );
		}
		_displayInputLength = static_cast<int>( _display.size() );
		_modifiedState = false;
		return;
	}
	Replxx::colors_t colors( _data.length(), Replxx::Color::DEFAULT );
	_utf8Buffer.assign( _data );
	if ( !! _highlighterCallback ) {
		IOModeGuard ioModeGuard( _terminal );
		_highlighterCallback( _utf8Buffer.get(), colors );
	}
	paren_info_t pi( matching_paren() );
	Replxx::Color ERROR( Replxx::Color::RED | color::bg( Replxx::Color::BRIGHTRED ) );
	if ( pi.index != -1 ) {
		colors[pi.index] = pi.error ? ERROR : Replxx::Color::BRIGHTRED;
	}
	Replxx::Color c( Replxx::Color::DEFAULT );
	for ( int i( 0 ); i < _data.length(); ++ i ) {
		if ( colors[i] != c ) {
			c = colors[i];
			set_color( c );
		}
		render( _data[i] );
	}
	set_color( Replxx::Color::DEFAULT );
	_displayInputLength = static_cast<int>( _display.size() );
	_modifiedState = false;
	return;
}

void Replxx::ReplxxImpl::handle_hints( HINT_ACTION hintAction_ ) {
	if ( _noColor ) {
		return;
	}
	if ( ! _hintCallback ) {
		return;
	}
	if ( ( _hintDelay > 0 ) && ( hintAction_ != HINT_ACTION::REPAINT ) ) {
		_hintSelection = -1;
		return;
	}
	if ( ( hintAction_ == HINT_ACTION::SKIP ) || ( hintAction_ == HINT_ACTION::TRIM ) ) {
		return;
	}
	if ( _pos != _data.length() ) {
		return;
	}
	_hint = UnicodeString();
	if ( hintAction_ == HINT_ACTION::REGENERATE ) {
		_hintSelection = -1;
	}
	_utf8Buffer.assign( _data, _pos );
	if ( ( _utf8Buffer != _hintSeed ) || ( _hintContextLenght < 0 ) ) {
		_hintSeed.assign( _utf8Buffer );
		_hintContextLenght = context_length();
		_hintColor = Replxx::Color::GRAY;
		IOModeGuard ioModeGuard( _terminal );
		_hintsCache = call_hinter( _utf8Buffer.get(), _hintContextLenght, _hintColor );
	}
	int hintCount( static_cast<int>( _hintsCache.size() ) );
	if ( hintCount == 1 ) {
		_hint = _hintsCache.front();
		int len( _hint.length() - _hintContextLenght );
		if ( len > 0 ) {
			set_color( _hintColor );
			for ( int i( 0 ); i < len; ++ i ) {
				_display.push_back( _hint[i + _hintContextLenght] );
			}
			set_color( Replxx::Color::DEFAULT );
		}
	} else if ( ( _maxHintRows > 0 ) && ( hintCount > 0 ) ) {
		int posInLine( pos_in_line() );
		int startCol( ( _indentMultiline || ( posInLine == _pos ) ? _prompt.indentation() : 0 ) + posInLine );
		int maxCol( _prompt.screen_columns() );
#ifdef _WIN32
		-- maxCol;
#endif
		if ( _hintSelection < -1 ) {
			_hintSelection = hintCount - 1;
		} else if ( _hintSelection >= hintCount ) {
			_hintSelection = -1;
		}
		if ( _hintSelection != -1 ) {
			_hint = _hintsCache[_hintSelection];
			int len( min<int>( _hint.length(), maxCol - ( startCol - _hintContextLenght ) ) );
			if ( _hintContextLenght < len ) {
				set_color( _hintColor );
				for ( int i( _hintContextLenght ); i < len; ++ i ) {
					_display.push_back( _hint[i] );
				}
				set_color( Replxx::Color::DEFAULT );
			}
		}
		startCol -= _hintContextLenght;
		for ( int hintRow( 0 ); hintRow < min( hintCount, _maxHintRows ); ++ hintRow ) {
#ifdef _WIN32
			_display.push_back( '\r' );
#endif
			_display.push_back( '\n' );
			int col( 0 );
			for ( int i( 0 ); ( i < startCol ) && ( col < maxCol ); ++ i, ++ col ) {
				_display.push_back( ' ' );
			}
			set_color( _hintColor );
			int hintNo( hintRow + _hintSelection + 1 );
			if ( hintNo == hintCount ) {
				for ( int i( _pos - _hintContextLenght ); ( i < _pos ) && ( col < maxCol ); ++ i, ++ col ) {
					_display.push_back( _data[i] );
				}
				continue;
			} else if ( hintNo > hintCount ) {
				-- hintNo;
			}
			UnicodeString const& h( _hintsCache[hintNo % hintCount] );
			for ( int i( 0 ); ( i < h.length() ) && ( col < maxCol ); ++ i, ++ col ) {
				_display.push_back( h[i] );
			}
			set_color( Replxx::Color::DEFAULT );
		}
	}
	return;
}

// check for a matching brace/bracket/paren, remember its position if found
Replxx::ReplxxImpl::paren_info_t Replxx::ReplxxImpl::matching_paren( void ) {
	if (_pos >= _data.length()) {
		return ( paren_info_t{ -1, false } );
	}
	/* this scans for a brace matching _data[_pos] to highlight */
	unsigned char part1, part2;
	int scanDirection = 0;
	if ( strchr( "}])", _data[_pos] ) ) {
		scanDirection = -1; /* backwards */
		if (_data[_pos] == '}') {
			part1 = '}'; part2 = '{';
		} else if (_data[_pos] == ']') {
			part1 = ']'; part2 = '[';
		} else {
			part1 = ')'; part2 = '(';
		}
	} else if ( strchr( "{[(", _data[_pos] ) ) {
		scanDirection = 1; /* forwards */
		if (_data[_pos] == '{') {
			//part1 = '{'; part2 = '}';
			part1 = '}'; part2 = '{';
		} else if (_data[_pos] == '[') {
			//part1 = '['; part2 = ']';
			part1 = ']'; part2 = '[';
		} else {
			//part1 = '('; part2 = ')';
			part1 = ')'; part2 = '(';
		}
	} else {
		return ( paren_info_t{ -1, false } );
	}
	int highlightIdx = -1;
	bool indicateError = false;
	int unmatched = scanDirection;
	int unmatchedOther = 0;
	for (int i = _pos + scanDirection; i >= 0 && i < _data.length(); i += scanDirection) {
		/* TODO: the right thing when inside a string */
		if (strchr("}])", _data[i])) {
			if (_data[i] == part1) {
				--unmatched;
			} else {
				--unmatchedOther;
			}
		} else if (strchr("{[(", _data[i])) {
			if (_data[i] == part2) {
				++unmatched;
			} else {
				++unmatchedOther;
			}
		}

		if (unmatched == 0) {
			highlightIdx = i;
			indicateError = (unmatchedOther != 0);
			break;
		}
	}
	return ( paren_info_t{ highlightIdx, indicateError } );
}

int Replxx::ReplxxImpl::virtual_render( char32_t const* buffer_, int len_, int& xPos_, int& yPos_, Prompt const* prompt_ ) {
	Prompt const& prompt( prompt_ ? *prompt_ : _prompt );
	return ( replxx::virtual_render( buffer_, len_, xPos_, yPos_, prompt.screen_columns(), _indentMultiline ? prompt.indentation() : 0 ) );
}

/**
 * Refresh the user's input line: the prompt is already onscreen and is not
 * redrawn here screen position
 */
void Replxx::ReplxxImpl::refresh_line( HINT_ACTION hintAction_ ) {
	int long long now( now_us() );
	int long long duration( now - _lastRefreshTime );
	if ( duration < RAPID_REFRESH_US ) {
		_lastRefreshTime = now;
		_refreshSkipped = true;
		return;
	}
	_refreshSkipped = false;
	render( hintAction_ );
	handle_hints( hintAction_ );
	// calculate the desired position of the cursor
	int xCursorPos( _prompt.indentation() );
	int yCursorPos( 0 );
	virtual_render( _data.get(), _pos, xCursorPos, yCursorPos );

	// calculate the position of the end of the input line
	int xEndOfInput( _prompt.indentation() );
	int yEndOfInput( 0 );
	// _data part of _display already contains the indent,
	// also newlines belonging to hints part of display shall be ignored
	// with respect to extra indent for multiline inputs
	// in other words _display should not be re-indented
	replxx::virtual_render( _display.data(), static_cast<int>( _display.size() ), xEndOfInput, yEndOfInput, _prompt.screen_columns(), 0 );

	// position at the end of the prompt, clear to end of previous input
	_terminal.set_cursor_visible( false );
	_terminal.jump_cursor(
		_prompt.indentation(), // 0-based on Win32
		-( _prompt._cursorRowOffset - _prompt._extraLines )
	);
	// display the input line
	if ( _hasNewlines ) {
		_terminal.clear_screen( Terminal::CLEAR_SCREEN::TO_END );
		_terminal.write32( _display.data(), static_cast<int>( _display.size() ) );
	} else {
		_terminal.write32( _display.data(), _displayInputLength );
		_terminal.clear_screen( Terminal::CLEAR_SCREEN::TO_END );
		_terminal.write32( _display.data() + _displayInputLength, static_cast<int>( _display.size() ) - _displayInputLength );
	}
#ifndef _WIN32
	// we have to generate our own newline on line wrap
	if ( ( xEndOfInput == 0 ) && ( yEndOfInput > 0 ) && ! _data.is_empty() && ( _data.back() != '\n' ) ) {
		_terminal.write8( "\n", 1 );
	}
#endif
	// position the cursor
	_terminal.jump_cursor( xCursorPos, -( yEndOfInput - yCursorPos ) );
	_terminal.set_cursor_visible( true );
	_prompt._cursorRowOffset = _prompt._extraLines + yCursorPos; // remember row for next pass
	_lastRefreshTime = now_us();
	_oldPos = _pos;
	_moveCursor = false;
}

void Replxx::ReplxxImpl::move_cursor( void ) {
	// calculate the desired position of the cursor
	int xCursorPos( _prompt.indentation() );
	int yCursorPos( 0 );
	virtual_render( _data.get(), _pos, xCursorPos, yCursorPos );
	// position the cursor
	_terminal.jump_cursor( xCursorPos, -( _prompt._cursorRowOffset - _prompt._extraLines - yCursorPos ) );
	_prompt._cursorRowOffset = _prompt._extraLines + yCursorPos;
	_oldPos = _pos;
	_moveCursor = false;
}

int Replxx::ReplxxImpl::context_length() {
	int prefixLength = _pos;
	while ( prefixLength > 0 ) {
		if ( is_word_break_character<false>( _data[prefixLength - 1] ) ) {
			break;
		}
		-- prefixLength;
	}
	return ( _pos - prefixLength );
}

void Replxx::ReplxxImpl::repaint( void ) {
	_prompt.write();
	for ( int i( _prompt._extraLines ); i < _prompt._cursorRowOffset; ++ i ) {
		_terminal.write8( "\n", 1 );
	}
	refresh_line( HINT_ACTION::SKIP );
}

void Replxx::ReplxxImpl::clear_self_to_end_of_screen( Prompt const* prompt_ ) {
	// position at the start of the prompt, clear to end of previous input
	_terminal.jump_cursor( 0, prompt_ ? -prompt_->_cursorRowOffset : -_prompt._cursorRowOffset );
	_terminal.clear_screen( Terminal::CLEAR_SCREEN::TO_END );
	return;
}

namespace {

int longest_common_prefix( Replxx::ReplxxImpl::completions_t const& completions, bool ignoreCase ) {
	int completionsCount( static_cast<int>( completions.size() ) );
	if ( completionsCount < 1 ) {
		return ( 0 );
	}
	int longestCommonPrefix( 0 );
	UnicodeString const& sample( completions.front().text() );
	while ( true ) {
		if ( longestCommonPrefix >= sample.length() ) {
			return ( longestCommonPrefix );
		}
		char32_t sc( sample[longestCommonPrefix] );
		for ( int i( 1 ); i < completionsCount; ++ i ) {
			UnicodeString const& candidate( completions[i].text() );
			if ( longestCommonPrefix >= candidate.length() ) {
				return ( longestCommonPrefix );
			}
			char32_t cc( candidate[longestCommonPrefix] );
			if ( ignoreCase ) {
				if ( !case_insensitive_equal( cc, sc ) ) {
					return longestCommonPrefix;
				}
			} else if ( cc != sc ) {
				return longestCommonPrefix;
			}
		}
		++ longestCommonPrefix;
	}
}

}

/**
 * Handle command completion, using a completionCallback() routine to provide
 * possible substitutions
 * This routine handles the mechanics of updating the user's input buffer with
 * possible replacement of text as the user selects a proposed completion string,
 * or cancels the completion attempt.
 * @param pi - Prompt struct holding information about the prompt and our
 * screen position
 */
char32_t Replxx::ReplxxImpl::do_complete_line( bool showCompletions_ ) {
	char32_t c = 0;

	// completionCallback() expects a parsable entity, so find the previous break
	// character and
	// extract a copy to parse.	we also handle the case where tab is hit while
	// not at end-of-line.

	_utf8Buffer.assign( _data, _pos );
	// get a list of completions
	_completionSelection = -1;
	_completionContextLength = context_length();
	/* IOModeGuard scope */ {
		IOModeGuard ioModeGuard( _terminal );
		_completions = call_completer( _utf8Buffer.get(), _completionContextLength );
	}

	// if no completions, we are done
	if ( _completions.empty() ) {
		beep();
		return 0;
	}

	// at least one completion
	int longestCommonPrefix = 0;
	int completionsCount( static_cast<int>( _completions.size() ) );
	int selectedCompletion( 0 );
	if ( ( completionsCount > 1 ) && ( _hintSelection != -1 ) ) {
		selectedCompletion = _hintSelection;
		completionsCount = 1;
	}
	if ( completionsCount == 1 ) {
		longestCommonPrefix = static_cast<int>( _completions[selectedCompletion].text().length() );
	} else {
		bool ignoreCase( _ignoreCase && std::none_of( _data.end() - _completionContextLength, _data.end(), []( char32_t x ) { return iswupper( static_cast<wint_t>( x ) ); } ) );
		longestCommonPrefix = longest_common_prefix( _completions, ignoreCase );
	}
	if ( _beepOnAmbiguousCompletion && ( completionsCount != 1 ) ) { // beep if ambiguous
		beep();
	}

	// if we can extend the item, extend it and return to main loop
	if ( ( longestCommonPrefix > _completionContextLength ) || ( completionsCount == 1 ) ) {
		UnicodeString const* cand( &_completions[selectedCompletion].text() );
		if ( _ignoreCase && ( _hintSelection == -1 ) ) {
			for ( int i( 0 ); i < completionsCount; ++ i ) {
				if ( _completions[i].text() < *cand ) {
					cand = &_completions[i].text();
				}
			}
		}
		_pos -= _completionContextLength;
		_data.erase( _pos, _completionContextLength );
		_data.insert( _pos, *cand, 0, longestCommonPrefix );
		_completionContextLength = longestCommonPrefix;
		if ( _ignoreCase && ( completionsCount > 1 ) ) {
			for ( int i( 0 ); i < longestCommonPrefix; ++ i ) {
				_data[_pos + i] = static_cast<char32_t>( towlower( static_cast<wint_t>( _data[_pos + i] ) ) );
			}
		}
		_pos += _completionContextLength;
		refresh_line();
		return 0;
	}

	if ( ! showCompletions_ ) {
		return ( 0 );
	}

	if ( _doubleTabCompletion ) {
		// we can't complete any further, wait for second tab
		do {
			c = read_char();
		} while ( c == static_cast<char32_t>( -1 ) );

		// if any character other than tab, pass it to the main loop
		if ( c != Replxx::KEY::TAB ) {
			return c;
		}
	}

	// we got a second tab, maybe show list of possible completions
	bool showCompletions = true;
	bool onNewLine = false;
	if ( static_cast<int>( _completions.size() ) > _completionCountCutoff ) {
		int savePos = _pos; // move cursor to EOL to avoid overwriting the command line
		_pos = _data.length();
		refresh_line();
		_pos = savePos;
		printf( "\nDisplay all %u possibilities? (y or n)", static_cast<unsigned int>( _completions.size() ) );
		fflush(stdout);
		onNewLine = true;
		while (c != 'y' && c != 'Y' && c != 'n' && c != 'N' && c != Replxx::KEY::control('C')) {
			do {
				c = read_char();
			} while (c == static_cast<char32_t>(-1));
		}
		switch (c) {
			case 'n':
			case 'N':
				showCompletions = false;
				break;
			case Replxx::KEY::control('C'):
				showCompletions = false;
				// Display the ^C we got
				_terminal.write8( "^C", 2 );
				c = 0;
				break;
		}
	}

	// if showing the list, do it the way readline does it
	bool stopList( false );
	if ( showCompletions ) {
		int longestCompletion( 0 );
		for ( size_t j( 0 ); j < _completions.size(); ++ j ) {
			int itemLength( static_cast<int>( _completions[j].text().length() ) );
			if ( itemLength > longestCompletion ) {
				longestCompletion = itemLength;
			}
		}
		longestCompletion += 2;
		int columnCount = _prompt.screen_columns() / longestCompletion;
		if ( columnCount < 1 ) {
			columnCount = 1;
		}
		if ( ! onNewLine ) {  // skip this if we showed "Display all %d possibilities?"
			int savePos = _pos; // move cursor to EOL to avoid overwriting the command line
			_pos = _data.length();
			refresh_line( HINT_ACTION::TRIM );
			_pos = savePos;
		} else {
			_terminal.clear_screen( Terminal::CLEAR_SCREEN::TO_END );
		}
		size_t pauseRow = _terminal.get_screen_rows() - 1;
		size_t rowCount = (_completions.size() + columnCount - 1) / columnCount;
		for (size_t row = 0; row < rowCount; ++row) {
			if (row == pauseRow) {
				printf("\n--More--");
				fflush(stdout);
				c = 0;
				bool doBeep = false;
				while (c != ' ' && c != Replxx::KEY::ENTER && c != 'y' && c != 'Y' &&
							 c != 'n' && c != 'N' && c != 'q' && c != 'Q' &&
							 c != Replxx::KEY::control('C')) {
					if (doBeep) {
						beep();
					}
					doBeep = true;
					do {
						c = read_char();
					} while (c == static_cast<char32_t>(-1));
				}
				switch (c) {
					case ' ':
					case 'y':
					case 'Y':
						printf("\r				\r");
						pauseRow += _terminal.get_screen_rows() - 1;
						break;
					case Replxx::KEY::ENTER:
						printf("\r				\r");
						++pauseRow;
						break;
					case 'n':
					case 'N':
					case 'q':
					case 'Q':
						printf("\r				\r");
						stopList = true;
						break;
					case Replxx::KEY::control('C'):
						// Display the ^C we got
						_terminal.write8( "^C", 2 );
						stopList = true;
						break;
				}
			} else {
				_terminal.write8( "\n", 1 );
			}
			if (stopList) {
				break;
			}
			static UnicodeString const res( ansi_color( Replxx::Color::DEFAULT ) );
			for (int column = 0; column < columnCount; ++column) {
				size_t index = (column * rowCount) + row;
				if ( index < _completions.size() ) {
					Completion const& c( _completions[index] );
					int itemLength = static_cast<int>(c.text().length());
					fflush(stdout);

					if ( longestCommonPrefix > 0 ) {
						static UnicodeString const col( ansi_color( Replxx::Color::BRIGHTMAGENTA ) );
						if (!_noColor) {
							_terminal.write32(col.get(), col.length());
						}
						_terminal.write32(c.text().get(), longestCommonPrefix);
						if (!_noColor) {
							_terminal.write32(res.get(), res.length());
						}
					}

					if ( !_noColor && ( c.color() != Replxx::Color::DEFAULT ) ) {
						UnicodeString ac( ansi_color( c.color() ) );
						_terminal.write32( ac.get(), ac.length() );
					}
					_terminal.write32( c.text().get() + longestCommonPrefix, itemLength - longestCommonPrefix );
					if ( !_noColor && ( c.color() != Replxx::Color::DEFAULT ) ) {
						_terminal.write32( res.get(), res.length() );
					}

					if ( ((column + 1) * rowCount) + row < _completions.size() ) {
						for ( int k( itemLength ); k < longestCompletion; ++k ) {
							printf( " " );
						}
					}
				}
			}
		}
		fflush(stdout);
	}

	// display the prompt on a new line, then redisplay the input buffer
	if (!stopList || c == Replxx::KEY::control('C')) {
		_terminal.write8( "\n", 1 );
	}
	_prompt.write();
	_prompt._cursorRowOffset = _prompt._extraLines;
	refresh_line();
	return 0;
}

int Replxx::ReplxxImpl::get_input_line( void ) {
	// The latest history entry is always our current buffer
	if ( _data.length() > 0 ) {
		_history.add( _data );
	} else {
		_history.add( UnicodeString() );
	}
	_history.jump( false, false );

	// display the prompt
	_prompt.write();

	// the cursor starts out at the end of the prompt
	_prompt._cursorRowOffset = _prompt._extraLines;

	// kill and yank start in "other" mode
	_killRing.lastAction = KillRing::actionOther;

	// if there is already text in the buffer, display it first
	if (_data.length() > 0) {
		refresh_line();
	}

	// loop collecting characters, respond to line editing characters
	Replxx::ACTION_RESULT next( Replxx::ACTION_RESULT::CONTINUE );
	while ( next == Replxx::ACTION_RESULT::CONTINUE ) {
		int c( read_char( HINT_ACTION::REPAINT ) ); // get a new keystroke

		if (c == 0) {
			return _data.length();
		}

		if (c == -1) {
			refresh_line();
			continue;
		}

		if (c == -2) {
			_prompt.write();
			refresh_line();
			continue;
		}

		key_press_handlers_t::iterator it( _keyPressHandlers.find( c ) );
		if ( it != _keyPressHandlers.end() ) {
			next = it->second( c );
			if ( _modifiedState ) {
				refresh_line();
			} else if ( _moveCursor ) {
				move_cursor();
			}
		} else {
			next = action( RESET_KILL_ACTION | HISTORY_RECALL_MOST_RECENT, &Replxx::ReplxxImpl::insert_character, c );
		}
	}
	return ( next == Replxx::ACTION_RESULT::RETURN ? _data.length() : -1 );
}

Replxx::ACTION_RESULT Replxx::ReplxxImpl::action( action_trait_t actionTrait_, key_press_handler_raw_t const& handler_, char32_t code_ ) {
	Replxx::ACTION_RESULT res( ( this->*handler_ )( code_ ) );
	call_modify_callback();
	if ( actionTrait_ & HISTORY_RECALL_MOST_RECENT ) {
		_history.reset_recall_most_recent();
	}
	if ( actionTrait_ & RESET_KILL_ACTION ) {
		_killRing.lastAction = KillRing::actionOther;
	}
	if ( actionTrait_ & SET_KILL_ACTION ) {
		_killRing.lastAction = KillRing::actionKill;
	}
	if ( ! ( actionTrait_ & DONT_RESET_PREFIX ) ) {
		_prefix = _pos;
	}
	if ( ! ( actionTrait_ & DONT_RESET_COMPLETIONS ) ) {
		_completions.clear();
		_completionSelection = -1;
		_completionContextLength = 0;
	}
	if ( ! ( actionTrait_ & DONT_RESET_HIST_YANK_INDEX ) ) {
		_history.reset_yank_iterator();
	}
	if ( actionTrait_ & WANT_REFRESH ) {
		_modifiedState = true;
	}
	if ( actionTrait_ & MOVE_CURSOR ) {
		_modifiedState = ( _pos != _oldPos ) && (
			( _pos == _data.length() )
			|| ( _oldPos == _data.length() )
			|| ( ( _pos < _data.length() ) && strchr( "{}[]()", _data[_pos] ) )
			|| ( ( _oldPos < _data.length() ) && strchr( "{}[]()", _data[_oldPos] ) )
		);
		_moveCursor = _pos != _oldPos;
	}
	return ( res );
}

Replxx::ACTION_RESULT Replxx::ReplxxImpl::insert_character( char32_t c ) {
	/*
	 * beep on unknown Ctrl and/or Meta keys
	 * don't insert control characters
	 */
	if ( ( c >= static_cast<int>( Replxx::KEY::BASE ) ) || ( is_control_code( c ) && ( c != '\n' ) ) ) {
		beep();
		return ( Replxx::ACTION_RESULT::CONTINUE );
	}
	if ( ! _overwrite || ( _pos >= _data.length() ) ) {
		_data.insert( _pos, c );
	} else {
		_data[_pos] = c;
	}
	_oldPos = _pos;
	++ _pos;
	call_modify_callback();
	int long long now( now_us() );
	int long long duration( now - _lastRefreshTime );
	if ( duration < RAPID_REFRESH_US ) {
		_lastRefreshTime = now;
		_refreshSkipped = true;
		return ( Replxx::ACTION_RESULT::CONTINUE );
	}
	int xCursorPos( _prompt.indentation() );
	int yCursorPos( 0 );
	virtual_render( _data.get(), _data.length(), xCursorPos, yCursorPos );
	if (
		( _pos == _data.length() )
		&& ! _modifiedState
		&& ( _noColor || ! ( !! _highlighterCallback || !! _hintCallback ) )
		&& ( yCursorPos == 0 )
	) {
		/* Avoid a full assign of the line in the
		 * trivial case. */
		render( c );
		_displayInputLength = static_cast<int>( _display.size() );
		_terminal.write32( reinterpret_cast<char32_t*>( &c ), 1 );
	} else {
		refresh_line();
	}
	_lastRefreshTime = now_us();
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// ctrl-J/linefeed/newline
Replxx::ACTION_RESULT Replxx::ReplxxImpl::new_line( char32_t ) {
	return ( insert_character( '\n' ) );
}

// ctrl-A, HOME: move cursor to start of line
Replxx::ACTION_RESULT Replxx::ReplxxImpl::go_to_begining_of_line( char32_t char_ ) {
	if ( _hasNewlines ) {
		bool onNewline( ( _pos > 0 ) && ( _pos < _data.length() ) && ( _data[_pos] == '\n' ) );
		int startPos( onNewline ? _pos - 1 : _pos );
		int newPos( prev_newline_position( startPos ) + 1 );
		_pos = ( newPos == _pos ) && ( char_ == Replxx::KEY::control( 'A' ) ) ? 0 : newPos;
	} else {
		_pos = 0;
	}
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

Replxx::ACTION_RESULT Replxx::ReplxxImpl::go_to_end_of_line( char32_t char_ ) {
	if ( _hasNewlines ) {
		int newPos( next_newline_position( _pos ) );
		_pos = ( newPos < 0 ) || ( ( newPos == _pos ) && ( char_ == Replxx::KEY::control( 'E' ) ) ) ? _data.length() : newPos;
	} else {
		_pos = _data.length();
	}
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// ctrl-B, move cursor left by one character
Replxx::ACTION_RESULT Replxx::ReplxxImpl::move_one_char_left( char32_t ) {
	if (_pos > 0) {
		--_pos;
	}
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// ctrl-F, move cursor right by one character
Replxx::ACTION_RESULT Replxx::ReplxxImpl::move_one_char_right( char32_t ) {
	if ( _pos < _data.length() ) {
		++_pos;
	}
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// meta-B, move cursor left by one word
template <bool subword>
Replxx::ACTION_RESULT Replxx::ReplxxImpl::move_one_word_left( char32_t ) {
	if (_pos > 0) {
		while (_pos > 0 && is_word_break_character<subword>( _data[_pos - 1] ) ) {
			--_pos;
		}
		while (_pos > 0 && !is_word_break_character<subword>( _data[_pos - 1] ) ) {
			--_pos;
		}
	}
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// meta-f, move cursor right by one word
template <bool subword>
Replxx::ACTION_RESULT Replxx::ReplxxImpl::move_one_word_right( char32_t ) {
	if ( _pos < _data.length() ) {
		while ( _pos < _data.length() && is_word_break_character<subword>( _data[_pos] ) ) {
			++_pos;
		}
		while ( _pos < _data.length() && !is_word_break_character<subword>( _data[_pos] ) ) {
			++_pos;
		}
	}
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// meta-Backspace, kill word to left of cursor
template <bool subword>
Replxx::ACTION_RESULT Replxx::ReplxxImpl::kill_word_to_left( char32_t ) {
	if ( _pos > 0 ) {
		int startingPos = _pos;
		while ( _pos > 0 && is_word_break_character<subword>( _data[_pos - 1] ) ) {
			-- _pos;
		}
		while ( _pos > 0 && !is_word_break_character<subword>( _data[_pos - 1] ) ) {
			-- _pos;
		}
		_killRing.kill( _data.get() + _pos, startingPos - _pos, false);
		_data.erase( _pos, startingPos - _pos );
		refresh_line();
	}
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// meta-D, kill word to right of cursor
template <bool subword>
Replxx::ACTION_RESULT Replxx::ReplxxImpl::kill_word_to_right( char32_t ) {
	if ( _pos < _data.length() ) {
		int endingPos = _pos;
		while ( endingPos < _data.length() && is_word_break_character<subword>( _data[endingPos] ) ) {
			++ endingPos;
		}
		while ( endingPos < _data.length() && !is_word_break_character<subword>( _data[endingPos] ) ) {
			++ endingPos;
		}
		_killRing.kill( _data.get() + _pos, endingPos - _pos, true );
		_data.erase( _pos, endingPos - _pos );
		refresh_line();
	}
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// ctrl-W, kill to whitespace (not word) to left of cursor
Replxx::ACTION_RESULT Replxx::ReplxxImpl::kill_to_whitespace_to_left( char32_t ) {
	if ( _pos > 0 ) {
		int startingPos = _pos;
		while ( ( _pos > 0 ) && isspace( _data[_pos - 1] ) ) {
			--_pos;
		}
		while ( ( _pos > 0 ) && ! isspace( _data[_pos - 1] ) ) {
			-- _pos;
		}
		_killRing.kill( _data.get() + _pos, startingPos - _pos, false );
		_data.erase( _pos, startingPos - _pos );
		refresh_line();
	}
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// ctrl-K, kill from cursor to end of line
Replxx::ACTION_RESULT Replxx::ReplxxImpl::kill_to_end_of_line( char32_t ) {
	int to( _data.length() );
	if ( _hasNewlines ) {
		to = next_newline_position( _pos );
		if ( ( to < 0 ) || ( to == _pos ) ) {
			to = _data.length();
		}
	}

	_killRing.kill( _data.get() + _pos, to - _pos, true );
	_data.erase( _pos, to - _pos );
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// ctrl-U, kill all characters to the left of the cursor
Replxx::ACTION_RESULT Replxx::ReplxxImpl::kill_to_begining_of_line( char32_t ) {
	if (_pos <= 0) {
		return ( Replxx::ACTION_RESULT::CONTINUE );
	}
	int newPos( 0 );
	if ( _hasNewlines ) {
		bool onNewline( ( _pos > 0 ) && ( _pos < _data.length() ) && ( _data[_pos] == '\n' ) );
		int startPos( onNewline ? _pos - 1 : _pos );
		newPos = prev_newline_position( startPos ) + 1;
		if ( newPos == _pos ) {
			newPos = 0;
		}
	}
	_killRing.kill( _data.get() + newPos, _pos - newPos, false );
	_data.erase( newPos, _pos - newPos );
	_pos = newPos;
	refresh_line();
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// ctrl-Y, yank killed text
Replxx::ACTION_RESULT Replxx::ReplxxImpl::yank( char32_t ) {
	UnicodeString* restoredText( _killRing.yank() );
	if ( restoredText ) {
		_data.insert( _pos, *restoredText, 0, restoredText->length() );
		_pos += restoredText->length();
		refresh_line();
		_killRing.lastAction = KillRing::actionYank;
		_lastYankSize = restoredText->length();
	} else {
		beep();
	}
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// meta-Y, "yank-pop", rotate popped text
Replxx::ACTION_RESULT Replxx::ReplxxImpl::yank_cycle( char32_t ) {
	if ( _killRing.lastAction != KillRing::actionYank ) {
		beep();
		return ( Replxx::ACTION_RESULT::CONTINUE );
	}
	UnicodeString* restoredText = _killRing.yankPop();
	if ( !restoredText ) {
		beep();
		return ( Replxx::ACTION_RESULT::CONTINUE );
	}
	_pos -= _lastYankSize;
	_data.erase( _pos, _lastYankSize );
	_data.insert( _pos, *restoredText, 0, restoredText->length() );
	_pos += restoredText->length();
	_lastYankSize = restoredText->length();
	refresh_line();
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// meta-., "yank-last-arg", on consecutive uses move back in history for popped text
Replxx::ACTION_RESULT Replxx::ReplxxImpl::yank_last_arg( char32_t ) {
	if ( _history.size() < 2 ) {
		return ( Replxx::ACTION_RESULT::CONTINUE );
	}
	if ( _history.next_yank_position() ) {
		_lastYankSize = 0;
	}
	UnicodeString const& histLine( _history.yank_line() );
	int endPos( histLine.length() );
	while ( ( endPos > 0 ) && isspace( histLine[endPos - 1] ) ) {
		-- endPos;
	}
	int startPos( endPos );
	while ( ( startPos > 0 ) && ! isspace( histLine[startPos - 1] ) ) {
		-- startPos;
	}
	_pos -= _lastYankSize;
	_data.erase( _pos, _lastYankSize );
	_lastYankSize = endPos - startPos;
	_data.insert( _pos, histLine, startPos, _lastYankSize );
	_pos += _lastYankSize;
	refresh_line();
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// meta-C, give word initial Cap
template <bool subword>
Replxx::ACTION_RESULT Replxx::ReplxxImpl::capitalize_word( char32_t ) {
	if (_pos < _data.length()) {
		while ( _pos < _data.length() && is_word_break_character<subword>( _data[_pos] ) ) {
			++_pos;
		}
		if (_pos < _data.length() && !is_word_break_character<subword>( _data[_pos] ) ) {
			if ( iswlower( static_cast<wint_t>( _data[_pos] ) ) ) {
				_data[_pos] = static_cast<char32_t>( towupper( static_cast<wint_t>( _data[_pos] ) ) );
			}
			++_pos;
		}
		while (_pos < _data.length() && !is_word_break_character<subword>( _data[_pos] ) ) {
			if ( iswupper( static_cast<wint_t>( _data[_pos] ) ) ) {
				_data[_pos] = static_cast<char32_t>( towlower( static_cast<wint_t>( _data[_pos] ) ) );
			}
			++_pos;
		}
		refresh_line();
	}
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// meta-L, lowercase word
template <bool subword>
Replxx::ACTION_RESULT Replxx::ReplxxImpl::lowercase_word( char32_t ) {
	if (_pos < _data.length()) {
		while ( _pos < _data.length() && is_word_break_character<subword>( _data[_pos] ) ) {
			++ _pos;
		}
		while (_pos < _data.length() && !is_word_break_character<subword>( _data[_pos] ) ) {
			if ( iswupper( static_cast<wint_t>( _data[_pos] ) ) ) {
				_data[_pos] = static_cast<char32_t>( towlower( static_cast<wint_t>( _data[_pos] ) ) );
			}
			++ _pos;
		}
		refresh_line();
	}
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// meta-U, uppercase word
template <bool subword>
Replxx::ACTION_RESULT Replxx::ReplxxImpl::uppercase_word( char32_t ) {
	if (_pos < _data.length()) {
		while ( _pos < _data.length() && is_word_break_character<subword>( _data[_pos] ) ) {
			++ _pos;
		}
		while ( _pos < _data.length() && !is_word_break_character<subword>( _data[_pos] ) ) {
			if ( iswlower( static_cast<wint_t>( _data[_pos] ) ) ) {
				_data[_pos] = static_cast<char32_t>( towupper( static_cast<wint_t>( _data[_pos] ) ) );
			}
			++ _pos;
		}
		refresh_line();
	}
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// ctrl-T, transpose characters
Replxx::ACTION_RESULT Replxx::ReplxxImpl::transpose_characters( char32_t ) {
	if ( _pos > 0 && _data.length() > 1 ) {
		size_t leftCharPos = ( _pos == _data.length() ) ? _pos - 2 : _pos - 1;
		char32_t aux = _data[leftCharPos];
		_data[leftCharPos] = _data[leftCharPos + 1];
		_data[leftCharPos + 1] = aux;
		if ( _pos != _data.length() ) {
			++_pos;
		}
		refresh_line();
	}
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// ctrl-C, abort this line
Replxx::ACTION_RESULT Replxx::ReplxxImpl::abort_line( char32_t keyCode_ ) {
	errno = EAGAIN;
	_history.drop_last();
	// we need one last refresh with the cursor at the end of the line
	// so we don't display the next prompt over the previous input line
	_pos = _data.length(); // pass _data.length() as _pos for EOL
	_lastRefreshTime = 0;
	refresh_line( _refreshSkipped ? HINT_ACTION::REGENERATE : HINT_ACTION::TRIM );
	if ( keyCode_ == Replxx::KEY::control( 'C' ) ) {
		_terminal.write8( "^C\r\n", 4 );
	}
	return ( Replxx::ACTION_RESULT::BAIL );
}

// DEL, delete the character under the cursor
Replxx::ACTION_RESULT Replxx::ReplxxImpl::delete_character( char32_t ) {
	if ( ( _data.length() > 0 ) && ( _pos < _data.length() ) ) {
		_data.erase( _pos );
		refresh_line();
	}
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// ctrl-D, delete the character under the cursor
// on an empty line, exit the shell
Replxx::ACTION_RESULT Replxx::ReplxxImpl::send_eof( char32_t key_ ) {
	if ( _data.length() == 0 ) {
		_history.drop_last();
		return ( Replxx::ACTION_RESULT::BAIL );
	}
	return ( delete_character( key_ ) );
}

// backspace/ctrl-H, delete char to left of cursor
Replxx::ACTION_RESULT Replxx::ReplxxImpl::backspace_character( char32_t ) {
	if ( _pos > 0 ) {
		-- _pos;
		_data.erase( _pos );
		refresh_line();
	}
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// ctrl-M/return/enter, accept line
Replxx::ACTION_RESULT Replxx::ReplxxImpl::commit_line( char32_t ) {
	// we need one last refresh with the cursor at the end of the line
	// so we don't display the next prompt over the previous input line
	_pos = _data.length(); // pass _data.length() as _pos for EOL
	_lastRefreshTime = 0;
	refresh_line( _refreshSkipped ? HINT_ACTION::REGENERATE : HINT_ACTION::TRIM );
	_history.commit_index();
	_history.drop_last();
	return ( Replxx::ACTION_RESULT::RETURN );
}

int Replxx::ReplxxImpl::prev_newline_position( int pos_ ) const {
	assert( ( pos_ >= 0 ) && ( pos_ <= _data.length() ) );
	if ( pos_ == _data.length() ) {
		-- pos_;
	}
	while ( pos_ >= 0 ) {
		if ( _data[pos_] == '\n' ) {
			break;
		}
		-- pos_;
	}
	return ( pos_ );
}

int Replxx::ReplxxImpl::next_newline_position( int pos_ ) const {
	assert( ( pos_ >= 0 ) && ( pos_ <= _data.length() ) );
	int len( _data.length() );
	while ( pos_ < len ) {
		if ( _data[pos_] == '\n' ) {
			break;
		}
		++ pos_;
	}
	return ( pos_ < len ? pos_ : -1 );
}

int Replxx::ReplxxImpl::pos_in_line( void ) const {
	if ( ! _hasNewlines ) {
		return ( _pos );
	}
	int lineStart( prev_newline_position( _pos ) + 1 );
	return ( _pos - lineStart );
}

// Up, recall previous line in history
Replxx::ACTION_RESULT Replxx::ReplxxImpl::line_previous( char32_t ) {
	assert( ( _pos >= 0 ) && ( _pos <= _data.length() ) );
	do {
		if ( ! _hasNewlines ) {
			break;
		}
		int prevNewlinePosition( prev_newline_position( _pos ) );
		if ( prevNewlinePosition == _pos ) {
			prevNewlinePosition = prev_newline_position( _pos - 1 );
		}
		if ( prevNewlinePosition < 0 ) {
			break;
		}
		int posInLine( _pos - prevNewlinePosition - 1 );
		int prevLineStart( prevNewlinePosition > 0 ? prev_newline_position( prevNewlinePosition - 1 ) + 1 : 0 );
		int prevLineLength( max( prevNewlinePosition - prevLineStart, 0 ) );
		int shift( ! _indentMultiline && ( prevLineStart == 0 ) ? _prompt.indentation() : 0 );
		posInLine = max( min( posInLine, prevLineLength + shift ) - shift, 0 );
		_pos = prevLineStart + posInLine;
		assert( ( _pos >= 0 ) && ( _pos <= _data.length() ) );
		return ( Replxx::ACTION_RESULT::CONTINUE );
	} while ( false );
	return ( history_move( true ) );
}

// Down, recall next line in history
Replxx::ACTION_RESULT Replxx::ReplxxImpl::line_next( char32_t ) {
	assert( ( _pos >= 0 ) && ( _pos <= _data.length() ) );
	do {
		if ( ! _hasNewlines ) {
			break;
		}
		int nextNewlinePosition( next_newline_position( _pos ) );
		if ( nextNewlinePosition < 0 ) {
			break;
		}
		int nextLineStart( nextNewlinePosition + 1 );
		int nextLineEnd( next_newline_position( nextLineStart ) );
		if ( nextLineEnd < 0 ) {
			nextLineEnd = _data.length();
		}
		int nextLineLength( nextLineEnd - nextLineStart );
		int prevNewlinePosition( prev_newline_position( _pos ) );
		if ( prevNewlinePosition == _pos ) {
			prevNewlinePosition = _pos > 0 ? prev_newline_position( _pos - 1 ) : -1;
		}
		int lineStartPosition( prevNewlinePosition + 1 );
		int posInLine( _pos - lineStartPosition );
		int shift( ! _indentMultiline && ( lineStartPosition == 0 ) ? _prompt.indentation() : 0 );
		posInLine = max( min( posInLine + shift, nextLineLength ), 0 );
		_pos = nextLineStart + posInLine;
		assert( ( _pos >= 0 ) && ( _pos <= _data.length() ) );
		return ( Replxx::ACTION_RESULT::CONTINUE );
	} while ( false );
	return ( history_move( false ) );
}

Replxx::ACTION_RESULT Replxx::ReplxxImpl::history_next( char32_t ) {
	return ( history_move( false ) );
}

Replxx::ACTION_RESULT Replxx::ReplxxImpl::history_previous( char32_t ) {
	return ( history_move( true ) );
}

Replxx::ACTION_RESULT Replxx::ReplxxImpl::history_move( bool previous_ ) {
	// if not already recalling, add the current line to the history list so
	// we don't have to special case it
	if ( _history.is_last() ) {
		_history.update_last( _data );
	}
	if ( _history.is_empty() ) {
		return ( Replxx::ACTION_RESULT::CONTINUE );
	}
	_history.set_current_scratch( _data );
	if ( ! _history.move( previous_ ) ) {
		return ( Replxx::ACTION_RESULT::CONTINUE );
	}
	_data.assign( _history.current() );
	_pos = _data.length();
	refresh_line();
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// meta-<, beginning of history
// Page Up, beginning of history
Replxx::ACTION_RESULT Replxx::ReplxxImpl::history_first( char32_t ) {
	do {
		if ( ! _hasNewlines ) {
			break;
		}
		if ( _pos == 0 ) {
			break;
		}
		_pos = 0;
		return ( Replxx::ACTION_RESULT::CONTINUE );
	} while ( false );
	return ( history_jump( true ) );
}

// meta->, end of history
// Page Down, end of history
Replxx::ACTION_RESULT Replxx::ReplxxImpl::history_last( char32_t ) {
	do {
		if ( ! _hasNewlines ) {
			break;
		}
		if ( _pos == _data.length() ) {
			break;
		}
		_pos = _data.length();
		return ( Replxx::ACTION_RESULT::CONTINUE );
	} while ( false );
	return ( history_jump( false ) );
}

// CTRL-g, restore current history entry
Replxx::ACTION_RESULT Replxx::ReplxxImpl::history_restore_current( char32_t ) {
	// if not already recalling, there is nothing to restore.
	if ( ! _history.is_last() ) {
		_history.reset_current_scratch();
		_data.assign( _history.current() );
		_pos = _data.length();
		refresh_line();
	}
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// meta-g restore all history entries
Replxx::ACTION_RESULT Replxx::ReplxxImpl::history_restore( char32_t ) {
	_history.reset_scratches();
	// if not already recalling, there is nothing to restore.
	if ( ! _history.is_last() ) {
		_data.assign( _history.current() );
		_pos = _data.length();
		refresh_line();
	}
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

Replxx::ACTION_RESULT Replxx::ReplxxImpl::history_jump( bool back_ ) {
	// if not already recalling, add the current line to the history list so
	// we don't
	// have to special case it
	if ( _history.is_last() ) {
		_history.update_last( _data );
	}
	if ( ! _history.is_empty() ) {
		_history.set_current_scratch( _data );
		_history.jump( back_ );
		_data.assign( _history.current() );
		_pos = _data.length();
		refresh_line();
	}
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

Replxx::ACTION_RESULT Replxx::ReplxxImpl::hint_next( char32_t ) {
	return ( hint_move( false ) );
}

Replxx::ACTION_RESULT Replxx::ReplxxImpl::hint_previous( char32_t ) {
	return ( hint_move( true ) );
}

Replxx::ACTION_RESULT Replxx::ReplxxImpl::hint_move( bool previous_ ) {
	if ( ! _noColor ) {
		_killRing.lastAction = KillRing::actionOther;
		if ( previous_ ) {
			-- _hintSelection;
		} else {
			++ _hintSelection;
		}
		refresh_line( HINT_ACTION::REPAINT );
	}
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

Replxx::ACTION_RESULT Replxx::ReplxxImpl::toggle_overwrite_mode( char32_t ) {
	_overwrite = ! _overwrite;
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

#ifndef _WIN32
Replxx::ACTION_RESULT Replxx::ReplxxImpl::verbatim_insert( char32_t ) {
	static int const MAX_ESC_SEQ( 32 );
	char32_t buf[MAX_ESC_SEQ];
	int len( _terminal.read_verbatim( buf, MAX_ESC_SEQ ) );
	_data.insert( _pos, UnicodeString( buf, len ), 0, len );
	_pos += len;
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// ctrl-Z, job control
Replxx::ACTION_RESULT Replxx::ReplxxImpl::suspend( char32_t ) {
	/* IOModeGuard scope */ {
		IOModeGuard ioModeGuard( _terminal );
		raise( SIGSTOP );   // Break out in mid-line
	}
	// Redraw prompt
	_prompt.write();
	return ( Replxx::ACTION_RESULT::CONTINUE );
}
#endif

Replxx::ACTION_RESULT Replxx::ReplxxImpl::complete_line( char32_t c ) {
	if ( !! _completionCallback && ( _completeOnEmpty || ( _pos > 0 ) ) ) {
		// complete_line does the actual completion and replacement
		c = do_complete_line( c != 0 );

		if ( static_cast<int>( c ) < 0 ) {
			return ( Replxx::ACTION_RESULT::BAIL );
		}
		if ( c != 0 ) {
			emulate_key_press( c );
		}
	} else {
		insert_character( c );
	}
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

Replxx::ACTION_RESULT Replxx::ReplxxImpl::complete( bool previous_ ) {
	if ( _completions.empty() ) {
		bool first( _completions.empty() );
		int dataLen( _data.length() );
		complete_line( 0 );
		if ( ! _immediateCompletion && first && ( _data.length() > dataLen ) ) {
			return ( Replxx::ACTION_RESULT::CONTINUE );
		}
	}
	int newSelection( _completionSelection + ( previous_ ? -1 : 1 ) );
	if ( newSelection >= static_cast<int>( _completions.size() ) ) {
		newSelection = -1;
	} else if ( newSelection == -2 ) {
		newSelection = static_cast<int>( _completions.size() ) - 1;
	}
	if ( _completionSelection != -1 ) {
		int oldCompletionLength( max( _completions[_completionSelection].text().length() - _completionContextLength, 0 ) );
		_pos -= oldCompletionLength;
		_data.erase( _pos, oldCompletionLength );
	}
	if ( newSelection != -1 ) {
		int newCompletionLength( max( _completions[newSelection].text().length() - _completionContextLength, 0 ) );
		_data.insert( _pos, _completions[newSelection].text(), _completionContextLength, newCompletionLength );
		_pos += newCompletionLength;
	}
	_completionSelection = newSelection;
	refresh_line();  // Refresh the line
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

Replxx::ACTION_RESULT Replxx::ReplxxImpl::complete_next( char32_t ) {
	return ( complete( false ) );
}

Replxx::ACTION_RESULT Replxx::ReplxxImpl::complete_previous( char32_t ) {
	return ( complete( true ) );
}

// Alt-P, reverse history search for prefix
// Alt-P, reverse history search for prefix
// Alt-N, forward history search for prefix
// Alt-N, forward history search for prefix
Replxx::ACTION_RESULT Replxx::ReplxxImpl::common_prefix_search( char32_t startChar ) {
	if (
		_history.common_prefix_search(
			_data, _prefix, ( startChar == ( Replxx::KEY::meta( 'p' ) ) ) || ( startChar == ( Replxx::KEY::meta( 'P' ) ) ), _ignoreCase
		)
	) {
		_data.assign( _history.current() );
		_pos = _data.length();
		refresh_line();
	}
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// ctrl-R, reverse history search
// ctrl-S, forward history search
/**
 * Incremental history search -- take over the prompt and keyboard as the user
 * types a search string, deletes characters from it, changes _direction,
 * and either accepts the found line (for execution orediting) or cancels.
 * @param startChar - the character that began the search, used to set the initial
 * _direction
 */
Replxx::ACTION_RESULT Replxx::ReplxxImpl::incremental_history_search( char32_t startChar ) {
	// if not already recalling, add the current line to the history list so we
	// don't have to special case it
	if ( _history.is_last() ) {
		_history.update_last( _data );
	}
	_history.save_pos();
	int historyLinePosition( _pos );
	clear_self_to_end_of_screen();
	bool seeded( startChar == Replxx::KEY::meta( 'r' ) );
	DynamicPrompt dp( _terminal, ( startChar == Replxx::KEY::control( 'R' ) ) || seeded ? -1 : 1 );
	if ( seeded ) {
		dp._searchText.assign( _data );
		dp.updateSearchPrompt();
	}

	// draw user's text with our prompt
	dynamic_refresh( _prompt, dp, _data.get(), _data.length(), historyLinePosition );

	// loop until we get an exit character
	char32_t c( 0 );
	bool keepLooping = true;
	bool useSearchedLine = true;
	bool searchAgain = false;
	UnicodeString activeHistoryLine;
	while ( keepLooping ) {
		if ( ! seeded ) {
			c = read_char();
		}

		switch ( c ) {
			case 0: break;
			// these characters keep the selected text but do not execute it
			case Replxx::KEY::control('A'): // ctrl-A, move cursor to start of line
			case Replxx::KEY::HOME:
			case Replxx::KEY::control('B'): // ctrl-B, move cursor left by one character
			case Replxx::KEY::LEFT:
			case Replxx::KEY::meta( 'b' ): // meta-B, move cursor left by one word
			case Replxx::KEY::meta( 'B' ):
			case Replxx::KEY::control( Replxx::KEY::LEFT ):
			case Replxx::KEY::meta( Replxx::KEY::LEFT ): // Emacs allows Meta, bash & readline don't
			case Replxx::KEY::control('D'):
			case Replxx::KEY::meta( 'd' ): // meta-D, kill word to right of cursor
			case Replxx::KEY::meta( 'D' ):
			case Replxx::KEY::control('E'): // ctrl-E, move cursor to end of line
			case Replxx::KEY::END:
			case Replxx::KEY::control('F'): // ctrl-F, move cursor right by one character
			case Replxx::KEY::RIGHT:
			case Replxx::KEY::meta( 'f' ): // meta-F, move cursor right by one word
			case Replxx::KEY::meta( 'F' ):
			case Replxx::KEY::control( Replxx::KEY::RIGHT ):
			case Replxx::KEY::meta( Replxx::KEY::RIGHT ): // Emacs allows Meta, bash & readline don't
			case Replxx::KEY::meta( Replxx::KEY::BACKSPACE ):
			case Replxx::KEY::control('J'):
			case Replxx::KEY::control('K'): // ctrl-K, kill from cursor to end of line
			case Replxx::KEY::ENTER:
			case Replxx::KEY::control('N'): // ctrl-N, recall next line in history
			case Replxx::KEY::control('P'): // ctrl-P, recall previous line in history
			case Replxx::KEY::DOWN:
			case Replxx::KEY::UP:
			case Replxx::KEY::control('T'): // ctrl-T, transpose characters
			case Replxx::KEY::control('U'): // ctrl-U, kill all characters to the left of the cursor
			case Replxx::KEY::control('W'):
			case Replxx::KEY::meta( 'y' ): // meta-Y, "yank-pop", rotate popped text
			case Replxx::KEY::meta( 'Y' ):
			case 127:
			case Replxx::KEY::DELETE:
			case Replxx::KEY::meta( '<' ): // start of history
			case Replxx::KEY::PAGE_UP:
			case Replxx::KEY::meta( '>' ): // end of history
			case Replxx::KEY::PAGE_DOWN: {
				keepLooping = false;
			} break;

			// these characters revert the input line to its previous state
			case Replxx::KEY::control('C'): // ctrl-C, abort this line
			case Replxx::KEY::control('G'):
			case Replxx::KEY::control('L'): { // ctrl-L, clear screen and redisplay line
				keepLooping = false;
				useSearchedLine = false;
				if (c != Replxx::KEY::control('L')) {
					c = -1; // ctrl-C and ctrl-G just abort the search and do nothing else
				}
			} break;

			// these characters stay in search mode and assign the display
			case Replxx::KEY::control( 'S' ):
			case Replxx::KEY::control( 'R' ):
			case Replxx::KEY::meta( 'r' ): {
				if ( ( dp._searchText.length() == 0 ) && ( _previousSearchText.length() > 0 ) ) {
					// if no current search text, recall previous text
					dp._searchText = _previousSearchText;
				}
				if (
					( ( dp._direction == 1 ) && ( ( c == Replxx::KEY::control( 'R' ) ) || ( c == Replxx::KEY::meta( 'r' ) ) ) )
					|| ( ( dp._direction == -1 ) && ( c == Replxx::KEY::control( 'S' ) ) )
				) {
					dp._direction = 0 - dp._direction; // reverse direction
					dp.updateSearchPrompt();           // change the prompt
				} else {
					searchAgain = true; // same direction, search again
				}
			} break;

// job control is its own thing
#ifndef _WIN32
			case Replxx::KEY::control('Z'): { // ctrl-Z, job control
				/* IOModeGuard scope */ {
					IOModeGuard ioModeGuard( _terminal );
					// Returning to Linux (whatever) shell, leave raw mode
					// Break out in mid-line
					// Back from Linux shell, re-enter raw mode
					raise( SIGSTOP );
				}
				dynamic_refresh( dp, dp, activeHistoryLine.get(), activeHistoryLine.length(), historyLinePosition );
				continue;
			} break;
#endif

			// these characters assign the search string, and hence the selected input line
			case Replxx::KEY::BACKSPACE: { // backspace/ctrl-H, delete char to left of cursor
				if ( dp._searchText.length() > 0 ) {
					dp._searchText.erase( dp._searchText.length() - 1 );
					dp.updateSearchPrompt();
					_history.restore_pos();
					historyLinePosition = _pos;
				} else {
					beep();
				}
			} break;

			case Replxx::KEY::control('Y'): { // ctrl-Y, yank killed text
			} break;

			default: {
				if ( ! is_control_code( c ) && ( c < static_cast<int>( Replxx::KEY::BASE ) ) ) { // not an action character
					dp._searchText.insert( dp._searchText.length(), c );
					dp.updateSearchPrompt();
				} else {
					beep();
				}
			}
		} // switch

		// if we are staying in search mode, search now
		if ( ! keepLooping ) {
			break;
		}
		activeHistoryLine.assign( _history.current() );
		if ( dp._searchText.length() > 0 ) {
			bool found = false;
			int lineSearchPos = historyLinePosition;
			if ( searchAgain ) {
				lineSearchPos += dp._direction;
			}
			searchAgain = false;
			bool ignoreCase( _ignoreCase && std::none_of( dp._searchText.begin(), dp._searchText.end(), []( char32_t x ) { return iswupper( static_cast<wint_t>( x ) ); } ) );
			while ( true ) {
				while (
					dp._direction < 0
						? ( lineSearchPos >= 0 )
						: ( ( lineSearchPos + dp._searchText.length() ) <= activeHistoryLine.length() )
				) {
					if (
						( lineSearchPos >= 0 )
						&& ( ( lineSearchPos + dp._searchText.length() ) <= activeHistoryLine.length() )
						&& std::equal(
							dp._searchText.begin(), dp._searchText.end(), activeHistoryLine.begin() + lineSearchPos,
							ignoreCase ? case_insensitive_equal : case_sensitive_equal
						)
					) {
						if ( ! seeded ) {
							found = true;
							break;
						} else {
							seeded = false;
						}
					}
					lineSearchPos += dp._direction;
				}
				if ( found ) {
					historyLinePosition = lineSearchPos;
					break;
				} else if ( _history.move( dp._direction < 0 ) ) {
					activeHistoryLine.assign( _history.current() );
					lineSearchPos = ( dp._direction > 0 ) ? 0 : ( activeHistoryLine.length() - dp._searchText.length() );
				} else {
					historyLinePosition = _pos;
					beep();
					break;
				}
			} // while
			if ( ! found ) {
				_history.restore_pos();
			}
		} else {
			_history.restore_pos();
			historyLinePosition = _pos;
		}
		activeHistoryLine.assign( _history.current() );
		dynamic_refresh( dp, dp, activeHistoryLine.get(), activeHistoryLine.length(), historyLinePosition ); // draw user's text with our prompt
		seeded = false;
	} // while

	// leaving history search, restore previous prompt, maybe make searched line
	// current
	Prompt pb( _terminal );
	UnicodeString tempUnicode( &_prompt._text[_prompt._lastLinePosition], _prompt._text.length() - _prompt._lastLinePosition );
	pb.set_text( tempUnicode );
	pb.update_screen_columns();
	if ( useSearchedLine && ( activeHistoryLine.length() > 0 ) ) {
		_history.commit_index();
		_data.assign( activeHistoryLine );
		_pos = historyLinePosition;
		_modifiedState = true;
	} else if ( ! useSearchedLine ) {
		_history.restore_pos();
	}
	dynamic_refresh(pb, _prompt, _data.get(), _data.length(), _pos); // redraw the original prompt with current input
	_previousSearchText = dp._searchText; // save search text for possible reuse on ctrl-R ctrl-R
	emulate_key_press( c ); // pass a character or -1 back to main loop
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

// ctrl-L, clear screen and redisplay line
Replxx::ACTION_RESULT Replxx::ReplxxImpl::clear_screen( char32_t c ) {
	_terminal.clear_screen( Terminal::CLEAR_SCREEN::WHOLE );
	if ( c ) {
		_prompt.write();
		_prompt._cursorRowOffset = _prompt._extraLines;
		refresh_line();
	}
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

Replxx::ACTION_RESULT Replxx::ReplxxImpl::bracketed_paste( char32_t ) {
	UnicodeString buf;
	while ( char32_t c = _terminal.read_char() ) {
		if ( c == KEY::PASTE_FINISH ) {
			break;
		}
		if ( ( c == '\r' ) || ( c == KEY::control( 'M' ) ) ) {
			c = '\n';
		} else if ( c == KEY::control( 'I' ) ) {
			c = '\t';
		}
		buf.push_back( c );
	}
	_data.insert( _pos, buf, 0, buf.length() );
	_pos += buf.length();
	return ( Replxx::ACTION_RESULT::CONTINUE );
}

template <bool subword>
bool Replxx::ReplxxImpl::is_word_break_character( char32_t char_ ) const {
	bool wbc( false );
	if ( char_ < 128 ) {
		wbc = strchr( subword ? _subwordBreakChars.c_str() : _wordBreakChars.c_str(), static_cast<char>( char_ ) ) != nullptr;
	}
	return ( wbc );
}

void Replxx::ReplxxImpl::history_add( std::string const& line ) {
	_history.add( UnicodeString( line ) );
}

bool Replxx::ReplxxImpl::history_save( std::string const& filename ) {
	return ( _history.save( filename, false ) );
}

void Replxx::ReplxxImpl::history_save( std::ostream& out ) {
	_history.save( out );
}

bool Replxx::ReplxxImpl::history_sync( std::string const& filename ) {
	return ( _history.save( filename, true ) );
}

bool Replxx::ReplxxImpl::history_load( std::string const& filename ) {
	return ( _history.load( filename ) );
}

void Replxx::ReplxxImpl::history_load( std::istream& in ) {
	_history.load( in );
}

void Replxx::ReplxxImpl::history_clear( void ) {
	_history.clear();
}

int Replxx::ReplxxImpl::history_size( void ) const {
	return ( _history.size() );
}

Replxx::HistoryScan::impl_t Replxx::ReplxxImpl::history_scan( void ) const {
	return ( _history.scan() );
}

void Replxx::ReplxxImpl::set_modify_callback( Replxx::modify_callback_t const& fn ) {
	_modifyCallback = fn;
}

void Replxx::ReplxxImpl::set_completion_callback( Replxx::completion_callback_t const& fn ) {
	_completionCallback = fn;
}

void Replxx::ReplxxImpl::set_highlighter_callback( Replxx::highlighter_callback_t const& fn ) {
	_highlighterCallback = fn;
}

void Replxx::ReplxxImpl::set_hint_callback( Replxx::hint_callback_t const& fn ) {
	_hintCallback = fn;
}

void Replxx::ReplxxImpl::set_max_history_size( int len ) {
	_history.set_max_size( len );
}

void Replxx::ReplxxImpl::set_completion_count_cutoff( int count ) {
	_completionCountCutoff = count;
}

void Replxx::ReplxxImpl::set_max_hint_rows( int count ) {
	_maxHintRows = count;
}

void Replxx::ReplxxImpl::set_hint_delay( int hintDelay_ ) {
	_hintDelay = hintDelay_;
}

void Replxx::ReplxxImpl::set_word_break_characters( char const* wordBreakers ) {
	_wordBreakChars = wordBreakers;
}

void Replxx::ReplxxImpl::set_subword_break_characters( char const* subwordBreakers ) {
	_subwordBreakChars = subwordBreakers;
}

void Replxx::ReplxxImpl::set_double_tab_completion( bool val ) {
	_doubleTabCompletion = val;
}

void Replxx::ReplxxImpl::set_complete_on_empty( bool val ) {
	_completeOnEmpty = val;
}

void Replxx::ReplxxImpl::set_beep_on_ambiguous_completion( bool val ) {
	_beepOnAmbiguousCompletion = val;
}

void Replxx::ReplxxImpl::set_immediate_completion( bool val ) {
	_immediateCompletion = val;
}

void Replxx::ReplxxImpl::set_unique_history( bool val ) {
	_history.set_unique( val );
}

void Replxx::ReplxxImpl::set_no_color( bool val ) {
	_noColor = val;
}

void Replxx::ReplxxImpl::set_indent_multiline( bool val ) {
	_indentMultiline = val;
}

/**
 * Display the dynamic incremental search prompt and the current user input
 * line.
 * @param pi    Prompt struct holding information about the prompt and our
 * screen position
 * @param buf32 input buffer to be displayed
 * @param len   count of characters in the buffer
 * @param pos   current cursor position within the buffer (0 <= pos <= len)
 */
void Replxx::ReplxxImpl::dynamic_refresh(Prompt& oldPrompt, Prompt& newPrompt, char32_t* buf32, int len, int pos) {
	clear_self_to_end_of_screen( &oldPrompt );
	// calculate the position of the end of the prompt
	int xEndOfPrompt( 0 );
	int yEndOfPrompt( 0 );
	replxx::virtual_render( newPrompt._text.get(), newPrompt._text.length(), xEndOfPrompt, yEndOfPrompt, newPrompt.screen_columns(), 0 );

	// calculate the desired position of the cursor
	int xCursorPos( xEndOfPrompt );
	int yCursorPos( yEndOfPrompt );
	virtual_render( buf32, pos, xCursorPos, yCursorPos, &newPrompt );

	// calculate the position of the end of the input line
	int xEndOfInput( xCursorPos );
	int yEndOfInput( yCursorPos );
	virtual_render( buf32 + pos, len - pos, xEndOfInput, yEndOfInput, &newPrompt );

	// display the prompt
	newPrompt.write();

	// display the input line
	_terminal.write32( buf32, len );

#ifndef _WIN32
	// we have to generate our own newline on line wrap
	if ( ( xEndOfInput == 0 ) && ( yEndOfInput > 0 ) && ( len > 0 ) && ( buf32[len - 1] != '\n' ) ) {
		_terminal.write8( "\n", 1 );
	}
#endif
	// position the cursor
	_terminal.jump_cursor(
		xCursorPos, // 0-based on Win32
		-( yEndOfInput - yCursorPos )
	);
	newPrompt._cursorRowOffset = newPrompt._extraLines + yCursorPos; // remember row for next pass
}

}

