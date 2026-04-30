#ifdef _WIN32

#include <conio.h>
#include <windows.h>
#include <io.h>
#if _MSC_VER < 1900 && defined (_MSC_VER)
#define snprintf _snprintf	// Microsoft headers use underscores in some names
#endif
#define strcasecmp _stricmp
#define strdup _strdup
#define write _write
#define STDIN_FILENO 0

#else /* _WIN32 */

#include <unistd.h>

#endif /* _WIN32 */

#include "prompt.hxx"
#include "util.hxx"

namespace replxx {

Prompt::Prompt( Terminal& terminal_ )
	: _extraLines( 0 )
	, _lastLinePosition( 0 )
	, _cursorRowOffset( 0 )
	, _screenColumns( 0 )
	, _terminal( terminal_ ) {
}

void Prompt::write() {
	_terminal.write32( _text.get(), _text.length() );
}

void Prompt::update_screen_columns( void ) {
	_screenColumns = _terminal.get_screen_columns();
}

void Prompt::set_text( UnicodeString const& text_ ) {
	_text = text_;
	update_state();
}

void Prompt::update_state() {
	_cursorRowOffset -= _extraLines;
	_extraLines = 0;
	_lastLinePosition = 0;
	_screenColumns = 0;
	update_screen_columns();
	// strip control characters from the prompt -- we do allow newline
	UnicodeString::const_iterator in( _text.begin() );

	int x = 0;
	int renderedSize( 0 );
	_characterCount = virtual_render( _text.get(), _text.length(), x, _extraLines, _screenColumns, 0, _text.get(), &renderedSize );
	_lastLinePosition = _characterCount - x;
	_text.erase( renderedSize, _text.length() - renderedSize );

	_cursorRowOffset += _extraLines;
}

int Prompt::indentation() const {
	return _characterCount - _lastLinePosition;
}

// Used with DynamicPrompt (history search)
//
const UnicodeString forwardSearchBasePrompt("(i-search)`");
const UnicodeString reverseSearchBasePrompt("(reverse-i-search)`");
const UnicodeString endSearchBasePrompt("': ");

DynamicPrompt::DynamicPrompt( Terminal& terminal_, int initialDirection )
	: Prompt( terminal_ )
	, _searchText()
	, _direction( initialDirection ) {
	updateSearchPrompt();
}

void DynamicPrompt::updateSearchPrompt(void) {
	update_screen_columns();
	const UnicodeString* basePrompt =
			(_direction > 0) ? &forwardSearchBasePrompt : &reverseSearchBasePrompt;
	_text.assign( *basePrompt ).append( _searchText ).append( endSearchBasePrompt );
	update_state();
}

}

