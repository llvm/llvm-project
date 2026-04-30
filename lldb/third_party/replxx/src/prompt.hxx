#ifndef REPLXX_PROMPT_HXX_INCLUDED
#define REPLXX_PROMPT_HXX_INCLUDED 1

#include <cstdlib>

#include "unicodestring.hxx"
#include "terminal.hxx"

namespace replxx {

class Prompt {              // a convenience struct for grouping prompt info
public:
	UnicodeString _text;      // our copy of the prompt text, edited
	int _characterCount{0};   // visible characters in _text
	int _extraLines{0};       // extra lines (beyond 1) occupied by prompt
	int _lastLinePosition{0}; // index into _text where last line begins
	int _cursorRowOffset{0};  // where the cursor is relative to the start of the prompt

private:
	int _screenColumns{0};    // width of screen in columns [cache]
	Terminal& _terminal;
public:
	Prompt( Terminal& );
	void set_text( UnicodeString const& textPtr );
	void update_state();
	void update_screen_columns( void );
	int screen_columns() const {
		return ( _screenColumns );
	}
	void write();
	int indentation() const;
};

// changing prompt for "(reverse-i-search)`text':" etc.
//
struct DynamicPrompt : public Prompt {
	UnicodeString _searchText; // text we are searching for
	int _direction;            // current search _direction, 1=forward, -1=reverse

	DynamicPrompt( Terminal&, int initialDirection );
	void updateSearchPrompt(void);
};

}

#endif
