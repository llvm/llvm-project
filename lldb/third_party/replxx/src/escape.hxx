#ifndef REPLXX_ESCAPE_HXX_INCLUDED
#define REPLXX_ESCAPE_HXX_INCLUDED 1

namespace replxx {

namespace EscapeSequenceProcessing {

// This is a typedef for the routine called by doDispatch().	It takes the
// current character
// as input, does any required processing including reading more characters and
// calling other
// dispatch routines, then eventually returns the final (possibly extended or
// special) character.
//
typedef char32_t (*CharacterDispatchRoutine)(char32_t);

// This structure is used by doDispatch() to hold a list of characters to test
// for and
// a list of routines to call if the character matches.	The dispatch routine
// list is one
// longer than the character list; the final entry is used if no character
// matches.
//
struct CharacterDispatch {
	unsigned int len;                   // length of the chars list
	const char* chars;                  // chars to test
	CharacterDispatchRoutine* dispatch; // array of routines to call
};

char32_t doDispatch(char32_t c);

}

}

#endif

