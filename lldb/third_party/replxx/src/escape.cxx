#include "escape.hxx"
#include "terminal.hxx"
#include "replxx.hxx"

#ifndef _WIN32

namespace replxx {

namespace EscapeSequenceProcessing { // move these out of global namespace

// This chunk of code does parsing of the escape sequences sent by various Linux
// terminals.
//
// It handles arrow keys, Home, End and Delete keys by interpreting the
// sequences sent by
// gnome terminal, xterm, rxvt, konsole, aterm and yakuake including the Alt and
// Ctrl key
// combinations that are understood by replxx.
//
// The parsing uses tables, a bunch of intermediate dispatch routines and a
// doDispatch
// loop that reads the tables and sends control to "deeper" routines to continue
// the
// parsing.	The starting call to doDispatch( c, initialDispatch ) will
// eventually return
// either a character (with optional CTRL and META bits set), or -1 if parsing
// fails, or
// zero if an attempt to read from the keyboard fails.
//
// This is rather sloppy escape sequence processing, since we're not paying
// attention to what the
// actual TERM is set to and are processing all key sequences for all terminals,
// but it works with
// the most common keystrokes on the most common terminals.	It's intricate, but
// the nested 'if'
// statements required to do it directly would be worse.	This way has the
// advantage of allowing
// changes and extensions without having to touch a lot of code.


static char32_t thisKeyMetaCtrl = 0;	// holds pre-set Meta and/or Ctrl modifiers

// This dispatch routine is given a dispatch table and then farms work out to
// routines
// listed in the table based on the character it is called with.	The dispatch
// routines can
// read more input characters to decide what should eventually be returned.
// Eventually,
// a called routine returns either a character or -1 to indicate parsing
// failure.
//
char32_t doDispatch(char32_t c, CharacterDispatch& dispatchTable) {
	for (unsigned int i = 0; i < dispatchTable.len; ++i) {
		if (static_cast<unsigned char>(dispatchTable.chars[i]) == c) {
			return dispatchTable.dispatch[i](c);
		}
	}
	return dispatchTable.dispatch[dispatchTable.len](c);
}

// Final dispatch routines -- return something
//
static char32_t normalKeyRoutine(char32_t c) { return thisKeyMetaCtrl | c; }
static char32_t upArrowKeyRoutine(char32_t) {
	return thisKeyMetaCtrl | Replxx::KEY::UP;;
}
static char32_t downArrowKeyRoutine(char32_t) {
	return thisKeyMetaCtrl | Replxx::KEY::DOWN;
}
static char32_t rightArrowKeyRoutine(char32_t) {
	return thisKeyMetaCtrl | Replxx::KEY::RIGHT;
}
static char32_t leftArrowKeyRoutine(char32_t) {
	return thisKeyMetaCtrl | Replxx::KEY::LEFT;
}
static char32_t homeKeyRoutine(char32_t) { return thisKeyMetaCtrl | Replxx::KEY::HOME; }
static char32_t endKeyRoutine(char32_t) { return thisKeyMetaCtrl | Replxx::KEY::END; }
static char32_t shiftTabRoutine(char32_t) { return Replxx::KEY::BASE_SHIFT | Replxx::KEY::TAB; }
static char32_t f1KeyRoutine(char32_t) { return thisKeyMetaCtrl | Replxx::KEY::F1; }
static char32_t f2KeyRoutine(char32_t) { return thisKeyMetaCtrl | Replxx::KEY::F2; }
static char32_t f3KeyRoutine(char32_t) { return thisKeyMetaCtrl | Replxx::KEY::F3; }
static char32_t f4KeyRoutine(char32_t) { return thisKeyMetaCtrl | Replxx::KEY::F4; }
static char32_t f5KeyRoutine(char32_t) { return thisKeyMetaCtrl | Replxx::KEY::F5; }
static char32_t f6KeyRoutine(char32_t) { return thisKeyMetaCtrl | Replxx::KEY::F6; }
static char32_t f7KeyRoutine(char32_t) { return thisKeyMetaCtrl | Replxx::KEY::F7; }
static char32_t f8KeyRoutine(char32_t) { return thisKeyMetaCtrl | Replxx::KEY::F8; }
static char32_t f9KeyRoutine(char32_t) { return thisKeyMetaCtrl | Replxx::KEY::F9; }
static char32_t f10KeyRoutine(char32_t) { return thisKeyMetaCtrl | Replxx::KEY::F10; }
static char32_t f11KeyRoutine(char32_t) { return thisKeyMetaCtrl | Replxx::KEY::F11; }
static char32_t f12KeyRoutine(char32_t) { return thisKeyMetaCtrl | Replxx::KEY::F12; }
static char32_t pageUpKeyRoutine(char32_t) {
	return thisKeyMetaCtrl | Replxx::KEY::PAGE_UP;
}
static char32_t pageDownKeyRoutine(char32_t) {
	return thisKeyMetaCtrl | Replxx::KEY::PAGE_DOWN;
}
static char32_t deleteCharRoutine(char32_t) {
	return thisKeyMetaCtrl | Replxx::KEY::BACKSPACE;
}	// key labeled Backspace
static char32_t insertKeyRoutine(char32_t) {
	return thisKeyMetaCtrl | Replxx::KEY::INSERT;
}	// key labeled Delete
static char32_t deleteKeyRoutine(char32_t) {
	return thisKeyMetaCtrl | Replxx::KEY::DELETE;
}	// key labeled Delete
static char32_t ctrlUpArrowKeyRoutine(char32_t) {
	return thisKeyMetaCtrl | Replxx::KEY::BASE_CONTROL | Replxx::KEY::UP;
}
static char32_t ctrlDownArrowKeyRoutine(char32_t) {
	return thisKeyMetaCtrl | Replxx::KEY::BASE_CONTROL | Replxx::KEY::DOWN;
}
static char32_t ctrlRightArrowKeyRoutine(char32_t) {
	return thisKeyMetaCtrl | Replxx::KEY::BASE_CONTROL | Replxx::KEY::RIGHT;
}
static char32_t ctrlLeftArrowKeyRoutine(char32_t) {
	return thisKeyMetaCtrl | Replxx::KEY::BASE_CONTROL | Replxx::KEY::LEFT;
}
static char32_t bracketPasteStartKeyRoutine(char32_t) {
	return thisKeyMetaCtrl | Replxx::KEY::PASTE_START;
}
static char32_t bracketPasteFinishKeyRoutine(char32_t) {
	return thisKeyMetaCtrl | Replxx::KEY::PASTE_FINISH;
}
static char32_t escFailureRoutine(char32_t) {
	beep();
	return -1;
}

// Handle ESC [ 1 ; 2 or 3 (or 5) <more stuff> escape sequences
//
static CharacterDispatchRoutine escLeftBracket1Semicolon2or3or5Routines[] = {
	upArrowKeyRoutine,
	downArrowKeyRoutine,
	rightArrowKeyRoutine,
	leftArrowKeyRoutine,
	homeKeyRoutine,
	endKeyRoutine,
	f1KeyRoutine,
	f2KeyRoutine,
	f3KeyRoutine,
	f4KeyRoutine,
	escFailureRoutine
};
static CharacterDispatch escLeftBracket1Semicolon2or3or5Dispatch = {
		10, "ABCDHFPQRS", escLeftBracket1Semicolon2or3or5Routines
};

// Handle ESC [ 1 ; <more stuff> escape sequences
//
static char32_t escLeftBracket1Semicolon2Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	thisKeyMetaCtrl |= Replxx::KEY::BASE_SHIFT;
	return doDispatch(c, escLeftBracket1Semicolon2or3or5Dispatch);
}
static char32_t escLeftBracket1Semicolon3Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	thisKeyMetaCtrl |= Replxx::KEY::BASE_META;
	return doDispatch(c, escLeftBracket1Semicolon2or3or5Dispatch);
}
static char32_t escLeftBracket1Semicolon5Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	thisKeyMetaCtrl |= Replxx::KEY::BASE_CONTROL;
	return doDispatch(c, escLeftBracket1Semicolon2or3or5Dispatch);
}
static CharacterDispatchRoutine escLeftBracket1SemicolonRoutines[] = {
	escLeftBracket1Semicolon2Routine,
	escLeftBracket1Semicolon3Routine,
	escLeftBracket1Semicolon5Routine,
	escFailureRoutine
};
static CharacterDispatch escLeftBracket1SemicolonDispatch = {
	3, "235", escLeftBracket1SemicolonRoutines
};

// Handle ESC [ 1 ; <more stuff> escape sequences
//
static char32_t escLeftBracket1SemicolonRoutine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket1SemicolonDispatch);
}

// (S)-F5
static CharacterDispatchRoutine escLeftBracket15Semicolon2Routines[] = {
	f5KeyRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket15Semicolon2Dispatch = {
	1, "~", escLeftBracket15Semicolon2Routines
};
static char32_t escLeftBracket15Semicolon2Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	thisKeyMetaCtrl |= Replxx::KEY::BASE_SHIFT;
	return doDispatch(c, escLeftBracket15Semicolon2Dispatch);
}

// (C)-F5
static CharacterDispatchRoutine escLeftBracket15Semicolon5Routines[] = {
	f5KeyRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket15Semicolon5Dispatch = {
	1, "~", escLeftBracket15Semicolon5Routines
};
static char32_t escLeftBracket15Semicolon5Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	thisKeyMetaCtrl |= Replxx::KEY::BASE_CONTROL;
	return doDispatch(c, escLeftBracket15Semicolon5Dispatch);
}

static CharacterDispatchRoutine escLeftBracket15SemicolonRoutines[] = {
	escLeftBracket15Semicolon2Routine, escLeftBracket15Semicolon5Routine, escFailureRoutine
};
static CharacterDispatch escLeftBracket15SemicolonDispatch = {
	2, "25", escLeftBracket15SemicolonRoutines
};
static char32_t escLeftBracket15SemicolonRoutine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket15SemicolonDispatch);
}

static CharacterDispatchRoutine escLeftBracket15Routines[] = {
	f5KeyRoutine, escLeftBracket15SemicolonRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket15Dispatch = {
	2, "~;", escLeftBracket15Routines
};
static char32_t escLeftBracket15Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket15Dispatch);
}

// (S)-F6
static CharacterDispatchRoutine escLeftBracket17Semicolon2Routines[] = {
	f6KeyRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket17Semicolon2Dispatch = {
	1, "~", escLeftBracket17Semicolon2Routines
};
static char32_t escLeftBracket17Semicolon2Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	thisKeyMetaCtrl |= Replxx::KEY::BASE_SHIFT;
	return doDispatch(c, escLeftBracket17Semicolon2Dispatch);
}

// (C)-F6
static CharacterDispatchRoutine escLeftBracket17Semicolon5Routines[] = {
	f6KeyRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket17Semicolon5Dispatch = {
	1, "~", escLeftBracket17Semicolon5Routines
};
static char32_t escLeftBracket17Semicolon5Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	thisKeyMetaCtrl |= Replxx::KEY::BASE_CONTROL;
	return doDispatch(c, escLeftBracket17Semicolon5Dispatch);
}

static CharacterDispatchRoutine escLeftBracket17SemicolonRoutines[] = {
	escLeftBracket17Semicolon2Routine, escLeftBracket17Semicolon5Routine, escFailureRoutine
};
static CharacterDispatch escLeftBracket17SemicolonDispatch = {
	2, "25", escLeftBracket17SemicolonRoutines
};
static char32_t escLeftBracket17SemicolonRoutine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket17SemicolonDispatch);
}

static CharacterDispatchRoutine escLeftBracket17Routines[] = {
	f6KeyRoutine, escLeftBracket17SemicolonRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket17Dispatch = {
	2, "~;", escLeftBracket17Routines
};
static char32_t escLeftBracket17Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket17Dispatch);
}

// (S)-F7
static CharacterDispatchRoutine escLeftBracket18Semicolon2Routines[] = {
	f7KeyRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket18Semicolon2Dispatch = {
	1, "~", escLeftBracket18Semicolon2Routines
};
static char32_t escLeftBracket18Semicolon2Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	thisKeyMetaCtrl |= Replxx::KEY::BASE_SHIFT;
	return doDispatch(c, escLeftBracket18Semicolon2Dispatch);
}

// (C)-F7
static CharacterDispatchRoutine escLeftBracket18Semicolon5Routines[] = {
	f7KeyRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket18Semicolon5Dispatch = {
	1, "~", escLeftBracket18Semicolon5Routines
};
static char32_t escLeftBracket18Semicolon5Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	thisKeyMetaCtrl |= Replxx::KEY::BASE_CONTROL;
	return doDispatch(c, escLeftBracket18Semicolon5Dispatch);
}

static CharacterDispatchRoutine escLeftBracket18SemicolonRoutines[] = {
	escLeftBracket18Semicolon2Routine, escLeftBracket18Semicolon5Routine, escFailureRoutine
};
static CharacterDispatch escLeftBracket18SemicolonDispatch = {
	2, "25", escLeftBracket18SemicolonRoutines
};
static char32_t escLeftBracket18SemicolonRoutine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket18SemicolonDispatch);
}

static CharacterDispatchRoutine escLeftBracket18Routines[] = {
	f7KeyRoutine, escLeftBracket18SemicolonRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket18Dispatch = {
	2, "~;", escLeftBracket18Routines
};
static char32_t escLeftBracket18Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket18Dispatch);
}

// (S)-F8
static CharacterDispatchRoutine escLeftBracket19Semicolon2Routines[] = {
	f8KeyRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket19Semicolon2Dispatch = {
	1, "~", escLeftBracket19Semicolon2Routines
};
static char32_t escLeftBracket19Semicolon2Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	thisKeyMetaCtrl |= Replxx::KEY::BASE_SHIFT;
	return doDispatch(c, escLeftBracket19Semicolon2Dispatch);
}

// (C)-F8
static CharacterDispatchRoutine escLeftBracket19Semicolon5Routines[] = {
	f8KeyRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket19Semicolon5Dispatch = {
	1, "~", escLeftBracket19Semicolon5Routines
};
static char32_t escLeftBracket19Semicolon5Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	thisKeyMetaCtrl |= Replxx::KEY::BASE_CONTROL;
	return doDispatch(c, escLeftBracket19Semicolon5Dispatch);
}

static CharacterDispatchRoutine escLeftBracket19SemicolonRoutines[] = {
	escLeftBracket19Semicolon2Routine, escLeftBracket19Semicolon5Routine, escFailureRoutine
};
static CharacterDispatch escLeftBracket19SemicolonDispatch = {
	2, "25", escLeftBracket19SemicolonRoutines
};
static char32_t escLeftBracket19SemicolonRoutine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket19SemicolonDispatch);
}

static CharacterDispatchRoutine escLeftBracket19Routines[] = {
	f8KeyRoutine, escLeftBracket19SemicolonRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket19Dispatch = {
	2, "~;", escLeftBracket19Routines
};
static char32_t escLeftBracket19Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket19Dispatch);
}

// Handle ESC [ 1 <more stuff> escape sequences
//
static CharacterDispatchRoutine escLeftBracket1Routines[] = {
	homeKeyRoutine, escLeftBracket1SemicolonRoutine,
	escLeftBracket15Routine,
	escLeftBracket17Routine,
	escLeftBracket18Routine,
	escLeftBracket19Routine,
	escFailureRoutine
};
static CharacterDispatch escLeftBracket1Dispatch = {
	6, "~;5789", escLeftBracket1Routines
};

// Handle ESC [ 2 <more stuff> escape sequences
//

// (S)-F9
static CharacterDispatchRoutine escLeftBracket20Semicolon2Routines[] = {
	f9KeyRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket20Semicolon2Dispatch = {
	1, "~", escLeftBracket20Semicolon2Routines
};
static char32_t escLeftBracket20Semicolon2Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	thisKeyMetaCtrl |= Replxx::KEY::BASE_SHIFT;
	return doDispatch(c, escLeftBracket20Semicolon2Dispatch);
}

// (C)-F9
static CharacterDispatchRoutine escLeftBracket20Semicolon5Routines[] = {
	f9KeyRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket20Semicolon5Dispatch = {
	1, "~", escLeftBracket20Semicolon5Routines
};
static char32_t escLeftBracket20Semicolon5Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	thisKeyMetaCtrl |= Replxx::KEY::BASE_CONTROL;
	return doDispatch(c, escLeftBracket20Semicolon5Dispatch);
}

static CharacterDispatchRoutine escLeftBracket20SemicolonRoutines[] = {
	escLeftBracket20Semicolon2Routine, escLeftBracket20Semicolon5Routine, escFailureRoutine
};
static CharacterDispatch escLeftBracket20SemicolonDispatch = {
	2, "25", escLeftBracket20SemicolonRoutines
};
static char32_t escLeftBracket20SemicolonRoutine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket20SemicolonDispatch);
}

static CharacterDispatchRoutine escLeftBracket200Routines[] = {
	bracketPasteStartKeyRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket200Dispatch = {
	1, "~", escLeftBracket200Routines
};
static char32_t escLeftBracket200Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket200Dispatch);
}

static CharacterDispatchRoutine escLeftBracket201Routines[] = {
	bracketPasteFinishKeyRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket201Dispatch = {
	1, "~", escLeftBracket201Routines
};
static char32_t escLeftBracket201Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket201Dispatch);
}

static CharacterDispatchRoutine escLeftBracket20Routines[] = {
	f9KeyRoutine, escLeftBracket20SemicolonRoutine, escLeftBracket200Routine, escLeftBracket201Routine, escFailureRoutine
};
static CharacterDispatch escLeftBracket20Dispatch = {
	4, "~;01", escLeftBracket20Routines
};
static char32_t escLeftBracket20Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket20Dispatch);
}

// (S)-F10
static CharacterDispatchRoutine escLeftBracket21Semicolon2Routines[] = {
	f10KeyRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket21Semicolon2Dispatch = {
	1, "~", escLeftBracket21Semicolon2Routines
};
static char32_t escLeftBracket21Semicolon2Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	thisKeyMetaCtrl |= Replxx::KEY::BASE_SHIFT;
	return doDispatch(c, escLeftBracket21Semicolon2Dispatch);
}

// (C)-F10
static CharacterDispatchRoutine escLeftBracket21Semicolon5Routines[] = {
	f10KeyRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket21Semicolon5Dispatch = {
	1, "~", escLeftBracket21Semicolon5Routines
};
static char32_t escLeftBracket21Semicolon5Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	thisKeyMetaCtrl |= Replxx::KEY::BASE_CONTROL;
	return doDispatch(c, escLeftBracket21Semicolon5Dispatch);
}

static CharacterDispatchRoutine escLeftBracket21SemicolonRoutines[] = {
	escLeftBracket21Semicolon2Routine, escLeftBracket21Semicolon5Routine, escFailureRoutine
};
static CharacterDispatch escLeftBracket21SemicolonDispatch = {
	2, "25", escLeftBracket21SemicolonRoutines
};
static char32_t escLeftBracket21SemicolonRoutine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket21SemicolonDispatch);
}

static CharacterDispatchRoutine escLeftBracket21Routines[] = {
	f10KeyRoutine, escLeftBracket21SemicolonRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket21Dispatch = {
	2, "~;", escLeftBracket21Routines
};
static char32_t escLeftBracket21Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket21Dispatch);
}

// (S)-F11
static CharacterDispatchRoutine escLeftBracket23Semicolon2Routines[] = {
	f11KeyRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket23Semicolon2Dispatch = {
	1, "~", escLeftBracket23Semicolon2Routines
};
static char32_t escLeftBracket23Semicolon2Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	thisKeyMetaCtrl |= Replxx::KEY::BASE_SHIFT;
	return doDispatch(c, escLeftBracket23Semicolon2Dispatch);
}

// (C)-F11
static CharacterDispatchRoutine escLeftBracket23Semicolon5Routines[] = {
	f11KeyRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket23Semicolon5Dispatch = {
	1, "~", escLeftBracket23Semicolon5Routines
};
static char32_t escLeftBracket23Semicolon5Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	thisKeyMetaCtrl |= Replxx::KEY::BASE_CONTROL;
	return doDispatch(c, escLeftBracket23Semicolon5Dispatch);
}

static CharacterDispatchRoutine escLeftBracket23SemicolonRoutines[] = {
	escLeftBracket23Semicolon2Routine, escLeftBracket23Semicolon5Routine, escFailureRoutine
};
static CharacterDispatch escLeftBracket23SemicolonDispatch = {
	2, "25", escLeftBracket23SemicolonRoutines
};
static char32_t escLeftBracket23SemicolonRoutine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket23SemicolonDispatch);
}

static CharacterDispatchRoutine escLeftBracket23Routines[] = {
	f11KeyRoutine, escLeftBracket23SemicolonRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket23Dispatch = {
	2, "~;", escLeftBracket23Routines
};
static char32_t escLeftBracket23Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket23Dispatch);
}

// (S)-F12
static CharacterDispatchRoutine escLeftBracket24Semicolon2Routines[] = {
	f12KeyRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket24Semicolon2Dispatch = {
	1, "~", escLeftBracket24Semicolon2Routines
};
static char32_t escLeftBracket24Semicolon2Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	thisKeyMetaCtrl |= Replxx::KEY::BASE_SHIFT;
	return doDispatch(c, escLeftBracket24Semicolon2Dispatch);
}

// (C)-F12
static CharacterDispatchRoutine escLeftBracket24Semicolon5Routines[] = {
	f12KeyRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket24Semicolon5Dispatch = {
	1, "~", escLeftBracket24Semicolon5Routines
};
static char32_t escLeftBracket24Semicolon5Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	thisKeyMetaCtrl |= Replxx::KEY::BASE_CONTROL;
	return doDispatch(c, escLeftBracket24Semicolon5Dispatch);
}

static CharacterDispatchRoutine escLeftBracket24SemicolonRoutines[] = {
	escLeftBracket24Semicolon2Routine, escLeftBracket24Semicolon5Routine, escFailureRoutine
};
static CharacterDispatch escLeftBracket24SemicolonDispatch = {
	2, "25", escLeftBracket24SemicolonRoutines
};
static char32_t escLeftBracket24SemicolonRoutine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket24SemicolonDispatch);
}

static CharacterDispatchRoutine escLeftBracket24Routines[] = {
	f12KeyRoutine, escLeftBracket24SemicolonRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket24Dispatch = {
	2, "~;", escLeftBracket24Routines
};
static char32_t escLeftBracket24Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket24Dispatch);
}

// Handle ESC [ 2 <more stuff> escape sequences
//
static CharacterDispatchRoutine escLeftBracket2Routines[] = {
	insertKeyRoutine,
	escLeftBracket20Routine,
	escLeftBracket21Routine,
	escLeftBracket23Routine,
	escLeftBracket24Routine,
	escFailureRoutine
};
static CharacterDispatch escLeftBracket2Dispatch = {
	5, "~0134", escLeftBracket2Routines
};

// Handle ESC [ 3 <more stuff> escape sequences
//
static CharacterDispatchRoutine escLeftBracket3Routines[] = {
	deleteKeyRoutine, escFailureRoutine
};

static CharacterDispatch escLeftBracket3Dispatch = {
	1, "~", escLeftBracket3Routines
};

// Handle ESC [ 4 <more stuff> escape sequences
//
static CharacterDispatchRoutine escLeftBracket4Routines[] = {
	endKeyRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket4Dispatch = {
	1, "~", escLeftBracket4Routines
};

// Handle ESC [ 5 <more stuff> escape sequences
//
static CharacterDispatchRoutine escLeftBracket5Semicolon5Routines[] = {
	pageUpKeyRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket5Semicolon5Dispatch = {
	1, "~", escLeftBracket5Semicolon5Routines
};
static char32_t escLeftBracket5Semicolon5Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	thisKeyMetaCtrl |= Replxx::KEY::BASE_CONTROL;
	return doDispatch(c, escLeftBracket5Semicolon5Dispatch);
}
static CharacterDispatchRoutine escLeftBracket5SemicolonRoutines[] = {
	escLeftBracket5Semicolon5Routine,
	escFailureRoutine
};
static CharacterDispatch escLeftBracket5SemicolonDispatch = {
	1, "5", escLeftBracket5SemicolonRoutines
};
static char32_t escLeftBracket5SemicolonRoutine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket5SemicolonDispatch);
}

static CharacterDispatchRoutine escLeftBracket5Routines[] = {
	pageUpKeyRoutine, escLeftBracket5SemicolonRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket5Dispatch = {
	2, "~;", escLeftBracket5Routines
};

// Handle ESC [ 6 <more stuff> escape sequences
//
static CharacterDispatchRoutine escLeftBracket6Semicolon5Routines[] = {
	pageDownKeyRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket6Semicolon5Dispatch = {
	1, "~", escLeftBracket6Semicolon5Routines
};
static char32_t escLeftBracket6Semicolon5Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	thisKeyMetaCtrl |= Replxx::KEY::BASE_CONTROL;
	return doDispatch(c, escLeftBracket6Semicolon5Dispatch);
}
static CharacterDispatchRoutine escLeftBracket6SemicolonRoutines[] = {
	escLeftBracket6Semicolon5Routine,
	escFailureRoutine
};
static CharacterDispatch escLeftBracket6SemicolonDispatch = {
	1, "5", escLeftBracket6SemicolonRoutines
};
static char32_t escLeftBracket6SemicolonRoutine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket6SemicolonDispatch);
}

static CharacterDispatchRoutine escLeftBracket6Routines[] = {
	pageDownKeyRoutine, escLeftBracket6SemicolonRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket6Dispatch = {
	2, "~;", escLeftBracket6Routines
};

// Handle ESC [ 7 <more stuff> escape sequences
//
static CharacterDispatchRoutine escLeftBracket7Routines[] = {
	homeKeyRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket7Dispatch = {
	1, "~", escLeftBracket7Routines
};

// Handle ESC [ 8 <more stuff> escape sequences
//
static CharacterDispatchRoutine escLeftBracket8Routines[] = {
	endKeyRoutine, escFailureRoutine
};
static CharacterDispatch escLeftBracket8Dispatch = {
	1, "~", escLeftBracket8Routines
};

// Handle ESC [ <digit> escape sequences
//
static char32_t escLeftBracket0Routine(char32_t c) {
	return escFailureRoutine(c);
}
static char32_t escLeftBracket1Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket1Dispatch);
}
static char32_t escLeftBracket2Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket2Dispatch);
}
static char32_t escLeftBracket3Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket3Dispatch);
}
static char32_t escLeftBracket4Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket4Dispatch);
}
static char32_t escLeftBracket5Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket5Dispatch);
}
static char32_t escLeftBracket6Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket6Dispatch);
}
static char32_t escLeftBracket7Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket7Dispatch);
}
static char32_t escLeftBracket8Routine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracket8Dispatch);
}
static char32_t escLeftBracket9Routine(char32_t c) {
	return escFailureRoutine(c);
}

// Handle ESC [ <more stuff> escape sequences
//
static CharacterDispatchRoutine escLeftBracketRoutines[] = {
	upArrowKeyRoutine,      downArrowKeyRoutine,    rightArrowKeyRoutine,
	leftArrowKeyRoutine,    homeKeyRoutine,         endKeyRoutine,
	shiftTabRoutine,
	escLeftBracket0Routine, escLeftBracket1Routine, escLeftBracket2Routine,
	escLeftBracket3Routine, escLeftBracket4Routine, escLeftBracket5Routine,
	escLeftBracket6Routine, escLeftBracket7Routine, escLeftBracket8Routine,
	escLeftBracket9Routine, escFailureRoutine
};
static CharacterDispatch escLeftBracketDispatch = {17, "ABCDHFZ0123456789",
																									 escLeftBracketRoutines};

// Handle ESC O <char> escape sequences
//
static CharacterDispatchRoutine escORoutines[] = {
	upArrowKeyRoutine,       downArrowKeyRoutine,     rightArrowKeyRoutine,
	leftArrowKeyRoutine,     homeKeyRoutine,          endKeyRoutine,
	f1KeyRoutine,            f2KeyRoutine,            f3KeyRoutine,
	f4KeyRoutine,
	ctrlUpArrowKeyRoutine,   ctrlDownArrowKeyRoutine, ctrlRightArrowKeyRoutine,
	ctrlLeftArrowKeyRoutine, escFailureRoutine
};
static CharacterDispatch escODispatch = {14, "ABCDHFPQRSabcd", escORoutines};

// Initial ESC dispatch -- could be a Meta prefix or the start of an escape
// sequence
//
static char32_t escLeftBracketRoutine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escLeftBracketDispatch);
}
static char32_t escORoutine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escODispatch);
}
static char32_t setMetaRoutine(char32_t c);	// need forward reference
static CharacterDispatchRoutine escRoutines[] = {
	escLeftBracketRoutine, escORoutine, setMetaRoutine
};
static CharacterDispatch escDispatch = {2, "[O", escRoutines};

// Initial dispatch -- we are not in the middle of anything yet
//
static char32_t escRoutine(char32_t c) {
	c = read_unicode_character();
	if (c == 0) return 0;
	return doDispatch(c, escDispatch);
}
static CharacterDispatchRoutine initialRoutines[] = {
	escRoutine, deleteCharRoutine, normalKeyRoutine
};
static CharacterDispatch initialDispatch = {2, "\x1B\x7F", initialRoutines};

// Special handling for the ESC key because it does double duty
//
static char32_t setMetaRoutine(char32_t c) {
	thisKeyMetaCtrl = Replxx::KEY::BASE_META;
	if (c == 0x1B) {	// another ESC, stay in ESC processing mode
		c = read_unicode_character();
		if (c == 0) return 0;
		return doDispatch(c, escDispatch);
	}
	return doDispatch(c, initialDispatch);
}

char32_t doDispatch(char32_t c) {
	EscapeSequenceProcessing::thisKeyMetaCtrl = 0;	// no modifiers yet at initialDispatch
	return doDispatch(c, initialDispatch);
}

}	// namespace EscapeSequenceProcessing // move these out of global namespace

}

#endif /* #ifndef _WIN32 */

