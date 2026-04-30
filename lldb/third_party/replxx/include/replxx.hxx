/*
 * Copyright (c) 2017-2018, Marcin Konarski (amok at codestation.org)
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of Redis nor the names of its contributors may be used
 *     to endorse or promote products derived from this software without
 *     specific prior written permission.
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
 */

#ifndef HAVE_REPLXX_HXX_INCLUDED
#define HAVE_REPLXX_HXX_INCLUDED 1

#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <iosfwd>

/*
 * For use in Windows DLLs:
 *
 * If you are building replxx into a DLL,
 * unless you are using supplied CMake based build,
 * ensure that 'REPLXX_BUILDING_DLL' is defined when
 * building the DLL so that proper symbols are exported.
 */
#if defined( _WIN32 ) && ! defined( REPLXX_STATIC )
#	ifdef REPLXX_BUILDING_DLL
#		define REPLXX_IMPEXP __declspec( dllexport )
#	else
#		define REPLXX_IMPEXP __declspec( dllimport )
#	endif
#else
#	define REPLXX_IMPEXP /**/
#endif

#ifdef ERROR
enum { ERROR_BB1CA97EC761FC37101737BA0AA2E7C5 = ERROR };
#undef ERROR
enum { ERROR = ERROR_BB1CA97EC761FC37101737BA0AA2E7C5 };
#endif
#ifdef ABORT
enum { ABORT_8D12A2CA7E5A64036D7251A3EDA51A38 = ABORT };
#undef ABORT
enum { ABORT = ABORT_8D12A2CA7E5A64036D7251A3EDA51A38 };
#endif
#ifdef DELETE
enum { DELETE_32F68A60CEF40FAEDBC6AF20298C1A1E = DELETE };
#undef DELETE
enum { DELETE = DELETE_32F68A60CEF40FAEDBC6AF20298C1A1E };
#endif

namespace replxx {

class REPLXX_IMPEXP Replxx {
public:
	enum class Color : int {
		BLACK         = 0,
		RED           = 1,
		GREEN         = 2,
		BROWN         = 3,
		BLUE          = 4,
		MAGENTA       = 5,
		CYAN          = 6,
		LIGHTGRAY     = 7,
		GRAY          = 8,
		BRIGHTRED     = 9,
		BRIGHTGREEN   = 10,
		YELLOW        = 11,
		BRIGHTBLUE    = 12,
		BRIGHTMAGENTA = 13,
		BRIGHTCYAN    = 14,
		WHITE         = 15,
		DEFAULT       = 1u << 16u
	};
	struct KEY {
		static char32_t const BASE         = 0x0010ffff + 1;
		static char32_t const BASE_SHIFT   = 0x01000000;
		static char32_t const BASE_CONTROL = 0x02000000;
		static char32_t const BASE_META    = 0x04000000;
		static char32_t const ESCAPE       = 27;
		static char32_t const PAGE_UP      = BASE      + 1;
		static char32_t const PAGE_DOWN    = PAGE_UP   + 1;
		static char32_t const DOWN         = PAGE_DOWN + 1;
		static char32_t const UP           = DOWN      + 1;
		static char32_t const LEFT         = UP        + 1;
		static char32_t const RIGHT        = LEFT      + 1;
		static char32_t const HOME         = RIGHT     + 1;
		static char32_t const END          = HOME      + 1;
		static char32_t const DELETE       = END       + 1;
		static char32_t const INSERT       = DELETE    + 1;
		static char32_t const F1           = INSERT    + 1;
		static char32_t const F2           = F1        + 1;
		static char32_t const F3           = F2        + 1;
		static char32_t const F4           = F3        + 1;
		static char32_t const F5           = F4        + 1;
		static char32_t const F6           = F5        + 1;
		static char32_t const F7           = F6        + 1;
		static char32_t const F8           = F7        + 1;
		static char32_t const F9           = F8        + 1;
		static char32_t const F10          = F9        + 1;
		static char32_t const F11          = F10       + 1;
		static char32_t const F12          = F11       + 1;
		static char32_t const F13          = F12       + 1;
		static char32_t const F14          = F13       + 1;
		static char32_t const F15          = F14       + 1;
		static char32_t const F16          = F15       + 1;
		static char32_t const F17          = F16       + 1;
		static char32_t const F18          = F17       + 1;
		static char32_t const F19          = F18       + 1;
		static char32_t const F20          = F19       + 1;
		static char32_t const F21          = F20       + 1;
		static char32_t const F22          = F21       + 1;
		static char32_t const F23          = F22       + 1;
		static char32_t const F24          = F23       + 1;
		static char32_t const MOUSE        = F24       + 1;
		static char32_t const PASTE_START  = MOUSE     + 1;
		static char32_t const PASTE_FINISH = PASTE_START + 1;
		static constexpr char32_t shift( char32_t key_ ) {
			return ( key_ | BASE_SHIFT );
		}
		static constexpr char32_t control( char32_t key_ ) {
			return ( key_ | BASE_CONTROL );
		}
		static constexpr char32_t meta( char32_t key_ ) {
			return ( key_ | BASE_META );
		}
		static char32_t const BACKSPACE    = 'H' | BASE_CONTROL;
		static char32_t const TAB          = 'I' | BASE_CONTROL;
		static char32_t const ENTER        = 'M' | BASE_CONTROL;
		static char32_t const ABORT        = 'C' | BASE_CONTROL | BASE_META;
	};
	/*! \brief List of built-in actions that act upon user input.
	 */
	enum class ACTION {
		INSERT_CHARACTER,
		NEW_LINE,
		DELETE_CHARACTER_UNDER_CURSOR,
		DELETE_CHARACTER_LEFT_OF_CURSOR,
		KILL_TO_END_OF_LINE,
		KILL_TO_BEGINING_OF_LINE,
		KILL_TO_END_OF_WORD,
		KILL_TO_BEGINING_OF_WORD,
		KILL_TO_END_OF_SUBWORD,
		KILL_TO_BEGINING_OF_SUBWORD,
		KILL_TO_WHITESPACE_ON_LEFT,
		YANK,
		YANK_CYCLE,
		YANK_LAST_ARG,
		MOVE_CURSOR_TO_BEGINING_OF_LINE,
		MOVE_CURSOR_TO_END_OF_LINE,
		MOVE_CURSOR_ONE_WORD_LEFT,
		MOVE_CURSOR_ONE_WORD_RIGHT,
		MOVE_CURSOR_ONE_SUBWORD_LEFT,
		MOVE_CURSOR_ONE_SUBWORD_RIGHT,
		MOVE_CURSOR_LEFT,
		MOVE_CURSOR_RIGHT,
		LINE_NEXT,
		LINE_PREVIOUS,
		HISTORY_NEXT,
		HISTORY_PREVIOUS,
		HISTORY_FIRST,
		HISTORY_LAST,
		HISTORY_RESTORE,
		HISTORY_RESTORE_CURRENT,
		HISTORY_INCREMENTAL_SEARCH,
		HISTORY_SEEDED_INCREMENTAL_SEARCH,
		HISTORY_COMMON_PREFIX_SEARCH,
		HINT_NEXT,
		HINT_PREVIOUS,
		CAPITALIZE_WORD,
		LOWERCASE_WORD,
		UPPERCASE_WORD,
		CAPITALIZE_SUBWORD,
		LOWERCASE_SUBWORD,
		UPPERCASE_SUBWORD,
		TRANSPOSE_CHARACTERS,
		TOGGLE_OVERWRITE_MODE,
#ifndef _WIN32
		VERBATIM_INSERT,
		SUSPEND,
#endif
		BRACKETED_PASTE,
		CLEAR_SCREEN,
		CLEAR_SELF,
		REPAINT,
		COMPLETE_LINE,
		COMPLETE_NEXT,
		COMPLETE_PREVIOUS,
		COMMIT_LINE,
		ABORT_LINE,
		SEND_EOF
	};
	/*! \brief Possible results of key-press handler actions.
	 */
	enum class ACTION_RESULT {
		CONTINUE, /*!< Continue processing user input. */
		RETURN,   /*!< Return user input entered so far. */
		BAIL      /*!< Stop processing user input, returns nullptr from the \e input() call. */
	};
	typedef std::vector<Color> colors_t;
	class Completion {
		std::string _text;
		Color _color;
	public:
		Completion( char const* text_ )
			: _text( text_ )
			, _color( Color::DEFAULT ) {
		}
		Completion( std::string const& text_ )
			: _text( text_ )
			, _color( Color::DEFAULT ) {
		}
		Completion( std::string const& text_, Color color_ )
			: _text( text_ )
			, _color( color_ ) {
		}
		std::string const& text( void ) const {
			return ( _text );
		}
		Color color( void ) const {
			return ( _color );
		}
	};
	typedef std::vector<Completion> completions_t;
	class HistoryEntry {
		std::string _timestamp;
		std::string _text;
	public:
		HistoryEntry( std::string const& timestamp_, std::string const& text_ )
			: _timestamp( timestamp_ )
			, _text( text_ ) {
		}
		std::string const& timestamp( void ) const {
			return ( _timestamp );
		}
		std::string const& text( void ) const {
			return ( _text );
		}
	};
	class HistoryScanImpl;
	class HistoryScan {
	public:
		typedef std::unique_ptr<HistoryScanImpl, void (*)( HistoryScanImpl* )> impl_t;
	private:
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4251)
#endif
		impl_t _impl;
#ifdef _MSC_VER
#pragma warning(pop)
#endif
	public:
		HistoryScan( impl_t );
		HistoryScan( HistoryScan&& ) = default;
		HistoryScan& operator = ( HistoryScan&& ) = default;
		bool next( void );
		HistoryEntry const& get( void ) const;
	private:
		HistoryScan( HistoryScan const& ) = delete;
		HistoryScan& operator = ( HistoryScan const& ) = delete;
	};
	typedef std::vector<std::string> hints_t;

	/*! \brief Line modification callback type definition.
	 *
	 * User can observe and modify line contents (and cursor position)
	 * in response to changes to both introduced by the user through
	 * normal interactions.
	 *
	 * When callback returns Replxx updates current line content
	 * and current cursor position to the ones updated by the callback.
	 *
	 * \param line[in,out] - a R/W reference to an UTF-8 encoded input entered by the user so far.
	 * \param cursorPosition[in,out] - a R/W reference to current cursor position.
	 */
	typedef std::function<void ( std::string& line, int& cursorPosition )> modify_callback_t;

	/*! \brief Completions callback type definition.
	 *
	 * \e contextLen is counted in Unicode code points (not in bytes!).
	 *
	 * For user input:
	 * if ( obj.me
	 *
	 * input == "if ( obj.me"
	 * contextLen == 2 (depending on \e set_word_break_characters())
	 *
	 * Client application is free to update \e contextLen to be 6 (or any other non-negative
	 * number not greater than the number of code points in input) if it makes better sense
	 * for given client application semantics.
	 *
	 * \param input - UTF-8 encoded input entered by the user until current cursor position.
	 * \param[in,out] contextLen - length of the additional context to provide while displaying completions.
	 * \return A list of user completions.
	 */
	typedef std::function<completions_t ( std::string const& input, int& contextLen )> completion_callback_t;

	/*! \brief Highlighter callback type definition.
	 *
	 * If user want to have colorful input she must simply install highlighter callback.
	 * The callback would be invoked by the library after each change to the input done by
	 * the user. After callback returns library uses data from colors buffer to colorize
	 * displayed user input.
	 *
	 * Size of \e colors buffer is equal to number of code points in user \e input
	 * which will be different from simple `input.length()`!
	 *
	 * \param input - an UTF-8 encoded input entered by the user so far.
	 * \param colors - output buffer for color information.
	 */
	typedef std::function<void ( std::string const& input, colors_t& colors )> highlighter_callback_t;

	/*! \brief Hints callback type definition.
	 *
	 * \e contextLen is counted in Unicode code points (not in bytes!).
	 *
	 * For user input:
	 * if ( obj.me
	 *
	 * input == "if ( obj.me"
	 * contextLen == 2 (depending on \e set_word_break_characters())
	 *
	 * Client application is free to update \e contextLen to be 6 (or any other non-negative
	 * number not greater than the number of code points in input) if it makes better sense
	 * for given client application semantics.
	 *
	 * \param input - UTF-8 encoded input entered by the user until current cursor position.
	 * \param contextLen[in,out] - length of the additional context to provide while displaying hints.
	 * \param color - a color used for displaying hints.
	 * \return A list of possible hints.
	 */
	typedef std::function<hints_t ( std::string const& input, int& contextLen, Color& color )> hint_callback_t;

	/*! \brief Key press handler type definition.
	 *
	 * \param code - the key code replxx got from terminal.
	 * \return Decision on how should input() behave after this key handler returns.
	 */
	typedef std::function<ACTION_RESULT ( char32_t code )> key_press_handler_t;

	struct State {
		char const* _text;
		int _cursorPosition;
		State( char const* text_, int cursorPosition_ = -1 )
			: _text( text_ )
			, _cursorPosition( cursorPosition_ ) {
		}
		State( State const& ) = default;
		State& operator = ( State const& ) = default;
		char const* text( void ) const {
			return ( _text );
		}
		int cursor_position( void ) const {
			return ( _cursorPosition );
		}
	};

	class ReplxxImpl;
private:
	typedef std::unique_ptr<ReplxxImpl, void (*)( ReplxxImpl* )> impl_t;
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4251)
#endif
	impl_t _impl;
#ifdef _MSC_VER
#pragma warning(pop)
#endif

public:
	Replxx( void );
	Replxx( Replxx&& ) = default;
	Replxx& operator = ( Replxx&& ) = default;

	/*! \brief Register modify callback.
	 *
	 * \param fn - user defined callback function.
	 */
	void set_modify_callback( modify_callback_t const& fn );

	/*! \brief Register completion callback.
	 *
	 * \param fn - user defined callback function.
	 */
	void set_completion_callback( completion_callback_t const& fn );

	/*! \brief Register highlighter callback.
	 *
	 * \param fn - user defined callback function.
	 */
	void set_highlighter_callback( highlighter_callback_t const& fn );

	/*! \brief Register hints callback.
	 *
	 * \param fn - user defined callback function.
	 */
	void set_hint_callback( hint_callback_t const& fn );

	/*! \brief Read line of user input.
	 *
	 * Returned pointer is managed by the library and is not to be freed in the client.
	 *
	 * \param prompt - prompt to be displayed before getting user input.
	 * \return An UTF-8 encoded input given by the user (or nullptr on EOF).
	 */
	char const* input( std::string const& prompt );

	/*! \brief Get current state data.
	 *
	 * This call is intended to be used in handlers.
	 *
	 * \return Current state of the model.
	 */
	State get_state( void ) const;

	/*! \brief Set new state data.
	 *
	 * This call is intended to be used in handlers.
	 *
	 * \param state - new state of the model.
	 */
	void set_state( State const& state );

	/*! \brief Enable/disable case insensitive history search and completion.
	 *
	 * \param val - if set to non-zero then history search and completion will be case insensitive.
	 */
	void set_ignore_case( bool val );

	/*! \brief Print formatted string to standard output.
	 *
	 * This function ensures proper handling of ANSI escape sequences
	 * contained in printed data, which is especially useful on Windows
	 * since Unixes handle them correctly out of the box.
	 *
	 * \param fmt - printf style format.
	 */
	void print( char const* fmt, ... );

	/*! \brief Prints a char array with the given length to standard output.
	 *
	 * \copydetails print
	 *
	 * \param str - The char array to print.
	 * \param length - The length of the array.
	 */
	void write( char const* str, int length );

	/*! \brief Asynchronously change the prompt while replxx_input() call is in efect.
	 *
	 * Can be used to change the prompt from callbacks or other threads.
	 *
	 * \param prompt - The prompt string to change to.
	 */
	void set_prompt( std::string prompt );

	/*! \brief Schedule an emulated key press event.
	 *
	 * \param code - key press code to be emulated.
	 */
	void emulate_key_press( char32_t code );

	/*! \brief Invoke built-in action handler.
	 *
	 * \pre This method can be called only from key-press handler.
	 *
	 * \param action - a built-in action to invoke.
	 * \param code - a supplementary key-code to consume by built-in action handler.
	 * \return The action result informing the replxx what shall happen next.
	 */
	ACTION_RESULT invoke( ACTION action, char32_t code );

	/*! \brief Bind user defined action to handle given key-press event.
	 *
	 * \param code - handle this key-press event with following handler.
	 * \param handle - use this handler to handle key-press event.
	 */
	void bind_key( char32_t code, key_press_handler_t handler );

	/*! \brief Bind internal `replxx` action (by name) to handle given key-press event.
	 *
	 * Action names are the same as names of Replxx::ACTION enumerations
	 * but in lower case, e.g.: an action for recalling previous history line
	 * is \e Replxx::ACTION::LINE_PREVIOUS so action name to be used in this
	 * interface for the same effect is "line_previous".
	 *
	 * \param code - handle this key-press event with following handler.
	 * \param actionName - name of internal action to be invoked on key press.
	 */
	void bind_key_internal( char32_t code, char const* actionName );

	void history_add( std::string const& line );

	/*! \brief Synchronize REPL's history with given file.
	 *
	 * Synchronizing means loading existing history from given file,
	 * merging it with current history sorted by timestamps,
	 * saving merged version to given file,
	 * keeping merged version as current REPL's history.
	 *
	 * This call is an equivalent of calling:
	 * history_save( "some-file" );
	 * history_load( "some-file" );
	 *
	 * \param filename - a path to the file with which REPL's current history should be synchronized.
	 * \return True iff history file was successfully created.
	 */
	bool history_sync( std::string const& filename );

	/*! \brief Save REPL's history into given file.
	 *
	 * Saving means loading existing history from given file,
	 * merging it with current history sorted by timestamps,
	 * saving merged version to given file,
	 * keeping original (NOT merged) version as current REPL's history.
	 *
	 * \param filename - a path to the file where REPL's history should be saved.
	 * \return True iff history file was successfully created.
	 */
	bool history_save( std::string const& filename );

	/*!
	 * \copydoc history_save
	 */
	void history_save( std::ostream& out );

	/*! \brief Load REPL's history from given file.
	 *
	 * \param filename - a path to the file which contains REPL's history that should be loaded.
	 * \return True iff history file was successfully opened.
	 */
	bool history_load( std::string const& filename );

	/*!
	 * \copydoc history_load
	 */
	void history_load( std::istream& in );

	/*! \brief Clear REPL's in-memory history.
	 */
	void history_clear( void );
	int history_size( void ) const;
	HistoryScan history_scan( void ) const;

	void set_preload_buffer( std::string const& preloadText );

	/*! \brief Set set of word break characters.
	 *
	 * This setting influences word based cursor movement and line editing capabilities.
	 *
	 * \param wordBreakers - 7-bit ASCII set of word breaking characters.
	 */
	void set_word_break_characters( char const* wordBreakers );

	/*! \brief How many completions should trigger pagination.
	 */
	void set_completion_count_cutoff( int count );

	/*! \brief Set maximum number of displayed hint rows.
	 */
	void set_max_hint_rows( int count );

	/*! \brief Set a delay before hint are shown after user stopped typing..
	 *
	 * \param milliseconds - a number of milliseconds to wait before showing hints.
	 */
	void set_hint_delay( int milliseconds );

	/*! \brief Set tab completion behavior.
	 *
	 * \param val - use double tab to invoke completions.
	 */
	void set_double_tab_completion( bool val );

	/*! \brief Set tab completion behavior.
	 *
	 * \param val - invoke completion even if user input is empty.
	 */
	void set_complete_on_empty( bool val );

	/*! \brief Set tab completion behavior.
	 *
	 * \param val - beep if completion is ambiguous.
	 */
	void set_beep_on_ambiguous_completion( bool val );

	/*! \brief Set complete next/complete previous behavior.
	 *
	 * COMPLETE_NEXT/COMPLETE_PREVIOUS actions have two modes of operations,
	 * in case when a partial completion is possible complete only partial part (`false` setting)
	 * or complete first proposed completion fully (`true` setting).
	 * The default is to complete fully (a `true` setting - complete immediately).
	 *
	 * \param val - complete immediately.
	 */
	void set_immediate_completion( bool val );

	/*! \brief Set history duplicate entries behaviour.
	 *
	 * \param val - should history contain only unique entries?
	 */
	void set_unique_history( bool val );

	/*! \brief Disable output coloring.
	 *
	 * \param val - if set to non-zero disable output colors.
	 */
	void set_no_color( bool val );

	/*! \brief Enable/disable (prompt width) indent for multiline entry.
	 *
	 * \param val - if set to true then multiline indent will be enabled.
	 */
	void set_indent_multiline( bool val );

	/*! \brief Set maximum number of entries in history list.
	 */
	void set_max_history_size( int len );
	void clear_screen( void );
	int install_window_change_handler( void );
	void enable_bracketed_paste( void );
	void disable_bracketed_paste( void );

private:
	Replxx( Replxx const& ) = delete;
	Replxx& operator = ( Replxx const& ) = delete;
};

/*! \brief Color definition related helper function.
 *
 * To be used to leverage 256 color terminal capabilities.
 */
namespace color {

/*! \brief Combine two color definitions to get encompassing color definition.
 *
 * To be used only for combining foreground and background colors.
 *
 * \param color1 - first input color.
 * \param color2 - second input color.
 * \return A new color definition that represent combined input colors.
 */
Replxx::Color operator | ( Replxx::Color color1, Replxx::Color color2 );

/*! \brief Transform foreground color definition into a background color definition.
 *
 * \param color - an input foreground color definition.
 * \return A background color definition that is a transformed input \e color.
 */
Replxx::Color bg( Replxx::Color color );

/*! \brief Add `bold` attribute to color definition.
 *
 * \param color - an input color definition.
 * \return A new color definition with bold attribute set.
 */
Replxx::Color bold( Replxx::Color color );

/*! \brief Add `underline` attribute to color definition.
 *
 * \param color - an input color definition.
 * \return A new color definition with underline attribute set.
 */
Replxx::Color underline( Replxx::Color color );

/*! \brief Create a new grayscale color of given brightness level.
 *
 * \param level - a brightness level for new color, must be between 0 (darkest) and 23 (brightest).
 * \return A new grayscale color of a given brightest \e level.
 */
Replxx::Color grayscale( int level );

/*! \brief Create a new color in 6×6×6 RGB color space from base component levels.
 *
 * \param red - a red (of RGB) component level, must be 0 and 5.
 * \param green - a green (of RGB) component level, must be 0 and 5.
 * \param blue - a blue (of RGB) component level, must be 0 and 5.
 * \return A new color in 6×6×6 RGB color space.
 */
Replxx::Color rgb666( int red, int green, int blue );

}

}

#endif /* HAVE_REPLXX_HXX_INCLUDED */

