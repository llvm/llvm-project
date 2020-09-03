//===-- EditLineWin.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// this file is only relevant for Visual C++
#if defined(_WIN32)

#include "lldb/Host/windows/windows.h"

#include "lldb/Host/windows/editlinewin.h"
#include "llvm/Support/ErrorHandling.h"
#include <assert.h>
#include <vector>

// edit line EL_ADDFN function pointer type
typedef unsigned char (*el_addfn_func)(EditLine *e, int ch);
typedef const char *(*el_prompt_func)(EditLine *);

// edit line wrapper binding container
struct el_binding {
  //
  const char *name;
  const char *help;
  // function pointer to callback routine
  el_addfn_func func;
  // ascii key this function is bound to
  const char *key;
};

// stored key bindings
static std::vector<el_binding *> _bindings;

// TODO: this should in fact be related to the exact edit line context we create
static void *clientData = NULL;

// store the current prompt string
// default to what we expect to receive anyway
static const char *_prompt = "(lldb) ";

#if !defined(_WIP_INPUT_METHOD)

static char *el_get_s(char *buffer, int chars) { return gets_s(buffer, chars); }
#else

static void con_output(char _in) {
  HANDLE hout = GetStdHandle(STD_OUTPUT_HANDLE);
  DWORD written = 0;
  // get the cursor position
  CONSOLE_SCREEN_BUFFER_INFO info;
  GetConsoleScreenBufferInfo(hout, &info);
  // output this char
  WriteConsoleOutputCharacterA(hout, &_in, 1, info.dwCursorPosition, &written);
  // advance cursor position
  info.dwCursorPosition.X++;
  SetConsoleCursorPosition(hout, info.dwCursorPosition);
}

static void con_backspace(void) {
  HANDLE hout = GetStdHandle(STD_OUTPUT_HANDLE);
  DWORD written = 0;
  // get cursor position
  CONSOLE_SCREEN_BUFFER_INFO info;
  GetConsoleScreenBufferInfo(hout, &info);
  // nudge cursor backwards
  info.dwCursorPosition.X--;
  SetConsoleCursorPosition(hout, info.dwCursorPosition);
  // blank out the last character
  WriteConsoleOutputCharacterA(hout, " ", 1, info.dwCursorPosition, &written);
}

static void con_return(void) {
  HANDLE hout = GetStdHandle(STD_OUTPUT_HANDLE);
  DWORD written = 0;
  // get cursor position
  CONSOLE_SCREEN_BUFFER_INFO info;
  GetConsoleScreenBufferInfo(hout, &info);
  // move onto the new line
  info.dwCursorPosition.X = 0;
  info.dwCursorPosition.Y++;
  SetConsoleCursorPosition(hout, info.dwCursorPosition);
}

static bool runBind(char _key) {
  for (int i = 0; i < _bindings.size(); i++) {
    el_binding *bind = _bindings[i];
    if (bind->key[0] == _key) {
      bind->func((EditLine *)-1, _key);
      return true;
    }
  }
  return false;
}

// replacement get_s which is EL_BIND aware
static char *el_get_s(char *buffer, int chars) {
  //
  char *head = buffer;
  //
  for (;; Sleep(10)) {
    //
    INPUT_RECORD _record;
    //
    DWORD _read = 0;
    if (ReadConsoleInputA(GetStdHandle(STD_INPUT_HANDLE), &_record, 1,
                          &_read) == FALSE)
      break;
    // if we didn't read a key
    if (_read == 0)
      continue;
    // only interested in key events
    if (_record.EventType != KEY_EVENT)
      continue;
    // is the key down
    if (!_record.Event.KeyEvent.bKeyDown)
      continue;
    // read the ascii key character
    char _key = _record.Event.KeyEvent.uChar.AsciiChar;
    // non ascii conformant key press
    if (_key == 0) {
      // check the scan code
      // if VK_UP scroll back through history
      // if VK_DOWN scroll forward through history
      continue;
    }
    // try to execute any bind this key may have
    if (runBind(_key))
      continue;
    // if we read a return key
    if (_key == '\n' || _key == '\r') {
      con_return();
      break;
    }
    // key is backspace
    if (_key == 0x8) {
      // avoid deleting past beginning
      if (head > buffer) {
        con_backspace();
        head--;
      }
      continue;
    }

    // add this key to the input buffer
    if ((head - buffer) < (chars - 1)) {
      con_output(_key);
      *(head++) = _key;
    }
  }
  // insert end of line character
  *head = '\0';

  return buffer;
}
#endif

// edit line initialize
EditLine *el_init(const char *, FILE *, FILE *, FILE *) {
  //
  SetConsoleTitleA("lldb");
  // return dummy handle
  return (EditLine *)-1;
}

const char *el_gets(EditLine *el, int *length) {
  // print the prompt if we have one
  if (_prompt != NULL)
    printf("%s", _prompt);
  // create a buffer for the user input
  char *buffer = new char[MAX_PATH];
  // try to get user input string
  if (el_get_s(buffer, MAX_PATH)) {
    // get the string length in 'length'
    while (buffer[*length] != '\0')
      (*length)++;
    // return the input buffer
    // remember that this memory has the be free'd somewhere
    return buffer;
  } else {
    // on error
    delete[] buffer;
    return NULL;
  }
}

int el_set(EditLine *el, int code, ...) {
  va_list vl;
  va_start(vl, code);
  //
  switch (code) {
  // edit line set prompt message
  case (EL_PROMPT): {
    // EL_PROMPT, char *(*f)( EditLine *)
    //      define a prompt printing function as 'f', which is to return a
    //      string that
    //      contains the prompt.

    // get the function pointer from the arg list
    void *func_vp = (void *)va_arg(vl, el_prompt_func);
    // cast to suitable prototype
    el_prompt_func func_fp = (el_prompt_func)func_vp;
    // call to get the prompt as a string
    _prompt = func_fp(el);
  } break;

  case (EL_PROMPT_ESC): {
    // EL_PROMPT, char *(*f)( EditLine *)
    //      define a prompt printing function as 'f', which is to return a
    //      string that
    //      contains the prompt.

    // get the function pointer from the arg list
    void *func_vp = (void *)va_arg(vl, el_prompt_func);
    va_arg(vl, int);
    // call to get the prompt as a string
    el_prompt_func func_fp = (el_prompt_func)func_vp;
    _prompt = func_fp(el);
  } break;

  case (EL_EDITOR): {
    // EL_EDITOR, const char *mode
    //      set editing mode to "emacs" or "vi"
  } break;
  case (EL_HIST): {
    // EL_HIST, History *(*fun)(History *, int op, ... ), const char *ptr
    //      defines which history function to use, which is usually history().
    //      Ptr should be the
    //      value returned by history_init().
  } break;
  case (EL_ADDFN): {
    // EL_ADDFN, const char *name, const char *help, unsigned char
    // (*func)(EditLine *e, int ch)
    //      add a user defined function, func), referred to as 'name' which is
    //      invoked when a key which is bound to 'name' is
    //      entered. 'help' is a description of 'name'. at invocation time, 'ch'
    //      is the key which caused the invocation. the
    //      return value of 'func()' should be one of:
    //          CC_NORM         add a normal character
    //          CC_NEWLINE      end of line was entered
    //          CC_EOF          EOF was entered
    //          CC_ARGHACK      expecting further command input as arguments, do
    //          nothing visually.
    //          CC_REFRESH      refresh display.
    //          CC_REFRESH_BEEP refresh display and beep.
    //          CC_CURSOR       cursor moved so update and perform CC_REFRESH
    //          CC_REDISPLAY        redisplay entire input line. this is useful
    //          if a key binding outputs extra information.
    //          CC_ERROR            an error occurred. beep and flush tty.
    //          CC_FATAL            fatal error, reset tty to known state.

    el_binding *binding = new el_binding;
    binding->name = va_arg(vl, const char *);
    binding->help = va_arg(vl, const char *);
    binding->func = va_arg(vl, el_addfn_func);
    binding->key = 0;
    // add this to the bindings list
    _bindings.push_back(binding);
  } break;
  case (EL_BIND): {
    // EL_BIND, const char *, ..., NULL
    //      perform the BIND built-in command.  Refer to editrc(5) for more
    //      information.

    const char *name = va_arg(vl, const char *);

    for (auto bind : _bindings) {
      if (strcmp(bind->name, name) == 0) {
        bind->key = va_arg(vl, const char *);
        break;
      }
    }

  } break;
  case (EL_CLIENTDATA): {
    clientData = va_arg(vl, void *);
  } break;
  }
  return 0;
}

void el_end(EditLine *el) {
  // assert( !"Not implemented!" );
}

void el_reset(EditLine *) { llvm_unreachable("Not implemented!"); }

int el_getc(EditLine *, char *) {
  llvm_unreachable("Not implemented!");
}

void el_push(EditLine *, const char *) {}

void el_beep(EditLine *) { Beep(1000, 500); }

int el_parse(EditLine *, int, const char **) {
  llvm_unreachable("Not implemented!");
}

int el_get(EditLine *el, int code, ...) {
  va_list vl;
  va_start(vl, code);

  switch (code) {
  case (EL_CLIENTDATA): {
    void **dout = va_arg(vl, void **);
    *dout = clientData;
  } break;
  default:
    llvm_unreachable("Not implemented!");
  }
  return 0;
}

int el_source(EditLine *el, const char *file) {
  // init edit line by reading the contents of 'file' nothing to do here on
  // windows...
  return 0;
}

void el_resize(EditLine *) { llvm_unreachable("Not implemented!"); }

const LineInfo *el_line(EditLine *el) { return 0; }

int el_insertstr(EditLine *, const char *) {
  //    assert( !"Not implemented!" );
  return 0;
}

void el_deletestr(EditLine *, int) { llvm_unreachable("Not implemented!"); }

History *history_init(void) {
  // return dummy handle
  return (History *)-1;
}

void history_end(History *) {
  //    assert( !"Not implemented!" );
}

int history(History *, HistEvent *, int op, ...) {
  // perform operation 'op' on the history list with optional arguments as
  // needed by the operation.
  return 0;
}

#endif
