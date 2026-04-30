//===-- Editline.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <climits>
#include <iomanip>
#include <optional>

#include "lldb/Host/Editline.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/StreamFile.h"
#include "lldb/Utility/AnsiTerminal.h"
#include "lldb/Utility/CompletionRequest.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/StringList.h"
#include "lldb/lldb-forward.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Locale.h"

#if !defined(_WIN32)
#include <pthread.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#include "lldb/Host/windows/windows.h"
#endif

using namespace lldb_private;
using namespace lldb_private::line_editor;

/// https://www.ecma-international.org/publications/files/ECMA-ST/Ecma-048.pdf
#define ESCAPE "\x1b"
#define ANSI_CLEAR_BELOW ESCAPE "[J"
#define ANSI_SET_COLUMN_N ESCAPE "[%dG"

static size_t ColumnWidth(llvm::StringRef str) {
  std::string stripped = ansi::StripAnsiTerminalCodes(str);
  return llvm::sys::locale::columnWidth(stripped);
}

// ===----------------------------------------------------------------------===//
// Private methods
// ===----------------------------------------------------------------------===//

void Editline::SetCurrentLine(int line_index) {
  m_current_line_index = line_index;
  m_current_prompt = PromptForIndex(line_index);
}

std::string Editline::PromptForIndex(int line_index) {
  bool use_line_numbers = m_base_line_number > 0;
  std::string prompt = m_set_prompt;
  if (use_line_numbers && prompt.empty())
    prompt = ": ";
  std::string continuation_prompt = prompt;
  if (!m_set_continuation_prompt.empty()) {
    continuation_prompt = m_set_continuation_prompt;
    const size_t prompt_width = ColumnWidth(prompt);
    const size_t cont_prompt_width = ColumnWidth(continuation_prompt);
    const size_t padded_width = std::max(prompt_width, cont_prompt_width);
    if (prompt_width < padded_width)
      prompt += std::string(padded_width - prompt_width, ' ');
    else if (cont_prompt_width < padded_width)
      continuation_prompt +=
          std::string(padded_width - cont_prompt_width, ' ');
  }

  if (use_line_numbers) {
    StreamString prompt_stream;
    prompt_stream.Printf(
        "%*d%s", m_line_number_digits, m_base_line_number + line_index,
        (line_index == 0) ? prompt.c_str() : continuation_prompt.c_str());
    return std::string(std::move(prompt_stream.GetString()));
  }
  return (line_index == 0) ? prompt : continuation_prompt;
}

size_t Editline::GetPromptWidth() { return ColumnWidth(PromptForIndex(0)); }

const char *Editline::GetHistoryFilePath() {
  if (m_history_path.empty() && !m_editor_name.empty()) {
    llvm::SmallString<128> lldb_history_file;
    FileSystem::Instance().GetHomeDirectory(lldb_history_file);
    llvm::sys::path::append(lldb_history_file, ".lldb");
    if (!llvm::sys::fs::create_directory(lldb_history_file)) {
      std::string filename = m_editor_name + "-history";
      llvm::sys::path::append(lldb_history_file, filename);
      m_history_path = std::string(lldb_history_file.str());
    }
  }
  return m_history_path.empty() ? nullptr : m_history_path.c_str();
}

void Editline::UpdateTerminalSize() {
#if defined(_WIN32)
  CONSOLE_SCREEN_BUFFER_INFO csbi;
  if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
    m_terminal_width = csbi.srWindow.Right - csbi.srWindow.Left + 1;
    m_terminal_height = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
  } else {
    m_terminal_width = 80;
    m_terminal_height = 24;
  }
#elif defined(TIOCGWINSZ)
  struct winsize ws;
  int fd = m_output_stream_sp
               ? fileno(m_output_stream_sp->Lock().GetFile().GetStream())
               : STDOUT_FILENO;
  if (ioctl(fd, TIOCGWINSZ, &ws) == 0) {
    m_terminal_width = ws.ws_col > 0 ? ws.ws_col : 80;
    m_terminal_height = ws.ws_row > 0 ? ws.ws_row : 24;
  } else {
    m_terminal_width = 80;
    m_terminal_height = 24;
  }
#else
  m_terminal_width = 80;
  m_terminal_height = 24;
#endif
}

// replxx completion callback: called when user presses Tab.
void Editline::CompletionCallback(const char *input,
                                  replxx_completions *completions,
                                  int *context_len, void *user_data) {
  Editline *self = static_cast<Editline *>(user_data);
  if (!self->m_completion_callback)
    return;

  llvm::StringRef line(input);
  unsigned cursor_index = static_cast<unsigned>(line.size());
  CompletionResult result;
  CompletionRequest request(line, cursor_index, result);
  self->m_completion_callback(request);

  llvm::ArrayRef<CompletionResult::Completion> results = result.GetResults();
  for (const CompletionResult::Completion &c : results) {
    if (!c.GetCompletion().empty())
      replxx_add_completion(completions, c.GetCompletion().c_str());
  }

  // Let replxx know the context (prefix already typed) length.
  std::string prefix = request.GetCursorArgumentPrefix().str();
  *context_len = static_cast<int>(prefix.size());
}

// replxx hint callback: called to provide inline autosuggestion.
void Editline::HintCallback(const char *input, replxx_hints *hints,
                            int *context_len, ReplxxColor *color,
                            void *user_data) {
  Editline *self = static_cast<Editline *>(user_data);
  if (!self->m_suggestion_callback)
    return;

  if (std::optional<std::string> suggestion =
          self->m_suggestion_callback(llvm::StringRef(input))) {
    replxx_add_hint(hints, suggestion->c_str());
    *color = REPLXX_COLOR_GRAY;
  }
}

// Key handler: Ctrl+F applies the current inline suggestion.
ReplxxActionResult Editline::ApplySuggestionHandler(int code, void *user_data) {
  Editline *self = static_cast<Editline *>(user_data);
  if (!self->m_replxx || !self->m_suggestion_callback)
    return REPLXX_ACTION_RESULT_CONTINUE;

  ReplxxState state;
  replxx_get_state(self->m_replxx, &state);
  llvm::StringRef current(state.text);

  if (std::optional<std::string> suggestion =
          self->m_suggestion_callback(current)) {
    std::string new_text = std::string(current) + *suggestion;
    ReplxxState new_state{new_text.c_str(),
                          static_cast<int>(new_text.size())};
    replxx_set_state(self->m_replxx, &new_state);
  }
  return REPLXX_ACTION_RESULT_CONTINUE;
}

void Editline::InitializeReplxx() {
#if LLDB_ENABLE_REPLXX
  if (m_replxx)
    return;

  m_replxx = replxx_init();
  replxx_install_window_change_handler(m_replxx);

  replxx_set_max_history_size(m_replxx, 800);
  replxx_set_unique_history(m_replxx, 1);
  replxx_set_word_break_characters(m_replxx, " \t\n\"\\'><=;|&{(");

  if (m_completion_callback)
    replxx_set_completion_callback(m_replxx, CompletionCallback, this);

  if (m_suggestion_callback) {
    replxx_set_hint_callback(m_replxx, HintCallback, this);
    replxx_bind_key(m_replxx, REPLXX_KEY_CONTROL('F'), ApplySuggestionHandler,
                    this);
  }

  replxx_set_no_color(m_replxx, m_color ? 0 : 1);
  replxx_set_indent_multiline(m_replxx, 1);

  const char *path = GetHistoryFilePath();
  if (path)
    replxx_history_load(m_replxx, path);
#endif
}

StringList Editline::GetInputAsStringList(int line_count) {
  StringList lines;
  for (const std::string &line : m_input_lines) {
    if (line_count == 0)
      break;
    lines.AppendString(line);
    --line_count;
  }
  return lines;
}

// ===----------------------------------------------------------------------===//
// Public methods
// ===----------------------------------------------------------------------===//

Editline::Editline(const char *editline_name, FILE *input_file,
                   lldb::LockableStreamFileSP output_stream_sp,
                   lldb::LockableStreamFileSP error_stream_sp, bool color)
    : m_input_file(input_file), m_output_stream_sp(output_stream_sp),
      m_error_stream_sp(error_stream_sp),
      m_editor_status(EditorStatus::Complete), m_color(color) {
  assert(output_stream_sp && error_stream_sp);
  m_editor_name = (editline_name == nullptr) ? "lldb-tmp" : editline_name;
  UpdateTerminalSize();
}

Editline::~Editline() {
#if LLDB_ENABLE_REPLXX
  if (m_replxx) {
    const char *path = GetHistoryFilePath();
    if (path)
      replxx_history_save(m_replxx, path);
    replxx_end(m_replxx);
    m_replxx = nullptr;
  }
#endif
}

void Editline::UseColor(bool use_color) {
  m_color = use_color;
#if LLDB_ENABLE_REPLXX
  if (m_replxx)
    replxx_set_no_color(m_replxx, use_color ? 0 : 1);
#endif
}

void Editline::SetPrompt(const char *prompt) {
  m_set_prompt = prompt == nullptr ? "" : prompt;
}

void Editline::SetContinuationPrompt(const char *continuation_prompt) {
  m_set_continuation_prompt =
      continuation_prompt == nullptr ? "" : continuation_prompt;
}

void Editline::TerminalSizeChanged() {
  m_terminal_size_has_changed = 1;
  UpdateTerminalSize();
}

const char *Editline::GetPrompt() { return m_set_prompt.c_str(); }

uint32_t Editline::GetCurrentLine() { return m_current_line_index; }

bool Editline::Interrupt() {
  m_editor_status = EditorStatus::Interrupted;
#if !defined(_WIN32)
  if (m_reading.load()) {
    ::pthread_kill(m_reading_thread, SIGINT);
    return true;
  }
#endif
  return false;
}

bool Editline::Cancel() {
  m_editor_status = EditorStatus::Interrupted;
#if !defined(_WIN32)
  if (m_reading.load()) {
    ::pthread_kill(m_reading_thread, SIGINT);
    return true;
  }
#endif
  return false;
}

/// Print completions list with pagination.
static size_t
PrintCompletion(FILE *output_file,
                llvm::ArrayRef<CompletionResult::Completion> results,
                size_t max_completion_length, size_t max_length,
                std::optional<size_t> max_height = std::nullopt) {
  constexpr size_t ellipsis_length = 3;
  constexpr size_t padding_length = 8;
  constexpr size_t separator_length = 4;

  const size_t description_col =
      std::min(max_completion_length + padding_length, max_length);

  size_t lines_printed = 0;
  size_t results_printed = 0;
  for (const CompletionResult::Completion &c : results) {
    if (max_height && lines_printed >= *max_height)
      break;

    results_printed++;

    if (c.GetCompletion().empty())
      continue;

    fprintf(output_file, "        ");

    const size_t completion_length = c.GetCompletion().size();
    if (padding_length + completion_length < max_length) {
      fprintf(output_file, "%-*s",
              static_cast<int>(description_col - padding_length),
              c.GetCompletion().c_str());
    } else {
      fprintf(output_file, "%.*s...\n",
              static_cast<int>(max_length - padding_length - ellipsis_length),
              c.GetCompletion().c_str());
      lines_printed++;
      continue;
    }

    if (c.GetDescription().empty() ||
        description_col + separator_length + ellipsis_length >= max_length) {
      fprintf(output_file, "\n");
      lines_printed++;
      continue;
    }

    fprintf(output_file, " -- ");

    bool first = true;
    for (llvm::StringRef line : llvm::split(c.GetDescription(), '\n')) {
      if (line.empty())
        break;
      if (max_height && lines_printed >= *max_height)
        break;
      if (!first)
        fprintf(output_file, "%*s",
                static_cast<int>(description_col + separator_length), "");

      first = false;
      const size_t position = description_col + separator_length;
      const size_t description_length = line.size();
      if (position + description_length < max_length) {
        fprintf(output_file, "%.*s\n", static_cast<int>(description_length),
                line.data());
        lines_printed++;
      } else {
        fprintf(output_file, "%.*s...\n",
                static_cast<int>(max_length - position - ellipsis_length),
                line.data());
        lines_printed++;
        continue;
      }
    }
  }
  return results_printed;
}

void Editline::DisplayCompletions(
    Editline &editline, llvm::ArrayRef<CompletionResult::Completion> results) {
  assert(!results.empty());

  LockedStreamFile locked_stream = editline.m_output_stream_sp->Lock();
  FILE *out = locked_stream.GetFile().GetStream();

  fprintf(out, "\n" ANSI_CLEAR_BELOW "Available completions:\n");

  const size_t page_size = editline.GetTerminalHeight() - 3;

  bool all = false;

  auto longest =
      std::max_element(results.begin(), results.end(), [](auto &c1, auto &c2) {
        return c1.GetCompletion().size() < c2.GetCompletion().size();
      });

  const size_t max_len = longest->GetCompletion().size();

  size_t cur_pos = 0;
  while (cur_pos < results.size()) {
    cur_pos += PrintCompletion(out, results.slice(cur_pos), max_len,
                               editline.GetTerminalWidth(),
                               all ? std::nullopt
                                   : std::optional<size_t>(page_size));

    if (cur_pos >= results.size())
      break;

    fprintf(out, "More (Y/n/a): ");
    fflush(out);

    int reply = fgetc(editline.m_input_file);
    fprintf(out, "\n");
    if (reply == EOF || reply == 'n')
      break;
    if (reply == 'a')
      all = true;
  }
}

bool Editline::GetLine(std::string &line, bool &interrupted) {
  InitializeReplxx();

  SetCurrentLine(0);
  m_input_lines.clear();
  m_editor_status = EditorStatus::Editing;
  interrupted = false;

  if (m_redraw_callback)
    m_redraw_callback();

#if LLDB_ENABLE_REPLXX
  std::string full_prompt =
      m_prompt_ansi_prefix + PromptForIndex(0) + m_prompt_ansi_suffix;

  m_reading.store(true);
#if !defined(_WIN32)
  m_reading_thread = ::pthread_self();
#endif

  const char *input = replxx_input(m_replxx, full_prompt.c_str());

  m_reading.store(false);

  if (m_editor_status == EditorStatus::Interrupted) {
    interrupted = true;
    return true;
  }

  if (input == nullptr) {
    if (m_editor_status == EditorStatus::Interrupted) {
      interrupted = true;
      return true;
    }
    LockedStreamFile locked_stream = m_output_stream_sp->Lock();
    fprintf(locked_stream.GetFile().GetStream(), "\n");
    m_editor_status = EditorStatus::EndOfInput;
    return false;
  }

  line = input;
  m_input_lines = {line};

  if (!line.empty())
    replxx_history_add(m_replxx, line.c_str());

  m_editor_status = EditorStatus::Complete;
  return true;
#else
  // Fallback without replxx
  LockedStreamFile locked_stream = m_output_stream_sp->Lock();
  fprintf(locked_stream.GetFile().GetStream(), "%s", PromptForIndex(0).c_str());
  fflush(locked_stream.GetFile().GetStream());
  char buf[4096];
  if (!fgets(buf, sizeof(buf), m_input_file)) {
    m_editor_status = EditorStatus::EndOfInput;
    return false;
  }
  line = buf;
  while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
    line.pop_back();
  m_editor_status = EditorStatus::Complete;
  return true;
#endif
}

bool Editline::GetLines(int first_line_number, StringList &lines,
                        bool &interrupted) {
  InitializeReplxx();

  m_base_line_number = first_line_number;
  m_line_number_digits =
      std::max(3, (int)std::to_string(first_line_number).length() + 1);

  m_input_lines.clear();
  interrupted = false;
  int line_index = 0;

  while (true) {
    SetCurrentLine(line_index);
    std::string full_prompt = m_prompt_ansi_prefix + PromptForIndex(line_index) +
                              m_prompt_ansi_suffix;

    if (m_redraw_callback)
      m_redraw_callback();

    m_editor_status = EditorStatus::Editing;

#if LLDB_ENABLE_REPLXX
    m_reading.store(true);
#if !defined(_WIN32)
    m_reading_thread = ::pthread_self();
#endif

    const char *input = replxx_input(m_replxx, full_prompt.c_str());

    m_reading.store(false);
#else
    LockedStreamFile locked_stream = m_output_stream_sp->Lock();
    fprintf(locked_stream.GetFile().GetStream(), "%s", full_prompt.c_str());
    fflush(locked_stream.GetFile().GetStream());
    char buf[4096];
    const char *input = nullptr;
    std::string buf_str;
    if (fgets(buf, sizeof(buf), m_input_file)) {
      buf_str = buf;
      while (!buf_str.empty() &&
             (buf_str.back() == '\n' || buf_str.back() == '\r'))
        buf_str.pop_back();
      input = buf_str.c_str();
    }
#endif

    if (m_editor_status == EditorStatus::Interrupted) {
      interrupted = true;
      return true;
    }

    if (input == nullptr) {
      // EOF or SIGINT
      interrupted = (m_editor_status == EditorStatus::Interrupted);
      if (!interrupted) {
        LockedStreamFile locked_stream = m_output_stream_sp->Lock();
        fprintf(locked_stream.GetFile().GetStream(), "\n");
        m_editor_status = EditorStatus::EndOfInput;
      }
      return !interrupted;
    }

    m_input_lines.push_back(input);

    // Check if input is complete
    StringList current_lines = GetInputAsStringList();
    if (!m_is_input_complete_callback ||
        m_is_input_complete_callback(this, current_lines)) {
      // Update m_input_lines from callback (it may have modified them)
      m_input_lines.clear();
      for (size_t i = 0; i < current_lines.GetSize(); ++i)
        m_input_lines.push_back(std::string(current_lines[i]));
      break;
    }

    // Handle auto-indentation for the next line
    if (m_fix_indentation_callback) {
      int indent =
          m_fix_indentation_callback(this, current_lines, 0);
      if (indent > 0)
        m_input_lines.push_back(std::string(indent, ' '));
    }

    ++line_index;
  }

  // Add multi-line entry to history
#if LLDB_ENABLE_REPLXX
  if (!m_input_lines.empty()) {
    std::string combined;
    for (const std::string &l : m_input_lines)
      combined += l + "\n";
    if (!combined.empty() && combined.back() == '\n')
      combined.pop_back();
    replxx_history_add(m_replxx, combined.c_str());
  }
#endif

  m_editor_status = EditorStatus::Complete;
  lines = GetInputAsStringList();
  return true;
}

void Editline::PrintAsync(lldb::LockableStreamFileSP stream_sp, const char *s,
                          size_t len) {
#if LLDB_ENABLE_REPLXX
  if (m_replxx && m_editor_status == EditorStatus::Editing) {
    replxx_write(m_replxx, s, static_cast<int>(len));
    return;
  }
#endif
  LockedStreamFile locked_stream = stream_sp->Lock();
  locked_stream.Write(s, len);
}

void Editline::Refresh() {}
