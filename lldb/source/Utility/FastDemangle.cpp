//===-- FastDemangle.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <functional>

#include "lldb/Utility/FastDemangle.h"
#include "lldb/lldb-private.h"

//#define DEBUG_FAILURES 1
//#define DEBUG_SUBSTITUTIONS 1
//#define DEBUG_TEMPLATE_ARGS 1
//#define DEBUG_HIGHWATER 1
//#define DEBUG_REORDERING 1

namespace {

/// @brief Represents the collection of qualifiers on a type

enum Qualifiers {
  QualifierNone = 0,
  QualifierConst = 1,
  QualifierRestrict = 2,
  QualifierVolatile = 4,
  QualifierReference = 8,
  QualifierRValueReference = 16,
  QualifierPointer = 32
};

/// @brief Categorizes the recognized operators

enum class OperatorKind {
  Unary,
  Postfix,
  Binary,
  Ternary,
  Other,
  ConversionOperator,
  Vendor,
  NoMatch
};

/// @brief Represents one of the recognized two-character operator
/// abbreviations used when parsing operators as names and expressions

struct Operator {
  const char *name;
  OperatorKind kind;
};

/// @brief Represents a range of characters in the output buffer, typically for
/// use with RewriteRange()

struct BufferRange {
  int offset;
  int length;
};

/// @brief Transient state required while parsing a name

struct NameState {
  bool parse_function_params;
  bool is_last_generic;
  bool has_no_return_type;
  BufferRange last_name_range;
};

/// @brief LLDB's fast C++ demangler
///
/// This is an incomplete implementation designed to speed up the demangling
/// process that is often a bottleneck when LLDB stops a process for the first
/// time.  Where the implementation doesn't know how to demangle a symbol it
/// fails gracefully to allow the caller to fall back to the existing demangler.
///
/// Over time the full mangling spec should be supported without compromising
/// performance for the most common cases.

class SymbolDemangler {
public:
  //----------------------------------------------------
  // Public API
  //----------------------------------------------------

  /// @brief Create a SymbolDemangler
  ///
  /// The newly created demangler allocates and owns scratch memory sufficient
  /// for demangling typical symbols.  Additional memory will be allocated if
  /// needed and managed by the demangler instance.

  SymbolDemangler() {
    m_buffer = (char *)malloc(8192);
    m_buffer_end = m_buffer + 8192;
    m_owns_buffer = true;

    m_rewrite_ranges = (BufferRange *)malloc(128 * sizeof(BufferRange));
    m_rewrite_ranges_size = 128;
    m_owns_m_rewrite_ranges = true;
  }

  /// @brief Create a SymbolDemangler that uses provided scratch memory
  ///
  /// The provided memory is not owned by the demangler.  It will be
  /// overwritten during calls to GetDemangledCopy() but can be used for
  /// other purposes between calls.  The provided memory will not be freed
  /// when this instance is destroyed.
  ///
  /// If demangling a symbol requires additional space it will be allocated
  /// and managed by the demangler instance.
  ///
  /// @param storage_ptr Valid pointer to at least storage_size bytes of
  /// space that the SymbolDemangler can use during demangling
  ///
  /// @param storage_size Number of bytes of space available scratch memory
  /// referenced by storage_ptr

  SymbolDemangler(void *storage_ptr, size_t storage_size,
                  std::function<void(const char *)> builtins_hook = nullptr)
      : m_builtins_hook(builtins_hook) {
    // Use up to 1/8th of the provided space for rewrite ranges
    m_rewrite_ranges_size = (storage_size >> 3) / sizeof(BufferRange);
    m_rewrite_ranges = (BufferRange *)storage_ptr;
    m_owns_m_rewrite_ranges = false;

    // Use the rest for the character buffer
    m_buffer =
        (char *)storage_ptr + m_rewrite_ranges_size * sizeof(BufferRange);
    m_buffer_end = (const char *)storage_ptr + storage_size;
    m_owns_buffer = false;
  }

  /// @brief Destroys the SymbolDemangler and deallocates any scratch
  /// memory that it owns

  ~SymbolDemangler() {
    if (m_owns_buffer)
      free(m_buffer);
    if (m_owns_m_rewrite_ranges)
      free(m_rewrite_ranges);
  }

#ifdef DEBUG_HIGHWATER
  int highwater_store = 0;
  int highwater_buffer = 0;
#endif

  /// @brief Parses the provided mangled name and returns a newly allocated
  /// demangling
  ///
  /// @param mangled_name Valid null-terminated C++ mangled name following
  /// the Itanium C++ ABI mangling specification as implemented by Clang
  ///
  /// @result Newly allocated null-terminated demangled name when demangling
  /// is successful, and nullptr when demangling fails.  The caller is
  /// responsible for freeing the allocated memory.

  char *GetDemangledCopy(const char *mangled_name,
                         long mangled_name_length = 0) {
    if (!ParseMangling(mangled_name, mangled_name_length))
      return nullptr;

#ifdef DEBUG_HIGHWATER
    int rewrite_count = m_next_substitute_index +
                        (m_rewrite_ranges_size - 1 - m_next_template_arg_index);
    int buffer_size = (int)(m_write_ptr - m_buffer);
    if (rewrite_count > highwater_store)
      highwater_store = rewrite_count;
    if (buffer_size > highwater_buffer)
      highwater_buffer = buffer_size;
#endif

    int length = (int)(m_write_ptr - m_buffer);
    char *copy = (char *)malloc(length + 1);
    memcpy(copy, m_buffer, length);
    copy[length] = '\0';
    return copy;
  }

private:
  //----------------------------------------------------
  // Grow methods
  //
  // Manage the storage used during demangling
  //----------------------------------------------------

  void GrowBuffer(long min_growth = 0) {
    // By default, double the size of the buffer
    long growth = m_buffer_end - m_buffer;

    // Avoid growing by more than 1MB at a time
    if (growth > 1 << 20)
      growth = 1 << 20;

    // ... but never grow by less than requested,
    // or 1K, whichever is greater
    if (min_growth < 1024)
      min_growth = 1024;
    if (growth < min_growth)
      growth = min_growth;

    // Allocate the new m_buffer and migrate content
    long new_size = (m_buffer_end - m_buffer) + growth;
    char *new_buffer = (char *)malloc(new_size);
    memcpy(new_buffer, m_buffer, m_write_ptr - m_buffer);
    if (m_owns_buffer)
      free(m_buffer);
    m_owns_buffer = true;

    // Update references to the new buffer
    m_write_ptr = new_buffer + (m_write_ptr - m_buffer);
    m_buffer = new_buffer;
    m_buffer_end = m_buffer + new_size;
  }

  void GrowRewriteRanges() {
    // By default, double the size of the array
    int growth = m_rewrite_ranges_size;

    // Apply reasonable minimum and maximum sizes for growth
    if (growth > 128)
      growth = 128;
    if (growth < 16)
      growth = 16;

    // Allocate the new array and migrate content
    int bytes = (m_rewrite_ranges_size + growth) * sizeof(BufferRange);
    BufferRange *new_ranges = (BufferRange *)malloc(bytes);
    for (int index = 0; index < m_next_substitute_index; index++) {
      new_ranges[index] = m_rewrite_ranges[index];
    }
    for (int index = m_rewrite_ranges_size - 1;
         index > m_next_template_arg_index; index--) {
      new_ranges[index + growth] = m_rewrite_ranges[index];
    }
    if (m_owns_m_rewrite_ranges)
      free(m_rewrite_ranges);
    m_owns_m_rewrite_ranges = true;

    // Update references to the new array
    m_rewrite_ranges = new_ranges;
    m_rewrite_ranges_size += growth;
    m_next_template_arg_index += growth;
  }

  //----------------------------------------------------
  // Range and state management
  //----------------------------------------------------

  int GetStartCookie() { return (int)(m_write_ptr - m_buffer); }

  BufferRange EndRange(int start_cookie) {
    return {start_cookie, (int)(m_write_ptr - (m_buffer + start_cookie))};
  }

  void ReorderRange(BufferRange source_range, int insertion_point_cookie) {
    // Ensure there's room the preserve the source range
    if (m_write_ptr + source_range.length > m_buffer_end) {
      GrowBuffer(m_write_ptr + source_range.length - m_buffer_end);
    }

    // Reorder the content
    memcpy(m_write_ptr, m_buffer + source_range.offset, source_range.length);
    memmove(m_buffer + insertion_point_cookie + source_range.length,
            m_buffer + insertion_point_cookie,
            source_range.offset - insertion_point_cookie);
    memcpy(m_buffer + insertion_point_cookie, m_write_ptr, source_range.length);

    // Fix up rewritable ranges, covering both substitutions and templates
    int index = 0;
    while (true) {
      if (index == m_next_substitute_index)
        index = m_next_template_arg_index + 1;
      if (index == m_rewrite_ranges_size)
        break;

      // Affected ranges are either shuffled forward when after the
      // insertion but before the source, or backward when inside the
      // source
      int candidate_offset = m_rewrite_ranges[index].offset;
      if (candidate_offset >= insertion_point_cookie) {
        if (candidate_offset < source_range.offset) {
          m_rewrite_ranges[index].offset += source_range.length;
        } else if (candidate_offset >= source_range.offset) {
          m_rewrite_ranges[index].offset -=
              (source_range.offset - insertion_point_cookie);
        }
      }
      ++index;
    }
  }

  void EndSubstitution(int start_cookie) {
    if (m_next_substitute_index == m_next_template_arg_index)
      GrowRewriteRanges();

    int index = m_next_substitute_index++;
    m_rewrite_ranges[index] = EndRange(start_cookie);
#ifdef DEBUG_SUBSTITUTIONS
    printf("Saved substitution # %d = %.*s\n", index,
           m_rewrite_ranges[index].length, m_buffer + start_cookie);
#endif
  }

  void EndTemplateArg(int start_cookie) {
    if (m_next_substitute_index == m_next_template_arg_index)
      GrowRewriteRanges();

    int index = m_next_template_arg_index--;
    m_rewrite_ranges[index] = EndRange(start_cookie);
#ifdef DEBUG_TEMPLATE_ARGS
    printf("Saved template arg # %d = %.*s\n",
           m_rewrite_ranges_size - index - 1, m_rewrite_ranges[index].length,
           m_buffer + start_cookie);
#endif
  }

  void ResetTemplateArgs() {
    // TODO: this works, but is it the right thing to do?
    // Should we push/pop somehow at the call sites?
    m_next_template_arg_index = m_rewrite_ranges_size - 1;
  }

  //----------------------------------------------------
  // Write methods
  //
  // Appends content to the existing output buffer
  //----------------------------------------------------

  void Write(char character) {
    if (m_write_ptr == m_buffer_end)
      GrowBuffer();
    *m_write_ptr++ = character;
  }

  void Write(const char *content) { Write(content, strlen(content)); }

  void Write(const char *content, long content_length) {
    char *end_m_write_ptr = m_write_ptr + content_length;
    if (end_m_write_ptr > m_buffer_end) {
      if (content >= m_buffer && content < m_buffer_end) {
        long offset = content - m_buffer;
        GrowBuffer(end_m_write_ptr - m_buffer_end);
        content = m_buffer + offset;
      } else {
        GrowBuffer(end_m_write_ptr - m_buffer_end);
      }
      end_m_write_ptr = m_write_ptr + content_length;
    }
    memcpy(m_write_ptr, content, content_length);
    m_write_ptr = end_m_write_ptr;
  }
#define WRITE(x) Write(x, sizeof(x) - 1)

  void WriteTemplateStart() { Write('<'); }

  void WriteTemplateEnd() {
    // Put a space between terminal > characters when nesting templates
    if (m_write_ptr != m_buffer && *(m_write_ptr - 1) == '>')
      WRITE(" >");
    else
      Write('>');
  }

  void WriteCommaSpace() { WRITE(", "); }

  void WriteNamespaceSeparator() { WRITE("::"); }

  void WriteStdPrefix() { WRITE("std::"); }

  void WriteQualifiers(int qualifiers, bool space_before_reference = true) {
    if (qualifiers & QualifierPointer)
      Write('*');
    if (qualifiers & QualifierConst)
      WRITE(" const");
    if (qualifiers & QualifierVolatile)
      WRITE(" volatile");
    if (qualifiers & QualifierRestrict)
      WRITE(" restrict");
    if (qualifiers & QualifierReference) {
      if (space_before_reference)
        WRITE(" &");
      else
        Write('&');
    }
    if (qualifiers & QualifierRValueReference) {
      if (space_before_reference)
        WRITE(" &&");
      else
        WRITE("&&");
    }
  }

  //----------------------------------------------------
  // Rewrite methods
  //
  // Write another copy of content already present
  // earlier in the output buffer
  //----------------------------------------------------

  void RewriteRange(BufferRange range) {
    Write(m_buffer + range.offset, range.length);
  }

  bool RewriteSubstitution(int index) {
    if (index < 0 || index >= m_next_substitute_index) {
#ifdef DEBUG_FAILURES
      printf("*** Invalid substitution #%d\n", index);
#endif
      return false;
    }
    RewriteRange(m_rewrite_ranges[index]);
    return true;
  }

  bool RewriteTemplateArg(int template_index) {
    int index = m_rewrite_ranges_size - 1 - template_index;
    if (template_index < 0 || index <= m_next_template_arg_index) {
#ifdef DEBUG_FAILURES
      printf("*** Invalid template arg reference #%d\n", template_index);
#endif
      return false;
    }
    RewriteRange(m_rewrite_ranges[index]);
    return true;
  }

  //----------------------------------------------------
  // TryParse methods
  //
  // Provide information with return values instead of
  // writing to the output buffer
  //
  // Values indicating failure guarantee that the pre-
  // call m_read_ptr is unchanged
  //----------------------------------------------------

  int TryParseNumber() {
    unsigned char digit = *m_read_ptr - '0';
    if (digit > 9)
      return -1;

    int count = digit;
    while (true) {
      digit = *++m_read_ptr - '0';
      if (digit > 9)
        break;

      count = count * 10 + digit;
    }
    return count;
  }

  int TryParseBase36Number() {
    char digit = *m_read_ptr;
    int count;
    if (digit >= '0' && digit <= '9')
      count = digit -= '0';
    else if (digit >= 'A' && digit <= 'Z')
      count = digit -= ('A' - 10);
    else
      return -1;

    while (true) {
      digit = *++m_read_ptr;
      if (digit >= '0' && digit <= '9')
        digit -= '0';
      else if (digit >= 'A' && digit <= 'Z')
        digit -= ('A' - 10);
      else
        break;

      count = count * 36 + digit;
    }
    return count;
  }

  // <builtin-type> ::= v    # void
  //                ::= w    # wchar_t
  //                ::= b    # bool
  //                ::= c    # char
  //                ::= a    # signed char
  //                ::= h    # unsigned char
  //                ::= s    # short
  //                ::= t    # unsigned short
  //                ::= i    # int
  //                ::= j    # unsigned int
  //                ::= l    # long
  //                ::= m    # unsigned long
  //                ::= x    # long long, __int64
  //                ::= y    # unsigned long long, __int64
  //                ::= n    # __int128
  //                ::= o    # unsigned __int128
  //                ::= f    # float
  //                ::= d    # double
  //                ::= e    # long double, __float80
  //                ::= g    # __float128
  //                ::= z    # ellipsis
  //                ::= Dd   # IEEE 754r decimal floating point (64 bits)
  //                ::= De   # IEEE 754r decimal floating point (128 bits)
  //                ::= Df   # IEEE 754r decimal floating point (32 bits)
  //                ::= Dh   # IEEE 754r half-precision floating point (16 bits)
  //                ::= Di   # char32_t
  //                ::= Ds   # char16_t
  //                ::= Da   # auto (in dependent new-expressions)
  //                ::= Dn   # std::nullptr_t (i.e., decltype(nullptr))
  //                ::= u <source-name>    # vendor extended type

  const char *TryParseBuiltinType() {
    if (m_builtins_hook)
      m_builtins_hook(m_read_ptr);

    switch (*m_read_ptr++) {
    case 'v':
      return "void";
    case 'w':
      return "wchar_t";
    case 'b':
      return "bool";
    case 'c':
      return "char";
    case 'a':
      return "signed char";
    case 'h':
      return "unsigned char";
    case 's':
      return "short";
    case 't':
      return "unsigned short";
    case 'i':
      return "int";
    case 'j':
      return "unsigned int";
    case 'l':
      return "long";
    case 'm':
      return "unsigned long";
    case 'x':
      return "long long";
    case 'y':
      return "unsigned long long";
    case 'n':
      return "__int128";
    case 'o':
      return "unsigned __int128";
    case 'f':
      return "float";
    case 'd':
      return "double";
    case 'e':
      return "long double";
    case 'g':
      return "__float128";
    case 'z':
      return "...";
    case 'D': {
      switch (*m_read_ptr++) {
      case 'd':
        return "decimal64";
      case 'e':
        return "decimal128";
      case 'f':
        return "decimal32";
      case 'h':
        return "decimal16";
      case 'i':
        return "char32_t";
      case 's':
        return "char16_t";
      case 'a':
        return "auto";
      case 'c':
        return "decltype(auto)";
      case 'n':
        return "std::nullptr_t";
      default:
        --m_read_ptr;
      }
    }
    }
    --m_read_ptr;
    return nullptr;
  }

  //   <operator-name>
  //                   ::= aa    # &&
  //                   ::= ad    # & (unary)
  //                   ::= an    # &
  //                   ::= aN    # &=
  //                   ::= aS    # =
  //                   ::= cl    # ()
  //                   ::= cm    # ,
  //                   ::= co    # ~
  //                   ::= da    # delete[]
  //                   ::= de    # * (unary)
  //                   ::= dl    # delete
  //                   ::= dv    # /
  //                   ::= dV    # /=
  //                   ::= eo    # ^
  //                   ::= eO    # ^=
  //                   ::= eq    # ==
  //                   ::= ge    # >=
  //                   ::= gt    # >
  //                   ::= ix    # []
  //                   ::= le    # <=
  //                   ::= ls    # <<
  //                   ::= lS    # <<=
  //                   ::= lt    # <
  //                   ::= mi    # -
  //                   ::= mI    # -=
  //                   ::= ml    # *
  //                   ::= mL    # *=
  //                   ::= mm    # -- (postfix in <expression> context)
  //                   ::= na    # new[]
  //                   ::= ne    # !=
  //                   ::= ng    # - (unary)
  //                   ::= nt    # !
  //                   ::= nw    # new
  //                   ::= oo    # ||
  //                   ::= or    # |
  //                   ::= oR    # |=
  //                   ::= pm    # ->*
  //                   ::= pl    # +
  //                   ::= pL    # +=
  //                   ::= pp    # ++ (postfix in <expression> context)
  //                   ::= ps    # + (unary)
  //                   ::= pt    # ->
  //                   ::= qu    # ?
  //                   ::= rm    # %
  //                   ::= rM    # %=
  //                   ::= rs    # >>
  //                   ::= rS    # >>=
  //                   ::= cv <type>    # (cast)
  //                   ::= v <digit> <source-name>        # vendor extended
  //                   operator

  Operator TryParseOperator() {
    switch (*m_read_ptr++) {
    case 'a':
      switch (*m_read_ptr++) {
      case 'a':
        return {"&&", OperatorKind::Binary};
      case 'd':
        return {"&", OperatorKind::Unary};
      case 'n':
        return {"&", OperatorKind::Binary};
      case 'N':
        return {"&=", OperatorKind::Binary};
      case 'S':
        return {"=", OperatorKind::Binary};
      }
      --m_read_ptr;
      break;
    case 'c':
      switch (*m_read_ptr++) {
      case 'l':
        return {"()", OperatorKind::Other};
      case 'm':
        return {",", OperatorKind::Other};
      case 'o':
        return {"~", OperatorKind::Unary};
      case 'v':
        return {nullptr, OperatorKind::ConversionOperator};
      }
      --m_read_ptr;
      break;
    case 'd':
      switch (*m_read_ptr++) {
      case 'a':
        return {" delete[]", OperatorKind::Other};
      case 'e':
        return {"*", OperatorKind::Unary};
      case 'l':
        return {" delete", OperatorKind::Other};
      case 'v':
        return {"/", OperatorKind::Binary};
      case 'V':
        return {"/=", OperatorKind::Binary};
      }
      --m_read_ptr;
      break;
    case 'e':
      switch (*m_read_ptr++) {
      case 'o':
        return {"^", OperatorKind::Binary};
      case 'O':
        return {"^=", OperatorKind::Binary};
      case 'q':
        return {"==", OperatorKind::Binary};
      }
      --m_read_ptr;
      break;
    case 'g':
      switch (*m_read_ptr++) {
      case 'e':
        return {">=", OperatorKind::Binary};
      case 't':
        return {">", OperatorKind::Binary};
      }
      --m_read_ptr;
      break;
    case 'i':
      switch (*m_read_ptr++) {
      case 'x':
        return {"[]", OperatorKind::Other};
      }
      --m_read_ptr;
      break;
    case 'l':
      switch (*m_read_ptr++) {
      case 'e':
        return {"<=", OperatorKind::Binary};
      case 's':
        return {"<<", OperatorKind::Binary};
      case 'S':
        return {"<<=", OperatorKind::Binary};
      case 't':
        return {"<", OperatorKind::Binary};
        // case 'i': return { "?", OperatorKind::Binary };
      }
      --m_read_ptr;
      break;
    case 'm':
      switch (*m_read_ptr++) {
      case 'i':
        return {"-", OperatorKind::Binary};
      case 'I':
        return {"-=", OperatorKind::Binary};
      case 'l':
        return {"*", OperatorKind::Binary};
      case 'L':
        return {"*=", OperatorKind::Binary};
      case 'm':
        return {"--", OperatorKind::Postfix};
      }
      --m_read_ptr;
      break;
    case 'n':
      switch (*m_read_ptr++) {
      case 'a':
        return {" new[]", OperatorKind::Other};
      case 'e':
        return {"!=", OperatorKind::Binary};
      case 'g':
        return {"-", OperatorKind::Unary};
      case 't':
        return {"!", OperatorKind::Unary};
      case 'w':
        return {" new", OperatorKind::Other};
      }
      --m_read_ptr;
      break;
    case 'o':
      switch (*m_read_ptr++) {
      case 'o':
        return {"||", OperatorKind::Binary};
      case 'r':
        return {"|", OperatorKind::Binary};
      case 'R':
        return {"|=", OperatorKind::Binary};
      }
      --m_read_ptr;
      break;
    case 'p':
      switch (*m_read_ptr++) {
      case 'm':
        return {"->*", OperatorKind::Binary};
      case 's':
        return {"+", OperatorKind::Unary};
      case 'l':
        return {"+", OperatorKind::Binary};
      case 'L':
        return {"+=", OperatorKind::Binary};
      case 'p':
        return {"++", OperatorKind::Postfix};
      case 't':
        return {"->", OperatorKind::Binary};
      }
      --m_read_ptr;
      break;
    case 'q':
      switch (*m_read_ptr++) {
      case 'u':
        return {"?", OperatorKind::Ternary};
      }
      --m_read_ptr;
      break;
    case 'r':
      switch (*m_read_ptr++) {
      case 'm':
        return {"%", OperatorKind::Binary};
      case 'M':
        return {"%=", OperatorKind::Binary};
      case 's':
        return {">>", OperatorKind::Binary};
      case 'S':
        return {">=", OperatorKind::Binary};
      }
      --m_read_ptr;
      break;
    case 'v':
      char digit = *m_read_ptr;
      if (digit >= '0' && digit <= '9') {
        m_read_ptr++;
        return {nullptr, OperatorKind::Vendor};
      }
      --m_read_ptr;
      break;
    }
    --m_read_ptr;
    return {nullptr, OperatorKind::NoMatch};
  }

  // <CV-qualifiers> ::= [r] [V] [K]
  // <ref-qualifier> ::= R                   # & ref-qualifier
  // <ref-qualifier> ::= O                   # && ref-qualifier

  int TryParseQualifiers(bool allow_cv, bool allow_ro) {
    int qualifiers = QualifierNone;
    char next = *m_read_ptr;
    if (allow_cv) {
      if (next == 'r') // restrict
      {
        qualifiers |= QualifierRestrict;
        next = *++m_read_ptr;
      }
      if (next == 'V') // volatile
      {
        qualifiers |= QualifierVolatile;
        next = *++m_read_ptr;
      }
      if (next == 'K') // const
      {
        qualifiers |= QualifierConst;
        next = *++m_read_ptr;
      }
    }
    if (allow_ro) {
      if (next == 'R') {
        ++m_read_ptr;
        qualifiers |= QualifierReference;
      } else if (next == 'O') {
        ++m_read_ptr;
        qualifiers |= QualifierRValueReference;
      }
    }
    return qualifiers;
  }

  // <discriminator> := _ <non-negative number>      # when number < 10
  //                 := __ <non-negative number> _   # when number >= 10
  //  extension      := decimal-digit+

  int TryParseDiscriminator() {
    const char *discriminator_start = m_read_ptr;

    // Test the extension first, since it's what Clang uses
    int discriminator_value = TryParseNumber();
    if (discriminator_value != -1)
      return discriminator_value;

    char next = *m_read_ptr;
    if (next == '_') {
      next = *++m_read_ptr;
      if (next == '_') {
        ++m_read_ptr;
        discriminator_value = TryParseNumber();
        if (discriminator_value != -1 && *m_read_ptr++ != '_') {
          return discriminator_value;
        }
      } else if (next >= '0' && next <= '9') {
        ++m_read_ptr;
        return next - '0';
      }
    }

    // Not a valid discriminator
    m_read_ptr = discriminator_start;
    return -1;
  }

  //----------------------------------------------------
  // Parse methods
  //
  // Consume input starting from m_read_ptr and produce
  // buffered output at m_write_ptr
  //
  // Failures return false and may leave m_read_ptr in an
  // indeterminate state
  //----------------------------------------------------

  bool Parse(char character) {
    if (*m_read_ptr++ == character)
      return true;
#ifdef DEBUG_FAILURES
    printf("*** Expected '%c'\n", character);
#endif
    return false;
  }

  // <number> ::= [n] <non-negative decimal integer>

  bool ParseNumber(bool allow_negative = false) {
    if (allow_negative && *m_read_ptr == 'n') {
      Write('-');
      ++m_read_ptr;
    }
    const char *before_digits = m_read_ptr;
    while (true) {
      unsigned char digit = *m_read_ptr - '0';
      if (digit > 9)
        break;
      ++m_read_ptr;
    }
    if (int digit_count = (int)(m_read_ptr - before_digits)) {
      Write(before_digits, digit_count);
      return true;
    }
#ifdef DEBUG_FAILURES
    printf("*** Expected number\n");
#endif
    return false;
  }

  // <substitution> ::= S <seq-id> _
  //                ::= S_
  // <substitution> ::= Sa # ::std::allocator
  // <substitution> ::= Sb # ::std::basic_string
  // <substitution> ::= Ss # ::std::basic_string < char,
  //                                               ::std::char_traits<char>,
  //                                               ::std::allocator<char> >
  // <substitution> ::= Si # ::std::basic_istream<char,  std::char_traits<char>
  // >
  // <substitution> ::= So # ::std::basic_ostream<char,  std::char_traits<char>
  // >
  // <substitution> ::= Sd # ::std::basic_iostream<char, std::char_traits<char>
  // >

  bool ParseSubstitution() {
    const char *substitution;
    switch (*m_read_ptr) {
    case 'a':
      substitution = "std::allocator";
      break;
    case 'b':
      substitution = "std::basic_string";
      break;
    case 's':
      substitution = "std::string";
      break;
    case 'i':
      substitution = "std::istream";
      break;
    case 'o':
      substitution = "std::ostream";
      break;
    case 'd':
      substitution = "std::iostream";
      break;
    default:
      // A failed attempt to parse a number will return -1 which turns out to be
      // perfect here as S_ is the first substitution, S0_ the next and so forth
      int substitution_index = TryParseBase36Number();
      if (*m_read_ptr++ != '_') {
#ifdef DEBUG_FAILURES
        printf("*** Expected terminal _ in substitution\n");
#endif
        return false;
      }
      return RewriteSubstitution(substitution_index + 1);
    }
    Write(substitution);
    ++m_read_ptr;
    return true;
  }

  // <function-type> ::= F [Y] <bare-function-type> [<ref-qualifier>] E
  //
  // <bare-function-type> ::= <signature type>+      # types are possible return
  // type, then parameter types

  bool ParseFunctionType(int inner_qualifiers = QualifierNone) {
#ifdef DEBUG_FAILURES
    printf("*** Function types not supported\n");
#endif
    // TODO: first steps toward an implementation follow, but they're far
    // from complete.  Function types tend to bracket other types eg:
    // int (*)() when used as the type for "name" becomes int (*name)().
    // This makes substitution et al ... interesting.
    return false;

#if 0  // TODO
        if (*m_read_ptr == 'Y')
            ++m_read_ptr;

        int return_type_start_cookie = GetStartCookie();
        if (!ParseType())
            return false;
        Write(' ');

        int insert_cookie = GetStartCookie();
        Write('(');
        bool first_param = true;
        int qualifiers = QualifierNone;
        while (true)
        {
            switch (*m_read_ptr)
            {
                case 'E':
                    ++m_read_ptr;
                    Write(')');
                    break;
                case 'v':
                    ++m_read_ptr;
                    continue;
                case 'R':
                case 'O':
                    if (*(m_read_ptr + 1) == 'E')
                    {
                        qualifiers = TryParseQualifiers (false, true);
                        Parse('E');
                        break;
                    }
                    // fallthrough
                default:
                {
                    if (first_param)
                        first_param = false;
                    else WriteCommaSpace();

                    if (!ParseType())
                        return false;
                    continue;
                }
            }
            break;
        }

        if (qualifiers)
        {
            WriteQualifiers (qualifiers);
            EndSubstitution (return_type_start_cookie);
        }

        if (inner_qualifiers)
        {
            int qualifier_start_cookie = GetStartCookie();
            Write ('(');
            WriteQualifiers (inner_qualifiers);
            Write (')');
            ReorderRange (EndRange (qualifier_start_cookie), insert_cookie);
        }
        return true;
#endif // TODO
  }

  // <array-type> ::= A <positive dimension number> _ <element type>
  //              ::= A [<dimension expression>] _ <element type>

  bool ParseArrayType(int qualifiers = QualifierNone) {
#ifdef DEBUG_FAILURES
    printf("*** Array type unsupported\n");
#endif
    // TODO: We fail horribly when recalling these as substitutions or
    // templates and trying to constify them eg:
    // _ZN4llvm2cl5applyIA28_cNS0_3optIbLb0ENS0_6parserIbEEEEEEvRKT_PT0_
    //
    // TODO: Chances are we don't do any better with references and pointers
    // that should be type (&) [] instead of type & []

    return false;

#if 0  // TODO
        if (*m_read_ptr == '_')
        {
            ++m_read_ptr;
            if (!ParseType())
                return false;
            if (qualifiers)
                WriteQualifiers(qualifiers);
            WRITE(" []");
            return true;
        }
        else
        {
            const char *before_digits = m_read_ptr;
            if (TryParseNumber() != -1)
            {
                const char *after_digits = m_read_ptr;
                if (!Parse('_'))
                    return false;
                if (!ParseType())
                    return false;
                if (qualifiers)
                    WriteQualifiers(qualifiers);
                Write(' ');
                Write('[');
                Write(before_digits, after_digits - before_digits);
            }
            else
            {
                int type_insertion_cookie = GetStartCookie();
                if (!ParseExpression())
                    return false;
                if (!Parse('_'))
                    return false;

                int type_start_cookie = GetStartCookie();
                if (!ParseType())
                    return false;
                if (qualifiers)
                    WriteQualifiers(qualifiers);
                Write(' ');
                Write('[');
                ReorderRange (EndRange (type_start_cookie), type_insertion_cookie);
            }
            Write(']');
            return true;
        }
#endif // TODO
  }

  // <pointer-to-member-type> ::= M <class type> <member type>

  // TODO: Determine how to handle pointers to function members correctly,
  // currently not an issue because we don't have function types at all...
  bool ParsePointerToMemberType() {
    int insertion_cookie = GetStartCookie();
    Write(' ');
    if (!ParseType())
      return false;
    WRITE("::*");

    int type_cookie = GetStartCookie();
    if (!ParseType())
      return false;
    ReorderRange(EndRange(type_cookie), insertion_cookie);
    return true;
  }

  // <template-param> ::= T_    # first template parameter
  //                  ::= T <parameter-2 non-negative number> _

  bool ParseTemplateParam() {
    int count = TryParseNumber();
    if (!Parse('_'))
      return false;

    // When no number is present we get -1, which is convenient since
    // T_ is the zeroth element T0_ is element 1, and so on
    return RewriteTemplateArg(count + 1);
  }

  // <vector-type>
  // Dv <dimension number> _ <vector type>
  bool TryParseVectorType() {
    const int dimension = TryParseNumber();
    if (dimension == -1)
      return false;

    if (*m_read_ptr++ != '_')
      return false;

    char vec_dimens[32] = {'\0'};
    ::snprintf(vec_dimens, sizeof vec_dimens - 1, " __vector(%d)", dimension);
    ParseType();
    Write(vec_dimens);
    return true;
  }

  // <type> ::= <builtin-type>
  //        ::= <function-type>
  //        ::= <class-enum-type>
  //        ::= <array-type>
  //        ::= <pointer-to-member-type>
  //        ::= <template-param>
  //        ::= <template-template-param> <template-args>
  //        ::= <decltype>
  //        ::= <substitution>
  //        ::= <CV-qualifiers> <type>
  //        ::= P <type>        # pointer-to
  //        ::= R <type>        # reference-to
  //        ::= O <type>        # rvalue reference-to (C++0x)
  //        ::= C <type>        # complex pair (C 2000)
  //        ::= G <type>        # imaginary (C 2000)
  //        ::= Dp <type>       # pack expansion (C++0x)
  //        ::= U <source-name> <type>  # vendor extended type qualifier
  // extension := U <objc-name> <objc-type>  # objc-type<identifier>
  // extension := <vector-type> # <vector-type> starts with Dv

  // <objc-name> ::= <k0 number> objcproto <k1 number> <identifier>  # k0 = 9 +
  // <number of digits in k1> + k1
  // <objc-type> := <source-name>  # PU<11+>objcproto 11objc_object<source-name>
  // 11objc_object -> id<source-name>

  bool ParseType() {
#ifdef DEBUG_FAILURES
    const char *failed_type = m_read_ptr;
#endif
    int type_start_cookie = GetStartCookie();
    bool suppress_substitution = false;

    int qualifiers = TryParseQualifiers(true, false);
    switch (*m_read_ptr) {
    case 'D':
      ++m_read_ptr;
      switch (*m_read_ptr++) {
      case 'p':
        if (!ParseType())
          return false;
        break;
      case 'v':
        if (!TryParseVectorType())
          return false;
        break;
      case 'T':
      case 't':
      default:
#ifdef DEBUG_FAILURES
        printf("*** Unsupported type: %.3s\n", failed_type);
#endif
        return false;
      }
      break;
    case 'T':
      ++m_read_ptr;
      if (!ParseTemplateParam())
        return false;
      break;
    case 'M':
      ++m_read_ptr;
      if (!ParsePointerToMemberType())
        return false;
      break;
    case 'A':
      ++m_read_ptr;
      if (!ParseArrayType())
        return false;
      break;
    case 'F':
      ++m_read_ptr;
      if (!ParseFunctionType())
        return false;
      break;
    case 'S':
      if (*++m_read_ptr == 't') {
        ++m_read_ptr;
        WriteStdPrefix();
        if (!ParseName())
          return false;
      } else {
        suppress_substitution = true;
        if (!ParseSubstitution())
          return false;
      }
      break;
    case 'P': {
      switch (*++m_read_ptr) {
      case 'F':
        ++m_read_ptr;
        if (!ParseFunctionType(QualifierPointer))
          return false;
        break;
      default:
        if (!ParseType())
          return false;
        Write('*');
        break;
      }
      break;
    }
    case 'R': {
      ++m_read_ptr;
      if (!ParseType())
        return false;
      Write('&');
      break;
    }
    case 'O': {
      ++m_read_ptr;
      if (!ParseType())
        return false;
      Write('&');
      Write('&');
      break;
    }
    case 'C':
    case 'G':
    case 'U':
#ifdef DEBUG_FAILURES
      printf("*** Unsupported type: %.3s\n", failed_type);
#endif
      return false;
    // Test for common cases to avoid TryParseBuiltinType() overhead
    case 'N':
    case 'Z':
    case 'L':
      if (!ParseName())
        return false;
      break;
    default:
      if (const char *builtin = TryParseBuiltinType()) {
        Write(builtin);
        suppress_substitution = true;
      } else {
        if (!ParseName())
          return false;
      }
      break;
    }

    // Allow base substitutions to be suppressed, but always record
    // substitutions for the qualified variant
    if (!suppress_substitution)
      EndSubstitution(type_start_cookie);
    if (qualifiers) {
      WriteQualifiers(qualifiers, false);
      EndSubstitution(type_start_cookie);
    }
    return true;
  }

  // <unnamed-type-name> ::= Ut [ <nonnegative number> ] _
  //                     ::= <closure-type-name>
  //
  // <closure-type-name> ::= Ul <lambda-sig> E [ <nonnegative number> ] _
  //
  // <lambda-sig> ::= <parameter type>+  # Parameter types or "v" if the lambda
  // has no parameters

  bool ParseUnnamedTypeName(NameState &name_state) {
    switch (*m_read_ptr++) {
    case 't': {
      int cookie = GetStartCookie();
      WRITE("'unnamed");
      const char *before_digits = m_read_ptr;
      if (TryParseNumber() != -1)
        Write(before_digits, m_read_ptr - before_digits);
      if (!Parse('_'))
        return false;
      Write('\'');
      name_state.last_name_range = EndRange(cookie);
      return true;
    }
    case 'b': {
      int cookie = GetStartCookie();
      WRITE("'block");
      const char *before_digits = m_read_ptr;
      if (TryParseNumber() != -1)
        Write(before_digits, m_read_ptr - before_digits);
      if (!Parse('_'))
        return false;
      Write('\'');
      name_state.last_name_range = EndRange(cookie);
      return true;
    }
    case 'l':
#ifdef DEBUG_FAILURES
      printf("*** Lambda type names unsupported\n");
#endif
      return false;
    }
#ifdef DEBUG_FAILURES
    printf("*** Unknown unnamed type %.3s\n", m_read_ptr - 2);
#endif
    return false;
  }

  // <ctor-dtor-name> ::= C1      # complete object constructor
  //                  ::= C2      # base object constructor
  //                  ::= C3      # complete object allocating constructor

  bool ParseCtor(NameState &name_state) {
    char next = *m_read_ptr;
    if (next == '1' || next == '2' || next == '3' || next == '5') {
      RewriteRange(name_state.last_name_range);
      name_state.has_no_return_type = true;
      ++m_read_ptr;
      return true;
    }
#ifdef DEBUG_FAILURES
    printf("*** Broken constructor\n");
#endif
    return false;
  }

  // <ctor-dtor-name> ::= D0      # deleting destructor
  //                  ::= D1      # complete object destructor
  //                  ::= D2      # base object destructor

  bool ParseDtor(NameState &name_state) {
    char next = *m_read_ptr;
    if (next == '0' || next == '1' || next == '2' || next == '5') {
      Write('~');
      RewriteRange(name_state.last_name_range);
      name_state.has_no_return_type = true;
      ++m_read_ptr;
      return true;
    }
#ifdef DEBUG_FAILURES
    printf("*** Broken destructor\n");
#endif
    return false;
  }

  // See TryParseOperator()

  bool ParseOperatorName(NameState &name_state) {
#ifdef DEBUG_FAILURES
    const char *operator_ptr = m_read_ptr;
#endif
    Operator parsed_operator = TryParseOperator();
    if (parsed_operator.name) {
      WRITE("operator");
      Write(parsed_operator.name);
      return true;
    }

    // Handle special operators
    switch (parsed_operator.kind) {
    case OperatorKind::Vendor:
      WRITE("operator ");
      return ParseSourceName();
    case OperatorKind::ConversionOperator:
      ResetTemplateArgs();
      name_state.has_no_return_type = true;
      WRITE("operator ");
      return ParseType();
    default:
#ifdef DEBUG_FAILURES
      printf("*** Unknown operator: %.2s\n", operator_ptr);
#endif
      return false;
    }
  }

  // <source-name> ::= <positive length number> <identifier>

  bool ParseSourceName() {
    int count = TryParseNumber();
    if (count == -1) {
#ifdef DEBUG_FAILURES
      printf("*** Malformed source name, missing length count\n");
#endif
      return false;
    }

    const char *next_m_read_ptr = m_read_ptr + count;
    if (next_m_read_ptr > m_read_end) {
#ifdef DEBUG_FAILURES
      printf("*** Malformed source name, premature termination\n");
#endif
      return false;
    }

    if (count >= 10 && strncmp(m_read_ptr, "_GLOBAL__N", 10) == 0)
      WRITE("(anonymous namespace)");
    else
      Write(m_read_ptr, count);

    m_read_ptr = next_m_read_ptr;
    return true;
  }

  // <unqualified-name> ::= <operator-name>
  //                    ::= <ctor-dtor-name>
  //                    ::= <source-name>
  //                    ::= <unnamed-type-name>

  bool ParseUnqualifiedName(NameState &name_state) {
    // Note that these are detected directly in ParseNestedName for
    // performance rather than switching on the same options twice
    char next = *m_read_ptr;
    switch (next) {
    case 'C':
      ++m_read_ptr;
      return ParseCtor(name_state);
    case 'D':
      ++m_read_ptr;
      return ParseDtor(name_state);
    case 'U':
      ++m_read_ptr;
      return ParseUnnamedTypeName(name_state);
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9': {
      int name_start_cookie = GetStartCookie();
      if (!ParseSourceName())
        return false;
      name_state.last_name_range = EndRange(name_start_cookie);
      return true;
    }
    default:
      return ParseOperatorName(name_state);
    };
  }

  // <unscoped-name> ::= <unqualified-name>
  //                 ::= St <unqualified-name>   # ::std::
  // extension       ::= StL<unqualified-name>

  bool ParseUnscopedName(NameState &name_state) {
    if (*m_read_ptr == 'S' && *(m_read_ptr + 1) == 't') {
      WriteStdPrefix();
      if (*(m_read_ptr += 2) == 'L')
        ++m_read_ptr;
    }
    return ParseUnqualifiedName(name_state);
  }

  bool ParseIntegerLiteral(const char *prefix, const char *suffix,
                           bool allow_negative) {
    if (prefix)
      Write(prefix);
    if (!ParseNumber(allow_negative))
      return false;
    if (suffix)
      Write(suffix);
    return Parse('E');
  }

  bool ParseBooleanLiteral() {
    switch (*m_read_ptr++) {
    case '0':
      WRITE("false");
      break;
    case '1':
      WRITE("true");
      break;
    default:
#ifdef DEBUG_FAILURES
      printf("*** Boolean literal not 0 or 1\n");
#endif
      return false;
    }
    return Parse('E');
  }

  // <expr-primary> ::= L <type> <value number> E                          #
  // integer literal
  //                ::= L <type> <value float> E                           #
  //                floating literal
  //                ::= L <string type> E                                  #
  //                string literal
  //                ::= L <nullptr type> E                                 #
  //                nullptr literal (i.e., "LDnE")
  //                ::= L <type> <real-part float> _ <imag-part float> E   #
  //                complex floating point literal (C 2000)
  //                ::= L <mangled-name> E                                 #
  //                external name

  bool ParseExpressionPrimary() {
    switch (*m_read_ptr++) {
    case 'b':
      return ParseBooleanLiteral();
    case 'x':
      return ParseIntegerLiteral(nullptr, "ll", true);
    case 'l':
      return ParseIntegerLiteral(nullptr, "l", true);
    case 'i':
      return ParseIntegerLiteral(nullptr, nullptr, true);
    case 'n':
      return ParseIntegerLiteral("(__int128)", nullptr, true);
    case 'j':
      return ParseIntegerLiteral(nullptr, "u", false);
    case 'm':
      return ParseIntegerLiteral(nullptr, "ul", false);
    case 'y':
      return ParseIntegerLiteral(nullptr, "ull", false);
    case 'o':
      return ParseIntegerLiteral("(unsigned __int128)", nullptr, false);
    case '_':
      if (*m_read_ptr++ == 'Z') {
        if (!ParseEncoding())
          return false;
        return Parse('E');
      }
      --m_read_ptr;
      LLVM_FALLTHROUGH;
    case 'w':
    case 'c':
    case 'a':
    case 'h':
    case 's':
    case 't':
    case 'f':
    case 'd':
    case 'e':
#ifdef DEBUG_FAILURES
      printf("*** Unsupported primary expression %.5s\n", m_read_ptr - 1);
#endif
      return false;
    case 'T':
// Invalid mangled name per
//   http://sourcerytools.com/pipermail/cxx-abi-dev/2011-August/002422.html
#ifdef DEBUG_FAILURES
      printf("*** Invalid primary expr encoding\n");
#endif
      return false;
    default:
      --m_read_ptr;
      Write('(');
      if (!ParseType())
        return false;
      Write(')');
      if (!ParseNumber())
        return false;
      return Parse('E');
    }
  }

  // <unresolved-type> ::= <template-param>
  //                   ::= <decltype>
  //                   ::= <substitution>

  bool ParseUnresolvedType() {
    int type_start_cookie = GetStartCookie();
    switch (*m_read_ptr++) {
    case 'T':
      if (!ParseTemplateParam())
        return false;
      EndSubstitution(type_start_cookie);
      return true;
    case 'S': {
      if (*m_read_ptr != 't')
        return ParseSubstitution();

      ++m_read_ptr;
      WriteStdPrefix();
      NameState type_name = {};
      if (!ParseUnqualifiedName(type_name))
        return false;
      EndSubstitution(type_start_cookie);
      return true;
    }
    case 'D':
    default:
#ifdef DEBUG_FAILURES
      printf("*** Unsupported unqualified type: %3s\n", m_read_ptr - 1);
#endif
      return false;
    }
  }

  // <base-unresolved-name> ::= <simple-id>                                #
  // unresolved name
  //          extension     ::= <operator-name>                            #
  //          unresolved operator-function-id
  //          extension     ::= <operator-name> <template-args>            #
  //          unresolved operator template-id
  //                        ::= on <operator-name>                         #
  //                        unresolved operator-function-id
  //                        ::= on <operator-name> <template-args>         #
  //                        unresolved operator template-id
  //                        ::= dn <destructor-name>                       #
  //                        destructor or pseudo-destructor;
  //                                                                         #
  //                                                                         e.g.
  //                                                                         ~X
  //                                                                         or
  //                                                                         ~X<N-1>

  bool ParseBaseUnresolvedName() {
#ifdef DEBUG_FAILURES
    printf("*** Base unresolved name unsupported\n");
#endif
    return false;
  }

  // <unresolved-name>
  //  extension        ::= srN <unresolved-type> [<template-args>]
  //  <unresolved-qualifier-level>* E <base-unresolved-name>
  //                   ::= [gs] <base-unresolved-name>                     # x
  //                   or (with "gs") ::x
  //                   ::= [gs] sr <unresolved-qualifier-level>+ E
  //                   <base-unresolved-name>
  //                                                                       #
  //                                                                       A::x,
  //                                                                       N::y,
  //                                                                       A<T>::z;
  //                                                                       "gs"
  //                                                                       means
  //                                                                       leading
  //                                                                       "::"
  //                   ::= sr <unresolved-type> <base-unresolved-name>     #
  //                   T::x / decltype(p)::x
  //  extension        ::= sr <unresolved-type> <template-args>
  //  <base-unresolved-name>
  //                                                                       #
  //                                                                       T::N::x
  //                                                                       /decltype(p)::N::x
  //  (ignored)        ::= srN <unresolved-type>  <unresolved-qualifier-level>+
  //  E <base-unresolved-name>

  bool ParseUnresolvedName() {
#ifdef DEBUG_FAILURES
    printf("*** Unresolved names not supported\n");
#endif
    // TODO: grammar for all of this seems unclear...
    return false;

#if 0  // TODO
        if (*m_read_ptr == 'g' && *(m_read_ptr + 1) == 's')
        {
            m_read_ptr += 2;
            WriteNamespaceSeparator();
        }
#endif // TODO
  }

  // <expression> ::= <unary operator-name> <expression>
  //              ::= <binary operator-name> <expression> <expression>
  //              ::= <ternary operator-name> <expression> <expression>
  //              <expression>
  //              ::= cl <expression>+ E                                   #
  //              call
  //              ::= cv <type> <expression>                               #
  //              conversion with one argument
  //              ::= cv <type> _ <expression>* E                          #
  //              conversion with a different number of arguments
  //              ::= [gs] nw <expression>* _ <type> E                     # new
  //              (expr-list) type
  //              ::= [gs] nw <expression>* _ <type> <initializer>         # new
  //              (expr-list) type (init)
  //              ::= [gs] na <expression>* _ <type> E                     #
  //              new[] (expr-list) type
  //              ::= [gs] na <expression>* _ <type> <initializer>         #
  //              new[] (expr-list) type (init)
  //              ::= [gs] dl <expression>                                 #
  //              delete expression
  //              ::= [gs] da <expression>                                 #
  //              delete[] expression
  //              ::= pp_ <expression>                                     #
  //              prefix ++
  //              ::= mm_ <expression>                                     #
  //              prefix --
  //              ::= ti <type>                                            #
  //              typeid (type)
  //              ::= te <expression>                                      #
  //              typeid (expression)
  //              ::= dc <type> <expression>                               #
  //              dynamic_cast<type> (expression)
  //              ::= sc <type> <expression>                               #
  //              static_cast<type> (expression)
  //              ::= cc <type> <expression>                               #
  //              const_cast<type> (expression)
  //              ::= rc <type> <expression>                               #
  //              reinterpret_cast<type> (expression)
  //              ::= st <type>                                            #
  //              sizeof (a type)
  //              ::= sz <expression>                                      #
  //              sizeof (an expression)
  //              ::= at <type>                                            #
  //              alignof (a type)
  //              ::= az <expression>                                      #
  //              alignof (an expression)
  //              ::= nx <expression>                                      #
  //              noexcept (expression)
  //              ::= <template-param>
  //              ::= <function-param>
  //              ::= dt <expression> <unresolved-name>                    #
  //              expr.name
  //              ::= pt <expression> <unresolved-name>                    #
  //              expr->name
  //              ::= ds <expression> <expression>                         #
  //              expr.*expr
  //              ::= sZ <template-param>                                  #
  //              size of a parameter pack
  //              ::= sZ <function-param>                                  #
  //              size of a function parameter pack
  //              ::= sp <expression>                                      #
  //              pack expansion
  //              ::= tw <expression>                                      #
  //              throw expression
  //              ::= tr                                                   #
  //              throw with no operand (rethrow)
  //              ::= <unresolved-name>                                    #
  //              f(p), N::f(p), ::f(p),
  //                                                                       #
  //                                                                       freestanding
  //                                                                       dependent
  //                                                                       name
  //                                                                       (e.g.,
  //                                                                       T::x),
  //                                                                       #
  //                                                                       objectless
  //                                                                       nonstatic
  //                                                                       member
  //                                                                       reference
  //              ::= <expr-primary>

  bool ParseExpression() {
    Operator expression_operator = TryParseOperator();
    switch (expression_operator.kind) {
    case OperatorKind::Unary:
      Write(expression_operator.name);
      Write('(');
      if (!ParseExpression())
        return false;
      Write(')');
      return true;
    case OperatorKind::Binary:
      if (!ParseExpression())
        return false;
      Write(expression_operator.name);
      return ParseExpression();
    case OperatorKind::Ternary:
      if (!ParseExpression())
        return false;
      Write('?');
      if (!ParseExpression())
        return false;
      Write(':');
      return ParseExpression();
    case OperatorKind::NoMatch:
      break;
    case OperatorKind::Other:
    default:
#ifdef DEBUG_FAILURES
      printf("*** Unsupported operator: %s\n", expression_operator.name);
#endif
      return false;
    }

    switch (*m_read_ptr++) {
    case 'T':
      return ParseTemplateParam();
    case 'L':
      return ParseExpressionPrimary();
    case 's':
      if (*m_read_ptr++ == 'r')
        return ParseUnresolvedName();
      --m_read_ptr;
      LLVM_FALLTHROUGH;
    default:
      return ParseExpressionPrimary();
    }
  }

  // <template-arg> ::= <type>                                             #
  // type or template
  //                ::= X <expression> E                                   #
  //                expression
  //                ::= <expr-primary>                                     #
  //                simple expressions
  //                ::= J <template-arg>* E                                #
  //                argument pack
  //                ::= LZ <encoding> E                                    #
  //                extension

  bool ParseTemplateArg() {
    switch (*m_read_ptr) {
    case 'J':
#ifdef DEBUG_FAILURES
      printf("*** Template argument packs unsupported\n");
#endif
      return false;
    case 'X':
      ++m_read_ptr;
      if (!ParseExpression())
        return false;
      return Parse('E');
    case 'L':
      ++m_read_ptr;
      return ParseExpressionPrimary();
    default:
      return ParseType();
    }
  }

  // <template-args> ::= I <template-arg>* E
  //     extension, the abi says <template-arg>+

  bool ParseTemplateArgs(bool record_template_args = false) {
    if (record_template_args)
      ResetTemplateArgs();

    bool first_arg = true;
    while (*m_read_ptr != 'E') {
      if (first_arg)
        first_arg = false;
      else
        WriteCommaSpace();

      int template_start_cookie = GetStartCookie();
      if (!ParseTemplateArg())
        return false;
      if (record_template_args)
        EndTemplateArg(template_start_cookie);
    }
    ++m_read_ptr;
    return true;
  }

  // <nested-name> ::= N [<CV-qualifiers>] [<ref-qualifier>] <prefix>
  // <unqualified-name> E
  //               ::= N [<CV-qualifiers>] [<ref-qualifier>] <template-prefix>
  //               <template-args> E
  //
  // <prefix> ::= <prefix> <unqualified-name>
  //          ::= <template-prefix> <template-args>
  //          ::= <template-param>
  //          ::= <decltype>
  //          ::= # empty
  //          ::= <substitution>
  //          ::= <prefix> <data-member-prefix>
  //  extension ::= L
  //
  // <template-prefix> ::= <prefix> <template unqualified-name>
  //                   ::= <template-param>
  //                   ::= <substitution>
  //
  // <unqualified-name> ::= <operator-name>
  //                    ::= <ctor-dtor-name>
  //                    ::= <source-name>
  //                    ::= <unnamed-type-name>

  bool ParseNestedName(NameState &name_state,
                       bool parse_discriminator = false) {
    int qualifiers = TryParseQualifiers(true, true);
    bool first_part = true;
    bool suppress_substitution = true;
    int name_start_cookie = GetStartCookie();
    while (true) {
      char next = *m_read_ptr;
      if (next == 'E') {
        ++m_read_ptr;
        break;
      }

      // Record a substitution candidate for all prefixes, but not the full name
      if (suppress_substitution)
        suppress_substitution = false;
      else
        EndSubstitution(name_start_cookie);

      if (next == 'I') {
        ++m_read_ptr;
        name_state.is_last_generic = true;
        WriteTemplateStart();
        if (!ParseTemplateArgs(name_state.parse_function_params))
          return false;
        WriteTemplateEnd();
        continue;
      }

      if (first_part)
        first_part = false;
      else
        WriteNamespaceSeparator();

      name_state.is_last_generic = false;
      switch (next) {
      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9': {
        int name_start_cookie = GetStartCookie();
        if (!ParseSourceName())
          return false;
        name_state.last_name_range = EndRange(name_start_cookie);
        continue;
      }
      case 'S':
        if (*++m_read_ptr == 't') {
          WriteStdPrefix();
          ++m_read_ptr;
          if (!ParseUnqualifiedName(name_state))
            return false;
        } else {
          if (!ParseSubstitution())
            return false;
          suppress_substitution = true;
        }
        continue;
      case 'T':
        ++m_read_ptr;
        if (!ParseTemplateParam())
          return false;
        continue;
      case 'C':
        ++m_read_ptr;
        if (!ParseCtor(name_state))
          return false;
        continue;
      case 'D': {
        switch (*(m_read_ptr + 1)) {
        case 't':
        case 'T':
#ifdef DEBUG_FAILURES
          printf("*** Decltype unsupported\n");
#endif
          return false;
        }
        ++m_read_ptr;
        if (!ParseDtor(name_state))
          return false;
        continue;
      }
      case 'U':
        ++m_read_ptr;
        if (!ParseUnnamedTypeName(name_state))
          return false;
        continue;
      case 'L':
        ++m_read_ptr;
        if (!ParseUnqualifiedName(name_state))
          return false;
        continue;
      default:
        if (!ParseOperatorName(name_state))
          return false;
      }
    }

    if (parse_discriminator)
      TryParseDiscriminator();
    if (name_state.parse_function_params &&
        !ParseFunctionArgs(name_state, name_start_cookie)) {
      return false;
    }
    if (qualifiers)
      WriteQualifiers(qualifiers);
    return true;
  }

  // <local-name> := Z <function encoding> E <entity name> [<discriminator>]
  //              := Z <function encoding> E s [<discriminator>]
  //              := Z <function encoding> Ed [ <parameter number> ] _ <entity
  //              name>

  bool ParseLocalName(bool parse_function_params) {
    if (!ParseEncoding())
      return false;
    if (!Parse('E'))
      return false;

    switch (*m_read_ptr) {
    case 's':
      ++m_read_ptr;
      TryParseDiscriminator(); // Optional and ignored
      WRITE("::string literal");
      break;
    case 'd':
      ++m_read_ptr;
      TryParseNumber(); // Optional and ignored
      if (!Parse('_'))
        return false;
      WriteNamespaceSeparator();
      if (!ParseName())
        return false;
      break;
    default:
      WriteNamespaceSeparator();
      if (!ParseName(parse_function_params, true))
        return false;
      TryParseDiscriminator(); // Optional and ignored
    }
    return true;
  }

  // <name> ::= <nested-name>
  //        ::= <local-name>
  //        ::= <unscoped-template-name> <template-args>
  //        ::= <unscoped-name>

  // <unscoped-template-name> ::= <unscoped-name>
  //                          ::= <substitution>

  bool ParseName(bool parse_function_params = false,
                 bool parse_discriminator = false) {
    NameState name_state = {parse_function_params, false, false, {0, 0}};
    int name_start_cookie = GetStartCookie();

    switch (*m_read_ptr) {
    case 'N':
      ++m_read_ptr;
      return ParseNestedName(name_state, parse_discriminator);
    case 'Z': {
      ++m_read_ptr;
      if (!ParseLocalName(parse_function_params))
        return false;
      break;
    }
    case 'L':
      ++m_read_ptr;
      LLVM_FALLTHROUGH;
    default: {
      if (!ParseUnscopedName(name_state))
        return false;

      if (*m_read_ptr == 'I') {
        EndSubstitution(name_start_cookie);

        ++m_read_ptr;
        name_state.is_last_generic = true;
        WriteTemplateStart();
        if (!ParseTemplateArgs(parse_function_params))
          return false;
        WriteTemplateEnd();
      }
      break;
    }
    }
    if (parse_discriminator)
      TryParseDiscriminator();
    if (parse_function_params &&
        !ParseFunctionArgs(name_state, name_start_cookie)) {
      return false;
    }
    return true;
  }

  // <call-offset> ::= h <nv-offset> _
  //               ::= v <v-offset> _
  //
  // <nv-offset> ::= <offset number>
  //                 # non-virtual base override
  //
  // <v-offset>  ::= <offset number> _ <virtual offset number>
  //                 # virtual base override, with vcall offset

  bool ParseCallOffset() {
    switch (*m_read_ptr++) {
    case 'h':
      if (*m_read_ptr == 'n')
        ++m_read_ptr;
      if (TryParseNumber() == -1 || *m_read_ptr++ != '_')
        break;
      return true;
    case 'v':
      if (*m_read_ptr == 'n')
        ++m_read_ptr;
      if (TryParseNumber() == -1 || *m_read_ptr++ != '_')
        break;
      if (*m_read_ptr == 'n')
        ++m_read_ptr;
      if (TryParseNumber() == -1 || *m_read_ptr++ != '_')
        break;
      return true;
    }
#ifdef DEBUG_FAILURES
    printf("*** Malformed call offset\n");
#endif
    return false;
  }

  // <special-name> ::= TV <type>    # virtual table
  //                ::= TT <type>    # VTT structure (construction vtable index)
  //                ::= TI <type>    # typeinfo structure
  //                ::= TS <type>    # typeinfo name (null-terminated byte
  //                string)
  //                ::= Tc <call-offset> <call-offset> <base encoding>
  //                    # base is the nominal target function of thunk
  //                    # first call-offset is 'this' adjustment
  //                    # second call-offset is result adjustment
  //                ::= T <call-offset> <base encoding>
  //                    # base is the nominal target function of thunk
  //      extension ::= TC <first type> <number> _ <second type> # construction
  //      vtable for second-in-first

  bool ParseSpecialNameT() {
    switch (*m_read_ptr++) {
    case 'V':
      WRITE("vtable for ");
      return ParseType();
    case 'T':
      WRITE("VTT for ");
      return ParseType();
    case 'I':
      WRITE("typeinfo for ");
      return ParseType();
    case 'S':
      WRITE("typeinfo name for ");
      return ParseType();
    case 'c':
    case 'C':
#ifdef DEBUG_FAILURES
      printf("*** Unsupported thunk or construction vtable name: %.3s\n",
             m_read_ptr - 1);
#endif
      return false;
    default:
      if (*--m_read_ptr == 'v') {
        WRITE("virtual thunk to ");
      } else {
        WRITE("non-virtual thunk to ");
      }
      if (!ParseCallOffset())
        return false;
      return ParseEncoding();
    }
  }

  // <special-name> ::= GV <object name> # Guard variable for one-time
  // initialization
  //                                     # No <type>
  //      extension ::= GR <object name> # reference temporary for object

  bool ParseSpecialNameG() {
    switch (*m_read_ptr++) {
    case 'V':
      WRITE("guard variable for ");
      if (!ParseName(true))
        return false;
      break;
    case 'R':
      WRITE("reference temporary for ");
      if (!ParseName(true))
        return false;
      break;
    default:
#ifdef DEBUG_FAILURES
      printf("*** Unknown G encoding\n");
#endif
      return false;
    }
    return true;
  }

  // <bare-function-type> ::= <signature type>+        # types are possible
  // return type, then parameter types

  bool ParseFunctionArgs(NameState &name_state, int return_insert_cookie) {
    char next = *m_read_ptr;
    if (next == 'E' || next == '\0' || next == '.')
      return true;

    // Clang has a bad habit of making unique manglings by just sticking numbers
    // on the end of a symbol,
    // which is ambiguous with malformed source name manglings
    const char *before_clang_uniquing_test = m_read_ptr;
    if (TryParseNumber()) {
      if (*m_read_ptr == '\0')
        return true;
      m_read_ptr = before_clang_uniquing_test;
    }

    if (name_state.is_last_generic && !name_state.has_no_return_type) {
      int return_type_start_cookie = GetStartCookie();
      if (!ParseType())
        return false;
      Write(' ');
      ReorderRange(EndRange(return_type_start_cookie), return_insert_cookie);
    }

    Write('(');
    bool first_param = true;
    while (true) {
      switch (*m_read_ptr) {
      case '\0':
      case 'E':
      case '.':
        break;
      case 'v':
        ++m_read_ptr;
        continue;
      case '_':
        // Not a formal part of the mangling specification, but clang emits
        // suffixes starting with _block_invoke
        if (strncmp(m_read_ptr, "_block_invoke", 13) == 0) {
          m_read_ptr += strlen(m_read_ptr);
          break;
        }
        LLVM_FALLTHROUGH;
      default:
        if (first_param)
          first_param = false;
        else
          WriteCommaSpace();

        if (!ParseType())
          return false;
        continue;
      }
      break;
    }
    Write(')');
    return true;
  }

  // <encoding> ::= <function name> <bare-function-type>
  //            ::= <data name>
  //            ::= <special-name>

  bool ParseEncoding() {
    switch (*m_read_ptr) {
    case 'T':
      ++m_read_ptr;
      if (!ParseSpecialNameT())
        return false;
      break;
    case 'G':
      ++m_read_ptr;
      if (!ParseSpecialNameG())
        return false;
      break;
    default:
      if (!ParseName(true))
        return false;
      break;
    }
    return true;
  }

  bool ParseMangling(const char *mangled_name, long mangled_name_length = 0) {
    if (!mangled_name_length)
      mangled_name_length = strlen(mangled_name);
    m_read_end = mangled_name + mangled_name_length;
    m_read_ptr = mangled_name;
    m_write_ptr = m_buffer;
    m_next_substitute_index = 0;
    m_next_template_arg_index = m_rewrite_ranges_size - 1;

    if (*m_read_ptr++ != '_' || *m_read_ptr++ != 'Z') {
#ifdef DEBUG_FAILURES
      printf("*** Missing _Z prefix\n");
#endif
      return false;
    }
    if (!ParseEncoding())
      return false;
    switch (*m_read_ptr) {
    case '.':
      Write(' ');
      Write('(');
      Write(m_read_ptr, m_read_end - m_read_ptr);
      Write(')');
      LLVM_FALLTHROUGH;
    case '\0':
      return true;
    default:
#ifdef DEBUG_FAILURES
      printf("*** Unparsed mangled content\n");
#endif
      return false;
    }
  }

private:
  // External scratch storage used during demanglings

  char *m_buffer;
  const char *m_buffer_end;
  BufferRange *m_rewrite_ranges;
  int m_rewrite_ranges_size;
  bool m_owns_buffer;
  bool m_owns_m_rewrite_ranges;

  // Internal state used during demangling

  const char *m_read_ptr;
  const char *m_read_end;
  char *m_write_ptr;
  int m_next_template_arg_index;
  int m_next_substitute_index;
  std::function<void(const char *s)> m_builtins_hook;
};

} // Anonymous namespace

// Public entry points referenced from Mangled.cpp
namespace lldb_private {
char *FastDemangle(const char *mangled_name) {
  char buffer[16384];
  SymbolDemangler demangler(buffer, sizeof(buffer));
  return demangler.GetDemangledCopy(mangled_name);
}

char *FastDemangle(const char *mangled_name, size_t mangled_name_length,
                   std::function<void(const char *s)> builtins_hook) {
  char buffer[16384];
  SymbolDemangler demangler(buffer, sizeof(buffer), builtins_hook);
  return demangler.GetDemangledCopy(mangled_name, mangled_name_length);
}
} // lldb_private namespace
