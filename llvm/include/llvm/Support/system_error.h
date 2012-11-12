//===---------------------------- system_error ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This was lifted from libc++ and modified for C++03. This is called
// system_error even though it does not define that class because that's what
// it's called in C++0x. We don't define system_error because it is only used
// for exception handling, which we don't use in LLVM.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_SYSTEM_ERROR_H
#define LLVM_SYSTEM_SYSTEM_ERROR_H

#include "llvm/Support/Compiler.h"

/*
    system_error synopsis

namespace std
{

class error_category
{
public:
    virtual ~error_category();

    error_category(const error_category&) = delete;
    error_category& operator=(const error_category&) = delete;

    virtual const char* name() const = 0;
    virtual error_condition default_error_condition(int ev) const;
    virtual bool equivalent(int code, const error_condition& condition) const;
    virtual bool equivalent(const error_code& code, int condition) const;
    virtual std::string message(int ev) const = 0;

    bool operator==(const error_category& rhs) const;
    bool operator!=(const error_category& rhs) const;
    bool operator<(const error_category& rhs) const;
};

const error_category& generic_category();
const error_category& system_category();

template <class T> struct is_error_code_enum
    : public false_type {};

template <class T> struct is_error_condition_enum
    : public false_type {};

class error_code
{
public:
    // constructors:
    error_code();
    error_code(int val, const error_category& cat);
    template <class ErrorCodeEnum>
        error_code(ErrorCodeEnum e);

    // modifiers:
    void assign(int val, const error_category& cat);
    template <class ErrorCodeEnum>
        error_code& operator=(ErrorCodeEnum e);
    void clear();

    // observers:
    int value() const;
    const error_category& category() const;
    error_condition default_error_condition() const;
    std::string message() const;
    explicit operator bool() const;
};

// non-member functions:
bool operator<(const error_code& lhs, const error_code& rhs);
template <class charT, class traits>
    basic_ostream<charT,traits>&
    operator<<(basic_ostream<charT,traits>& os, const error_code& ec);

class error_condition
{
public:
    // constructors:
    error_condition();
    error_condition(int val, const error_category& cat);
    template <class ErrorConditionEnum>
        error_condition(ErrorConditionEnum e);

    // modifiers:
    void assign(int val, const error_category& cat);
    template <class ErrorConditionEnum>
        error_condition& operator=(ErrorConditionEnum e);
    void clear();

    // observers:
    int value() const;
    const error_category& category() const;
    std::string message() const;
    explicit operator bool() const;
};

bool operator<(const error_condition& lhs, const error_condition& rhs);

class system_error
    : public runtime_error
{
public:
    system_error(error_code ec, const std::string& what_arg);
    system_error(error_code ec, const char* what_arg);
    system_error(error_code ec);
    system_error(int ev, const error_category& ecat, const std::string& what_arg);
    system_error(int ev, const error_category& ecat, const char* what_arg);
    system_error(int ev, const error_category& ecat);

    const error_code& code() const throw();
    const char* what() const throw();
};

enum class errc
{
    address_family_not_supported,       // EAFNOSUPPORT
    address_in_use,                     // EADDRINUSE
    address_not_available,              // EADDRNOTAVAIL
    already_connected,                  // EISCONN
    argument_list_too_long,             // E2BIG
    argument_out_of_domain,             // EDOM
    bad_address,                        // EFAULT
    bad_file_descriptor,                // EBADF
    bad_message,                        // EBADMSG
    broken_pipe,                        // EPIPE
    connection_aborted,                 // ECONNABORTED
    connection_already_in_progress,     // EALREADY
    connection_refused,                 // ECONNREFUSED
    connection_reset,                   // ECONNRESET
    cross_device_link,                  // EXDEV
    destination_address_required,       // EDESTADDRREQ
    device_or_resource_busy,            // EBUSY
    directory_not_empty,                // ENOTEMPTY
    executable_format_error,            // ENOEXEC
    file_exists,                        // EEXIST
    file_too_large,                     // EFBIG
    filename_too_long,                  // ENAMETOOLONG
    function_not_supported,             // ENOSYS
    host_unreachable,                   // EHOSTUNREACH
    identifier_removed,                 // EIDRM
    illegal_byte_sequence,              // EILSEQ
    inappropriate_io_control_operation, // ENOTTY
    interrupted,                        // EINTR
    invalid_argument,                   // EINVAL
    invalid_seek,                       // ESPIPE
    io_error,                           // EIO
    is_a_directory,                     // EISDIR
    message_size,                       // EMSGSIZE
    network_down,                       // ENETDOWN
    network_reset,                      // ENETRESET
    network_unreachable,                // ENETUNREACH
    no_buffer_space,                    // ENOBUFS
    no_child_process,                   // ECHILD
    no_link,                            // ENOLINK
    no_lock_available,                  // ENOLCK
    no_message_available,               // ENODATA
    no_message,                         // ENOMSG
    no_protocol_option,                 // ENOPROTOOPT
    no_space_on_device,                 // ENOSPC
    no_stream_resources,                // ENOSR
    no_such_device_or_address,          // ENXIO
    no_such_device,                     // ENODEV
    no_such_file_or_directory,          // ENOENT
    no_such_process,                    // ESRCH
    not_a_directory,                    // ENOTDIR
    not_a_socket,                       // ENOTSOCK
    not_a_stream,                       // ENOSTR
    not_connected,                      // ENOTCONN
    not_enough_memory,                  // ENOMEM
    not_supported,                      // ENOTSUP
    operation_canceled,                 // ECANCELED
    operation_in_progress,              // EINPROGRESS
    operation_not_permitted,            // EPERM
    operation_not_supported,            // EOPNOTSUPP
    operation_would_block,              // EWOULDBLOCK
    owner_dead,                         // EOWNERDEAD
    permission_denied,                  // EACCES
    protocol_error,                     // EPROTO
    protocol_not_supported,             // EPROTONOSUPPORT
    read_only_file_system,              // EROFS
    resource_deadlock_would_occur,      // EDEADLK
    resource_unavailable_try_again,     // EAGAIN
    result_out_of_range,                // ERANGE
    state_not_recoverable,              // ENOTRECOVERABLE
    stream_timeout,                     // ETIME
    text_file_busy,                     // ETXTBSY
    timed_out,                          // ETIMEDOUT
    too_many_files_open_in_system,      // ENFILE
    too_many_files_open,                // EMFILE
    too_many_links,                     // EMLINK
    too_many_symbolic_link_levels,      // ELOOP
    value_too_large,                    // EOVERFLOW
    wrong_protocol_type                 // EPROTOTYPE
};

template <> struct is_error_condition_enum<errc> : true_type { }

error_code make_error_code(errc e);
error_condition make_error_condition(errc e);

// Comparison operators:
bool operator==(const error_code& lhs, const error_code& rhs);
bool operator==(const error_code& lhs, const error_condition& rhs);
bool operator==(const error_condition& lhs, const error_code& rhs);
bool operator==(const error_condition& lhs, const error_condition& rhs);
bool operator!=(const error_code& lhs, const error_code& rhs);
bool operator!=(const error_code& lhs, const error_condition& rhs);
bool operator!=(const error_condition& lhs, const error_code& rhs);
bool operator!=(const error_condition& lhs, const error_condition& rhs);

template <> struct hash<std::error_code>;

}  // std

*/

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/type_traits.h"
#include <cerrno>
#include <string>

// This must be here instead of a .inc file because it is used in the definition
// of the enum values below.
#ifdef LLVM_ON_WIN32

  // The following numbers were taken from VS2010.
# ifndef EAFNOSUPPORT
#   define EAFNOSUPPORT 102
# endif
# ifndef EADDRINUSE
#   define EADDRINUSE 100
# endif
# ifndef EADDRNOTAVAIL
#   define EADDRNOTAVAIL 101
# endif
# ifndef EISCONN
#   define EISCONN 113
# endif
# ifndef E2BIG
#   define E2BIG 7
# endif
# ifndef EDOM
#   define EDOM 33
# endif
# ifndef EFAULT
#   define EFAULT 14
# endif
# ifndef EBADF
#   define EBADF 9
# endif
# ifndef EBADMSG
#   define EBADMSG 104
# endif
# ifndef EPIPE
#   define EPIPE 32
# endif
# ifndef ECONNABORTED
#   define ECONNABORTED 106
# endif
# ifndef EALREADY
#   define EALREADY 103
# endif
# ifndef ECONNREFUSED
#   define ECONNREFUSED 107
# endif
# ifndef ECONNRESET
#   define ECONNRESET 108
# endif
# ifndef EXDEV
#   define EXDEV 18
# endif
# ifndef EDESTADDRREQ
#   define EDESTADDRREQ 109
# endif
# ifndef EBUSY
#   define EBUSY 16
# endif
# ifndef ENOTEMPTY
#   define ENOTEMPTY 41
# endif
# ifndef ENOEXEC
#   define ENOEXEC 8
# endif
# ifndef EEXIST
#   define EEXIST 17
# endif
# ifndef EFBIG
#   define EFBIG 27
# endif
# ifndef ENAMETOOLONG
#   define ENAMETOOLONG 38
# endif
# ifndef ENOSYS
#   define ENOSYS 40
# endif
# ifndef EHOSTUNREACH
#   define EHOSTUNREACH 110
# endif
# ifndef EIDRM
#   define EIDRM 111
# endif
# ifndef EILSEQ
#   define EILSEQ 42
# endif
# ifndef ENOTTY
#   define ENOTTY 25
# endif
# ifndef EINTR
#   define EINTR 4
# endif
# ifndef EINVAL
#   define EINVAL 22
# endif
# ifndef ESPIPE
#   define ESPIPE 29
# endif
# ifndef EIO
#   define EIO 5
# endif
# ifndef EISDIR
#   define EISDIR 21
# endif
# ifndef EMSGSIZE
#   define EMSGSIZE 115
# endif
# ifndef ENETDOWN
#   define ENETDOWN 116
# endif
# ifndef ENETRESET
#   define ENETRESET 117
# endif
# ifndef ENETUNREACH
#   define ENETUNREACH 118
# endif
# ifndef ENOBUFS
#   define ENOBUFS 119
# endif
# ifndef ECHILD
#   define ECHILD 10
# endif
# ifndef ENOLINK
#   define ENOLINK 121
# endif
# ifndef ENOLCK
#   define ENOLCK 39
# endif
# ifndef ENODATA
#   define ENODATA 120
# endif
# ifndef ENOMSG
#   define ENOMSG 122
# endif
# ifndef ENOPROTOOPT
#   define ENOPROTOOPT 123
# endif
# ifndef ENOSPC
#   define ENOSPC 28
# endif
# ifndef ENOSR
#   define ENOSR 124
# endif
# ifndef ENXIO
#   define ENXIO 6
# endif
# ifndef ENODEV
#   define ENODEV 19
# endif
# ifndef ENOENT
#   define ENOENT 2
# endif
# ifndef ESRCH
#   define ESRCH 3
# endif
# ifndef ENOTDIR
#   define ENOTDIR 20
# endif
# ifndef ENOTSOCK
#   define ENOTSOCK 128
# endif
# ifndef ENOSTR
#   define ENOSTR 125
# endif
# ifndef ENOTCONN
#   define ENOTCONN 126
# endif
# ifndef ENOMEM
#   define ENOMEM 12
# endif
# ifndef ENOTSUP
#   define ENOTSUP 129
# endif
# ifndef ECANCELED
#   define ECANCELED 105
# endif
# ifndef EINPROGRESS
#   define EINPROGRESS 112
# endif
# ifndef EPERM
#   define EPERM 1
# endif
# ifndef EOPNOTSUPP
#   define EOPNOTSUPP 130
# endif
# ifndef EWOULDBLOCK
#   define EWOULDBLOCK 140
# endif
# ifndef EOWNERDEAD
#   define EOWNERDEAD 133
# endif
# ifndef EACCES
#   define EACCES 13
# endif
# ifndef EPROTO
#   define EPROTO 134
# endif
# ifndef EPROTONOSUPPORT
#   define EPROTONOSUPPORT 135
# endif
# ifndef EROFS
#   define EROFS 30
# endif
# ifndef EDEADLK
#   define EDEADLK 36
# endif
# ifndef EAGAIN
#   define EAGAIN 11
# endif
# ifndef ERANGE
#   define ERANGE 34
# endif
# ifndef ENOTRECOVERABLE
#   define ENOTRECOVERABLE 127
# endif
# ifndef ETIME
#   define ETIME 137
# endif
# ifndef ETXTBSY
#   define ETXTBSY 139
# endif
# ifndef ETIMEDOUT
#   define ETIMEDOUT 138
# endif
# ifndef ENFILE
#   define ENFILE 23
# endif
# ifndef EMFILE
#   define EMFILE 24
# endif
# ifndef EMLINK
#   define EMLINK 31
# endif
# ifndef ELOOP
#   define ELOOP 114
# endif
# ifndef EOVERFLOW
#   define EOVERFLOW 132
# endif
# ifndef EPROTOTYPE
#   define EPROTOTYPE 136
# endif
#endif

namespace llvm {

// is_error_code_enum

template <class Tp> struct is_error_code_enum : public false_type {};

// is_error_condition_enum

template <class Tp> struct is_error_condition_enum : public false_type {};

// Some error codes are not present on all platforms, so we provide equivalents
// for them:

//enum class errc
struct errc {
enum _ {
  success                             = 0,
  address_family_not_supported        = EAFNOSUPPORT,
  address_in_use                      = EADDRINUSE,
  address_not_available               = EADDRNOTAVAIL,
  already_connected                   = EISCONN,
  argument_list_too_long              = E2BIG,
  argument_out_of_domain              = EDOM,
  bad_address                         = EFAULT,
  bad_file_descriptor                 = EBADF,
#ifdef EBADMSG
  bad_message                         = EBADMSG,
#else
  bad_message                         = EINVAL,
#endif
  broken_pipe                         = EPIPE,
  connection_aborted                  = ECONNABORTED,
  connection_already_in_progress      = EALREADY,
  connection_refused                  = ECONNREFUSED,
  connection_reset                    = ECONNRESET,
  cross_device_link                   = EXDEV,
  destination_address_required        = EDESTADDRREQ,
  device_or_resource_busy             = EBUSY,
  directory_not_empty                 = ENOTEMPTY,
  executable_format_error             = ENOEXEC,
  file_exists                         = EEXIST,
  file_too_large                      = EFBIG,
  filename_too_long                   = ENAMETOOLONG,
  function_not_supported              = ENOSYS,
  host_unreachable                    = EHOSTUNREACH,
  identifier_removed                  = EIDRM,
  illegal_byte_sequence               = EILSEQ,
  inappropriate_io_control_operation  = ENOTTY,
  interrupted                         = EINTR,
  invalid_argument                    = EINVAL,
  invalid_seek                        = ESPIPE,
  io_error                            = EIO,
  is_a_directory                      = EISDIR,
  message_size                        = EMSGSIZE,
  network_down                        = ENETDOWN,
  network_reset                       = ENETRESET,
  network_unreachable                 = ENETUNREACH,
  no_buffer_space                     = ENOBUFS,
  no_child_process                    = ECHILD,
#ifdef ENOLINK
  no_link                             = ENOLINK,
#else
  no_link                             = EINVAL,
#endif
  no_lock_available                   = ENOLCK,
#ifdef ENODATA
  no_message_available                = ENODATA,
#else
  no_message_available                = ENOMSG,
#endif
  no_message                          = ENOMSG,
  no_protocol_option                  = ENOPROTOOPT,
  no_space_on_device                  = ENOSPC,
#ifdef ENOSR
  no_stream_resources                 = ENOSR,
#else
  no_stream_resources                 = ENOMEM,
#endif
  no_such_device_or_address           = ENXIO,
  no_such_device                      = ENODEV,
  no_such_file_or_directory           = ENOENT,
  no_such_process                     = ESRCH,
  not_a_directory                     = ENOTDIR,
  not_a_socket                        = ENOTSOCK,
#ifdef ENOSTR
  not_a_stream                        = ENOSTR,
#else
  not_a_stream                        = EINVAL,
#endif
  not_connected                       = ENOTCONN,
  not_enough_memory                   = ENOMEM,
  not_supported                       = ENOTSUP,
#ifdef ECANCELED
  operation_canceled                  = ECANCELED,
#else
  operation_canceled                  = EINVAL,
#endif
  operation_in_progress               = EINPROGRESS,
  operation_not_permitted             = EPERM,
  operation_not_supported             = EOPNOTSUPP,
  operation_would_block               = EWOULDBLOCK,
#ifdef EOWNERDEAD
  owner_dead                          = EOWNERDEAD,
#else
  owner_dead                          = EINVAL,
#endif
  permission_denied                   = EACCES,
#ifdef EPROTO
  protocol_error                      = EPROTO,
#else
  protocol_error                      = EINVAL,
#endif
  protocol_not_supported              = EPROTONOSUPPORT,
  read_only_file_system               = EROFS,
  resource_deadlock_would_occur       = EDEADLK,
  resource_unavailable_try_again      = EAGAIN,
  result_out_of_range                 = ERANGE,
#ifdef ENOTRECOVERABLE
  state_not_recoverable               = ENOTRECOVERABLE,
#else
  state_not_recoverable               = EINVAL,
#endif
#ifdef ETIME
  stream_timeout                      = ETIME,
#else
  stream_timeout                      = ETIMEDOUT,
#endif
  text_file_busy                      = ETXTBSY,
  timed_out                           = ETIMEDOUT,
  too_many_files_open_in_system       = ENFILE,
  too_many_files_open                 = EMFILE,
  too_many_links                      = EMLINK,
  too_many_symbolic_link_levels       = ELOOP,
  value_too_large                     = EOVERFLOW,
  wrong_protocol_type                 = EPROTOTYPE
};

  _ v_;

  errc(_ v) : v_(v) {}
  operator int() const {return v_;}
};

template <> struct is_error_condition_enum<errc> : true_type { };

template <> struct is_error_condition_enum<errc::_> : true_type { };

class error_condition;
class error_code;

// class error_category

class _do_message;

class error_category
{
public:
  virtual ~error_category();

private:
  error_category();
  error_category(const error_category&) LLVM_DELETED_FUNCTION;
  error_category& operator=(const error_category&) LLVM_DELETED_FUNCTION;

public:
  virtual const char* name() const = 0;
  virtual error_condition default_error_condition(int _ev) const;
  virtual bool equivalent(int _code, const error_condition& _condition) const;
  virtual bool equivalent(const error_code& _code, int _condition) const;
  virtual std::string message(int _ev) const = 0;

  bool operator==(const error_category& _rhs) const {return this == &_rhs;}

  bool operator!=(const error_category& _rhs) const {return !(*this == _rhs);}

  bool operator< (const error_category& _rhs) const {return this < &_rhs;}

  friend class _do_message;
};

class _do_message : public error_category
{
public:
  virtual std::string message(int ev) const LLVM_OVERRIDE;
};

const error_category& generic_category();
const error_category& system_category();

/// Get the error_category used for errno values from POSIX functions. This is
/// the same as the system_category on POSIX systems, but is the same as the
/// generic_category on Windows.
const error_category& posix_category();

class error_condition
{
  int _val_;
  const error_category* _cat_;
public:
  error_condition() : _val_(0), _cat_(&generic_category()) {}

  error_condition(int _val, const error_category& _cat)
    : _val_(_val), _cat_(&_cat) {}

  template <class E>
  error_condition(E _e, typename enable_if_c<
                          is_error_condition_enum<E>::value
                        >::type* = 0)
    {*this = make_error_condition(_e);}

  void assign(int _val, const error_category& _cat) {
    _val_ = _val;
    _cat_ = &_cat;
  }

  template <class E>
    typename enable_if_c
    <
      is_error_condition_enum<E>::value,
      error_condition&
    >::type
    operator=(E _e)
      {*this = make_error_condition(_e); return *this;}

  void clear() {
    _val_ = 0;
    _cat_ = &generic_category();
  }

  int value() const {return _val_;}

  const error_category& category() const {return *_cat_;}
  std::string message() const;

  typedef void (*unspecified_bool_type)();
  static void unspecified_bool_true() {}

  operator unspecified_bool_type() const { // true if error
    return _val_ == 0 ? 0 : unspecified_bool_true;
  }
};

inline error_condition make_error_condition(errc _e) {
  return error_condition(static_cast<int>(_e), generic_category());
}

inline bool operator<(const error_condition& _x, const error_condition& _y) {
  return _x.category() < _y.category()
      || (_x.category() == _y.category() && _x.value() < _y.value());
}

// error_code

class error_code {
  int _val_;
  const error_category* _cat_;
public:
  error_code() : _val_(0), _cat_(&system_category()) {}

  static error_code success() {
    return error_code();
  }

  error_code(int _val, const error_category& _cat)
    : _val_(_val), _cat_(&_cat) {}

  template <class E>
  error_code(E _e, typename enable_if_c<
                     is_error_code_enum<E>::value
                   >::type* = 0) {
    *this = make_error_code(_e);
  }

  void assign(int _val, const error_category& _cat) {
      _val_ = _val;
      _cat_ = &_cat;
  }

  template <class E>
    typename enable_if_c
    <
      is_error_code_enum<E>::value,
      error_code&
    >::type
    operator=(E _e)
      {*this = make_error_code(_e); return *this;}

  void clear() {
    _val_ = 0;
    _cat_ = &system_category();
  }

  int value() const {return _val_;}

  const error_category& category() const {return *_cat_;}

  error_condition default_error_condition() const
    {return _cat_->default_error_condition(_val_);}

  std::string message() const;

  typedef void (*unspecified_bool_type)();
  static void unspecified_bool_true() {}

  operator unspecified_bool_type() const { // true if error
    return _val_ == 0 ? 0 : unspecified_bool_true;
  }
};

inline error_code make_error_code(errc _e) {
  return error_code(static_cast<int>(_e), generic_category());
}

inline bool operator<(const error_code& _x, const error_code& _y) {
  return _x.category() < _y.category()
      || (_x.category() == _y.category() && _x.value() < _y.value());
}

inline bool operator==(const error_code& _x, const error_code& _y) {
  return _x.category() == _y.category() && _x.value() == _y.value();
}

inline bool operator==(const error_code& _x, const error_condition& _y) {
  return _x.category().equivalent(_x.value(), _y)
      || _y.category().equivalent(_x, _y.value());
}

inline bool operator==(const error_condition& _x, const error_code& _y) {
  return _y == _x;
}

inline bool operator==(const error_condition& _x, const error_condition& _y) {
   return _x.category() == _y.category() && _x.value() == _y.value();
}

inline bool operator!=(const error_code& _x, const error_code& _y) {
  return !(_x == _y);
}

inline bool operator!=(const error_code& _x, const error_condition& _y) {
  return !(_x == _y);
}

inline bool operator!=(const error_condition& _x, const error_code& _y) {
  return !(_x == _y);
}

inline bool operator!=(const error_condition& _x, const error_condition& _y) {
  return !(_x == _y);
}

// Windows errors.

//  To construct an error_code after an API error:
//
//      error_code( ::GetLastError(), system_category() )
struct windows_error {
enum _ {
  success = 0,
  // These names and values are based on Windows WinError.h
  // This is not a complete list. Add to this list if you need to explicitly
  // check for it.
  invalid_function        = 1, // ERROR_INVALID_FUNCTION,
  file_not_found          = 2, // ERROR_FILE_NOT_FOUND,
  path_not_found          = 3, // ERROR_PATH_NOT_FOUND,
  too_many_open_files     = 4, // ERROR_TOO_MANY_OPEN_FILES,
  access_denied           = 5, // ERROR_ACCESS_DENIED,
  invalid_handle          = 6, // ERROR_INVALID_HANDLE,
  arena_trashed           = 7, // ERROR_ARENA_TRASHED,
  not_enough_memory       = 8, // ERROR_NOT_ENOUGH_MEMORY,
  invalid_block           = 9, // ERROR_INVALID_BLOCK,
  bad_environment         = 10, // ERROR_BAD_ENVIRONMENT,
  bad_format              = 11, // ERROR_BAD_FORMAT,
  invalid_access          = 12, // ERROR_INVALID_ACCESS,
  outofmemory             = 14, // ERROR_OUTOFMEMORY,
  invalid_drive           = 15, // ERROR_INVALID_DRIVE,
  current_directory       = 16, // ERROR_CURRENT_DIRECTORY,
  not_same_device         = 17, // ERROR_NOT_SAME_DEVICE,
  no_more_files           = 18, // ERROR_NO_MORE_FILES,
  write_protect           = 19, // ERROR_WRITE_PROTECT,
  bad_unit                = 20, // ERROR_BAD_UNIT,
  not_ready               = 21, // ERROR_NOT_READY,
  bad_command             = 22, // ERROR_BAD_COMMAND,
  crc                     = 23, // ERROR_CRC,
  bad_length              = 24, // ERROR_BAD_LENGTH,
  seek                    = 25, // ERROR_SEEK,
  not_dos_disk            = 26, // ERROR_NOT_DOS_DISK,
  sector_not_found        = 27, // ERROR_SECTOR_NOT_FOUND,
  out_of_paper            = 28, // ERROR_OUT_OF_PAPER,
  write_fault             = 29, // ERROR_WRITE_FAULT,
  read_fault              = 30, // ERROR_READ_FAULT,
  gen_failure             = 31, // ERROR_GEN_FAILURE,
  sharing_violation       = 32, // ERROR_SHARING_VIOLATION,
  lock_violation          = 33, // ERROR_LOCK_VIOLATION,
  wrong_disk              = 34, // ERROR_WRONG_DISK,
  sharing_buffer_exceeded = 36, // ERROR_SHARING_BUFFER_EXCEEDED,
  handle_eof              = 38, // ERROR_HANDLE_EOF,
  handle_disk_full        = 39, // ERROR_HANDLE_DISK_FULL,
  rem_not_list            = 51, // ERROR_REM_NOT_LIST,
  dup_name                = 52, // ERROR_DUP_NAME,
  bad_net_path            = 53, // ERROR_BAD_NETPATH,
  network_busy            = 54, // ERROR_NETWORK_BUSY,
  file_exists             = 80, // ERROR_FILE_EXISTS,
  cannot_make             = 82, // ERROR_CANNOT_MAKE,
  broken_pipe             = 109, // ERROR_BROKEN_PIPE,
  open_failed             = 110, // ERROR_OPEN_FAILED,
  buffer_overflow         = 111, // ERROR_BUFFER_OVERFLOW,
  disk_full               = 112, // ERROR_DISK_FULL,
  insufficient_buffer     = 122, // ERROR_INSUFFICIENT_BUFFER,
  lock_failed             = 167, // ERROR_LOCK_FAILED,
  busy                    = 170, // ERROR_BUSY,
  cancel_violation        = 173, // ERROR_CANCEL_VIOLATION,
  already_exists          = 183  // ERROR_ALREADY_EXISTS
};
  _ v_;

  windows_error(_ v) : v_(v) {}
  explicit windows_error(int v) : v_(_(v)) {}
  operator int() const {return v_;}
};


template <> struct is_error_code_enum<windows_error> : true_type { };

template <> struct is_error_code_enum<windows_error::_> : true_type { };

inline error_code make_error_code(windows_error e) {
  return error_code(static_cast<int>(e), system_category());
}

} // end namespace llvm

#endif
