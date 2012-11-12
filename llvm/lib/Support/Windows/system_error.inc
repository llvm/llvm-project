//===- llvm/Support/Win32/system_error.inc - Windows error_code --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the Windows specific implementation of the error_code
// and error_condition classes.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic Windows code that
//===          is guaranteed to work on *all* Windows variants.
//===----------------------------------------------------------------------===//

#include <windows.h>
#include <winerror.h>

using namespace llvm;

std::string
_system_error_category::message(int ev) const {
  LPVOID lpMsgBuf = 0;
  DWORD retval = ::FormatMessageA(
    FORMAT_MESSAGE_ALLOCATE_BUFFER |
    FORMAT_MESSAGE_FROM_SYSTEM |
    FORMAT_MESSAGE_IGNORE_INSERTS,
    NULL,
    ev,
    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
    (LPSTR) &lpMsgBuf,
    0,
    NULL);
  if (retval == 0) {
    ::LocalFree(lpMsgBuf);
    return std::string("Unknown error");
  }

  std::string str( static_cast<LPCSTR>(lpMsgBuf) );
  ::LocalFree(lpMsgBuf);

  while (str.size()
     && (str[str.size()-1] == '\n' || str[str.size()-1] == '\r'))
    str.erase( str.size()-1 );
  if (str.size() && str[str.size()-1] == '.')
    str.erase( str.size()-1 );
  return str;
}

// I'd rather not double the line count of the following.
#define MAP_ERR_TO_COND(x, y) case x: return make_error_condition(errc::y)

error_condition
_system_error_category::default_error_condition(int ev) const {
  switch (ev) {
  MAP_ERR_TO_COND(0, success);
  // Windows system -> posix_errno decode table  ---------------------------//
  // see WinError.h comments for descriptions of errors
  MAP_ERR_TO_COND(ERROR_ACCESS_DENIED,       permission_denied);
  MAP_ERR_TO_COND(ERROR_ALREADY_EXISTS,      file_exists);
  MAP_ERR_TO_COND(ERROR_BAD_UNIT,            no_such_device);
  MAP_ERR_TO_COND(ERROR_BUFFER_OVERFLOW,     filename_too_long);
  MAP_ERR_TO_COND(ERROR_BUSY,                device_or_resource_busy);
  MAP_ERR_TO_COND(ERROR_BUSY_DRIVE,          device_or_resource_busy);
  MAP_ERR_TO_COND(ERROR_CANNOT_MAKE,         permission_denied);
  MAP_ERR_TO_COND(ERROR_CANTOPEN,            io_error);
  MAP_ERR_TO_COND(ERROR_CANTREAD,            io_error);
  MAP_ERR_TO_COND(ERROR_CANTWRITE,           io_error);
  MAP_ERR_TO_COND(ERROR_CURRENT_DIRECTORY,   permission_denied);
  MAP_ERR_TO_COND(ERROR_DEV_NOT_EXIST,       no_such_device);
  MAP_ERR_TO_COND(ERROR_DEVICE_IN_USE,       device_or_resource_busy);
  MAP_ERR_TO_COND(ERROR_DIR_NOT_EMPTY,       directory_not_empty);
  MAP_ERR_TO_COND(ERROR_DIRECTORY,           invalid_argument);
  MAP_ERR_TO_COND(ERROR_DISK_FULL,           no_space_on_device);
  MAP_ERR_TO_COND(ERROR_FILE_EXISTS,         file_exists);
  MAP_ERR_TO_COND(ERROR_FILE_NOT_FOUND,      no_such_file_or_directory);
  MAP_ERR_TO_COND(ERROR_HANDLE_DISK_FULL,    no_space_on_device);
  MAP_ERR_TO_COND(ERROR_HANDLE_EOF,          value_too_large);
  MAP_ERR_TO_COND(ERROR_INVALID_ACCESS,      permission_denied);
  MAP_ERR_TO_COND(ERROR_INVALID_DRIVE,       no_such_device);
  MAP_ERR_TO_COND(ERROR_INVALID_FUNCTION,    function_not_supported);
  MAP_ERR_TO_COND(ERROR_INVALID_HANDLE,      invalid_argument);
  MAP_ERR_TO_COND(ERROR_INVALID_NAME,        invalid_argument);
  MAP_ERR_TO_COND(ERROR_LOCK_VIOLATION,      no_lock_available);
  MAP_ERR_TO_COND(ERROR_LOCKED,              no_lock_available);
  MAP_ERR_TO_COND(ERROR_NEGATIVE_SEEK,       invalid_argument);
  MAP_ERR_TO_COND(ERROR_NOACCESS,            permission_denied);
  MAP_ERR_TO_COND(ERROR_NOT_ENOUGH_MEMORY,   not_enough_memory);
  MAP_ERR_TO_COND(ERROR_NOT_READY,           resource_unavailable_try_again);
  MAP_ERR_TO_COND(ERROR_NOT_SAME_DEVICE,     cross_device_link);
  MAP_ERR_TO_COND(ERROR_OPEN_FAILED,         io_error);
  MAP_ERR_TO_COND(ERROR_OPEN_FILES,          device_or_resource_busy);
  MAP_ERR_TO_COND(ERROR_OPERATION_ABORTED,   operation_canceled);
  MAP_ERR_TO_COND(ERROR_OUTOFMEMORY,         not_enough_memory);
  MAP_ERR_TO_COND(ERROR_PATH_NOT_FOUND,      no_such_file_or_directory);
  MAP_ERR_TO_COND(ERROR_BAD_NETPATH,         no_such_file_or_directory);
  MAP_ERR_TO_COND(ERROR_READ_FAULT,          io_error);
  MAP_ERR_TO_COND(ERROR_RETRY,               resource_unavailable_try_again);
  MAP_ERR_TO_COND(ERROR_SEEK,                io_error);
  MAP_ERR_TO_COND(ERROR_SHARING_VIOLATION,   permission_denied);
  MAP_ERR_TO_COND(ERROR_TOO_MANY_OPEN_FILES, too_many_files_open);
  MAP_ERR_TO_COND(ERROR_WRITE_FAULT,         io_error);
  MAP_ERR_TO_COND(ERROR_WRITE_PROTECT,       permission_denied);
  MAP_ERR_TO_COND(ERROR_SEM_TIMEOUT,         timed_out);
  MAP_ERR_TO_COND(WSAEACCES,                 permission_denied);
  MAP_ERR_TO_COND(WSAEADDRINUSE,             address_in_use);
  MAP_ERR_TO_COND(WSAEADDRNOTAVAIL,          address_not_available);
  MAP_ERR_TO_COND(WSAEAFNOSUPPORT,           address_family_not_supported);
  MAP_ERR_TO_COND(WSAEALREADY,               connection_already_in_progress);
  MAP_ERR_TO_COND(WSAEBADF,                  bad_file_descriptor);
  MAP_ERR_TO_COND(WSAECONNABORTED,           connection_aborted);
  MAP_ERR_TO_COND(WSAECONNREFUSED,           connection_refused);
  MAP_ERR_TO_COND(WSAECONNRESET,             connection_reset);
  MAP_ERR_TO_COND(WSAEDESTADDRREQ,           destination_address_required);
  MAP_ERR_TO_COND(WSAEFAULT,                 bad_address);
  MAP_ERR_TO_COND(WSAEHOSTUNREACH,           host_unreachable);
  MAP_ERR_TO_COND(WSAEINPROGRESS,            operation_in_progress);
  MAP_ERR_TO_COND(WSAEINTR,                  interrupted);
  MAP_ERR_TO_COND(WSAEINVAL,                 invalid_argument);
  MAP_ERR_TO_COND(WSAEISCONN,                already_connected);
  MAP_ERR_TO_COND(WSAEMFILE,                 too_many_files_open);
  MAP_ERR_TO_COND(WSAEMSGSIZE,               message_size);
  MAP_ERR_TO_COND(WSAENAMETOOLONG,           filename_too_long);
  MAP_ERR_TO_COND(WSAENETDOWN,               network_down);
  MAP_ERR_TO_COND(WSAENETRESET,              network_reset);
  MAP_ERR_TO_COND(WSAENETUNREACH,            network_unreachable);
  MAP_ERR_TO_COND(WSAENOBUFS,                no_buffer_space);
  MAP_ERR_TO_COND(WSAENOPROTOOPT,            no_protocol_option);
  MAP_ERR_TO_COND(WSAENOTCONN,               not_connected);
  MAP_ERR_TO_COND(WSAENOTSOCK,               not_a_socket);
  MAP_ERR_TO_COND(WSAEOPNOTSUPP,             operation_not_supported);
  MAP_ERR_TO_COND(WSAEPROTONOSUPPORT,        protocol_not_supported);
  MAP_ERR_TO_COND(WSAEPROTOTYPE,             wrong_protocol_type);
  MAP_ERR_TO_COND(WSAETIMEDOUT,              timed_out);
  MAP_ERR_TO_COND(WSAEWOULDBLOCK,            operation_would_block);
  default: return error_condition(ev, system_category());
  }
}
