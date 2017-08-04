#
#//===----------------------------------------------------------------------===//
#//
#//                     The LLVM Compiler Infrastructure
#//
#// This file is dual licensed under the MIT and the University of Illinois Open
#// Source Licenses. See LICENSE.txt for details.
#//
#//===----------------------------------------------------------------------===//
#

# void libomptarget_say(string message_to_user);
# - prints out message_to_user
macro(libomptarget_say message_to_user)
  message(STATUS "LIBOMPTARGET: ${message_to_user}")
endmacro()

# void libomptarget_warning_say(string message_to_user);
# - prints out message_to_user with a warning
macro(libomptarget_warning_say message_to_user)
  message(WARNING "LIBOMPTARGET: ${message_to_user}")
endmacro()

# void libomptarget_error_say(string message_to_user);
# - prints out message_to_user with an error and exits cmake
macro(libomptarget_error_say message_to_user)
  message(FATAL_ERROR "LIBOMPTARGET: ${message_to_user}")
endmacro()
