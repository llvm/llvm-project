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

# void libomp_say(string message_to_user);
# - prints out message_to_user
macro(libomp_say message_to_user)
  message(STATUS "LIBOMP: ${message_to_user}")
endmacro()

# void libomp_warning_say(string message_to_user);
# - prints out message_to_user with a warning
macro(libomp_warning_say message_to_user)
  message(WARNING "LIBOMP: ${message_to_user}")
endmacro()

# void libomp_error_say(string message_to_user);
# - prints out message_to_user with an error and exits cmake
macro(libomp_error_say message_to_user)
  message(FATAL_ERROR "LIBOMP: ${message_to_user}")
endmacro()

# libomp_append(<flag> <flags_list> [(IF_TRUE | IF_FALSE | IF_TRUE_1_0 ) BOOLEAN])
#
# libomp_append(<flag> <flags_list>)
#   - unconditionally appends <flag> to the list of definitions
#
# libomp_append(<flag> <flags_list> <BOOLEAN>)
#   - appends <flag> to the list of definitions if BOOLEAN is true
#
# libomp_append(<flag> <flags_list> IF_TRUE <BOOLEAN>)
#   - appends <flag> to the list of definitions if BOOLEAN is true
#
# libomp_append(<flag> <flags_list> IF_FALSE <BOOLEAN>)
#   - appends <flag> to the list of definitions if BOOLEAN is false
#
# libomp_append(<flag> <flags_list> IF_DEFINED <VARIABLE>)
#   - appends <flag> to the list of definitions if VARIABLE is defined
#
# libomp_append(<flag> <flags_list> IF_TRUE_1_0 <BOOLEAN>)
#   - appends <flag>=1 to the list of definitions if <BOOLEAN> is true, <flag>=0 otherwise
# e.g., libomp_append("-D USE_FEATURE" IF_TRUE_1_0 HAVE_FEATURE)
#     appends "-D USE_FEATURE=1" if HAVE_FEATURE is true
#     or "-D USE_FEATURE=0" if HAVE_FEATURE is false
macro(libomp_append flags flag)
  if(NOT (${ARGC} EQUAL 2 OR ${ARGC} EQUAL 3 OR ${ARGC} EQUAL 4))
    libomp_error_say("libomp_append: takes 2, 3, or 4 arguments")
  endif()
  if(${ARGC} EQUAL 2)
    list(APPEND ${flags} "${flag}")
  elseif(${ARGC} EQUAL 3)
    if(${ARGV2})
      list(APPEND ${flags} "${flag}")
    endif()
  else()
    if(${ARGV2} STREQUAL "IF_TRUE")
      if(${ARGV3})
        list(APPEND ${flags} "${flag}")
      endif()
    elseif(${ARGV2} STREQUAL "IF_FALSE")
      if(NOT ${ARGV3})
        list(APPEND ${flags} "${flag}")
      endif()
    elseif(${ARGV2} STREQUAL "IF_DEFINED")
      if(DEFINED ${ARGV3})
        list(APPEND ${flags} "${flag}")
      endif()
    elseif(${ARGV2} STREQUAL "IF_TRUE_1_0")
      if(${ARGV3})
        list(APPEND ${flags} "${flag}=1")
      else()
        list(APPEND ${flags} "${flag}=0")
      endif()
    else()
      libomp_error_say("libomp_append: third argument must be one of IF_TRUE, IF_FALSE, IF_DEFINED, IF_TRUE_1_0")
    endif()
  endif()
endmacro()

# void libomp_get_legal_arch(string* return_arch_string);
# - returns (through return_arch_string) the formal architecture
#   string or warns user of unknown architecture
function(libomp_get_legal_arch return_arch_string)
  if(${IA32})
    set(${return_arch_string} "IA-32" PARENT_SCOPE)
  elseif(${INTEL64})
    set(${return_arch_string} "Intel(R) 64" PARENT_SCOPE)
  elseif(${MIC})
    set(${return_arch_string} "Intel(R) Many Integrated Core Architecture" PARENT_SCOPE)
  elseif(${ARM})
    set(${return_arch_string} "ARM" PARENT_SCOPE)
  elseif(${PPC64BE})
    set(${return_arch_string} "PPC64BE" PARENT_SCOPE)
  elseif(${PPC64LE})
    set(${return_arch_string} "PPC64LE" PARENT_SCOPE)
  elseif(${AARCH64})
    set(${return_arch_string} "AARCH64" PARENT_SCOPE)
  else()
    set(${return_arch_string} "${LIBOMP_ARCH}" PARENT_SCOPE)
    libomp_warning_say("libomp_get_legal_arch(): Warning: Unknown architecture: Using ${LIBOMP_ARCH}")
  endif()
endfunction()

# void libomp_check_variable(string var, ...);
# - runs through all values checking if ${var} == value
# - uppercase and lowercase do not matter
# - if the var is found, then just print it out
# - if the var is not found, then error out
function(libomp_check_variable var)
  set(valid_flag 0)
  string(TOLOWER "${${var}}" var_lower)
  foreach(value IN LISTS ARGN)
    string(TOLOWER "${value}" value_lower)
    if("${var_lower}" STREQUAL "${value_lower}")
      set(valid_flag 1)
      set(the_value "${value}")
    endif()
  endforeach()
  if(${valid_flag} EQUAL 0)
    libomp_error_say("libomp_check_variable(): ${var} = ${${var}} is unknown")
  endif()
endfunction()

# void libomp_get_build_number(string src_dir, string* return_build_number);
# - grab the eight digit build number (or 00000000) from kmp_version.c
function(libomp_get_build_number src_dir return_build_number)
  # sets file_lines_list to a list of all lines in kmp_version.c
  file(STRINGS "${src_dir}/src/kmp_version.c" file_lines_list)

  # runs through each line in kmp_version.c
  foreach(line IN LISTS file_lines_list)
    # if the line begins with "#define KMP_VERSION_BUILD" then we take not of the build number
    string(REGEX MATCH "^[ \t]*#define[ \t]+KMP_VERSION_BUILD" valid "${line}")
    if(NOT "${valid}" STREQUAL "") # if we matched "#define KMP_VERSION_BUILD", then grab the build number
      string(REGEX REPLACE "^[ \t]*#define[ \t]+KMP_VERSION_BUILD[ \t]+([0-9]+)" "\\1"
           build_number "${line}"
      )
    endif()
  endforeach()
  set(${return_build_number} "${build_number}" PARENT_SCOPE) # return build number
endfunction()

# void libomp_get_legal_type(string* return_legal_type);
# - set the legal type name Performance/Profiling/Stub
function(libomp_get_legal_type return_legal_type)
  if(${NORMAL_LIBRARY})
    set(${return_legal_type} "Performance" PARENT_SCOPE)
  elseif(${PROFILE_LIBRARY})
    set(${return_legal_type} "Profiling" PARENT_SCOPE)
  elseif(${STUBS_LIBRARY})
    set(${return_legal_type} "Stub" PARENT_SCOPE)
  endif()
endfunction()

# void libomp_add_suffix(string suffix, list<string>* list_of_items);
# - returns list_of_items with suffix appended to all items
# - original list is modified
function(libomp_add_suffix suffix list_of_items)
  set(local_list "")
  foreach(item IN LISTS "${list_of_items}")
    if(NOT "${item}" STREQUAL "")
      list(APPEND local_list "${item}${suffix}")
    endif()
  endforeach()
  set(${list_of_items} "${local_list}" PARENT_SCOPE)
endfunction()

# void libomp_list_to_string(list<string> list_of_things, string* return_string);
# - converts a list to a space separated string
function(libomp_list_to_string list_of_things return_string)
  string(REPLACE ";" " " output_variable "${list_of_things}")
  set(${return_string} "${output_variable}" PARENT_SCOPE)
endfunction()

# void libomp_string_to_list(string str, list<string>* return_list);
# - converts a string to a semicolon separated list
# - what it really does is just string_replace all running whitespace to a semicolon
# - in cmake, a list is strings separated by semicolons: i.e., list of four items, list = "item1;item2;item3;item4"
function(libomp_string_to_list str return_list)
  set(outstr)
  string(REGEX REPLACE "[ \t]+" ";" outstr "${str}")
  set(${return_list} "${outstr}" PARENT_SCOPE)
endfunction()

