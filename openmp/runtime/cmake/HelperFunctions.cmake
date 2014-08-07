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

####################################### FUNCTIONS/MACROS ###########################################
# It should be noted that in cmake, functions can only be used on a single line with the return value
# stored in a parameter you send to the function.  There isn't a true return value.  So technically, 
# all functions would have a C/C++ prototype of:
# void function_name(parameter1, parameter2, ...);
#  
# If you want a return value, you have to use a parameter so the function prototype would be:
# void function_name(input_parameter1, input_parameter2, ...,  return_value)
# ##################

# void say(string message_to_user);
# - prints out message_to_user
macro(say message_to_user)
    message("${message_to_user}")
endmacro()

# void warning_say(string message_to_user);
# - prints out message_to_user with a warning
macro(warning_say message_to_user)
    message(WARNING "${message_to_user}")
endmacro()

# void error_say(string message_to_user);
# - prints out message_to_user with an error and exits cmake
macro(error_say message_to_user)
    message(FATAL_ERROR "${message_to_user}")
endmacro()

# void debug_say(string message_to_developer);
# - prints out message when GLOBAL_DEBUG == 1 (for debugging cmake build)
macro(debug_say message_to_developer)
    if(${GLOBAL_DEBUG} STREQUAL "1")
        say("DEBUG: ${message_to_developer}")
    endif()
endmacro()

# void debug_say_var(variable var);
# - prints the variable name and its value (for debugging cmake build)
macro(debug_say_var var)
    if(${GLOBAL_DEBUG} STREQUAL "1")
        say("DEBUG: Variable: ${var} = ${${var}} ")
    endif()
endmacro()

# void set_legal_arch(string* return_arch_string);
# - returns (through return_arch_string) the formal architecture 
#   string or warns user of unknown architecture
function(set_legal_arch return_arch_string)
    if(${IA32}) 
        set(${return_arch_string} "IA-32" PARENT_SCOPE)
    elseif(${INTEL64})
        set(${return_arch_string} "Intel(R) 64" PARENT_SCOPE)
    elseif(${MIC})
        set(${return_arch_string} "Intel(R) Many Integrated Core Architecture" PARENT_SCOPE)
    elseif(${arch} STREQUAL "l1")
        set(${return_arch_string} "L1OM" PARENT_SCOPE)
    elseif(${ARM})
        set(${return_arch_string} "ARM" PARENT_SCOPE)
	elseif(${PPC64})
	    set(${return_arch_string} "PPC64" PARENT_SCOPE)
    else()
        warning_say("set_legal_arch(): Warning: Unknown architecture...")
    endif()
endfunction()

# void check_variable(string var, string var_name, list<string>values_list);
# - runs through values_list checking if ${var} == values_list[i] for any i.
# - if the var is found, then just print it out
# - if the var is not found, then warn user
function(check_variable var values_list)
    set(valid_flag 0)
    foreach(value IN LISTS values_list)
        if("${${var}}" STREQUAL "${value}")
            set(valid_flag 1)
            set(the_value "${value}")
        endif()
    endforeach()
    if(${valid_flag} EQUAL 0)
        error_say("check_variable(): ${var} = ${${var}} is unknown")
    endif()
endfunction()

# void _export_lib_dir(string export_dir, string platform, string suffix, string* return_value);
# - basically a special case for mac platforms where it adds '.thin' to the output lib directory
function(_export_lib_dir pltfrm return_value)
    if(${MAC})
        set(${return_value} "${export_dir}/${pltfrm}${suffix}/lib.thin" PARENT_SCOPE)
    else()
        set(${return_value} "${export_dir}/${pltfrm}${suffix}/lib" PARENT_SCOPE)
    endif()
endfunction()

# void _export_lib_fat_dir(string export_dir, string platform, string suffix, string* return_value);
# - another mac specialty case for fat libraries.
# - this sets export_lib_fat_dir in the MAIN part of CMakeLists.txt
function(_export_lib_fat_dir pltfrm return_value)
    set(${return_value} "${export_dir}/${pltfrm}${suffix}/lib" PARENT_SCOPE)
endfunction()

# void get_build_number(string src_dir, string* return_build_number);
# - grab the eight digit build number (or 00000000) from kmp_version.c
function(get_build_number src_dir return_build_number)
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

# void set_legal_type(string* return_legal_type);
# - set the legal type name Performance/Profiling/Stub
function(set_legal_type return_legal_type)
    if(${NORMAL_LIBRARY})
        set(${return_legal_type} "Performance" PARENT_SCOPE)
    elseif(${PROFILE_LIBRARY})
        set(${return_legal_type} "Profiling" PARENT_SCOPE)
    elseif(${STUBS_LIBRARY})
        set(${return_legal_type} "Stub" PARENT_SCOPE)
    endif()
endfunction()

# void set_mac_os_new(bool* return_mac_os_new);
# - sets the return_mac_os_new variable to true or false based on macosx version
# - no real "cmakey" way to do this.  Have to call execute_process()
function(set_mac_os_new return_mac_os_new)
    execute_process(COMMAND "sw_vers" "-productVersion" OUTPUT_VARIABLE mac_osx_version)
    if("${mac_osx_version}" VERSION_GREATER "10.6")
        set(${return_mac_os_new} TRUE PARENT_SCOPE)
    else()
        set(${return_mac_os_new} FALSE PARENT_SCOPE)
    endif()
endfunction()

# void add_prefix(string prefix, list<string>* list_of_items);
# - returns list_of_items with prefix prepended to all items
# - original list is modified
function(add_prefix prefix list_of_items)
    set(local_list "")
    foreach(item IN LISTS "${list_of_items}")
        if(NOT "${item}" STREQUAL "")
            list(APPEND local_list "${prefix}${item}")
        endif()
    endforeach()
    set(${list_of_items} "${local_list}" PARENT_SCOPE)
endfunction()

# void add_suffix(string suffix, list<string>* list_of_items);
# - returns list_of_items with suffix appended to all items
# - original list is modified
function(add_suffix suffix list_of_items)
    set(local_list "")
    foreach(item IN LISTS "${list_of_items}")
        if(NOT "${item}" STREQUAL "")
            list(APPEND local_list "${item}${suffix}")
        endif()
    endforeach()
    set(${list_of_items} "${local_list}" PARENT_SCOPE)
endfunction()

# void strip_suffix(list<string> list_of_items, list<string>* return_list);
# - returns a new list with suffix stripped (i.e., foo.c => foo)
# - list_of_items is not modified, return_list is modified
function(strip_suffix list_of_items return_list)
    set(local_list "")
    foreach(item IN LISTS "${list_of_items}")
        if(NOT "${item}" STREQUAL "")
            get_filename_component(filename "${item}" NAME_WE)
            list(APPEND local_list "${filename}")
        endif()
    endforeach()
    set(${return_list} "${local_list}" PARENT_SCOPE)
endfunction()

# void list_to_string(list<string> list_of_things, string* return_string);
# - converts a list to a space separated string
function(list_to_string list_of_things return_string)
    string(REPLACE ";" " " output_variable "${list_of_things}")
    set(${return_string} "${output_variable}" PARENT_SCOPE)
endfunction()

# void string_to_list(string str, list<string>* return_list);
# - converts a string to a semicolon separated list
# - what it really does is just string_replace all running whitespace to a semicolon
# - in cmake, a list is strings separated by semicolons: i.e., list of four items, list = "item1;item2;item3;item4"
function(string_to_list str return_list)
    set(outstr)
    string(REGEX REPLACE "[ \t]+" ";" outstr "${str}")
    set(${return_list} "${outstr}" PARENT_SCOPE) 
endfunction()

# void get_date(string* return_date);
# - returns the current date "yyyy-mm-dd hh:mm:ss UTC"
# - this function alone causes the need for CMake v2.8.11 (TIMESTAMP)
#function(get_date return_date)
#    string(TIMESTAMP local_date "%Y-%m-%d %H:%M:%S UTC" UTC)
#    set(${return_date} ${local_date} PARENT_SCOPE)
#endfunction()

# void find_a_program(string program_name, list<string> extra_paths, bool fail_on_not_found, string return_variable_name);
# - returns the full path of a program_name
# - first looks in the list of extra_paths
# - if not found in extra_paths, then look through system path
# - errors out if fail_on_not_found == true and cmake could not find program_name.
function(find_a_program program_name extra_paths fail_on_not_found return_variable_name)
    # first try to find the program in the extra_paths
    find_program(${return_variable_name} "${program_name}" PATHS "${extra_paths}" DOC "Path to ${program_name}" NO_CMAKE_ENVIRONMENT_PATH NO_CMAKE_PATH NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_SYSTEM_PATH)
    if("${${return_variable_name}}" MATCHES NOTFOUND)
        # if no extra_paths, or couldn't find it, then look in system $PATH
        find_program(${return_variable_name} "${program_name}" DOC "Path to ${program_name}")
        if("${${return_variable_name}}" MATCHES NOTFOUND AND ${fail_on_not_found})
            error_say("Error: Could not find program: ${program_name}")
        endif()
    endif()

    if(NOT "${${return_variable_name}}" MATCHES NOTFOUND)
        say("-- Found ${program_name}: ${${return_variable_name}}")
    endif()

    set(${return_variable_name} ${${return_variable_name}} PARENT_SCOPE)
endfunction()

# WINDOWS SPECIFIC 
# void replace_md_with_mt(string flags_var)
# - This macro replaces the /MD compiler flags (Windows specific) with /MT compiler flags
# - This does nothing if no /MD flags were replaced.
macro(replace_md_with_mt flags_var)
    set(flags_var_name  ${flags_var}) # i.e., CMAKE_C_FLAGS_RELEASE
    set(flags_var_value ${${flags_var}}) # i.e., "/MD /O2 ..."
    string(REPLACE /MD /MT temp_out "${flags_var_value}")
    string(COMPARE NOTEQUAL "${temp_out}" "${flags_var_value}" something_was_replaced)
    if("${something_was_replaced}")
        unset(${flags_var_name} CACHE)
        set(${flags_var_name} ${temp_out} CACHE STRING "Replaced /MD with /MT compiler flags")
    endif()
endmacro()

