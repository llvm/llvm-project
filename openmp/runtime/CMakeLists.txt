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

################
# CMAKE libiomp5
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(libiomp C CXX)

#########
# GLOBALS 
set(GLOBAL_DEBUG 0)

# Add cmake directory to search for custom cmake functions
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# Set base libomp directory (directory with exports/ , src/ , tools/ , etc.)
# The top-level CMakeLists.txt should define this variable.
set(LIBOMP_WORK ${CMAKE_CURRENT_SOURCE_DIR})

# These include files are in cmake/ subdirectory except for FindPerl which is a cmake standard module
include(HelperFunctions)
include(Definitions) # -D definitions when compiling 
include(CommonFlags) # compiler, assembler, fortran, linker flags common for all compilers
include(SourceFiles) # source files to compile 
include(PerlFlags)   # Perl flags for generate-def.pl and expand-vars.pl
include(FindPerl) # Standard cmake module to check for Perl

####################################################################
#                         CONFIGURATION
#
# * Any variable/value that is CACHE-ed can be changed after the initial run of cmake
# through the file, CMakeCache.txt which is in the build directory.
# * If you change any value in CMakeCache.txt, then just run cmake ..
# and the changed will be picked up.  One can also use -DVARIABLE=VALUE
# when calling cmake to changed configuration values.  
# * CMAKE_C_COMPILER, CMAKE_CXX_COMPILER, CMAKE_ASM_MASM_COMPILER, CMAKE_ASM_COMPILER,
# CMAKE_Fortran_COMPILER can only by specified on the initial run of cmake. 
# This means you cannot specify -DCMAKE_C_COMPILER= on a subsequent run of cmake
# in the same build directory until that build directory is emptied.
# If you want to change the compiler, then empty the build directory and rerun cmake.

# Build Configuration
set(os_possible_values          lin mac win mic)
set(arch_possible_values        32e 32 arm ppc64)
set(build_type_possible_values  release debug relwithdebinfo)
set(omp_version_possible_values 40 30)
set(lib_type_possible_values    normal profile stubs)
set(mic_arch_possible_values    knf knc)
set(mic_os_possible_values      bsd lin)

# Below, cmake will try and determine the operating system and architecture for you (it assumes Intel architecture)
# These values are set in CMakeCache.txt when cmake is first run (-Dvar_name=... will take precedence)
#  parameter  | default value             
# ----------------------------
# Right now, this build system considers os=lin to mean "Unix-like that is not MAC"
if(${APPLE}) # Apple goes first because CMake considers Mac to be a Unix based operating system, while libiomp5 considers it a special case
    set(os             mac         CACHE STRING "The operating system to build for (lin/mac/win/mic)")
elseif(${UNIX})
    set(os             lin         CACHE STRING "The operating system to build for (lin/mac/win/mic)")
elseif(${WIN32})       
    set(os             win         CACHE STRING "The operating system to build for (lin/mac/win/mic)")
else()
    set(os             lin         CACHE STRING "The operating system to build for (lin/mac/win/mic)")
endif()

# set to default architecture if the user did not specify an architecture explicitly
if(NOT arch)	
	if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "4")
    	set(arch           32          CACHE STRING "The architecture to build for (32e/32/arm/ppc64).  32e is Intel(R) 64 architecture, 32 is IA-32 architecture")
	else()
    	set(arch           32e         CACHE STRING "The architecture to build for (32e/32/arm/ppc64).  32e is Intel(R) 64 architecture, 32 is IA-32 architecture")
	endif()
endif()

set(lib_type       normal        CACHE STRING "Performance,Profiling,Stubs library (normal/profile/stubs)")
set(version        5             CACHE STRING "Produce libguide (version 4) or libiomp5 (version 5)")
set(omp_version    40            CACHE STRING "The OpenMP version (40/30)")
set(mic_arch       knc           CACHE STRING "Intel(R) Many Integrated Core Architecture (Intel(R) MIC Architecture) (knf/knc).   Ignored if not Intel(R) MIC Architecture build.")
set(mic_os         lin           CACHE STRING "Intel(R) MIC Architecture operating system (bsd/lin).   Ignored if not Intel(R) MIC Architecture build.")
set(create_fortran_modules false CACHE STRING "Create Fortran module files? (requires fortran compiler)")

# - These tests are little tests performed after the library is formed.
# - The library won't be copied to the exports directory until it has passed/skipped all below tests
# - To skip these tests, just pass -Dtests=OFF to cmake or change tests value in CMakeCache.txt to OFF after running cmake
set(test_touch      true         CACHE BOOL   "Perform a small touch test?"                            )
set(test_relo       true         CACHE BOOL   "Perform a relocation test for dynamic libraries?"       )
set(test_execstack  true         CACHE BOOL   "Perform a execstack test for linux dynamic libraries?"  )
set(test_instr      true         CACHE BOOL   "Perform an instruction test for Intel(R) MIC Architecture libraries?" )
set(test_deps       true         CACHE BOOL   "Perform a library dependency test?"                     )
set(tests           false        CACHE BOOL   "Perform touch, relo, execstack, instr, and deps tests?" )

# - stats-gathering enables OpenMP stats where things like the number of parallel regions, clock ticks spent in
#   particular openmp regions are recorded.
set(stats           false         CACHE BOOL   "Stats-Gathering functionality?"                         )

# User specified flags.  These are appended to the predetermined flags found in CommonFlags.cmake and ${CMAKE_C_COMPILER_ID}/*Flags.cmake (i.e., GNU/CFlags.cmake)
set(USER_C_FLAGS      "" CACHE STRING "Appended user specified C compiler flags."             )
set(USER_CXX_FLAGS    "" CACHE STRING "Appended user specified C++ compiler flags."           )
set(USER_CPP_FLAGS    "" CACHE STRING "Appended user specified C preprocessor flags."         )
set(USER_ASM_FLAGS    "" CACHE STRING "Appended user specified assembler flags."              )
set(USER_LD_FLAGS     "" CACHE STRING "Appended user specified linker flags."                 )
set(USER_LD_LIB_FLAGS "" CACHE STRING "Appended user specified linked libs flags. (i.e., -lm)")
set(USER_F_FLAGS      "" CACHE STRING "Appended user specified Fortran compiler flags.  These are only used if create_fortran_modules==true."       )

# - Allow three build types: Release, Debug, RelWithDebInfo (these relate to build.pl's release, debug, and diag settings respectively)
# - default is Release (when CMAKE_BUILD_TYPE is not defined)
# - CMAKE_BUILD_TYPE affects the -O and -g flags (CMake magically includes correct version of them on per compiler basis)
# - typical: Release = -O3 -DNDEBUG
#            RelWithDebInfo = -O2 -g -DNDEBUG
#            Debug = -g
if(CMAKE_BUILD_TYPE)
    # CMAKE_BUILD_TYPE was defined, check for validity 
    string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_lowercase)
    check_variable(cmake_build_type_lowercase  "${build_type_possible_values}")
else()
    # CMAKE_BUILD_TYPE was not defined, set default to Release
    unset(CMAKE_BUILD_TYPE CACHE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: Release/Debug/RelWithDebInfo")
    string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_lowercase)
    check_variable(cmake_build_type_lowercase  "${build_type_possible_values}")
endif()

# Check valid values
check_variable(os          "${os_possible_values}"         )
check_variable(arch        "${arch_possible_values}"       )
check_variable(omp_version "${omp_version_possible_values}")
check_variable(lib_type    "${lib_type_possible_values}"   )
if("${os}" STREQUAL "mic")
    check_variable(mic_arch "${mic_arch_possible_values}"  )
    check_variable(mic_os   "${mic_os_possible_values}"    )
endif()
# Get the build number from kmp_version.c
get_build_number("${LIBOMP_WORK}" build_number)

# Getting time and date 
# As of now, no timestamp will be created.
set(date "No Timestamp")

#################################################################
# Set some useful flags variables for other parts of cmake to use
# Operating System
set(LINUX FALSE)
set(MAC FALSE)
set(WINDOWS FALSE)
set(MIC FALSE)
set(FREEBSD FALSE)
if("${os}" STREQUAL "lin")
    set(LINUX TRUE)
    set(real_os lin)
elseif("${os}" STREQUAL "mac")
    set(MAC TRUE)
    set(real_os mac)
elseif("${os}" STREQUAL "win")
    set(WINDOWS TRUE)
    set(real_os win)
elseif("${os}" STREQUAL "mic")
    set(MIC TRUE)
    set(real_os lrb)
endif()
if("${CMAKE_SYSTEM_NAME}" STREQUAL "FreeBSD")
    set(FREEBSD TRUE)
endif()

# Architecture
set(IA32 FALSE)
set(INTEL64 FALSE)
set(ARM FALSE)
set(PPC64 FALSE)
if("${arch}" STREQUAL "32")      # IA-32 architecture
    set(IA32 TRUE)
elseif("${arch}" STREQUAL "32e") # Intel(R) 64 architecture
    set(INTEL64 TRUE)
elseif("${arch}" STREQUAL "arm") # ARM architecture
    set(ARM TRUE)
elseif("${arch}" STREQUAL "ppc64") # PPC64 architecture
    set(PPC64 TRUE)
endif()

# Set some flags based on build_type
# cmake_build_type_lowercase is based off of CMAKE_BUILD_TYPE, just put in lowercase.
set(RELEASE_BUILD        FALSE)
set(DEBUG_BUILD          FALSE)
set(RELWITHDEBINFO_BUILD FALSE)
if("${cmake_build_type_lowercase}" STREQUAL "release")
    set(RELEASE_BUILD TRUE)
elseif("${cmake_build_type_lowercase}" STREQUAL "debug")
    set(DEBUG_BUILD TRUE)
elseif("${cmake_build_type_lowercase}" STREQUAL "relwithdebinfo")
    set(RELWITHDEBINFO_BUILD TRUE)
endif()

# Stats-gathering on or off?
set(STATS_GATHERING FALSE)
if("${stats}") # string "on" or "ON" is seen as boolean TRUE
    set(STATS_GATHERING TRUE)
endif()

# Include itt notify interface? Right now, always.
set(USE_ITT_NOTIFY TRUE)

# normal, profile, stubs library.
set(NORMAL_LIBRARY FALSE)
set(STUBS_LIBRARY FALSE)
set(PROFILE_LIBRARY FALSE)
if("${lib_type}" STREQUAL "normal")
    set(NORMAL_LIBRARY TRUE)
elseif("${lib_type}" STREQUAL "profile")
    set(PROFILE_LIBRARY TRUE)
elseif("${lib_type}" STREQUAL "stubs")
    set(STUBS_LIBRARY TRUE)
endif()

###############################################
# Features for compilation and build in general

# - Does the compiler support a 128-bit floating point data type? Default is false
# - If a compiler does, then change it in the CMakeCache.txt file (or using the cmake GUI)
#   or send to cmake -DCOMPILER_SUPPORTS_QUAD_PRECISION=true
# - If COMPILER_SUPPORTS_QUAD_PRECISION is true, then a corresponding COMPILER_QUAD_TYPE must be given
#   This is the compiler's quad-precision data type.
# ** TODO: This isn't complete yet. Finish it. Requires changing macros in kmp_os.h **
set(COMPILER_SUPPORTS_QUAD_PRECISION false CACHE BOOL   "*INCOMPLETE* Does the compiler support a 128-bit floating point type?")
set(COMPILER_QUAD_TYPE               ""    CACHE STRING "*INCOMPLETE* The quad precision data type (i.e., for gcc, __float128)")

# - Should the orignal build rules for builds be used? (cmake/OriginalBuildRules.cmake).  This setting is off by default.
# - This always compiles with -g.  And if it is a release build, the debug info is stripped out via objcopy and put into libiomp5.dbg.
set(USE_BUILDPL_RULES false CACHE BOOL "Should the build follow build.pl rules/recipes?")

# - Should the build use the predefined linker flags (OS-dependent) in CommonFlags.cmake?
# - these predefined linker flags should work for Windows, Mac, and True Linux for the most popular compilers/linkers
set(USE_PREDEFINED_LINKER_FLAGS true CACHE BOOL "Should the build use the predefined linker flags in CommonFlags.cmake?")

# - TSX based locks have __asm code which can be troublesome for some compilers.  This feature is also x86 specific.
if({${IA32} OR ${INTEL64})
    set(USE_ADAPTIVE_LOCKS true CACHE BOOL "Should TSX-based lock be compiled (adaptive lock in kmp_lock.cpp).  These are x86 specific.")
else()
    set(USE_ADAPTIVE_LOCKS false CACHE BOOL "Should TSX-based lock be compiled (adaptive lock in kmp_lock.cpp).  These are x86 specific.")
endif()

##################################
# Error checking the configuration 
if(${STATS_GATHERING} AND (${WINDOWS} OR ${MAC}))
    error_say("Stats-gathering functionality is only supported on x86-Linux and Intel(R) MIC Architecture")
endif()
if(${STATS_GATHERING} AND NOT (${IA32} OR ${INTEL64}))
    error_say("Stats-gathering functionality is only supported on x86-Linux and Intel(R) MIC Architecture")
endif()
if(${USE_ADAPTIVE_LOCKS} AND NOT(${IA32} OR ${INTEL64}))
    error_say("Adaptive locks (TSX) functionality is only supported on x86 Architecture")
endif()

###############################################
# - Create the suffix for the export directory
# - Only add to suffix when not a default value
# - Example suffix: .deb.30.s1 
#   final export directory: exports/lin_32e.deb.30.s1/lib
# - These suffixes imply the build is a Debug, OpenMP 3.0, Stats-Gathering version of the library
if(NOT "${cmake_build_type_lowercase}" STREQUAL "release")
    string(SUBSTRING "${cmake_build_type_lowercase}" 0 3 build_type_suffix)
    set(suffix "${suffix}.${build_type_suffix}")
endif()
if(NOT "${omp_version}" STREQUAL "40")
    set(suffix "${suffix}.${omp_version}")
endif()
if(${STATS_GATHERING})
    set(suffix "${suffix}.s1")
endif()
if(${MIC})
    if(NOT "${mic_arch}" STREQUAL "knf")
        set(suffix "${suffix}.${mic_arch}")
    endif()
    if(NOT "${mic_os}" STREQUAL "bsd")
        set(suffix "${suffix}.${mic_os}")
    endif()
endif()

####################################
# Setting file extensions / suffixes
set(obj ${CMAKE_C_OUTPUT_EXTENSION}   )
set(lib ${CMAKE_STATIC_LIBRARY_SUFFIX})
set(dll ${CMAKE_SHARED_LIBRARY_SUFFIX})
set(exe ${CMAKE_EXECUTABLE_SUFFIX}    )

######################
# Find perl executable
# Perl is used to create omp.h (and other headers) along with kmp_i18n_id.inc and kmp_i18n_default.inc (see below in Rules section)
if(NOT "${PERL_FOUND}") # variable is defined in FindPerl Standard CMake Module
    error_say("Error: Could not find valid perl")
endif()

#########################
# Setting directory names
set(platform       "${real_os}_${arch}"                ) # i.e., lin_32e, mac_32
set(build_dir      "${CMAKE_CURRENT_BINARY_DIR}"       ) # build directory (Where CMakeCache.txt is created, build files generated)
set(src_dir        "${LIBOMP_WORK}/src"                )
set(tools_dir      "${LIBOMP_WORK}/tools"              ) 
set(export_dir     "${LIBOMP_WORK}/exports"            ) 
set(export_cmn_dir "${export_dir}/common${suffix}"     ) 
set(export_ptf_dir "${export_dir}/${platform}${suffix}")
_export_lib_dir(${platform} export_lib_dir)  # set exports directory (relative to build_dir) i.e., ../exports/lin_32e/lib/
                                             # or i.e., ../exports/mac_32e/lib.thin/ for mac
if(${MAC})
    # macs use lib.thin/ subdirectory for non-fat libraries that only contain one architecture
    # macs use lib/ subdirectory for fat libraries that contain both IA-32 architecture and Intel(R) 64 architecture code.
    _export_lib_fat_dir(${platform} export_lib_fat_dir)
endif()
set(inc_dir        "${LIBOMP_WORK}/src/include/${omp_version}")

############################
# Setting final library name
set(lib_item "libiomp")
if(${PROFILE_LIBRARY})
    set(lib_item "${lib_item}prof")
endif()
if(${STUBS_LIBRARY})
    set(lib_item "${lib_item}stubs")
endif()
set(lib_item "${lib_item}${version}")
if(${WINDOWS})
    set(lib_item "${lib_item}md")
endif()
set(lib_ext "${dll}")
# ${lib_file} is real library name:
# libiomp5.so    for Linux
# libiomp5.dylib for Mac
# libiomp5md.dll for Windows
set(lib_file "${lib_item}${lib_ext}")

########################################
# Setting export file names (full paths)
if(${WINDOWS})
    set(imp_file "${lib_item}${lib}") # this is exported (libiomp5md.lib)
    set(def_file "${lib_item}.def") # this is not exported
    set(rc_file  "${lib_item}.rc")  # this is not exported
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR "${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo" OR ${USE_BUILDPL_RULES})
        set(pdb_file "${lib_file}.pdb") # this is exported if it exists (libiomp5md.dll.pdb)
    endif()
endif()
set(export_lib_files  "${lib_file}" "${imp_file}" "${pdb_file}") 
set(export_inc_files  "iomp_lib.h")
set(export_mod_files  "omp_lib.mod" "omp_lib_kinds.mod")
set(export_cmn_files1 "omp.h" "omp_lib.h" "omp_lib.f" "omp_lib.f90")
set(export_cmn_files2 "iomp.h")
add_prefix("${export_lib_dir}/"                export_lib_files)
add_prefix("${export_ptf_dir}/include_compat/" export_inc_files)
add_prefix("${export_ptf_dir}/include/"        export_mod_files)
add_prefix("${export_cmn_dir}/include/"        export_cmn_files1)
add_prefix("${export_cmn_dir}/include_compat/" export_cmn_files2)
set(export_cmn_files "${export_cmn_files1}" "${export_cmn_files2}")
if("${export_lib_fat_dir}")
    set(export_lib_fat_files "${lib_file}" "${imp_file}")
    add_prefix("${export_lib_fat_dir}/" export_lib_fat_files)
endif()

#########################
# Getting legal type/arch
set_legal_type(legal_type)
set_legal_arch(legal_arch)

#################################################
# Preprocessor Definitions (cmake/Definitions.cmake)
# Preprocessor Includes 
# Compiler (C/C++) Flags (cmake/CommonFlags.cmake)
# Assembler Flags (cmake/CommonFlags.cmake)
# Fortran   Flags (cmake/CommonFlags.cmake)
# Linker    Flags (cmake/CommonFlags.cmake)
# Archiver  Flags (cmake/CommonFlags.cmake)
# Helper Perl Script Flags (cmake/PerlFlags.cmake)
# * Inside the cmake/CommonFlags.cmake file, the USER_*_FLAGS are added.
# * Cannot use CMAKE_*_FLAGS directly because -x c++ is put in the linker command and mangles the linking phase.

# preprocessor flags (-D definitions and -I includes)
# Grab environment variable CPPFLAGS and append those to definitions
set(include_dirs ${CMAKE_CURRENT_BINARY_DIR} ${src_dir} ${src_dir}/i18n ${inc_dir} ${src_dir}/thirdparty/ittnotify)
include_directories(${include_dirs})

# Grab assembler-dependent flags
# CMake will look for cmake/${CMAKE_ASM_COMPILER_ID}/AsmFlags.cmake to append additional assembler flags.
if(${WINDOWS})
    # Windows based systems use CMAKE_ASM_MASM_COMPILER
    # The windows assembly files are in MASM format, and they require a tool that can handle MASM syntax (ml.exe or ml64.exe typically)
    enable_language(ASM_MASM) 
    set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${CMAKE_ASM_MASM_COMPILER_ID} ${CMAKE_MODULE_PATH})
    find_file(assembler_specific_include_file_found AsmFlags.cmake ${CMAKE_MODULE_PATH})
    if(assembler_specific_include_file_found)
        include(AsmFlags)
        append_assembler_specific_asm_flags(ASM_FLAGS)
    else()
        warning_say("Could not find cmake/${CMAKE_ASM_MASM_COMPILER_ID}/AsmFlags.cmake: will only use default flags")
    endif()
else()
    # Unix (including Mac) based systems use CMAKE_ASM_COMPILER
    # Unix assembly files can be handled by compiler usually.
    enable_language(ASM)
    set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${CMAKE_ASM_COMPILER_ID} ${CMAKE_MODULE_PATH})
    find_file(assembler_specific_include_file_found AsmFlags.cmake ${CMAKE_MODULE_PATH})
    if(assembler_specific_include_file_found)
        include(AsmFlags)
        append_assembler_specific_asm_flags(ASM_FLAGS)
    else()
        warning_say("Could not find cmake/${CMAKE_ASM_COMPILER_ID}/AsmFlags.cmake: will only use default flags")
    endif()
endif()
# Grab compiler-dependent flags
# Cmake will look for cmake/${CMAKE_C_COMPILER_ID}/CFlags.cmake to append additional c, cxx, and linker flags.
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${CMAKE_C_COMPILER_ID} ${CMAKE_MODULE_PATH})
find_file(compiler_specific_include_file_found CFlags.cmake ${CMAKE_MODULE_PATH})
if(compiler_specific_include_file_found)
    include(CFlags) # COMPILER_SUPPORTS_QUAD_PRECISION changed in here
    append_compiler_specific_c_and_cxx_flags(C_FLAGS CXX_FLAGS)
    append_compiler_specific_linker_flags(LD_FLAGS LD_LIB_FLAGS)
else()
    warning_say("Could not find cmake/${CMAKE_C_COMPILER_ID}/CFlags.cmake: will only use default flags")
endif()

# Grab all the compiler-independent flags
append_c_and_cxx_flags_common(C_FLAGS CXX_FLAGS) 
append_asm_flags_common(ASM_FLAGS)
append_fort_flags_common(F_FLAGS)
append_linker_flags_common(LD_FLAGS LD_LIB_FLAGS)
append_archiver_flags_common(AR_FLAGS)
append_cpp_flags(DEFINITIONS_FLAGS)

# Setup the flags correctly for cmake (covert to string)
# Pretty them up (STRIP any beginning and trailing whitespace)
list_to_string("${DEFINITIONS_FLAGS}" DEFINITIONS_FLAGS)
list_to_string("${C_FLAGS}"           C_FLAGS          )
list_to_string("${CXX_FLAGS}"         CXX_FLAGS        )
list_to_string("${ASM_FLAGS}"         ASM_FLAGS        )
list_to_string("${LD_FLAGS}"          LD_FLAGS         )
list_to_string("${LD_LIB_FLAGS}"      LD_LIB_FLAGS     )
list_to_string("${AR_FLAGS}"          AR_FLAGS         ) # Windows specific for creating import library
string(STRIP   "${DEFINITIONS_FLAGS}" DEFINITIONS_FLAGS)
string(STRIP   "${C_FLAGS}"           C_FLAGS          )
string(STRIP   "${CXX_FLAGS}"         CXX_FLAGS        )
string(STRIP   "${ASM_FLAGS}"         ASM_FLAGS        )
string(STRIP   "${LD_FLAGS}"          LD_FLAGS         )
string(STRIP   "${LD_LIB_FLAGS}"      LD_LIB_FLAGS     )
string(STRIP   "${AR_FLAGS}"          AR_FLAGS         ) # Windows specific for creating import library

# Grab the Perl flags
set_ev_flags(ev_flags) # expand-vars.pl flags
set_gd_flags(gd_flags) # generate-def.pl flags (Windows only)
set(oa_opts "--os=${real_os}" "--arch=${arch}") # sent to the perl scripts

#########################################################
# Getting correct source files (cmake/SourceFiles.cmake)
set_c_files(lib_c_items)
set_cpp_files(lib_cxx_items)
set_asm_files(lib_asm_items)
set_imp_c_files(imp_c_items) # Windows-specific

###################################
# Setting all source file variables
set(lib_src_files "${lib_c_items}" "${lib_cxx_items}" "${lib_asm_items}")
set(imp_src_files "${imp_c_items}")
add_prefix("${src_dir}/" lib_src_files)
add_prefix("${src_dir}/" imp_src_files) # Windows-specific
add_prefix("${src_dir}/" lib_c_items  )
add_prefix("${src_dir}/" lib_cxx_items)
add_prefix("${src_dir}/" lib_asm_items)
add_prefix("${src_dir}/" imp_c_items  ) # Windows-specific

#####################################################################
# Debug print outs.  Will print "variable = ${variable}" if GLOBAL_DEBUG == 1
if(GLOBAL_DEBUG)
    include(CMakePrintSystemInformation)
endif()
debug_say_var(CMAKE_ASM_COMPILE_OBJECT)
debug_say_var(CMAKE_RC_COMPILER)
debug_say_var(CMAKE_C_COMPILER_ID)
debug_say_var(LIBOMP_WORK)
debug_say_var(date)
debug_say_var(stats)
debug_say_var(lib_file)
debug_say_var(export_lib_files)
debug_say_var(DEFINITIONS_FLAGS)
debug_say_var(C_FLAGS)
debug_say_var(CXX_FLAGS)
debug_say_var(ASM_FLAGS)
debug_say_var(F_FLAGS)
debug_say_var(LD_FLAGS)
debug_say_var(LD_LIB_FLAGS)
debug_say_var(AR_FLAGS)
debug_say_var(ev_flags)
debug_say_var(gd_flags)
debug_say_var(oa_opts)
debug_say_var(lib_c_items)
debug_say_var(lib_cxx_items)
debug_say_var(lib_asm_items)
debug_say_var(imp_c_items)
debug_say_var(lib_src_files)
debug_say_var(imp_src_files)

####################################################################
#                     ---------------------                        #
#                     --- Rules/Recipes ---                        #
#                     ---------------------                        #
####################################################################
# Below, ${ldeps} always stands for "local dependencies" for the 
# next immediate target to be created via add_custom_target() or 
# add_custom_command()

####################
# --- Create all ---
add_custom_target(lib ALL DEPENDS ${export_lib_files})
add_custom_target(inc ALL DEPENDS ${export_inc_files})
if(${create_fortran_modules})
add_custom_target(mod ALL DEPENDS ${export_mod_files})
endif()
# --- Enforce the tests to be completed/skipped before copying to exports directory ---
if(${tests})
    if(${WINDOWS})
        set(test-dependencies test-touch-mt/.success test-touch-md/.success test-relo/.success test-execstack/.success test-instr/.success test-deps/.success)
    else()
        set(test-dependencies test-touch-rt/.success test-relo/.success test-execstack/.success test-instr/.success test-deps/.success)
    endif()
    set_source_files_properties(${export_lib_files} PROPERTIES OBJECT_DEPENDS "${test-dependencies}")
endif()

#############################
# --- Create Common Files ---
add_custom_target(common DEPENDS ${export_cmn_files})
add_custom_target(clean-common COMMAND ${CMAKE_COMMAND} -E remove -f ${export_cmn_files})

##########################################
# --- Copy files to export directories ---
# - just a simple copy recipe which acts as an install step
# - copies out of the src_dir into the dest_dir
#
# dest_dir/target : src_dir/target
#    cp src_dir/target dest_dir/target
macro (simple_copy_recipe target src_dir dest_dir)
    get_source_file_property(extra_depends ${dest_dir}/${target} OBJECT_DEPENDS)
    if("${extra_depends}" MATCHES "NOTFOUND")
        set(extra_depends)
    endif()
    set(ldeps ${src_dir}/${target} "${extra_depends}")
    if(NOT "${target}" STREQUAL "")
        file(MAKE_DIRECTORY ${dest_dir}) # make sure destination directory exists
        add_custom_command(
            OUTPUT   ${dest_dir}/${target}
            COMMAND  ${CMAKE_COMMAND} -E copy ${src_dir}/${target} ${dest_dir}/${target}
            DEPENDS  ${ldeps}
        )
    endif()
endmacro()
# copy from build directory to final resting places in exports directory
simple_copy_recipe("omp.h"             "${build_dir}"  "${export_cmn_dir}/include")
simple_copy_recipe("omp_lib.h"         "${build_dir}"  "${export_cmn_dir}/include")
simple_copy_recipe("omp_lib.f"         "${build_dir}"  "${export_cmn_dir}/include")
simple_copy_recipe("omp_lib.f90"       "${build_dir}"  "${export_cmn_dir}/include")
simple_copy_recipe("iomp.h"            "${build_dir}"  "${export_cmn_dir}/include_compat")
simple_copy_recipe("${lib_file}"       "${build_dir}"  "${export_lib_dir}")
simple_copy_recipe("${imp_file}"       "${build_dir}"  "${export_lib_dir}")
simple_copy_recipe("${pdb_file}"       "${build_dir}"  "${export_lib_dir}")
simple_copy_recipe("${dbg_file}"       "${build_dir}"  "${export_lib_dir}")
simple_copy_recipe("omp_lib.mod"       "${build_dir}"  "${export_ptf_dir}/include")
simple_copy_recipe("omp_lib_kinds.mod" "${build_dir}"  "${export_ptf_dir}/include")
simple_copy_recipe("iomp_lib.h"        "${build_dir}"  "${export_ptf_dir}/include_compat")

######################################################
# --- Build the main library ---
# $(lib_file) <== Main library file to create

# objects depend on : .inc files and omp.h
# This way the *.inc and omp.h are generated before any compilations take place
add_custom_target(needed-headers DEPENDS ${build_dir}/kmp_i18n_id.inc ${build_dir}/kmp_i18n_default.inc ${build_dir}/omp.h)

# For Windows, there is a definitions file (.def) and resource file (.res) created using generate-def.pl and rc.exe respectively.
if(${WINDOWS})
    add_custom_target(needed-windows-files DEPENDS ${build_dir}/${def_file} ${build_dir}/${rc_file})
    list(APPEND lib_src_files ${build_dir}/${rc_file})
    # The windows assembly files are in MASM format, and they require a tool that can handle MASM syntax (ml.exe or ml64.exe typically)
    enable_language(ASM_MASM) 
else()
    # Unix assembly files can be handled by compiler.
    enable_language(ASM)
endif()

# Remove any cmake-automatic linking of libraries by linker, This is so linux 
# and mac don't include libstdc++ just because we compile c++ files.
if(${USE_PREDEFINED_LINKER_FLAGS})
    set(CMAKE_C_IMPLICIT_LINK_LIBRARIES   "")
    set(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES "")
    set(CMAKE_ASM_IMPLICIT_LINK_LIBRARIES "")
endif()

# --- ${lib_file} rule ---
add_library(iomp5 SHARED ${lib_src_files})
set_target_properties(iomp5 PROPERTIES 
    PREFIX "" SUFFIX ""          # Take control 
    OUTPUT_NAME "${lib_file}"    # of output name
    LINK_FLAGS  "${LD_FLAGS}"     
    LINKER_LANGUAGE C            # use C Compiler for linking step
    SKIP_BUILD_RPATH true        # have Mac linker -install_name just be "-install_name libiomp5.dylib"
)
# Target lib (export files) depend on the library (iomp5) being built
add_dependencies(lib iomp5)

# Linking command will include libraries in LD_LIB_FLAGS
target_link_libraries(iomp5 ${LD_LIB_FLAGS})

# Create *.inc and omp.h before compiling any sources
add_dependencies(iomp5 needed-headers)
if(${WINDOWS})
# Create .def and .rc file before compiling any sources
    add_dependencies(iomp5 needed-windows-files)
endif()

# Set the compiler flags for each type of source
set_source_files_properties(${lib_c_items} 
                            ${imp_c_items}   PROPERTIES COMPILE_FLAGS "${C_FLAGS}"  )
set_source_files_properties(${lib_cxx_items} PROPERTIES COMPILE_FLAGS "${CXX_FLAGS}")
set_source_files_properties(${lib_asm_items} PROPERTIES COMPILE_FLAGS "${ASM_FLAGS}")
# Set the -D definitions for all sources
add_definitions(${DEFINITIONS_FLAGS})

# If creating a build that imitates build.pl's rules then set USE_BUILDPL_RULES to true
if(${USE_BUILDPL_RULES})
    include(BuildPLRules)
endif()

######################################################
# --- Source file specific flags ---
# kmp_version.o : -D _KMP_BUILD_TIME="\"$(date)}\""
set_source_files_properties(${src_dir}/kmp_version.c  PROPERTIES COMPILE_DEFINITIONS "_KMP_BUILD_TIME=\"\\\"${date}\\\"\"")

# z_Linux_asm.o : -D KMP_ARCH_*
if(${ARM})
    set_source_files_properties(${src_dir}/z_Linux_asm.s PROPERTIES COMPILE_DEFINITIONS "KMP_ARCH_ARM")
elseif(${INTEL64})
    set_source_files_properties(${src_dir}/z_Linux_asm.s PROPERTIES COMPILE_DEFINITIONS "KMP_ARCH_X86_64")
elseif(${IA32})
    set_source_files_properties(${src_dir}/z_Linux_asm.s PROPERTIES COMPILE_DEFINITIONS "KMP_ARCH_X86")
elseif(${PPC64})
	set_source_files_properties(${src_dir}/z_Linux_asm.s PROPERTIES COMPILE_DEFINITIONS "KMP_ARCH_PPC64")
endif()

if(${WINDOWS})
    set_source_files_properties(${src_dir}/thirdparty/ittnotify/ittnotify_static.c PROPERTIES COMPILE_DEFINITIONS "UNICODE")
endif()

######################################################
# MAC specific build rules
if(${MAC})
    # fat library rules
    if(${INTEL64})
        _export_lib_fat_dir( "mac_32e" export_fat_mac_32e)
        _export_lib_dir(     "mac_32"  export_mac_32     )
        _export_lib_dir(     "mac_32e" export_mac_32e    )
        file(MAKE_DIRECTORY ${export_fat_mac_32e})
        add_custom_target(fat
            COMMAND ${CMAKE_COMMAND} -E echo Building 32 and 32e fat libraries from ${export_mac_32}/${lib_file} and ${export_mac_32e}/${lib_file}
            COMMAND ${CMAKE_COMMAND} -E echo Will put fat library in ${export_fat_mac_32e} directory
            COMMAND lipo -create -output ${export_fat_mac_32e}/${lib_file} ${export_mac_32}/${lib_file} ${export_mac_32e}/${lib_file}
        )
    endif()
endif()

######################################################
# Windows specific build rules
if(${WINDOWS})

    # --- Create $(imp_file) ---
    # This file is first created in the unstripped/${lib_file} creation step.
    # It is then "re-linked" to include kmp_import.c which prevents linking of both Visual Studio OpenMP and newly built OpenMP
    if(NOT "${imp_file}" STREQUAL "")
        set(generated_import_file ${lib_file}${lib})
        add_library(iomp5imp STATIC ${generated_import_file} ${imp_src_files})
        set_source_files_properties(${generated_import_file} PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
        set_target_properties(iomp5imp PROPERTIES
            PREFIX "" SUFFIX ""
            OUTPUT_NAME "${imp_file}"
            STATIC_LIBRARY_FLAGS "${AR_FLAGS}"
            LINKER_LANGUAGE C
            SKIP_BUILD_RPATH true
        )
        add_custom_command(TARGET iomp5imp PRE_BUILD COMMAND ${CMAKE_COMMAND} -E remove -f ${imp_file})
        add_dependencies(iomp5imp iomp5)
    endif()

    # --- Create $(def_file) ---
    if(NOT "${def_file}" STREQUAL "")
        string_to_list("${gd_flags}" gd_flags)
        add_custom_command(
            OUTPUT  ${def_file}
            COMMAND ${PERL_EXECUTABLE} ${tools_dir}/generate-def.pl ${gd_flags} -o ${def_file} ${src_dir}/dllexports
            DEPENDS ${src_dir}/dllexports ${tools_dir}/generate-def.pl
        )
    endif()

    # --- Create $(rc_file) ---
    if(NOT "${rc_file}" STREQUAL "")
        add_custom_command(
            OUTPUT  ${rc_file}
            COMMAND ${CMAKE_COMMAND} -E copy libiomp.rc ${rc_file}
            DEPENDS libiomp.rc
        )
    endif()
endif()

######################################################
# kmp_i18n_id.inc and kmp_i18n_default.inc
set(perlcmd "${PERL_EXECUTABLE}" "${tools_dir}/message-converter.pl" "${oa_opts}" "--prefix=kmp_i18n" "--enum=kmp_i18n_id.inc" "${src_dir}/i18n/en_US.txt")
add_custom_command(
    OUTPUT  ${build_dir}/kmp_i18n_id.inc
    COMMAND ${perlcmd}
    DEPENDS ${src_dir}/i18n/en_US.txt ${tools_dir}/message-converter.pl
)
set(perlcmd "${PERL_EXECUTABLE}" "${tools_dir}/message-converter.pl" "${oa_opts}" "--prefix=kmp_i18n" "--default=kmp_i18n_default.inc" "${src_dir}/i18n/en_US.txt")
add_custom_command(
    OUTPUT  ${build_dir}/kmp_i18n_default.inc
    COMMAND ${perlcmd}
    DEPENDS ${src_dir}/i18n/en_US.txt ${tools_dir}/message-converter.pl
)

######################################################
# Micro test rules for after library has been built (cmake/MicroTests.cmake)
# - Only perform if ${tests} == true (specify when invoking: cmake -Dtests=on ...)
if(${tests})
    include(MicroTests)
endif()

######################################################
# --- Create Fortran Files ---
# omp_lib.mod
if(${create_fortran_modules})
    # Grab fortran-compiler-dependent flags
    # Cmake will look for cmake/${CMAKE_Fortran_COMPILER_ID}/FortranFlags.cmake to append additional fortran flags.
    enable_language(Fortran)
    set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${CMAKE_Fortran_COMPILER_ID} ${CMAKE_MODULE_PATH})
    find_file(fortran_specific_include_file_found FortranFlags.cmake ${CMAKE_MODULE_PATH})
    if(fortran_specific_include_file_found)
        include(FortranFlags)
        append_fortran_compiler_specific_fort_flags(F_FLAGS)
    else()
        warning_say("Could not find cmake/${CMAKE_Fortran_COMPILER_ID}/FortranFlags.cmake: will only use default flags in CommonFlags.cmake")
    endif()
    set(omp_lib_f "omp_lib.f90" )
    add_custom_command(
        OUTPUT "omp_lib.mod"
        COMMAND ${CMAKE_Fortran_COMPILER} -c ${F_FLAGS} ${omp_lib_f}
        DEPENDS ${omp_lib_f}
    )
    add_custom_command(
        OUTPUT "omp_lib_kinds.mod"
        COMMAND ${CMAKE_Fortran_COMPILER} -c ${F_FLAGS} ${omp_lib_f}
        DEPENDS ${omp_lib_f}
    )
    # clean omp_lib.o from build directory when "make clean" 
    set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES omp_lib${obj})
endif()

###############################################################
# --- Using expand-vars.pl to generate files ---
# - 'file' is generated using expand-vars.pl and 'file'.var
# - Any .h .f .f90 .rc files should be created with this recipe
macro(expand_vars_recipe filename)
    get_source_file_property(extra_ev_flags ${filename} COMPILE_DEFINITIONS)
    if("${extra_ev_flags}" MATCHES "NOTFOUND")
        set(extra_ev_flags)
    else()
        string_to_list("${extra_ev_flags}" extra_ev_flags)
    endif()
    find_file(${filename}_path ${filename}.var PATHS ${src_dir}/include/${omp_version} ${src_dir})
    set(ldeps "${${filename}_path}" "${src_dir}/kmp_version.c" "${tools_dir}/expand-vars.pl")
    set(expandvarscmd ${PERL_EXECUTABLE} ${tools_dir}/expand-vars.pl --strict ${ev_flags} ${extra_ev_flags} ${${filename}_path} ${filename})
    if(NOT "${filename}" STREQUAL "")
        add_custom_command(
            OUTPUT  ${filename}
            COMMAND ${expandvarscmd}
            DEPENDS ${ldeps}
        )
    endif()
endmacro()
string_to_list("${ev_flags}" ev_flags)
# omp_lib.h  : ev-flags += -D KMP_INT_PTR_KIND="int_ptr_kind()"
set_source_files_properties(omp_lib.h PROPERTIES COMPILE_DEFINITIONS "-D KMP_INT_PTR_KIND=\"int_ptr_kind()\"") 
# iomp_lib.h : ev-flags += -D KMP_INT_PTR_KIND=$(if $(filter 32,$(arch)),4,8)
if(${IA32}) # 32 bit archs
    set_source_files_properties(iomp_lib.h PROPERTIES COMPILE_DEFINITIONS "-D KMP_INT_PTR_KIND=4") 
else()
    set_source_files_properties(iomp_lib.h PROPERTIES COMPILE_DEFINITIONS "-D KMP_INT_PTR_KIND=8") 
endif()
# libiomp.rc : ev-flags += -D KMP_FILE=$(lib_file)
set_source_files_properties(libiomp.rc PROPERTIES COMPILE_DEFINITIONS "-D KMP_FILE=${lib_file}") 
expand_vars_recipe(omp.h)
expand_vars_recipe(omp_lib.h)
expand_vars_recipe(omp_lib.f)
expand_vars_recipe(omp_lib.f90)
expand_vars_recipe(iomp.h)
expand_vars_recipe(iomp_lib.h)
expand_vars_recipe(libiomp.rc)


####################################################################
# Print configuration after all variables are set.
say("")
say("----------------------- CONFIGURATION -----------------------")
say("Operating System   : ${os}")
say("Architecture       : ${arch}")
say("Build Type         : ${CMAKE_BUILD_TYPE}")
say("OpenMP Version     : ${omp_version}")
say("Lib Type           : ${lib_type}")
if(${MIC})
    say("Intel(R) MIC Architecture    : ${mic_arch}")
    say("Intel(R) MIC Architecture OS : ${mic_os}")
endif()
say("Fortran Modules    : ${create_fortran_modules}")
# will say development if all zeros
if("${build_number}" STREQUAL "00000000")
    set(build "development")
else()
    set(build "${build_number}")
endif()
say("Build              : ${build}")
say("Stats-Gathering    : ${stats}")
say("Use build.pl rules : ${USE_BUILDPL_RULES}")
say("Adaptive locks     : ${USE_ADAPTIVE_LOCKS}")
say("Use predefined linker flags      : ${USE_PREDEFINED_LINKER_FLAGS}")
say("Compiler supports quad precision : ${COMPILER_SUPPORTS_QUAD_PRECISION}")
say("-------------------------------------------------------------")
say("")

