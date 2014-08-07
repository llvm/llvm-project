######################################################
# MICRO TESTS
# The following micro-tests are small tests to perform on 
# the library just created in ${build_dir}/, there are currently
# five micro-tests: 
# (1) test-touch 
#    - Compile and run a small program using newly created libiomp5 library
#    - Fails if test-touch.c does not compile or if test-touch.c does not run after compilation
#    - Program dependencies: gcc or g++, grep, bourne shell
#    - Available for all Linux,Mac,Windows builds.  Not availble on Intel(R) MIC Architecture builds.
# (2) test-relo
#    - Tests dynamic libraries for position-dependent code (can not have any position dependent code)
#    - Fails if TEXTREL is in output of readelf -d libiomp5.so command
#    - Program dependencies: readelf, grep, bourne shell
#    - Available for Linux, Intel(R) MIC Architecture dynamic library builds. Not available otherwise.
# (3) test-execstack 
#    - Tests if stack is executable
#    - Fails if stack is executable. Should only be readable and writable. Not exectuable.
#    - Program dependencies: perl, readelf
#    - Available for Linux dynamic library builds. Not available otherwise.
# (4) test-instr (Intel(R) MIC Architecutre only) 
#    - Tests Intel(R) MIC Architecture libraries for valid instruction set
#    - Fails if finds invalid instruction for Intel(R) MIC Architecture (wasn't compiled with correct flags)
#    - Program dependencies: perl, objdump 
#    - Available for Intel(R) MIC Architecture builds. Not available otherwise.
# (5) test-deps      
#    - Tests newly created libiomp5 for library dependencies
#    - Fails if sees a dependence not listed in td_exp variable below
#    - Program dependencies: perl, (linux)readelf, (mac)otool[64], (windows)link.exe
#    - Available for Linux,Mac,Windows, Intel(R) MIC Architecture dynamic builds and Windows static builds. Not available otherwise.
#
# All tests can be turned off by including -Dtests=off when calling cmake
# An individual test can be turned off by issuing something like -Dtest_touch=off when calling cmake

# test-touch
if(NOT ${MIC} AND ${test_touch} AND ${tests})
    if(${WINDOWS})
        set(do_test_touch_mt TRUE)
        if(${do_test_touch_mt})
            set(test_touch_items ${test_touch_items} test-touch-md test-touch-mt)
        else()
            set(test_touch_items ${test_touch_items} test-touch-md)
        endif()
    else()
        set(test_touch_items ${test_touch_items} test-touch-rt)
    endif()
    set(regular_test_touch_items "${test_touch_items}")
    add_suffix("/.success"  regular_test_touch_items)
    # test-touch : ${test_touch_items}/.success
    set(ldeps "${regular_test_touch_items}")
    add_custom_target( test-touch DEPENDS ${ldeps})

    if(${WINDOWS})
        # pick test-touch compiler
        set(tt-c cl)
        # test-touch compilation flags
        list(APPEND tt-c-flags -nologo)
        if(${RELEASE_BUILD} OR ${RELWITHDEBINFO_BUILD})
            list(APPEND tt-c-flags-mt -MT)
            list(APPEND tt-c-flags-md -MD)
        else()
            list(APPEND tt-c-flags-mt -MTd)
            list(APPEND tt-c-flags-md -MDd)
        endif()
        list(APPEND tt-libs ${build_dir}/${imp_file})
        list(APPEND tt-ld-flags -link -nodefaultlib:oldnames)
        if(${IA32})
            list(APPEND tt-ld-flags -safeseh)
        endif()
        list(APPEND tt-ld-flags-v -verbose)
    else() # (Unix based systems, Intel(R) MIC Architecture, and Mac)
        # pick test-touch compiler
        if(${STD_CPP_LIB})
            set(tt-c ${CMAKE_CXX_COMPILER})
        else()
            set(tt-c ${CMAKE_C_COMPILER})
        endif()
        # test-touch compilation flags
        if(${LINUX})
            list(APPEND tt-c-flags -pthread)
        endif()
        if(${IA32})
            list(APPEND tt-c-flags -m32)
        elseif(${INTEL64})
            list(APPEND tt-c-flags -m64)
        endif()
        list(APPEND tt-libs ${build_dir}/${lib_file})
        if(${MAC})
            list(APPEND tt-ld-flags-v -Wl,-t)
            set(tt-env "DYLD_LIBRARY_PATH=.:$ENV{DYLD_LIBRARY_PATH}")
        else()
            list(APPEND tt-ld-flags-v -Wl,--verbose)
            set(tt-env LD_LIBRARY_PATH=".:${build_dir}:$ENV{LD_LIBRARY_PATH}")
        endif()
    endif()
    list(APPEND tt-c-flags "${tt-c-flags-rt}")
    list(APPEND tt-env "KMP_VERSION=1")

    macro(test_touch_recipe test_touch_dir)
        file(MAKE_DIRECTORY ${build_dir}/${test_touch_dir})
        set(ldeps ${src_dir}/test-touch.c ${build_dir}/${lib_file})
        set(tt-exe-file ${test_touch_dir}/test-touch${exe})
        if(${WINDOWS})
            # ****** list(APPEND tt-c-flags -Fo$(dir $@)test-touch${obj} -Fe$(dir $@)test-touch${exe}) *******
            set(tt-c-flags-out -Fo${test_touch_dir}/test-touch${obj} -Fe${test_touch_dir}/test-touch${exe})
            list(APPEND ldeps ${build_dir}/${imp_file})
        else()
            # ****** list(APPEND tt-c-flags -o $(dir $@)test-touch${exe}) ********
            set(tt-c-flags-out -o ${test_touch_dir}/test-touch${exe})
        endif()
        add_custom_command(
            OUTPUT  ${test_touch_dir}/.success
            COMMAND ${CMAKE_COMMAND} -E remove -f ${test_touch_dir}/*
            COMMAND ${tt-c} ${tt-c-flags-out} ${tt-c-flags} ${src_dir}/test-touch.c ${tt-libs} ${tt-ld-flags}
            COMMAND ${CMAKE_COMMAND} -E remove -f ${tt-exe-file}
            COMMAND ${tt-c} ${tt-c-flags-out} ${tt-c-flags} ${src_dir}/test-touch.c ${tt-libs} ${tt-ld-flags} ${tt-ld-flags-v} > ${test_touch_dir}/build.log 2>&1
            COMMAND ${tt-env} ${tt-exe-file}
            #COMMAND grep -i -e \"[^_]libirc\" ${test_touch_dir}/build.log > ${test_touch_dir}/libirc.log \; [ $$? -eq 1 ]
            COMMAND ${CMAKE_COMMAND} -E touch ${test_touch_dir}/.success
            DEPENDS ${ldeps}
        )
    endmacro()
    if(${WINDOWS})
        test_touch_recipe(test-touch-mt)
        test_touch_recipe(test-touch-md)
    else()
        test_touch_recipe(test-touch-rt)
    endif()
else()
    add_custom_target(test-touch DEPENDS test-touch/.success)
    macro(test_touch_recipe_skip test_touch_dir)
        if(${tests} AND ${test_touch})
            set(test_touch_message 'test-touch is not available for the Intel(R) MIC Architecture.  Will not perform it.')
        else()
            set(test_touch_message "test-touch is turned off.  Will not perform it.")
        endif()
        add_custom_command(
            OUTPUT ${test_touch_dir}/.success
            COMMAND ${CMAKE_COMMAND} -E echo ${test_touch_message}
        )
    endmacro()
    test_touch_recipe_skip(test-touch-rt)
    test_touch_recipe_skip(test-touch-mt)
    test_touch_recipe_skip(test-touch-md)
endif()

# test-relo 
add_custom_target(test-relo DEPENDS test-relo/.success)
if((${LINUX} OR ${MIC}) AND ${test_relo} AND ${tests})
    file(MAKE_DIRECTORY ${build_dir}/test-relo)
    add_custom_command(
        OUTPUT  test-relo/.success
        COMMAND readelf -d ${build_dir}/${lib_file} > test-relo/readelf.log
        COMMAND grep -e TEXTREL test-relo/readelf.log \; [ $$? -eq 1 ]
        COMMAND ${CMAKE_COMMAND} -E touch test-relo/.success
        DEPENDS ${build_dir}/${lib_file}
    )
else()
    if(${tests} AND ${test_relo})
        set(test_relo_message 'test-relo is only available for dynamic library on Linux or Intel(R) MIC Architecture.  Will not perform it.')
    else()
        set(test_relo_message "test-relo is turned off.  Will not perform it.")
    endif()
    add_custom_command(
        OUTPUT  test-relo/.success
        COMMAND ${CMAKE_COMMAND} -E echo ${test_relo_message}
    )
endif()

# test-execstack
add_custom_target(test-execstack DEPENDS test-execstack/.success)
if(${LINUX} AND ${test_execstack} AND ${tests})
    file(MAKE_DIRECTORY ${build_dir}/test-execstack)
    add_custom_command(
        OUTPUT  test-execstack/.success
        COMMAND ${PERL_EXECUTABLE} ${tools_dir}/check-execstack.pl ${build_dir}/${lib_file}
        COMMAND ${CMAKE_COMMAND} -E touch test-execstack/.success
        DEPENDS ${build_dir}/${lib_file}
    )
else()
    if(${tests} AND ${test_execstack})
        set(test_execstack_message "test-execstack is only available for dynamic library on Linux.  Will not perform it.")
    else()
        set(test_execstack_message "test-execstack is turned off.  Will not perform it.")
    endif()
    add_custom_command(
        OUTPUT  test-execstack/.success
        COMMAND ${CMAKE_COMMAND} -E echo ${test_execstack_message}
    )
endif()

# test-instr
add_custom_target(test-instr DEPENDS test-instr/.success)
if(${MIC} AND ${test_instr} AND ${tests})
    file(MAKE_DIRECTORY ${build_dir}/test-instr)
    add_custom_command(
        OUTPUT  test-instr/.success
        COMMAND ${PERL_EXECUTABLE} ${tools_dir}/check-instruction-set.pl ${oa_opts} --show --mic-arch=${mic_arch} --mic-os=${mic_os} ${build_dir}/${lib_file}
        COMMAND ${CMAKE_COMMAND} -E touch test-instr/.success
        DEPENDS ${build_dir}/${lib_file} ${tools_dir}/check-instruction-set.pl
    )
else()
    if(${tests} AND ${test_instr})
        set(test_instr_message 'test-instr is only available for Intel(R) MIC Architecture libraries.  Will not perform it.')
    else()
        set(test_instr_message "test-instr is turned off.  Will not perform it.")
    endif()
    add_custom_command(
        OUTPUT  test-instr/.success
        COMMAND ${CMAKE_COMMAND} -E echo ${test_instr_message}
    )
endif()

# test-deps
add_custom_target(test-deps DEPENDS test-deps/.success)
if(${test_deps} AND ${tests})
    set(td_exp)
    if(${FREEBSD})
        set(td_exp libc.so.7 libthr.so.3 libunwind.so.5)
    elseif(${LINUX})
        set(td_exp libdl.so.2,libgcc_s.so.1)
        if(NOT ${IA32} AND NOT ${INTEL64})
            set(td_exp ${td_exp},libffi.so.6,libffi.so.5)
        endif()
        if(${IA32})
            set(td_exp ${td_exp},libc.so.6,ld-linux.so.2)  
        elseif(${INTEL64})
            set(td_exp ${td_exp},libc.so.6,ld-linux-x86-64.so.2)  
        elseif(${ARM})
            set(td_exp ${td_exp},libc.so.6,ld-linux-armhf.so.3)  
        endif()
        if(${STD_CPP_LIB})
            set(td_exp ${td_exp},libstdc++.so.6)
        endif()
        if(NOT ${STUBS_LIBRARY})
            set(td_exp ${td_exp},libpthread.so.0)
        endif()
    elseif(${MIC})
        if("${mic_os}" STREQUAL "lin")
            set(td_exp libc.so.6,libpthread.so.0,libdl.so.2)
            if(${STD_CPP_LIB})
                set(td_exp ${td_exp},libstdc++.so.6)
            endif()
            if("${mic_arch}" STREQUAL "knf")
                set(td_exp ${td_exp},ld-linux-l1om.so.2,libgcc_s.so.1)
            elseif("${mic_arch}" STREQUAL "knc")
                set(td_exp ${td_exp},ld-linux-k1om.so.2)
            endif()
        elseif("${mic_os}" STREQUAL "bsd")
            set(td_exp libc.so.7,libthr.so.3,libunwind.so.5)
        endif()
    elseif(${MAC})
        set(td_exp /usr/lib/libSystem.B.dylib)
    elseif(${WINDOWS})
        set(td_exp kernel32.dll)
    endif()

    file(MAKE_DIRECTORY ${build_dir}/test-deps)
    add_custom_command(
        OUTPUT  test-deps/.success
        COMMAND ${PERL_EXECUTABLE} ${tools_dir}/check-depends.pl ${oa_opts} --expected="${td_exp}" ${build_dir}/${lib_file}
        COMMAND ${CMAKE_COMMAND} -E touch test-deps/.success
        DEPENDS ${build_dir}/${lib_file} ${tools_dir}/check-depends.pl
    )
else()
    if(${tests} AND ${test_deps})
        set(test_deps_message 'test-deps is available for dynamic libraries on Linux, Mac, Intel(R) MIC Architecture, Windows and static libraries on Windows.  Will not perform it.')
    else()
        set(test_deps_message "test-deps is turned off.  Will not perform it.")
    endif()
    add_custom_command(
        OUTPUT  test-deps/.success
        COMMAND ${CMAKE_COMMAND} -E echo ${test_deps_message}
    )
endif()
# END OF TESTS
######################################################
