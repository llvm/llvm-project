macro(get_all_targets_recursive targets dir)
    get_property(subdirectories DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
    foreach(subdir ${subdirectories})
        get_all_targets_recursive(${targets} ${subdir})
    endforeach()

    get_property(current_targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
    list(APPEND ${targets} ${current_targets})
endmacro()

function(get_all_targets dir outvar)
    set(targets)
    get_all_targets_recursive(targets ${dir})
    set(${outvar} ${targets} PARENT_SCOPE)
endfunction()

function(add_llvm_lib_precompiled_headers target)
    get_target_property(target_type ${target} TYPE)
    if (target_type STREQUAL "STATIC_LIBRARY")
        target_precompile_headers(${target} REUSE_FROM LLVMPchTarget)
    endif()
endfunction()

function(add_llvm_subdir_pch subdir)
    get_all_targets("${LLVM_MAIN_SRC_DIR}/lib/${subdir}" lib_targets)
    foreach(target ${lib_targets})
        add_llvm_lib_precompiled_headers(${target})
    endforeach()
endfunction()

function(llvm_lib_precompiled_headers)
    # target_precompile_headers was introduced in 3.16
    if(${CMAKE_VERSION} VERSION_LESS "3.16")
        message(STATUS "LLVM Lib Precompiled Headers requires CMake version 3.16")
        set(LLVM_ENABLE_LIB_PRECOMPILED_HEADERS OFF CACHE BOOL "" FORCE)
    endif()

    if (LLVM_ENABLE_LIB_PRECOMPILED_HEADERS)
        message(STATUS "LLVM Lib Precompiled Headers are enabled")
        # Create a dummy source file to compile the PCH target.
        set(pch_dummy_cpp ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp)
        if (NOT EXISTS ${pch_dummy_cpp})
            FILE(TOUCH ${pch_dummy_cpp})
        endif()

        set(precompiled_header_path ${LLVM_MAIN_INCLUDE_DIR}/llvm/PrecompiledHeaders.h)
        add_llvm_component_library(LLVMPchTarget ${pch_dummy_cpp} ${precompiled_header_path})
        target_precompile_headers(LLVMPchTarget PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${LLVM_MAIN_INCLUDE_DIR}/llvm/PrecompiledHeaders.h>")

        if (NOT ${LLVM_LIB_DIRETORIES_FOR_PRECOMPILED_HEADERS})
            set(default_lib_dirs_for_pch
                "Analysis"
                "Codegen"
                "DebugInfo"
                "ExecutionEngine"
                "IR"
                "MC"
                "MCA"
                "ObjCopy"
                "Object"
                "Passes"
                "Support"
                "Target"
                "Transforms"
            )
            set(LLVM_LIB_DIRETORIES_FOR_PRECOMPILED_HEADERS ${default_lib_dirs_for_pch})
        endif()
        foreach(subdir ${LLVM_LIB_DIRETORIES_FOR_PRECOMPILED_HEADERS})
	        add_llvm_subdir_pch(${subdir})
        endforeach()
    endif()
endfunction()
