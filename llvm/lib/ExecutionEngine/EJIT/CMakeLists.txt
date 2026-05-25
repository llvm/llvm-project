#===-- EmbeddedJIT target triple configuration ----------------------------===#
# When cross-compiling for ARM embedded, set this to the target triple
# so the JIT engine targets the correct architecture at runtime.
# Default: detect host (sufficient for native builds).
set(EJIT_DEFAULT_TARGET_TRIPLE "" CACHE STRING
  "Default target triple for the EmbeddedJIT engine (empty = detect host)")

if(EJIT_DEFAULT_TARGET_TRIPLE)
  message(STATUS "EmbeddedJIT: forcing target triple = ${EJIT_DEFAULT_TARGET_TRIPLE}")
endif()

#===-- Bare-metal configuration -------------------------------------------===#
option(EJIT_BARE_METAL "Build EJIT for bare-metal (no OS, no threads)" ON)
if(EJIT_BARE_METAL)
  message(STATUS "EmbeddedJIT: bare-metal mode enabled")
  add_definitions(-DEJIT_BARE_METAL)
endif()

# Build target component list from LLVM_TARGETS_TO_BUILD.
# Each target needs CodeGen, Desc, and Info for native target initialization.
# AsmPrinter init lives in CodeGen, so no separate AsmPrinter component needed.
set(EJIT_TARGET_COMPONENTS "")
foreach(_t ${LLVM_TARGETS_TO_BUILD})
  list(APPEND EJIT_TARGET_COMPONENTS ${_t}CodeGen ${_t}Desc ${_t}Info)
endforeach()

set(EJIT_SOURCES
  EJit.cpp
  EJitRuntime.cpp
  EJitRegistration.cpp
  EJitOrcEngine.cpp
  EJitCompileDriver.cpp
  EJitCache.cpp
  EJitRuntimeState.cpp
  EJitModuleLoader.cpp
  EJitLogger.cpp
  EJitRegistrationStore.cpp
  EJitStructFieldPass.cpp
  EJitOptimizer.cpp
  EJitPassBuilder.cpp
)

# Async/sync compiler — excluded in bare-metal (single-threaded, no std::thread)
if(NOT EJIT_BARE_METAL)
  list(APPEND EJIT_SOURCES
    EJitSyncCompiler.cpp
    EJitAsyncCompiler.cpp
  )
endif()

add_llvm_component_library(LLVMEJIT
  ${EJIT_SOURCES}

  PARTIAL_SOURCES_INTENDED

  LINK_COMPONENTS
  BitReader
  Core
  OrcJIT
  JITLink
  Support
  InstCombine
  TransformUtils
  Analysis
  ${EJIT_TARGET_COMPONENTS}
  # IPO and Scalar are not listed here — their LINK_COMPONENTS would
  # recursively pull in BitWriter, Instrumentation, Linker, AsmParser,
  # and AggressiveInstCombine.  Instead, they are added manually to the
  # ejit_minimal target below (see §6.5 of EJIT_LIBRARY_TRIMMING.md).
  # ExecutionEngine is not listed — it's already pulled by OrcJIT.
  )

# Propagate the target triple setting to the C++ code
if(EJIT_DEFAULT_TARGET_TRIPLE)
  target_compile_definitions(LLVMEJIT PUBLIC
    EJIT_DEFAULT_TRIPLE="${EJIT_DEFAULT_TARGET_TRIPLE}")
endif()

#===-- Minimal embedded runtime target -----------------------------------===#
# Builds a single combined static library (libejit_minimal.a) that bundles
# EJIT + all transitive LLVM dependencies. Designed for ARM embedded targets
# with a ~10MB RAM budget.
#
# Usage:
#   cmake --preset ejit-minimal
#   ninja LLVMEJIT              # build LLVMEJIT + its dependency libraries
#   ninja ejit_minimal           # combine into fat archive
#   ninja check-ejit-size        # verify size fits in budget

# Resolve LLVM component names → library names (e.g. "Core" → "LLVMCore")
llvm_map_components_to_libnames(EJIT_LLVM_LIBS
  BitReader Core OrcJIT JITLink Support
  InstCombine TransformUtils Analysis
  ${EJIT_TARGET_COMPONENTS}
  )
# IPO and Scalar are added manually to avoid their LINK_COMPONENTS
# (BitWriter, Instrumentation, Linker, AsmParser, AggressiveInstCombine).
# See EJIT_LIBRARY_TRIMMING.md §6.5.
list(APPEND EJIT_LLVM_LIBS LLVMipo LLVMScalarOpts)

# Build a file listing all .a files to combine (one per line)
# LLVM_LIBRARY_DIR is the build output directory for static libraries
set(EJIT_LIBS_LIST "${CMAKE_CURRENT_BINARY_DIR}/ejit_libs.txt")
set(EJIT_MINIMAL_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/libejit_minimal.a")

# Generate the library path list at configure time
# LLVMEJIT is special: its .a name includes the component name
set(_ejit_lib_paths "${LLVM_LIBRARY_DIR}/libLLVMEJIT.a")
foreach(_lib ${EJIT_LLVM_LIBS})
  list(APPEND _ejit_lib_paths "${LLVM_LIBRARY_DIR}/lib${_lib}.a")
endforeach()
string(REPLACE ";" "\n" _ejit_lib_paths_str "${_ejit_lib_paths}")
file(WRITE "${EJIT_LIBS_LIST}" "${_ejit_lib_paths_str}\n")

# Combine all libraries into one fat archive
add_custom_target(ejit_minimal
  COMMAND ${CMAKE_COMMAND}
    -DLIB_LIST_FILE="${EJIT_LIBS_LIST}"
    -DOUTPUT="${EJIT_MINIMAL_OUTPUT}"
    -DCMAKE_AR="${CMAKE_AR}"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/CombineArchives.cmake"
  DEPENDS LLVMEJIT ${EJIT_LLVM_LIBS}
  COMMENT "Combining EJIT + LLVM dependencies into libejit_minimal.a"
  )

# Size verification target — checks the combined archive fits within budget
add_custom_target(check-ejit-size
  COMMAND ${CMAKE_COMMAND}
    -DARCHIVE_FILE="${EJIT_MINIMAL_OUTPUT}"
    -DMAX_SIZE_MB=10
    -P "${CMAKE_CURRENT_SOURCE_DIR}/CheckSize.cmake"
  DEPENDS ejit_minimal
  )
