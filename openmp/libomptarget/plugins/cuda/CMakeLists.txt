##===----------------------------------------------------------------------===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
# 
##===----------------------------------------------------------------------===##
#
# Build a plugin for a CUDA machine if available.
#
##===----------------------------------------------------------------------===##
if(LIBOMPTARGET_DEP_LIBELF_FOUND)
  if(LIBOMPTARGET_DEP_CUDA_FOUND)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(ppc64le)$" AND CMAKE_SYSTEM_NAME MATCHES "Linux")
    
      libomptarget_say("Building CUDA offloading plugin.")
    
      # Define the suffix for the runtime messaging dumps.
      add_definitions(-DTARGET_NAME=CUDA)
    
      if(LIBOMPTARGET_CMAKE_BUILD_TYPE MATCHES debug)
          add_definitions(-DCUDA_ERROR_REPORT)
      endif()
      
      include_directories(${LIBOMPTARGET_DEP_CUDA_INCLUDE_DIRS})
    
      add_library(omptarget.rtl.cuda SHARED src/rtl.cpp)
      
      # Install plugin under the lib destination folder.
      install(TARGETS omptarget.rtl.cuda LIBRARY DESTINATION lib${LIBOMPTARGET_LIBDIR_SUFFIX})
      
      target_link_libraries(omptarget.rtl.cuda 
        ${LIBOMPTARGET_DEP_CUDA_LIBRARIES} 
        cuda 
        ${LIBOMPTARGET_DEP_LIBELF_LIBRARIES}
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports")
    
      # Report to the parent scope that we are building a plugin for CUDA.
      set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS} nvptx64-nvidia-cuda" PARENT_SCOPE)
    else()
      libomptarget_say("Not building CUDA offloading plugin: only support CUDA in Linux x86_64 or ppc64le hosts.")
    endif()
  else()
    libomptarget_say("Not building CUDA offloading plugin: CUDA not found in system.")
  endif()
else(LIBOMPTARGET_DEP_LIBELF_FOUND)
  libomptarget_say("Not building CUDA offloading plugin: libelf dependency not found.")
endif(LIBOMPTARGET_DEP_LIBELF_FOUND)