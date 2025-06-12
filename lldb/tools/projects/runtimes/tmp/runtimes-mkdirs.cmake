# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/anth/Developer/llvm-project/llvm/runtimes/../../runtimes")
  file(MAKE_DIRECTORY "/Users/anth/Developer/llvm-project/llvm/runtimes/../../runtimes")
endif()
file(MAKE_DIRECTORY
  "/Users/anth/Developer/llvm-project/lldb/tools/runtimes/runtimes-bins"
  "/Users/anth/Developer/llvm-project/lldb/tools/projects/runtimes"
  "/Users/anth/Developer/llvm-project/lldb/tools/projects/runtimes/tmp"
  "/Users/anth/Developer/llvm-project/lldb/tools/runtimes/runtimes-stamps"
  "/Users/anth/Developer/llvm-project/lldb/tools/projects/runtimes/src"
  "/Users/anth/Developer/llvm-project/lldb/tools/runtimes/runtimes-stamps"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/anth/Developer/llvm-project/lldb/tools/runtimes/runtimes-stamps/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/anth/Developer/llvm-project/lldb/tools/runtimes/runtimes-stamps${cfgdir}") # cfgdir has leading slash
endif()
