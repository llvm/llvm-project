# Builds the documentation of all of the LLVM projects, with hermetic links
# between them.
#
# The docs of the projects link to each other with `inv:` links, which Sphinx
# resolves against the other project's `objects.inv`. A plain
# `ninja docs-<project>-html` fetches the published inventories from llvm.org.
# This project builds them from the sources in this checkout instead, so there
# is no network access, and links to labels that the same change adds, and
# circular links, resolve.
#
#   cmake -G Ninja -S utils/docs/hermetic_build -B build-docs
#   ninja -C build-docs docs
#
# The tools (LLVM_ENABLE_PROJECTS) and the runtimes (LLVM_ENABLE_RUNTIMES) are
# separate CMake builds, so there are two sub-builds, each of which is built in
# two steps:
#
#   1. `docs-inventory`: the `docs-inventories` target of each, which only
#      reads the sources of each project, and writes <build>/inventories.
#   2. `docs`: the html and man pages of each, which read all of the
#      inventories. They are collected in <build>/docs/<project>.
#
# Sphinx is mostly serial, so the two sub-builds of each step run in parallel,
# and each one's output is kept together. Pass -DSPHINX_EXECUTABLE=... to use a
# particular Sphinx. Building the docs of all of the projects needs everything
# that their docs need, for example Python and SWIG for LLDB.

cmake_minimum_required(VERSION 3.20.0)
project(llvm-docs NONE)

include(ExternalProject)

get_filename_component(llvm_project_dir "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(inventory_dir "${CMAKE_BINARY_DIR}/inventories")

# Adds the docs targets of a project to build in the `tools` or `runtimes`
# sub-build, and collects the docs in the directory `docs_dir` of the sub-build.
function(add_project_docs name build docs_dir)
  set_property(GLOBAL APPEND PROPERTY ${build}_docs_targets ${ARGN})
  set_property(GLOBAL APPEND PROPERTY collect_docs_commands
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_BINARY_DIR}/${build}/${docs_dir}"
            "${CMAKE_BINARY_DIR}/docs/${name}")
endfunction()

add_project_docs(llvm tools docs docs-llvm-html docs-llvm-man)
add_project_docs(bolt tools tools/bolt/docs docs-bolt-html)
add_project_docs(clang tools tools/clang/docs docs-clang-html docs-clang-man)
add_project_docs(clang-tools-extra tools tools/clang/tools/extra/docs
                 docs-clang-tools-html docs-clang-tools-man)
add_project_docs(lldb tools tools/lldb/docs docs-lldb-html docs-lldb-man)
add_project_docs(lld tools tools/lld/docs docs-lld-html)
add_project_docs(polly tools tools/polly/docs docs-polly-html docs-polly-man)
add_project_docs(flang tools tools/flang/docs docs-flang-html docs-flang-man)
add_project_docs(libunwind runtimes libunwind/docs docs-libunwind-html)
add_project_docs(libcxx runtimes libcxx/docs docs-libcxx-html)
add_project_docs(libc runtimes libc/docs docs-libc-html)
add_project_docs(openmp runtimes openmp/docs docs-openmp-html)

# Defines the sub-build `name`, with steps `build` (the inventories) and `docs`.
function(add_docs_build name source_dir)
  set(binary_dir "${CMAKE_BINARY_DIR}/${name}")
  get_property(targets GLOBAL PROPERTY ${name}_docs_targets)
  set(sphinx_arg)
  if (SPHINX_EXECUTABLE)
    set(sphinx_arg -DSPHINX_EXECUTABLE=${SPHINX_EXECUTABLE})
  endif()
  ExternalProject_Add(${name}
    SOURCE_DIR "${source_dir}"
    BINARY_DIR "${binary_dir}"
    LIST_SEPARATOR |
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=Release
      -DLLVM_ENABLE_SPHINX=ON
      -DLLVM_SPHINX_INVENTORY_DIR=${inventory_dir}
      ${sphinx_arg}
      ${ARGN}
    BUILD_COMMAND ${CMAKE_COMMAND} --build "${binary_dir}" --target docs-inventories
    BUILD_ALWAYS ON
    INSTALL_COMMAND ""
    STEP_TARGETS build)
  ExternalProject_Add_Step(${name} docs
    COMMAND ${CMAKE_COMMAND} -E env TZ=UTC
            ${CMAKE_COMMAND} --build "${binary_dir}" --target ${targets}
    DEPENDEES build
    ALWAYS ON)
  ExternalProject_Add_StepTargets(${name} docs)
endfunction()

add_docs_build(tools "${llvm_project_dir}/llvm"
  -DLLVM_ENABLE_PROJECTS=bolt|clang|clang-tools-extra|lld|lldb|polly|mlir|flang)
add_docs_build(runtimes "${llvm_project_dir}/runtimes"
  -DLLVM_ENABLE_RUNTIMES=libcxx|libcxxabi|libunwind|libc|openmp
  -DLLVM_INCLUDE_TESTS=OFF)

# Every docs step reads all of the inventories, so wait for both builds of them.
ExternalProject_Add_StepDependencies(tools docs tools-build runtimes-build)
ExternalProject_Add_StepDependencies(runtimes docs tools-build runtimes-build)

# The inventories of all of the projects, in ${inventory_dir}/<project>.
add_custom_target(docs-inventory DEPENDS tools-build runtimes-build)

get_property(collect_commands GLOBAL PROPERTY collect_docs_commands)
add_custom_target(docs ${collect_commands})
add_dependencies(docs tools-docs runtimes-docs)
