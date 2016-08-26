include(ExternalProject)
include(ProcessorCount)

# Provide a config.h which exposes build system information.
configure_file(
  cmd/gllgo/config.h.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmd/gllgo/config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/cmd/gllgo)

list(APPEND LLVM_GO_PACKAGES "llvm.org/llgo=${CMAKE_CURRENT_SOURCE_DIR}")

llvm_add_go_executable(llgo llvm.org/llgo/cmd/gllgo ALL DEPENDS
  build/context.go
  cmd/gllgo/gllgo.go
  debug/debug.go
  driver/parser.go
  irgen/annotations.go
  irgen/attribute.go
  irgen/builtins.go
  irgen/cabi.go
  irgen/call.go
  irgen/channels.go
  irgen/closures.go
  irgen/compiler.go
  irgen/errors.go
  irgen/indirect.go
  irgen/interfaces.go
  irgen/maps.go
  irgen/predicates.go
  irgen/println.go
  irgen/runtime.go
  irgen/slice.go
  irgen/ssa.go
  irgen/strings.go
  irgen/switches.go
  irgen/targets.go
  irgen/typemap.go
  irgen/types.go
  irgen/utils.go
  irgen/value.go
  irgen/version.go
  ssaopt/esc.go
)

string(REGEX MATCH "[0-9]+\\.[0-9]+(\\.[0-9]+)?" LLGO_VERSION
  ${PACKAGE_VERSION})

configure_file(
  cmd/go/zdefaultcc.go.in
  ${CMAKE_CURRENT_BINARY_DIR}/cmd/go/zdefaultcc.go)

set(LLGO_GO_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/build.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/clean.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/context.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/discovery.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/doc.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/env.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/fix.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/fmt.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/generate.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/get.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/go11.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/help.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/http.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/list.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/main.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/note.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/pkg.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/run.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/signal.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/signal_unix.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/testflag.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/test.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/tool.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/vcs.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/version.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/go/vet.go
  ${CMAKE_CURRENT_BINARY_DIR}/cmd/go/zdefaultcc.go
)

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/bin/llgo-go${CMAKE_EXECUTABLE_SUFFIX}
  COMMAND ${CMAKE_BINARY_DIR}/bin/llgo -static-libgo
          -I ${CMAKE_CURRENT_BINARY_DIR}/libgo
          -o ${CMAKE_BINARY_DIR}/bin/llgo-go${CMAKE_EXECUTABLE_SUFFIX}
          ${LLGO_GO_SOURCES}
          ${CMAKE_CURRENT_BINARY_DIR}/libgo/zstdpkglist.go
  DEPENDS llgo libgo ${LLGO_GO_SOURCES}
  COMMENT "Building Go executable llgo-go"
  VERBATIM)
add_custom_target(llgo-go ALL DEPENDS ${CMAKE_BINARY_DIR}/bin/llgo-go${CMAKE_EXECUTABLE_SUFFIX})

set(LLGO_CGO_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/cgo/ast.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/cgo/doc.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/cgo/gcc.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/cgo/godefs.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/cgo/main.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/cgo/out.go
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo/go/cmd/cgo/util.go
  ${CMAKE_CURRENT_SOURCE_DIR}/cmd/cgo/zdefaultcc.go
)

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/lib/go/llgo-${LLGO_VERSION}/cgo
  COMMAND ${CMAKE_BINARY_DIR}/bin/llgo -static-libgo
          -o ${CMAKE_BINARY_DIR}/lib/go/llgo-${LLGO_VERSION}/cgo
          ${LLGO_CGO_SOURCES}
  DEPENDS llgo libgo ${LLGO_CGO_SOURCES}
  COMMENT "Building Go executable cgo"
  VERBATIM)
add_custom_target(cgo ALL DEPENDS ${CMAKE_BINARY_DIR}/lib/go/llgo-${LLGO_VERSION}/cgo)

llvm_add_go_executable(llgo-stage2 llvm.org/llgo/cmd/gllgo
  DEPENDS libgo ${CMAKE_BINARY_DIR}/bin/llgo${CMAKE_EXECUTABLE_SUFFIX}
          ${CMAKE_BINARY_DIR}/lib/go/llgo-${LLGO_VERSION}/cgo
          ${CMAKE_BINARY_DIR}/bin/llgo-go${CMAKE_EXECUTABLE_SUFFIX}
  GOFLAGS "cc=${CMAKE_BINARY_DIR}/bin/clang"
          "cxx=${CMAKE_BINARY_DIR}/bin/clang++"
          "go=${CMAKE_BINARY_DIR}/bin/llgo-go"
)

llvm_add_go_executable(llgo-stage3 llvm.org/llgo/cmd/gllgo
  DEPENDS libgo ${CMAKE_BINARY_DIR}/bin/llgo-stage2${CMAKE_EXECUTABLE_SUFFIX}
  GOFLAGS "cc=${CMAKE_BINARY_DIR}/bin/clang"
          "cxx=${CMAKE_BINARY_DIR}/bin/clang++"
          "go=${CMAKE_BINARY_DIR}/bin/llgo-go"
          "llgo=${CMAKE_BINARY_DIR}/bin/llgo-stage2${CMAKE_EXECUTABLE_SUFFIX}"
)

llvm_add_go_executable(cc-wrapper llvm.org/llgo/cmd/cc-wrapper DEPENDS
  cmd/cc-wrapper/main.go
)

llvm_add_go_executable(llgoi llvm.org/llgo/cmd/llgoi ALL
  DEPENDS libgo ${CMAKE_BINARY_DIR}/bin/llgo${CMAKE_EXECUTABLE_SUFFIX}
          ${CMAKE_BINARY_DIR}/lib/go/llgo-${LLGO_VERSION}/cgo
          ${CMAKE_BINARY_DIR}/bin/llgo-go${CMAKE_EXECUTABLE_SUFFIX}
          cmd/llgoi/llgoi.go
  GOFLAGS "cc=${CMAKE_BINARY_DIR}/bin/clang"
          "cxx=${CMAKE_BINARY_DIR}/bin/clang++"
          "go=${CMAKE_BINARY_DIR}/bin/llgo-go"
)

install(FILES ${CMAKE_BINARY_DIR}/bin/llgo${CMAKE_EXECUTABLE_SUFFIX}
              ${CMAKE_BINARY_DIR}/bin/llgoi${CMAKE_EXECUTABLE_SUFFIX}
              ${CMAKE_BINARY_DIR}/bin/llgo-go${CMAKE_EXECUTABLE_SUFFIX}
        DESTINATION bin
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE)

function(add_clobber_steps name)
  ExternalProject_Add_Step(${name} force-reconfigure
    DEPENDERS configure
    ALWAYS 1
    )

  ExternalProject_Add_Step(${name} clobber
    COMMAND ${CMAKE_COMMAND} -E remove_directory <BINARY_DIR>
    COMMAND ${CMAKE_COMMAND} -E make_directory <BINARY_DIR>
    COMMENT "Clobbering ${name} build directory..."
    DEPENDERS configure
    DEPENDS ${ARGN}
    )
endfunction()

processorcount(PROCESSOR_COUNT)

function(add_libgo_variant suffix cflags gocflags deps exclude_from_all)
  externalproject_add(libbacktrace${suffix}
    DEPENDS clang ${deps}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libbacktrace
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${suffix}/libbacktrace
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --disable-multilib --enable-host-shared "CC=${CMAKE_BINARY_DIR}/bin/clang ${cflags}"
    BUILD_COMMAND make -j${PROCESSOR_COUNT}
    INSTALL_COMMAND ""
    LOG_CONFIGURE 1
    LOG_BUILD 1
  )
  set_property(TARGET libbacktrace${suffix}
               PROPERTY EXCLUDE_FROM_ALL ${exclude_from_all})

  add_clobber_steps(libbacktrace${suffix} clang ${deps})

  externalproject_add(libffi${suffix}
    DEPENDS clang ${deps}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libffi
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${suffix}/libffi
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --disable-multilib "CC=${CMAKE_BINARY_DIR}/bin/clang ${cflags}"
    BUILD_COMMAND make -j${PROCESSOR_COUNT}
    INSTALL_COMMAND ""
    LOG_CONFIGURE 1
    LOG_BUILD 1
  )
  set_property(TARGET libffi${suffix}
               PROPERTY EXCLUDE_FROM_ALL ${exclude_from_all})

  add_clobber_steps(libffi${suffix} clang ${deps})

  externalproject_add(libgo${suffix}
    DEPENDS clang llgo cc-wrapper libbacktrace${suffix} libffi${suffix} ${deps}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${suffix}/libgo
    INSTALL_DIR ${CMAKE_BINARY_DIR}
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --disable-multilib --without-libatomic --prefix=<INSTALL_DIR> "CC=env REAL_CC=${CMAKE_BINARY_DIR}/bin/clang@SPACE@${cflags} ${CMAKE_BINARY_DIR}/bin/cc-wrapper" "GOC=${CMAKE_BINARY_DIR}/bin/llgo -no-prefix -fcompilerrt-prefix=${CMAKE_BINARY_DIR} ${gocflags}"
    BUILD_COMMAND make -j${PROCESSOR_COUNT}
    LOG_CONFIGURE 1
    LOG_BUILD 1
    LOG_INSTALL 1
  )
  set_property(TARGET libgo${suffix}
               PROPERTY EXCLUDE_FROM_ALL ${exclude_from_all})

  add_clobber_steps(libgo${suffix} clang
    ${CMAKE_BINARY_DIR}/bin/llgo${CMAKE_EXECUTABLE_SUFFIX}
    ${CMAKE_BINARY_DIR}/bin/cc-wrapper${CMAKE_EXECUTABLE_SUFFIX})
endfunction()

add_libgo_variant("" "" "" "" FALSE)

if(TARGET asan)
  add_libgo_variant("_asan" "-fsanitize=address" "-fsanitize=address" asan TRUE)
endif()

if(TARGET tsan)
  add_libgo_variant("_tsan" "-fsanitize=thread" "-fsanitize=thread" tsan TRUE)
endif()

if(TARGET msan)
  add_libgo_variant("_msan" "-fsanitize=memory" "-fsanitize=memory" msan TRUE)
endif()

if(TARGET dfsan)
  add_libgo_variant("_dfsan" "-fsanitize=dataflow" "-fsanitize=dataflow" dfsan TRUE)
endif()

set(LLGO_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})

install(FILES ${LLGO_LIBRARY_DIR}/libgo-llgo.a
              ${LLGO_LIBRARY_DIR}/libgo-llgo.so
              ${LLGO_LIBRARY_DIR}/libgo-llgo.so.8
              ${LLGO_LIBRARY_DIR}/libgo-llgo.so.8.0.0
              ${LLGO_LIBRARY_DIR}/libgobegin-llgo.a
        DESTINATION lib${LLVM_LIBDIR_SUFFIX})

install(DIRECTORY ${LLGO_LIBRARY_DIR}/go
        DESTINATION lib${LLVM_LIBDIR_SUFFIX})

add_custom_target(check-libgo
  COMMAND make -C ${CMAKE_CURRENT_BINARY_DIR}/libgo -j${PROCESSOR_COUNT} check
  DEPENDS libgo
  ${cmake_3_2_USES_TERMINAL}
  COMMENT "Running libgo tests")

add_custom_target(check-llgo-bootstrap
  COMMAND strip -R .note.gnu.build-id -o ${CMAKE_CURRENT_BINARY_DIR}/llgo-stage2.stripped
          ${CMAKE_BINARY_DIR}/bin/llgo-stage2${CMAKE_EXECUTABLE_SUFFIX}
  COMMAND strip -R .note.gnu.build-id -o ${CMAKE_CURRENT_BINARY_DIR}/llgo-stage3.stripped
          ${CMAKE_BINARY_DIR}/bin/llgo-stage3${CMAKE_EXECUTABLE_SUFFIX}
  COMMAND cmp ${CMAKE_CURRENT_BINARY_DIR}/llgo-stage2.stripped
          ${CMAKE_CURRENT_BINARY_DIR}/llgo-stage3.stripped
  DEPENDS llgo-stage2 llgo-stage3
  COMMENT "Checking llgo bootstrap")

add_subdirectory(test)
