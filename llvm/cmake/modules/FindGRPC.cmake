# FIXME(kirillbobyrev): Check if gRPC and Protobuf headers can be included at
# configure time.
find_package(Threads REQUIRED)
if (GRPC_INSTALL_PATH)
  # This setup requires gRPC to be built from sources using CMake and installed
  # to ${GRPC_INSTALL_PATH} via -DCMAKE_INSTALL_PREFIX=${GRPC_INSTALL_PATH}.
  # Libraries will be linked according to gRPC build policy which generates
  # static libraries when BUILD_SHARED_LIBS is Off and dynamic libraries when
  # it's On (NOTE: This is a variable passed to gRPC CMake build invocation,
  # LLVM's BUILD_SHARED_LIBS has no effect).
  set(protobuf_MODULE_COMPATIBLE TRUE)
  find_package(Protobuf CONFIG REQUIRED HINTS ${GRPC_INSTALL_PATH})
  message(STATUS "Using protobuf ${protobuf_VERSION}")
  find_package(gRPC CONFIG REQUIRED HINTS ${GRPC_INSTALL_PATH})
  message(STATUS "Using gRPC ${gRPC_VERSION}")

  include_directories(${Protobuf_INCLUDE_DIRS})

  # gRPC CMake CONFIG gives the libraries slightly odd names, make them match
  # the conventional system-installed names.
  set_target_properties(protobuf::libprotobuf PROPERTIES IMPORTED_GLOBAL TRUE)
  add_library(protobuf ALIAS protobuf::libprotobuf)
  set_target_properties(gRPC::grpc++ PROPERTIES IMPORTED_GLOBAL TRUE)
  add_library(grpc++ ALIAS gRPC::grpc++)

  set(GRPC_CPP_PLUGIN $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
  set(PROTOC ${Protobuf_PROTOC_EXECUTABLE})
else()
  # This setup requires system-installed gRPC and Protobuf.
  # We always link dynamically in this mode. While the static libraries are
  # usually installed, the CMake files telling us *which* static libraries to
  # link are not.
  if (NOT BUILD_SHARED_LIBS)
    message(NOTICE "gRPC and Protobuf will be linked dynamically. If you want static linking, build gRPC from sources with -DBUILD_SHARED_LIBS=Off.")
  endif()
  find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)
  find_program(PROTOC protoc)
  if (NOT GRPC_CPP_PLUGIN OR NOT PROTOC)
    message(FATAL_ERROR "gRPC C++ Plugin and Protoc must be on $PATH for Clangd remote index build.")
  endif()
  # On macOS the libraries are typically installed via Homebrew and are not on
  # the system path.
  set(GRPC_OPTS "")
  set(PROTOBUF_OPTS "")
  if (${APPLE})
    find_program(HOMEBREW brew)
    # If Homebrew is not found, the user might have installed libraries
    # manually. Fall back to the system path.
    if (HOMEBREW)
      execute_process(COMMAND ${HOMEBREW} --prefix grpc
        OUTPUT_VARIABLE GRPC_HOMEBREW_PATH
        RESULT_VARIABLE GRPC_HOMEBREW_RETURN_CODE
        OUTPUT_STRIP_TRAILING_WHITESPACE)
      execute_process(COMMAND ${HOMEBREW} --prefix protobuf
        OUTPUT_VARIABLE PROTOBUF_HOMEBREW_PATH
        RESULT_VARIABLE PROTOBUF_HOMEBREW_RETURN_CODE
        OUTPUT_STRIP_TRAILING_WHITESPACE)
      # If either library is not installed via Homebrew, fall back to the
      # system path.
      if (GRPC_HOMEBREW_RETURN_CODE EQUAL "0")
        include_directories(${GRPC_HOMEBREW_PATH}/include)
        list(APPEND GRPC_OPTS PATHS ${GRPC_HOMEBREW_PATH}/lib NO_DEFAULT_PATH)
      endif()
      if (PROTOBUF_HOMEBREW_RETURN_CODE EQUAL "0")
        include_directories(${PROTOBUF_HOMEBREW_PATH}/include)
        list(APPEND PROTOBUF_OPTS PATHS ${PROTOBUF_HOMEBREW_PATH}/lib NO_DEFAULT_PATH)
      endif()
    endif()
  endif()
  find_library(GRPC_LIBRARY grpc++ $GRPC_OPTS REQUIRED)
  add_library(grpc++ UNKNOWN IMPORTED GLOBAL)
  message(STATUS "Using grpc++: " ${GRPC_LIBRARY})
  set_target_properties(grpc++ PROPERTIES IMPORTED_LOCATION ${GRPC_LIBRARY})
  find_library(PROTOBUF_LIBRARY protobuf $PROTOBUF_OPTS REQUIRED)
  message(STATUS "Using protobuf: " ${PROTOBUF_LIBRARY})
  add_library(protobuf UNKNOWN IMPORTED GLOBAL)
  set_target_properties(protobuf PROPERTIES IMPORTED_LOCATION ${PROTOBUF_LIBRARY})
endif()

# Proto headers are generated in ${CMAKE_CURRENT_BINARY_DIR}.
# Libraries that use these headers should adjust the include path.
# If the "GRPC" argument is given, services are also generated.
# The DEPENDS list should name *.proto source files that are imported.
# They may be relative to the source dir or absolute (for generated protos).
function(generate_protos LibraryName ProtoFile)
  cmake_parse_arguments(PARSE_ARGV 2 PROTO "GRPC" "" "DEPENDS")
  get_filename_component(ProtoSourceAbsolutePath "${CMAKE_CURRENT_SOURCE_DIR}/${ProtoFile}" ABSOLUTE)
  get_filename_component(ProtoSourcePath ${ProtoSourceAbsolutePath} PATH)
  get_filename_component(Basename ${ProtoSourceAbsolutePath} NAME_WLE)

  set(GeneratedProtoSource "${CMAKE_CURRENT_BINARY_DIR}/${Basename}.pb.cc")
  set(GeneratedProtoHeader "${CMAKE_CURRENT_BINARY_DIR}/${Basename}.pb.h")
  set(Flags
    --cpp_out="${CMAKE_CURRENT_BINARY_DIR}"
    --proto_path="${ProtoSourcePath}")
  if (PROTO_GRPC)
    list(APPEND Flags
      --grpc_out="${CMAKE_CURRENT_BINARY_DIR}"
      --plugin=protoc-gen-grpc="${GRPC_CPP_PLUGIN}")
    list(APPEND GeneratedProtoSource "${CMAKE_CURRENT_BINARY_DIR}/${Basename}.grpc.pb.cc")
    list(APPEND GeneratedProtoHeader "${CMAKE_CURRENT_BINARY_DIR}/${Basename}.grpc.pb.h")
  endif()
  add_custom_command(
        OUTPUT ${GeneratedProtoSource} ${GeneratedProtoHeader}
        COMMAND ${PROTOC}
        ARGS ${Flags} "${ProtoSourceAbsolutePath}"
        DEPENDS "${ProtoSourceAbsolutePath}")

  add_clang_library(${LibraryName} ${GeneratedProtoSource}
    PARTIAL_SOURCES_INTENDED
    LINK_LIBS PUBLIC grpc++ protobuf)

  # Ensure dependency headers are generated before dependent protos are built.
  # DEPENDS arg is a list of "Foo.proto". While they're logically relative to
  # the source dir, the generated headers we need are in the binary dir.
  foreach(ImportedProto IN LISTS PROTO_DEPENDS)
    # Foo.proto -> Foo.pb.h
    STRING(REGEX REPLACE "\\.proto$" ".pb.h" ImportedHeader "${ImportedProto}")
    # Foo.pb.h -> ${CMAKE_CURRENT_BINARY_DIR}/Foo.pb.h
    get_filename_component(ImportedHeader "${ImportedHeader}"
      ABSOLUTE
      BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    # Compilation of each generated source depends on ${BINARY}/Foo.pb.h.
    foreach(Generated IN LISTS GeneratedProtoSource)
      # FIXME: CMake docs suggest OBJECT_DEPENDS isn't needed, but I can't get
      #        the recommended add_dependencies() approach to work.
      set_source_files_properties("${Generated}"
        PROPERTIES OBJECT_DEPENDS "${ImportedHeader}")
    endforeach(Generated)
  endforeach(ImportedProto)
endfunction()
