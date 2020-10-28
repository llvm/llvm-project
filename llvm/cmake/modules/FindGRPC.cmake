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
        find_library(GRPC_LIBRARY
                     grpc++
                     PATHS ${GRPC_HOMEBREW_PATH}/lib
                     NO_DEFAULT_PATH
                     REQUIRED)
        add_library(grpc++ UNKNOWN IMPORTED GLOBAL)
        set_target_properties(grpc++ PROPERTIES
                              IMPORTED_LOCATION ${GRPC_LIBRARY})
      endif()
      if (PROTOBUF_HOMEBREW_RETURN_CODE EQUAL "0")
        include_directories(${PROTOBUF_HOMEBREW_PATH}/include)
        find_library(PROTOBUF_LIBRARY
                     protobuf
                     PATHS ${PROTOBUF_HOMEBREW_PATH}/lib
                     NO_DEFAULT_PATH
                     REQUIRED)
        add_library(protobuf UNKNOWN IMPORTED GLOBAL)
        set_target_properties(protobuf PROPERTIES
                              IMPORTED_LOCATION ${PROTOBUF_LIBRARY})
      endif()
    endif()
  endif()
endif()

# Proto headers are generated in ${CMAKE_CURRENT_BINARY_DIR}.
# Libraries that use these headers should adjust the include path.
# FIXME(kirillbobyrev): Allow optional generation of gRPC code and give callers
# control over it via additional parameters.
function(generate_grpc_protos LibraryName ProtoFile)
  get_filename_component(ProtoSourceAbsolutePath "${CMAKE_CURRENT_SOURCE_DIR}/${ProtoFile}" ABSOLUTE)
  get_filename_component(ProtoSourcePath ${ProtoSourceAbsolutePath} PATH)

  set(GeneratedProtoSource "${CMAKE_CURRENT_BINARY_DIR}/Index.pb.cc")
  set(GeneratedProtoHeader "${CMAKE_CURRENT_BINARY_DIR}/Index.pb.h")
  set(GeneratedGRPCSource "${CMAKE_CURRENT_BINARY_DIR}/Index.grpc.pb.cc")
  set(GeneratedGRPCHeader "${CMAKE_CURRENT_BINARY_DIR}/Index.grpc.pb.h")
  add_custom_command(
        OUTPUT "${GeneratedProtoSource}" "${GeneratedProtoHeader}" "${GeneratedGRPCSource}" "${GeneratedGRPCHeader}"
        COMMAND ${PROTOC}
        ARGS --grpc_out="${CMAKE_CURRENT_BINARY_DIR}"
          --cpp_out="${CMAKE_CURRENT_BINARY_DIR}"
          --proto_path="${ProtoSourcePath}"
          --plugin=protoc-gen-grpc="${GRPC_CPP_PLUGIN}"
          "${ProtoSourceAbsolutePath}"
          DEPENDS "${ProtoSourceAbsolutePath}")

  add_clang_library(${LibraryName} ${GeneratedProtoSource} ${GeneratedGRPCSource}
    PARTIAL_SOURCES_INTENDED
    LINK_LIBS grpc++ protobuf)
endfunction()
