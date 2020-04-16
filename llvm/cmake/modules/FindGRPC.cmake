# This setup requires gRPC to be built from sources using CMake and installed to
# ${GRPC_INSTALL_PATH} via -DCMAKE_INSTALL_PREFIX=${GRPC_INSTALL_PATH}.
if (GRPC_INSTALL_PATH)
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
  find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)
  find_program(PROTOC protoc)
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

  add_library(${LibraryName} ${GeneratedProtoSource} ${GeneratedGRPCSource})
  target_link_libraries(${LibraryName} grpc++ protobuf)
endfunction()
