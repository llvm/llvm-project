cmake_minimum_required(VERSION 3.16)

project(nsapi_fort)

# Manually call nextfort to make NSAPI fortran wrapper library
execute_process(
  COMMAND ${CMAKE_FORTDIR}/nextfort -shared -o libnsapi_fort.so ${CMAKE_CURRENT_SOURCE_DIR}/telem.f90
  OUTPUT_VARIABLE fortOut
)

# Install fortran library and module "header" file
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libnsapi_fort.so"
  DESTINATION "lib")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/nsapi.mod"
  DESTINATION "include")
