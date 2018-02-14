cmake_minimum_required(VERSION 2.8)
project(DetectTestCompiler C CXX)

function(write_compiler_information lang)
  set(information "${CMAKE_${lang}_COMPILER}")
  set(information "${information}\\;${CMAKE_${lang}_COMPILER_ID}")
  set(information "${information}\\;${CMAKE_${lang}_COMPILER_VERSION}")
  set(information "${information}\\;${OpenMP_${lang}_FLAGS}")
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${lang}CompilerInformation.txt ${information})
endfunction(write_compiler_information)

find_package(OpenMP)
if (NOT OpenMP_Found)
  set(OpenMP_C_FLAGS "-fopenmp")
  set(OpenMP_CXX_FLAGS "-fopenmp")
endif()

write_compiler_information(C)
write_compiler_information(CXX)
