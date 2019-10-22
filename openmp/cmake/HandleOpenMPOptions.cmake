if (${OPENMP_STANDALONE_BUILD})
  # From HandleLLVMOptions.cmake
  function(append_if condition value)
    if (${condition})
      foreach(variable ${ARGN})
        set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
      endforeach(variable)
    endif()
  endfunction()
endif()

if (${OPENMP_ENABLE_WERROR})
  append_if(OPENMP_HAVE_WERROR_FLAG "-Werror" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
endif()

append_if(OPENMP_HAVE_STD_GNUPP11_FLAG "-std=gnu++11" CMAKE_CXX_FLAGS)
if (NOT OPENMP_HAVE_STD_GNUPP11_FLAG)
  append_if(OPENMP_HAVE_STD_CPP11_FLAG "-std=c++11" CMAKE_CXX_FLAGS)
endif()
