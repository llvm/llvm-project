# Copyright 2021 Matt Borland
# Distributed under the Boost Software License, Version 1.0.
# https://www.boost.org/LICENSE_1_0.txt

include(BoostTestJamfile OPTIONAL RESULT_VARIABLE HAVE_BOOST_TEST)

if(HAVE_BOOST_TEST)

    boost_test(SOURCES check_cmake_version.cpp ARGUMENTS ${PROJECT_VERSION} LINK_LIBRARIES Boost::core Boost::config)    

    if (BOOST_MATH_ENABLE_CUDA)

        message(STATUS "Building boost.math with CUDA")

        find_package(CUDA REQUIRED)
        enable_language(CUDA)
        set(CMAKE_CUDA_EXTENSIONS OFF)

        enable_testing()

        boost_test_jamfile(FILE cuda_jamfile LINK_LIBRARIES Boost::math Boost::assert Boost::concept_check Boost::config Boost::core Boost::integer Boost::lexical_cast Boost::multiprecision Boost::predef Boost::random Boost::static_assert Boost::throw_exception Boost::unit_test_framework ${CUDA_LIBRARIES} INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS}  )

    elseif (BOOST_MATH_ENABLE_NVRTC)

        message(STATUS "Building boost.math with NVRTC")

        find_package(CUDA REQUIRED)

        enable_testing()

        set(CUDA_nvrtc_LIBRARY /usr/local/cuda/lib64/libnvrtc.so)
        
        if (BOOST_MATH_NVRTC_CI_RUN)
            boost_test_jamfile(FILE nvrtc_jamfile LINK_LIBRARIES Boost::math Boost::assert Boost::concept_check Boost::config Boost::core Boost::integer Boost::lexical_cast Boost::multiprecision Boost::predef Boost::random Boost::static_assert Boost::throw_exception ${CUDA_nvrtc_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} COMPILE_DEFINITIONS BOOST_MATH_NVRTC_CI_RUN=1 INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS} )
        else ()
            boost_test_jamfile(FILE nvrtc_jamfile LINK_LIBRARIES Boost::math Boost::assert Boost::concept_check Boost::config Boost::core Boost::integer Boost::lexical_cast Boost::multiprecision Boost::predef Boost::random Boost::static_assert Boost::throw_exception ${CUDA_nvrtc_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS} )
        endif()

    elseif (BOOST_MATH_ENABLE_SYCL)

        message(STATUS "Building boost.math with SYCL")

        set(CMAKE_CXX_COMPILER "icpx")
        set(CMAKE_C_COMPILER "icx")

        enable_testing()

        boost_test_jamfile(FILE sycl_jamfile LINK_LIBRARIES Boost::math Boost::assert Boost::concept_check Boost::config Boost::core Boost::integer Boost::lexical_cast Boost::multiprecision Boost::predef Boost::random Boost::static_assert Boost::throw_exception Boost::unit_test_framework sycl COMPILE_OPTIONS -fsycl )
    else()

        add_subdirectory(compile_test)

    endif()

endif()
