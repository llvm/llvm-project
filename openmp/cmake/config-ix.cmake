include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

check_c_compiler_flag(-Werror OPENMP_HAVE_WERROR_FLAG)

check_cxx_compiler_flag(-std=c++11 OPENMP_HAVE_STD_CPP11_FLAG)