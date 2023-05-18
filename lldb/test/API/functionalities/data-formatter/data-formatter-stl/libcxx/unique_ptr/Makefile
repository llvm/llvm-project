CXX_SOURCES := main.cpp

USE_LIBCPP := 1

# We need debug info tuning for lldb in order to emit the preferred name for
# std::string. See https://reviews.llvm.org/D145803.
CXXFLAGS_EXTRAS := -std=c++14 -glldb
include Makefile.rules
