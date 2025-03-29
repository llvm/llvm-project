CXX_SOURCES := main.cpp
CFLAGS_EXTRAS := -std=c++20

ifeq "1" "$(USE_LIBSTDCPP)"
  CFLAGS_EXTRAS += -DUSE_LIBSTDCPP
endif

ifeq "1" "$(USE_LIBCPP)"
  CFLAGS_EXTRAS += -DUSE_LIBCPP
endif

include Makefile.rules
