C_SOURCES := main.c
DYLIB_CXX_SOURCES := unrelated.cpp
DYLIB_NAME := Unrelated

ifeq ($(RUNTIME_LOCATION),main)
C_SOURCES += runtime.c
else ifeq ($(RUNTIME_LOCATION),dylib)
DYLIB_C_SOURCES := library_runtime.c
else ifeq ($(RUNTIME_LOCATION),cxx)
DYLIB_CXX_SOURCES += cxx_runtime.cpp
else ifeq ($(RUNTIME_LOCATION),both)
C_SOURCES += runtime.c
DYLIB_CXX_SOURCES += cxx_runtime.cpp
endif

include Makefile.rules
