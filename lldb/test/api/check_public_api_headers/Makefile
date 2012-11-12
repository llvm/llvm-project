LEVEL = ../../make

CXX_SOURCES := main.cpp

MY_OS = $(shell uname -s)
ifeq "$(MY_OS)" "Darwin"
    LD_EXTRAS ?= -framework LLDB
else
    LD_EXTRAS ?= $(LLDB_BUILD_DIR)/_lldb.so
endif

# Example dictionary to pass to the Python build method:
# 
# FRAMEWORK_INCLUDES=-F/Volumes/data/lldb/svn/trunk/build/Debug

include $(LEVEL)/Makefile.rules
