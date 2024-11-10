USE_LIBCPP := 1
CXX_SOURCES := main.cpp

ifneq ($(OS),Darwin)
	LDFLAGS := -Xlinker --push-state -Xlinker --whole-archive -lc++ -lc++abi -Xlinker --pop-state
endif

include Makefile.rules
