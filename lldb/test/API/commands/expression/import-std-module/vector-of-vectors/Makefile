USE_LIBCPP := 1
CXX_SOURCES := main.cpp

ifneq ($(OS),Darwin)
	LD_EXTRAS := -Xlinker --whole-archive -Xlinker --allow-multiple-definition
endif

include Makefile.rules
