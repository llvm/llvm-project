SWIFT_SOURCES := main.swift
SWIFTFLAGS_EXTRAS = -L$(BUILDDIR) -I$(BUILDDIR) -I$(SRCDIR) -lWrapper
LD_EXTRAS = -Xlinker -rpath -Xlinker $(BUILDDIR)

all: sdkroot.txt libWrapper.dylib a.out

include Makefile.rules

libCppLib.dylib: CppLib/CppLib.cpp \
                 $(call tool_prereq,$(CXX_PATH))
	$(call MKDIR_P,CppLib)
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_NAME=CppLib DYLIB_ONLY=YES CXXFLAGS_EXTRAS=-fPIC \
		DYLIB_CXX_SOURCES="CppLib/CppLib.cpp CppLib/CppLibWrapper.cpp"
	install_name_tool -id $@ $@
ifneq "$(CODESIGN)" ""
	$(CODESIGN) -f -s - "$@"
endif

libWrapper.dylib: libCppLib.dylib \
                  $(call tool_prereq,$(SWIFTC_PATH)) $(call tool_prereq,$(SWIFT_FRONTEND_PATH))
	"$(MAKE)" -f $(MAKEFILE_RULES) \
		DYLIB_NAME=Wrapper DYLIB_SWIFT_SOURCES=wrapper.swift \
		LD_EXTRAS="$(LD_EXTRAS)" \
		SWIFTFLAGS_EXTRAS="-I$(SRCDIR) -L. -lCppLib -module-link-name Wrapper"

# Make sure Makefile and test agree on the SDK.
sdkroot.txt:
	echo $(SDKROOT) > $@
