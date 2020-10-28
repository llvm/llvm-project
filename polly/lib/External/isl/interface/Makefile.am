ACLOCAL_AMFLAGS = -I ../m4
AUTOMAKE_OPTIONS = nostdinc

noinst_PROGRAMS = extract_interface

includes = -I$(top_builddir) -I$(top_srcdir) \
	-I$(top_builddir)/include -I$(top_srcdir)/include

extract_interface_CPPFLAGS = $(includes)
extract_interface_CXXFLAGS = $(CLANG_CXXFLAGS)
extract_interface_SOURCES = \
	generator.h \
	generator.cc \
	python.h \
	python.cc \
	cpp.h \
	cpp.cc \
	cpp_conversion.h \
	cpp_conversion.cc \
	extract_interface.h \
	extract_interface.cc
extract_interface_LDFLAGS = $(CLANG_LDFLAGS) $(CLANG_RFLAG)
extract_interface_LDADD = $(CLANG_LIBS) $(CLANG_LDFLAGS)
