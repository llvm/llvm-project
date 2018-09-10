## Process this with automake to create Makefile.in

AUTOMAKE_OPTIONS = foreign subdir-objects

ACLOCAL_AMFLAGS = -I .. -I ../config

SUBDIRS = include testsuite man

EXTRA_DIST = LICENSE ChangeLog.v1 ChangeLog.libgcj			\
	ChangeLog.libffi ChangeLog.libffi-3.1				\
	m4/libtool.m4 m4/lt~obsolete.m4					\
	 m4/ltoptions.m4 m4/ltsugar.m4 m4/ltversion.m4			\
	 m4/ltversion.m4 src/debug.c msvcc.sh				\
	generate-darwin-source-and-headers.py				\
	libffi.xcodeproj/project.pbxproj				\
	libtool-ldflags

# Automake Documentation:
# If your package has Texinfo files in many directories, you can use the
# variable TEXINFO_TEX to tell Automake where to find the canonical
# `texinfo.tex' for your package. The value of this variable should be
# the relative path from the current `Makefile.am' to `texinfo.tex'.
TEXINFO_TEX   = ../gcc/doc/include/texinfo.tex

# Defines info, dvi, pdf and html targets
MAKEINFOFLAGS = -I $(srcdir)/../gcc/doc/include
info_TEXINFOS = doc/libffi.texi

# AM_CONDITIONAL on configure option --generated-files-in-srcdir
if GENINSRC
STAMP_GENINSRC = stamp-geninsrc
else
STAMP_GENINSRC =
endif

# AM_CONDITIONAL on configure check ACX_CHECK_PROG_VER([MAKEINFO])
if BUILD_INFO
STAMP_BUILD_INFO = stamp-build-info
else
STAMP_BUILD_INFO =
endif

all-local: $(STAMP_GENINSRC)

stamp-geninsrc: doc/libffi.info
	cp -p $(top_builddir)/doc/libffi.info $(srcdir)/doc/libffi.info
	@touch $@

doc/libffi.info: $(STAMP_BUILD_INFO)

stamp-build-info: doc/libffi.texi $(srcdir)/doc/version.texi doc/$(am__dirstamp)
	$(MAKEINFO) $(AM_MAKEINFOFLAGS) $(MAKEINFOFLAGS) -I $(srcdir)/doc -o doc/libffi.info $(srcdir)/doc/libffi.texi
	@touch $@

CLEANFILES = $(STAMP_GENINSRC) $(STAMP_BUILD_INFO) doc/libffi.info
MAINTAINERCLEANFILES = $(srcdir)/doc/libffi.info

## ################################################################

##
## This section is for make and multilib madness.
##

# Work around what appears to be a GNU make bug handling MAKEFLAGS
# values defined in terms of make variables, as is the case for CC and
# friends when we are called from the top level Makefile.
AM_MAKEFLAGS = \
	'AR_FLAGS=$(AR_FLAGS)' \
	'CC_FOR_BUILD=$(CC_FOR_BUILD)' \
	'CFLAGS=$(CFLAGS)' \
	'CXXFLAGS=$(CXXFLAGS)' \
	'CFLAGS_FOR_BUILD=$(CFLAGS_FOR_BUILD)' \
	'CFLAGS_FOR_TARGET=$(CFLAGS_FOR_TARGET)' \
	'INSTALL=$(INSTALL)' \
	'INSTALL_DATA=$(INSTALL_DATA)' \
	'INSTALL_PROGRAM=$(INSTALL_PROGRAM)' \
	'INSTALL_SCRIPT=$(INSTALL_SCRIPT)' \
	'JC1FLAGS=$(JC1FLAGS)' \
	'LDFLAGS=$(LDFLAGS)' \
	'LIBCFLAGS=$(LIBCFLAGS)' \
	'LIBCFLAGS_FOR_TARGET=$(LIBCFLAGS_FOR_TARGET)' \
	'MAKE=$(MAKE)' \
	'MAKEINFO=$(MAKEINFO) $(MAKEINFOFLAGS)' \
	'PICFLAG=$(PICFLAG)' \
	'PICFLAG_FOR_TARGET=$(PICFLAG_FOR_TARGET)' \
	'RUNTESTFLAGS=$(RUNTESTFLAGS)' \
	'SHELL=$(SHELL)' \
	'exec_prefix=$(exec_prefix)' \
	'infodir=$(infodir)' \
	'libdir=$(libdir)' \
	'mandir=$(mandir)' \
	'prefix=$(prefix)' \
	'AR=$(AR)' \
	'AS=$(AS)' \
	'CC=$(CC)' \
	'CXX=$(CXX)' \
	'LD=$(LD)' \
	'NM=$(NM)' \
	'RANLIB=$(RANLIB)' \
	'DESTDIR=$(DESTDIR)'

# Subdir rules rely on $(FLAGS_TO_PASS)
FLAGS_TO_PASS = $(AM_MAKEFLAGS)

MAKEOVERRIDES=

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = libffi.pc

toolexeclib_LTLIBRARIES = libffi.la
noinst_LTLIBRARIES = libffi_convenience.la

libffi_la_SOURCES = src/prep_cif.c src/types.c \
		src/raw_api.c src/java_raw_api.c src/closures.c

if FFI_DEBUG
libffi_la_SOURCES += src/debug.c
endif

noinst_HEADERS = \
	src/aarch64/ffitarget.h src/aarch64/internal.h			\
	src/alpha/ffitarget.h src/alpha/internal.h			\
	src/arc/ffitarget.h						\
	src/arm/ffitarget.h src/arm/internal.h				\
	src/avr32/ffitarget.h						\
	src/bfin/ffitarget.h						\
	src/cris/ffitarget.h						\
	src/frv/ffitarget.h						\
	src/ia64/ffitarget.h src/ia64/ia64_flags.h			\
	src/m32r/ffitarget.h						\
	src/m68k/ffitarget.h						\
	src/m88k/ffitarget.h						\
	src/metag/ffitarget.h						\
	src/microblaze/ffitarget.h					\
	src/mips/ffitarget.h						\
	src/moxie/ffitarget.h						\
	src/nios2/ffitarget.h						\
	src/or1k/ffitarget.h						\
	src/pa/ffitarget.h						\
	src/powerpc/ffitarget.h src/powerpc/asm.h src/powerpc/ffi_powerpc.h \
	src/s390/ffitarget.h						\
	src/sh/ffitarget.h						\
	src/sh64/ffitarget.h						\
	src/sparc/ffitarget.h src/sparc/internal.h			\
	src/tile/ffitarget.h						\
	src/vax/ffitarget.h						\
	src/x86/ffitarget.h src/x86/internal.h src/x86/internal64.h	\
	src/xtensa/ffitarget.h						\
	src/dlmalloc.c

EXTRA_libffi_la_SOURCES = \
	src/aarch64/ffi.c src/aarch64/sysv.S				\
	src/alpha/ffi.c src/alpha/osf.S					\
	src/arc/ffi.c src/arc/arcompact.S				\
	src/arm/ffi.c src/arm/sysv.S					\
	src/avr32/ffi.c src/avr32/sysv.S				\
	src/bfin/ffi.c src/bfin/sysv.S					\
	src/cris/ffi.c src/cris/sysv.S					\
	src/frv/ffi.c src/frv/eabi.S					\
	src/ia64/ffi.c src/ia64/unix.S					\
	src/m32r/ffi.c src/m32r/sysv.S					\
	src/m68k/ffi.c src/m68k/sysv.S					\
	src/m88k/ffi.c src/m88k/obsd.S					\
	src/metag/ffi.c src/metag/sysv.S				\
	src/microblaze/ffi.c src/microblaze/sysv.S			\
	src/mips/ffi.c src/mips/o32.S src/mips/n32.S			\
	src/moxie/ffi.c src/moxie/eabi.S				\
	src/nios2/ffi.c src/nios2/sysv.S				\
	src/or1k/ffi.c src/or1k/sysv.S					\
	src/pa/ffi.c src/pa/linux.S src/pa/hpux32.S			\
	src/powerpc/ffi.c src/powerpc/ffi_sysv.c src/powerpc/ffi_linux64.c \
	 src/powerpc/sysv.S src/powerpc/linux64.S			\
	 src/powerpc/linux64_closure.S src/powerpc/ppc_closure.S	\
	 src/powerpc/aix.S src/powerpc/darwin.S src/powerpc/aix_closure.S \
	 src/powerpc/darwin_closure.S src/powerpc/ffi_darwin.c		\
	src/s390/ffi.c src/s390/sysv.S					\
	src/sh/ffi.c src/sh/sysv.S					\
	src/sh64/ffi.c src/sh64/sysv.S					\
	src/sparc/ffi.c src/sparc/ffi64.c src/sparc/v8.S src/sparc/v9.S	\
	src/tile/ffi.c src/tile/tile.S					\
	src/vax/ffi.c src/vax/elfbsd.S					\
	src/x86/ffi.c src/x86/sysv.S					\
	 src/x86/ffiw64.c src/x86/win64.S 				\
	 src/x86/ffi64.c src/x86/unix64.S				\
	 src/x86/darwin64.S src/x86/darwin.S				\
	 src/x86/darwin64_c.c src/x86/darwin_c.c			\
	src/xtensa/ffi.c src/xtensa/sysv.S

TARGET_OBJ = @TARGET_OBJ@
libffi_la_LIBADD = $(TARGET_OBJ)
libffi_la_DEPENDENCIES = $(TARGET_OBJ)

libffi_convenience_la_SOURCES = $(libffi_la_SOURCES)
EXTRA_libffi_convenience_la_SOURCES = $(EXTRA_libffi_la_SOURCES)
libffi_convenience_la_LIBADD = $(libffi_la_LIBADD)
libffi_convenience_la_DEPENDENCIES = $(libffi_la_DEPENDENCIES)
nodist_libffi_convenience_la_SOURCES = $(nodist_libffi_la_SOURCES)

LTLDFLAGS = $(shell $(SHELL) $(top_srcdir)/../libtool-ldflags $(LDFLAGS))

AM_CFLAGS = -Wall -g -fexceptions
if FFI_DEBUG
# Build debug. Define FFI_DEBUG on the commandline so that, when building with
# MSVC, it can link against the debug CRT.
AM_CFLAGS += -DFFI_DEBUG
endif

libffi_la_LDFLAGS = -no-undefined -version-info `grep -v '^\#' $(srcdir)/libtool-version` $(LTLDFLAGS) $(AM_LTLDFLAGS)

AM_CPPFLAGS = -I. -I$(top_srcdir)/include -Iinclude -I$(top_srcdir)/src
AM_CCASFLAGS = $(AM_CPPFLAGS)

# Multilib support.  Automake should provide these on its own.
all-recursive: all-multi
install-recursive: install-multi
mostlyclean-recursive: mostlyclean-multi
clean-recursive: clean-multi
distclean-recursive: distclean-multi
maintainer-clean-recursive: maintainer-clean-multi
