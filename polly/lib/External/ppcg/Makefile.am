if BUNDLED_ISL
    MAYBE_ISL = isl
    ISL_LA = $(top_builddir)/isl/libisl.la
    LOCAL_ISL_LA = isl/libisl.la
endif
if BUNDLED_PET
    MAYBE_PET = pet
    PET_LA = $(top_builddir)/pet/libpet.la
endif

SUBDIRS = $(MAYBE_ISL) $(MAYBE_PET) .

FORCE:
isl/libisl.la: FORCE
	cd isl; $(MAKE) $(AM_MAKEFLAGS) libisl.la
pet/libpet.la: FORCE
	cd pet; $(MAKE) $(AM_MAKEFLAGS) libpet.la

ACLOCAL_AMFLAGS = -I m4

LIB_ISL = $(ISL_LA) @ISL_LIBS@
LIB_PET = $(PET_LA) @PET_LIBS@

AM_CPPFLAGS = @ISL_CFLAGS@ @PET_CFLAGS@
LDADD = $(LIB_PET) $(LIB_ISL)

bin_PROGRAMS = ppcg
ppcg_SOURCES = \
	cpu.c \
	cpu.h \
	cuda.c \
	cuda.h \
	opencl.c \
	opencl.h \
	cuda_common.h \
	cuda_common.c \
	gpu.c \
	gpu.h \
	gpu_array_tile.c \
	gpu_array_tile.h \
	gpu_group.c \
	gpu_group.h \
	gpu_hybrid.c \
	gpu_hybrid.h \
	gpu_print.c \
	gpu_print.h \
	gpu_tree.c \
	gpu_tree.h \
	grouping.c \
	hybrid.c \
	hybrid.h \
	schedule.c \
	schedule.h \
	ppcg_options.c \
	ppcg_options.h \
	ppcg.c \
	ppcg.h \
	print.c \
	print.h \
	util.c \
	util.h \
	version.c

TESTS = @extra_tests@
EXTRA_TESTS = opencl_test.sh polybench_test.sh
TEST_EXTENSIONS = .sh

EXTRA_DIST = \
	ocl_utilities.c \
	ocl_utilities.h \
	tests

dist-hook:
	echo @GIT_HEAD_VERSION@ > $(distdir)/GIT_HEAD_ID

gitversion.h: @GIT_HEAD@
	$(AM_V_GEN)echo '#define GIT_HEAD_ID "'@GIT_HEAD_VERSION@'"' > $@
