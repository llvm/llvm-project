##===- cmake/modules/Makefile ------------------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LEVEL = ../..

LINK_COMPONENTS := all

include $(LEVEL)/Makefile.common

PROJ_cmake := $(DESTDIR)$(PROJ_prefix)/share/llvm/cmake

ifeq ($(DISABLE_ASSERTIONS),1)
	LLVM_ENABLE_ASSERTIONS := 0
else
	LLVM_ENABLE_ASSERTIONS := 1
endif

ifeq ($(REQUIRES_EH),1)
	LLVM_ENABLE_EH := 1
else
	LLVM_ENABLE_EH := 0
endif

ifeq ($(REQUIRES_RTTI),1)
	LLVM_ENABLE_RTTI := 1
else
	LLVM_ENABLE_RTTI := 0
endif

# Don't try to run llvm-config during clean because it won't be available
ifneq ($(MAKECMDGOALS),clean)
LLVM_LIBS_TO_EXPORT := $(subst -l,,$(shell $(LLVM_CONFIG) --libs $(LINK_COMPONENTS) || echo Error))

ifeq ($(LLVM_LIBS_TO_EXPORT),Error)
$(error llvm-config --libs failed)
endif

ifndef LLVM_LIBS_TO_EXPORT
$(error LLVM_LIBS_TO_EXPORT cannot be empty)
endif
endif

OBJMODS := LLVMConfig.cmake LLVMConfigVersion.cmake LLVMExports.cmake

$(PROJ_OBJ_DIR)/LLVMConfig.cmake: LLVMConfig.cmake.in $(LLVMBuildCMakeFrag)
	$(Echo) 'Generating LLVM CMake package config file'
	$(Verb) ( \
	 cat $< | sed \
	  -e 's/@LLVM_CONFIG_CODE@/set(LLVM_INSTALL_PREFIX "'"$(subst /,\/,$(PROJ_prefix))"'")/' \
	  -e 's/@LLVM_VERSION_MAJOR@/'"$(LLVM_VERSION_MAJOR)"'/' \
	  -e 's/@LLVM_VERSION_MINOR@/'"$(LLVM_VERSION_MINOR)"'/' \
	  -e 's/@LLVM_VERSION_PATCH@/'"$(LLVM_VERSION_PATCH)"'/' \
	  -e 's/@PACKAGE_VERSION@/'"$(LLVMVersion)"'/' \
	  -e 's/@LLVM_COMMON_DEPENDS@//' \
	  -e 's/@LLVM_AVAILABLE_LIBS@/'"$(LLVM_LIBS_TO_EXPORT)"'/' \
	  -e 's/@LLVM_ALL_TARGETS@/'"$(ALL_TARGETS)"'/' \
	  -e 's/@LLVM_TARGETS_TO_BUILD@/'"$(TARGETS_TO_BUILD)"'/' \
	  -e 's/@LLVM_TARGETS_WITH_JIT@/'"$(TARGETS_WITH_JIT)"'/' \
	  -e 's/@TARGET_TRIPLE@/'"$(TARGET_TRIPLE)"'/' \
	  -e 's/@LLVM_ENABLE_ASSERTIONS@/'"$(LLVM_ENABLE_ASSERTIONS)"'/' \
	  -e 's/@LLVM_ENABLE_EH@/'"$(LLVM_ENABLE_EH)"'/' \
	  -e 's/@LLVM_ENABLE_RTTI@/'"$(LLVM_ENABLE_RTTI)"'/' \
	  -e 's/@LLVM_ENABLE_TERMINFO@/'"$(ENABLE_TERMINFO)"'/' \
	  -e 's/@LLVM_ENABLE_THREADS@/'"$(ENABLE_THREADS)"'/' \
	  -e 's/@LLVM_ENABLE_ZLIB@/'"$(ENABLE_ZLIB)"'/' \
	  -e 's/@LLVM_NATIVE_ARCH@/'"$(LLVM_NATIVE_ARCH)"'/' \
	  -e 's/@LLVM_ENABLE_PIC@/'"$(ENABLE_PIC)"'/' \
	  -e 's/@LLVM_ON_UNIX@/'"$(LLVM_ON_UNIX)"'/' \
	  -e 's/@LLVM_ON_WIN32@/'"$(LLVM_ON_WIN32)"'/' \
	  -e 's/@LLVM_CONFIG_INCLUDE_DIRS@/'"$(subst /,\/,$(PROJ_includedir))"'/' \
	  -e 's/@LLVM_CONFIG_LIBRARY_DIRS@/'"$(subst /,\/,$(PROJ_libdir))"'/' \
	  -e 's/@LLVM_CONFIG_CMAKE_DIR@/'"$(subst /,\/,$(PROJ_cmake))"'/' \
	  -e 's/@LLVM_CONFIG_TOOLS_BINARY_DIR@/'"$(subst /,\/,$(PROJ_bindir))"'/' \
	  -e 's/@LLVM_CONFIG_EXPORTS_FILE@/$${LLVM_CMAKE_DIR}\/LLVMExports.cmake/' \
	  -e 's/@all_llvm_lib_deps@//' \
	 && \
	 grep '^set_property.*LLVMBUILD_LIB_DEPS_' "$(LLVMBuildCMakeFrag)" \
	) > $@

$(PROJ_OBJ_DIR)/LLVMConfigVersion.cmake: LLVMConfigVersion.cmake.in
	$(Echo) 'Generating LLVM CMake package version file'
	$(Verb) cat $< | sed \
	  -e 's/@PACKAGE_VERSION@/'"$(LLVMVersion)"'/' \
	  -e 's/@LLVM_VERSION_MAJOR@/'"$(LLVM_VERSION_MAJOR)"'/' \
	  -e 's/@LLVM_VERSION_MINOR@/'"$(LLVM_VERSION_MINOR)"'/' \
	  -e 's/@LLVM_VERSION_PATCH@/'"$(LLVM_VERSION_PATCH)"'/' \
	  > $@

$(PROJ_OBJ_DIR)/LLVMExports.cmake: $(LLVMBuildCMakeExportsFrag)
	$(Echo) 'Generating LLVM CMake target exports file'
	$(Verb) ( \
	  echo '# LLVM CMake target exports.  Do not include directly.' && \
	  for lib in $(LLVM_LIBS_TO_EXPORT); do \
	    echo 'add_library('"$$lib"' STATIC IMPORTED)' && \
	    echo 'set_property(TARGET '"$$lib"' PROPERTY IMPORTED_LOCATION "'"$(PROJ_libdir)/lib$$lib.a"'")' ; \
	  done && \
	  cat "$(LLVMBuildCMakeExportsFrag)" && \
	  echo 'set_property(TARGET LLVMSupport APPEND PROPERTY IMPORTED_LINK_INTERFACE_LIBRARIES '"$(subst -l,,$(LIBS))"')' \
	) | grep -v gtest > $@

all-local:: $(addprefix $(PROJ_OBJ_DIR)/, $(OBJMODS))

SKIPSRCMODS := \
  CheckAtomic.cmake \
  GetHostTriple.cmake \
  LLVMBuildExports.cmake \
  LLVMConfig.cmake \
  LLVMConfigVersion.cmake \
  LLVMExports.cmake \
  VersionFromVCS.cmake

SRCMODS := $(notdir $(wildcard $(PROJ_SRC_DIR)/*.cmake))
SRCMODS := $(filter-out $(SKIPSRCMODS),$(SRCMODS))
INSTSRCMODS := $(addprefix $(PROJ_cmake)/, $(SRCMODS))
INSTOBJMODS := $(addprefix $(PROJ_cmake)/, $(OBJMODS))

$(PROJ_cmake):
	$(Echo) Making install directory: $@
	$(Verb) $(MKDIR) $@

$(INSTSRCMODS): $(PROJ_cmake)/%.cmake: $(PROJ_SRC_DIR)/%.cmake | $(PROJ_cmake)
	$(Echo) Installing cmake modules: $(notdir $<)
	$(Verb) $(DataInstall) $< $(PROJ_cmake)

$(INSTOBJMODS): $(PROJ_cmake)/%.cmake: $(PROJ_OBJ_DIR)/%.cmake | $(PROJ_cmake)
	$(Echo) Installing cmake modules: $(notdir $<)
	$(Verb) $(DataInstall) $< $(PROJ_cmake)

install-local:: $(INSTSRCMODS) $(INSTOBJMODS)
