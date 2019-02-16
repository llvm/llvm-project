#===-- Android.mk --------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.TXT for details.
#
#===----------------------------------------------------------------------===##

export proj_root?=$(NDK_PROJECT_PATH)/..

ifeq (armeabi-v7a,$(APP_ABI))
	export SYSROOT:=$(NDK_ROOT)/platforms/$(APP_PLATFORM)/arch-arm
else ifeq (arm64-v8a,$(APP_ABI))
	export SYSROOT:=$(NDK_ROOT)/platforms/$(APP_PLATFORM)/arch-arm64
else
	export SYSROOT:=$(NDK_ROOT)/platforms/$(APP_PLATFORM)/arch-$(APP_ABI)
endif

ifeq (windows,$(os_name))
	export CPATH_SEPARATOR :=;
else
	export CPATH_SEPARATOR :=:
endif

export ANDROID_NDK_ROOT:=$(NDK_ROOT)
export ndk_version:=$(lastword $(subst -, ,$(ANDROID_NDK_ROOT)))
ndk_version:= $(firstword $(subst /, ,$(ndk_version)))
ndk_version:= $(firstword $(subst \, ,$(ndk_version)))

ifeq (clang,$(compiler))
	# "TBB_RTL :=llvm-libc++/libcxx" should be used for ndk_version r13 r13b r14.
	TBB_RTL :=llvm-libc++
	TBB_RTL_LIB :=llvm-libc++
	TBB_RTL_FILE :=libc++_shared.so
else
	TBB_RTL :=gnu-libstdc++/$(NDK_TOOLCHAIN_VERSION)
	TBB_RTL_LIB :=$(TBB_RTL)
	TBB_RTL_FILE :=libgnustl_shared.so
endif

export CPATH := $(INCLUDE)$(CPATH_SEPARATOR)$(SYSROOT)/usr/include$(CPATH_SEPARATOR)$(NDK_ROOT)/sources/cxx-stl/$(TBB_RTL)/include$(CPATH_SEPARATOR)$(NDK_ROOT)/sources/cxx-stl/$(TBB_RTL)/libs/$(APP_ABI)/include$(CPATH_SEPARATOR)$(NDK_ROOT)/sources/android/support/include

LIB_STL_ANDROID_DIR := $(NDK_ROOT)/sources/cxx-stl/$(TBB_RTL_LIB)/libs/$(APP_ABI)
#LIB_STL_ANDROID is required to be set up for copying Android specific library to a device next to test
export LIB_STL_ANDROID := $(LIB_STL_ANDROID_DIR)/$(TBB_RTL_FILE)
export CPLUS_LIB_PATH := $(SYSROOT)/usr/lib -L$(LIB_STL_ANDROID_DIR)
export target_os_version:=$(APP_PLATFORM)
export tbb_tool_prefix:=$(TOOLCHAIN_PREFIX)
export TARGET_CXX
export TARGET_CC
export TARGET_CFLAGS

include $(proj_root)/build/Makefile
