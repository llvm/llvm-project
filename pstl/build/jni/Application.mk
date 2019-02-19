#===-- Application.mk ----------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##


ifndef os_name
  # Windows sets environment variable OS; for other systems, ask uname
  ifeq ($(OS),)
    OS:=$(shell uname)
    ifeq ($(OS),)
      $(error "Cannot detect operating system")
    endif
    export os_name=$(OS)
  endif

  ifeq ($(OS), Windows_NT)
    export os_name=windows
  endif
  ifeq ($(OS), Linux)
    export os_name=linux
  endif
  ifeq ($(OS), Darwin)
    export os_name=macos
  endif
endif

export compiler?=clang
export arch?=ia32
export target?=android

ifeq (ia32,$(arch))
    APP_ABI:=x86
    export TRIPLE:=i686-linux-android
else ifeq (intel64,$(arch))
    APP_ABI:=x86_64
    export TRIPLE:=x86_64-linux-android
else ifeq (arm,$(arch))
    APP_ABI:=armeabi-v7a
    export TRIPLE:=arm-linux-androideabi
else ifeq (arm64,$(arch))
    APP_ABI:=arm64-v8a
    export TRIPLE:=aarch64-linux-android
else
    APP_ABI:=$(arch)
endif

api_version?=21
export API_LEVEL:=$(api_version)
APP_PLATFORM:=android-$(api_version)

ifeq (clang,$(compiler))
    NDK_TOOLCHAIN_VERSION:=clang
    APP_STL:=c++_shared
else
    NDK_TOOLCHAIN_VERSION:=4.9
endif
