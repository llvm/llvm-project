##===- source/Plugins/Platform/MacOSX/Makefile -------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

LLDB_LEVEL := ../../../..
LEVEL := $(LLDB_LEVEL)/../..

include $(LEVEL)/Makefile.config

SOURCES += PlatformDarwin.cpp \
		   PlatformDarwinKernel.cpp \
		   PlatformMacOSX.cpp \
		   PlatformRemoteiOS.cpp \
		   PlatformRemoteAppleTV.cpp \
		   PlatformRemoteAppleWatch.cpp

ifeq ($(HOST_OS),Darwin)
SOURCES += PlatformAppleSimulator.cpp \
		   PlatformiOSSimulator.cpp \
		   PlatformiOSSimulatorCoreSimulatorSupport.mm \
		   PlatformAppleTVSimulator.cpp \
		   PlatformAppleWatchSimulator.cpp
endif

LIBRARYNAME := lldbPluginPlatformMacOSX
BUILD_ARCHIVE = 1

include $(LLDB_LEVEL)/Makefile

