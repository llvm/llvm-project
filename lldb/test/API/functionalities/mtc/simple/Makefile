OBJC_SOURCES := main.m
UI_FRAMEWORK = AppKit

ifneq ("$(SDKROOT)", "")
	ifeq (,$(findstring macOS,$(SDKROOT)))
		ifeq (,$(findstring MacOS,$(SDKROOT)))
			UI_FRAMEWORK = UIKit
		endif
	endif
endif

LD_EXTRAS = -lobjc -framework Foundation -framework $(UI_FRAMEWORK)

include Makefile.rules
