OBJC_SOURCES := main.m

include Makefile.rules

ifneq (,$(findstring arm,$(ARCH)))
    LD_EXTRAS = -framework Foundation -framework UIKit
else
    LD_EXTRAS = -framework Foundation -framework Cocoa
endif
