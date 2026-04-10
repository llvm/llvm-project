OBJC_SOURCES := main.m

override TRIPLE := arm64e-apple-darwin

# We need an arm64e stdlib.
USE_SYSTEM_STDLIB := 1

LD_EXTRAS := -framework Foundation

include Makefile.rules
